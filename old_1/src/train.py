import os
import random
import json
from typing import Dict, List
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import HandGestureDataset, check_image_mask_resolutions, SegAugment
from mambavision import MambaVision
from cnn_model_1 import YOLO11_ALL, YOLO11_v2


GESTURE_NAMES = [
	"call",
	"dislike",
	"like",
	"ok",
	"one",
	"palm",
	"peace",
	"rock",
	"stop",
	"three",
]


def set_seed(seed: int = 42) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def to_classification_logits(model_output: torch.Tensor | list | tuple, num_classes: int) -> torch.Tensor:
	"""
	Convert different model output formats into classification logits with shape [B, num_classes].

	This supports:
	- standard classifier output: Tensor[B, C]
	- dense predictions: Tensor[B, C, A] or Tensor[B, C, H, W]
	- multi-scale outputs: list/tuple of dense prediction tensors
	"""
	if isinstance(model_output, torch.Tensor):
		if model_output.ndim == 2:
			if model_output.shape[1] != num_classes:
				raise ValueError(
					f"Expected logits with {num_classes} classes, got shape {tuple(model_output.shape)}"
				)
			return model_output

		if model_output.ndim == 3:
			# [B, C, A] -> aggregate over anchors/locations
			return model_output[:, -num_classes:, :].amax(dim=-1)

		if model_output.ndim == 4:
			# [B, C, H, W] -> aggregate over spatial dims
			return model_output[:, -num_classes:, :, :].amax(dim=(-1, -2))

		raise ValueError(f"Unsupported tensor output shape: {tuple(model_output.shape)}")

	if isinstance(model_output, (list, tuple)):
		if len(model_output) == 0:
			raise ValueError("Received empty model output list/tuple")

		# For multi-head outputs (e.g., YOLO11_ALL), first element is class logits [B, C].
		if isinstance(model_output[0], torch.Tensor) and model_output[0].ndim == 2:
			logits = model_output[0]
			if logits.shape[1] != num_classes:
				raise ValueError(
					f"Expected logits with {num_classes} classes, got shape {tuple(logits.shape)}"
				)
			return logits

		scale_logits: list[torch.Tensor] = []
		for out in model_output:
			if not isinstance(out, torch.Tensor):
				raise TypeError(f"Unsupported output type in list/tuple: {type(out)}")
			if out.ndim == 4:
				scale_logits.append(out[:, -num_classes:, :, :].amax(dim=(-1, -2)))
			elif out.ndim == 3:
				scale_logits.append(out[:, -num_classes:, :].amax(dim=-1))
			else:
				raise ValueError(f"Unsupported output tensor in list/tuple: shape={tuple(out.shape)}")

		# Average logits from all scales
		return torch.stack(scale_logits, dim=0).mean(dim=0)

	raise TypeError(f"Unsupported model output type: {type(model_output)}")


def compute_detection_losses(
	boxes_cxcywh: torch.Tensor,
	det_probs: torch.Tensor,
	class_ids: torch.Tensor,
	gt_boxes: torch.Tensor,
	image_h: int,
	image_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	Compute simple single-object detection losses for YOLO11_ALL.

	- Select one predicted anchor per image based on highest probability for the GT class.
	- Regress that anchor's (cx, cy, w, h) to the GT box with SmoothL1.
	- Encourage confidence on GT class for that selected anchor with BCE.
	"""
	batch_size = boxes_cxcywh.size(0)
	idx = torch.arange(batch_size, device=boxes_cxcywh.device)

	# Probability of ground-truth class for each anchor: [B, N]
	class_index = class_ids.view(batch_size, 1, 1).expand(-1, det_probs.size(1), 1)
	gt_class_probs = det_probs.gather(dim=2, index=class_index).squeeze(-1)

	# Best anchor index per image
	best_anchor_idx = gt_class_probs.argmax(dim=1)

	selected_boxes = boxes_cxcywh[idx, best_anchor_idx]  # [B, 4]
	selected_class_probs = gt_class_probs[idx, best_anchor_idx].clamp(min=1e-6, max=1.0 - 1e-6)

	# Normalize (cx, cy, w, h) by image size to keep bbox loss on a stable scale.
	norm = torch.tensor([image_w, image_h, image_w, image_h], device=gt_boxes.device, dtype=gt_boxes.dtype)
	selected_boxes_norm = selected_boxes / norm
	gt_boxes_norm = gt_boxes / norm
	loss_bbox = F.smooth_l1_loss(selected_boxes_norm, gt_boxes_norm, reduction="mean")
	loss_det_cls = F.binary_cross_entropy(selected_class_probs, torch.ones_like(selected_class_probs))
	return loss_bbox, loss_det_cls


def run_one_epoch(
	model: torch.nn.Module,
	dataloader: DataLoader,
	criterion_classification: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	num_classes: int,
	lambda_bbox: float = 2.0,
	lambda_det_cls: float = 0.05,
	grad_clip_norm: float = 1.0,
) -> Dict[str, float]:
	model.train()
	running_loss = 0.0
	running_cls_loss = 0.0
	running_bbox_loss = 0.0
	running_det_cls_loss = 0.0
	correct = 0
	total = 0

	for images, _masks, class_ids, boxes in dataloader:
		images = images.to(device)
		class_ids = class_ids.to(device)
		boxes = boxes.to(device=device, dtype=torch.float32)

		optimizer.zero_grad(set_to_none=True)
		class_logits, pred_boxes, det_probs, _det_class_ids, _det_scores = model(images)

		loss_cls = criterion_classification(class_logits, class_ids)
		loss_bbox, loss_det_cls = compute_detection_losses(
			boxes_cxcywh=pred_boxes,
			det_probs=det_probs,
			class_ids=class_ids,
			gt_boxes=boxes,
			image_h=images.shape[-2],
			image_w=images.shape[-1],
		)
		loss = loss_cls + lambda_bbox * loss_bbox + lambda_det_cls * loss_det_cls
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
		optimizer.step()

		running_loss += loss.item() * images.size(0)
		running_cls_loss += loss_cls.item() * images.size(0)
		running_bbox_loss += loss_bbox.item() * images.size(0)
		running_det_cls_loss += loss_det_cls.item() * images.size(0)

		preds = class_logits.argmax(dim=1)
		correct += (preds == class_ids).sum().item()
		total += images.size(0)

	return {
		"loss": running_loss / max(total, 1),
		"cls_loss": running_cls_loss / max(total, 1),
		"bbox_loss": running_bbox_loss / max(total, 1),
		"det_cls_loss": running_det_cls_loss / max(total, 1),
		"acc": correct / max(total, 1),
	}


@torch.no_grad()
def evaluate(
	model: torch.nn.Module,
	dataloader: DataLoader,
	criterion_classification: torch.nn.Module,
	device: torch.device,
	num_classes: int,
	lambda_bbox: float = 2.0,
	lambda_det_cls: float = 0.05,
) -> Dict[str, float]:
	model.eval()
	running_loss = 0.0
	running_cls_loss = 0.0
	running_bbox_loss = 0.0
	running_det_cls_loss = 0.0
	correct = 0
	total = 0

	for images, _masks, class_ids, boxes in dataloader:
		images = images.to(device)
		class_ids = class_ids.to(device)
		boxes = boxes.to(device=device, dtype=torch.float32)

		class_logits, pred_boxes, det_probs, _det_class_ids, _det_scores = model(images)

		loss_cls = criterion_classification(class_logits, class_ids)
		loss_bbox, loss_det_cls = compute_detection_losses(
			boxes_cxcywh=pred_boxes,
			det_probs=det_probs,
			class_ids=class_ids,
			gt_boxes=boxes,
			image_h=images.shape[-2],
			image_w=images.shape[-1],
		)
		loss = loss_cls + lambda_bbox * loss_bbox + lambda_det_cls * loss_det_cls

		running_loss += loss.item() * images.size(0)
		running_cls_loss += loss_cls.item() * images.size(0)
		running_bbox_loss += loss_bbox.item() * images.size(0)
		running_det_cls_loss += loss_det_cls.item() * images.size(0)

		preds = class_logits.argmax(dim=1)
		correct += (preds == class_ids).sum().item()
		total += images.size(0)

	return {
		"loss": running_loss / max(total, 1),
		"cls_loss": running_cls_loss / max(total, 1),
		"bbox_loss": running_bbox_loss / max(total, 1),
		"det_cls_loss": running_det_cls_loss / max(total, 1),
		"acc": correct / max(total, 1),
	}


@torch.no_grad()
def visualize_predictions(
	model: torch.nn.Module,
	dataloader: DataLoader,
	device: torch.device,
	class_names: List[str],
	num_samples: int = 8,
	save_path: str | None = None,
) -> None:
	model.eval()
	# NOTE:
	# - If the dataset returns ImageNet-normalized tensors, we need to *undo* that for display.
	# - In this project, the pre-saved `image_tensors/*.pt` are RGB floats in [0, 1]
	#   (see `utils.image_to_tensor_dataset()`), so blindly undoing ImageNet normalization
	#   causes a yellow/warm color shift.
	def _to_display_rgb(img_chw: torch.Tensor) -> torch.Tensor:
		"""Convert a CHW float tensor to a displayable RGB tensor in [0, 1]."""
		img = img_chw.detach().cpu()
		if img.ndim != 3 or img.shape[0] != 3:
			raise ValueError(f"Expected image tensor shape [3,H,W], got {tuple(img.shape)}")

		# Heuristics:
		# - ImageNet-normalized tensors typically contain negative values.
		# - Some pipelines keep images as 0..255 floats.
		img_min = float(img.min().item())
		img_max = float(img.max().item())
		if img_min < -0.1:
			# Likely ImageNet normalization: x_norm = (x - mean) / std
			imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype).view(3, 1, 1)
			imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype).view(3, 1, 1)
			img = img * imagenet_std + imagenet_mean
		elif img_max > 1.1:
			# Probably 0..255 (or otherwise unnormalized) range.
			if img_max <= 255.0:
				img = img / 255.0

		return img.clamp(0.0, 1.0)

	images_shown = 0
	cols = 4
	rows = int(np.ceil(num_samples / cols))
	fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
	axes = np.array(axes).reshape(-1)

	for images, _masks, class_ids, _boxes in dataloader:
		images = images.to(device)
		class_ids = class_ids.to(device)
		logits = to_classification_logits(model(images), num_classes=len(class_names))
		preds = logits.argmax(dim=1)

		for i in range(images.size(0)):
			if images_shown >= num_samples:
				break

			img = _to_display_rgb(images[i])
			img = img.permute(1, 2, 0).numpy()
			gt = int(class_ids[i].item())
			pred = int(preds[i].item())

			ax = axes[images_shown]
			ax.imshow(img)
			ax.set_title(f"GT: {class_names[gt]}\nPred: {class_names[pred]}")
			ax.axis("off")
			images_shown += 1

		if images_shown >= num_samples:
			break

	for j in range(images_shown, len(axes)):
		axes[j].axis("off")

	plt.tight_layout()
	if save_path is not None:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		plt.savefig(save_path, dpi=180)
		print(f"Saved prediction visualization to: {save_path}")
	plt.show()


def plot_and_save_training_metrics(
	history: Dict[str, List[float]],
	output_dir: str,
) -> None:
	"""
	Save six training metric plots:
	- 4 individual plots: train_loss, val_loss, train_acc, val_acc
	- 2 comparison plots: loss (train vs val), acc (train vs val)
	"""
	os.makedirs(output_dir, exist_ok=True)
	epochs = np.arange(1, len(history["train_loss"]) + 1)

	individual_specs = [
		("train_loss", "Train Loss", "Loss", "tab:blue"),
		("val_loss", "Validation Loss", "Loss", "tab:orange"),
		("train_acc", "Train Accuracy", "Accuracy", "tab:green"),
		("val_acc", "Validation Accuracy", "Accuracy", "tab:red"),
	]

	# 4 individual plots
	for key, title, y_label, color in individual_specs:
		plt.figure(figsize=(8, 5))
		plt.plot(epochs, history[key], marker="o", linewidth=2, color=color)
		plt.title(title)
		plt.xlabel("Epoch")
		plt.ylabel(y_label)
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		file_path = os.path.join(output_dir, f"{key}.png")
		plt.savefig(file_path, dpi=180)
		plt.close()
		print(f"Saved plot: {file_path}")

	# Comparison plot 1: train vs val loss
	plt.figure(figsize=(8, 5))
	plt.plot(epochs, history["train_loss"], marker="o", linewidth=2, label="Train Loss")
	plt.plot(epochs, history["val_loss"], marker="o", linewidth=2, label="Val Loss")
	plt.title("Train vs Validation Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	loss_cmp_path = os.path.join(output_dir, "loss_comparison.png")
	plt.savefig(loss_cmp_path, dpi=180)
	plt.close()
	print(f"Saved plot: {loss_cmp_path}")

	# Comparison plot 2: train vs val accuracy
	plt.figure(figsize=(8, 5))
	plt.plot(epochs, history["train_acc"], marker="o", linewidth=2, label="Train Acc")
	plt.plot(epochs, history["val_acc"], marker="o", linewidth=2, label="Val Acc")
	plt.title("Train vs Validation Accuracy")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	acc_cmp_path = os.path.join(output_dir, "acc_comparison.png")
	plt.savefig(acc_cmp_path, dpi=180)
	plt.close()
	print(f"Saved plot: {acc_cmp_path}")


def main() -> None:
	set_seed(42)

	processed_dataset_path = "/home/yuxin/Object_Detection/project_23047126_Liu/dataset/processed_full_dataset"
	output_dir = "/home/yuxin/Object_Detection/project_23047126_Liu/outputs"
	os.makedirs(output_dir, exist_ok=True)

	check_image_mask_resolutions(processed_dataset_path)
	dataset = HandGestureDataset(dataset_path=processed_dataset_path, transform=None)

	train_size = int(0.8 * len(dataset))
	val_size = len(dataset) - train_size
	train_dataset, val_dataset = torch.utils.data.random_split(
		dataset,
		[train_size, val_size],
		generator=torch.Generator().manual_seed(42),
	)

	train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
	val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	# model = MambaVision(
	# 	dim=64,
	# 	in_dim=32,
	# 	depths=[2, 2, 6, 2],
	# 	window_size=[8, 8, 8, 8],
	# 	mlp_ratio=4.0,
	# 	num_heads=[2, 4, 8, 16],
	# 	num_classes=len(GESTURE_NAMES),
	# 	drop_path_rate=0.1,
	# ).to(device)

	model = YOLO11_ALL().to(device)

	# from torchinfo import summary

	# summary(model, input_size=(16, 3, 224, 224))
	num_epochs = 100
	criterion_classification = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
	
	optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)

	scheduler = torch.optim.lr_scheduler.SequentialLR(
		optimizer,
		schedulers=[
			torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=5),
			torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - 5),
		],
		milestones=[5],
	)

	
	best_val_acc = 0.0
	best_ckpt_path = os.path.join(output_dir, "best_yolov11_ALL_gesture_v1.pt")
	metrics_history: Dict[str, List[float]] = {
		"train_loss": [],
		"val_loss": [],
		"train_acc": [],
		"val_acc": [],
	}

	for epoch in range(1, num_epochs + 1):
		train_metrics = run_one_epoch(
			model,
			train_dataloader,
			criterion_classification,
			optimizer,
			device,
			num_classes=len(GESTURE_NAMES),
		)
		scheduler.step()

		val_metrics = evaluate(
			model,
			val_dataloader,
			criterion_classification,
			device,
			num_classes=len(GESTURE_NAMES),
		)

		print(
			f"Epoch [{epoch:02d}/{num_epochs}] | "
			f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.4f}, "
			f"Train BBox: {train_metrics['bbox_loss']:.4f} | "
			f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.4f}, "
			f"Val BBox: {val_metrics['bbox_loss']:.4f}"
		)

		metrics_history["train_loss"].append(float(train_metrics["loss"]))
		metrics_history["val_loss"].append(float(val_metrics["loss"]))
		metrics_history["train_acc"].append(float(train_metrics["acc"]))
		metrics_history["val_acc"].append(float(val_metrics["acc"]))

		if val_metrics["acc"] > best_val_acc:
			best_val_acc = val_metrics["acc"]
			torch.save(
				{
					"epoch": epoch,
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"val_acc": best_val_acc,
					"class_names": GESTURE_NAMES,
				},
				best_ckpt_path,
			)
			print(f"New best checkpoint saved at epoch {epoch} (val_acc={best_val_acc:.4f})")

	if os.path.exists(best_ckpt_path):
		checkpoint = torch.load(best_ckpt_path, map_location=device)
		model.load_state_dict(checkpoint["model_state_dict"])
		print(f"Loaded best checkpoint from {best_ckpt_path}")

	metrics_json_path = os.path.join(output_dir, "metrics_history.json")
	with open(metrics_json_path, "w", encoding="utf-8") as f:
		json.dump(metrics_history, f, indent=2)
	print(f"Saved metric history to: {metrics_json_path}")

	metrics_plot_dir = os.path.join(output_dir, "training_metrics")
	plot_and_save_training_metrics(history=metrics_history, output_dir=metrics_plot_dir)

	vis_path = os.path.join(output_dir, "val_prediction_samples.png")
	visualize_predictions(
		model=model,
		dataloader=val_dataloader,
		device=device,
		class_names=GESTURE_NAMES,
		num_samples=8,
		save_path=vis_path,
	)


if __name__ == "__main__":
	main()






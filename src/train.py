import os
import random
from typing import Dict, List
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import HandGestureDataset, check_image_mask_resolutions
from mambavision import MambaVision
from cnn_model_1 import YOLO11_v2, YOLOv11


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


def run_one_epoch(
	model: torch.nn.Module,
	dataloader: DataLoader,
	criterion: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	num_classes: int,
) -> Dict[str, float]:
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0

	for images, _masks, class_ids in tqdm.tqdm(dataloader, desc="Training"):
		images = images.to(device)
		class_ids = class_ids.to(device)

		optimizer.zero_grad(set_to_none=True)
		logits = model(images)
		# print(f"logits: {logits.shape}, _2: {_2.shape}, _3: {_3.shape}")
		# print(f"class_ids: {class_ids.shape}")
		loss = criterion(logits, class_ids)
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * images.size(0)
		preds = logits.argmax(dim=1)
		correct += (preds == class_ids).sum().item()
		total += images.size(0)

	return {
		"loss": running_loss / max(total, 1),
		"acc": correct / max(total, 1),
	}


@torch.no_grad()
def evaluate(
	model: torch.nn.Module,
	dataloader: DataLoader,
	criterion: torch.nn.Module,
	device: torch.device,
	num_classes: int,
) -> Dict[str, float]:
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0

	for images, _masks, class_ids in tqdm.tqdm(dataloader, desc="Evaluating"):
		images = images.to(device)
		class_ids = class_ids.to(device)

		logits = to_classification_logits(model(images), num_classes=num_classes)
		loss = criterion(logits, class_ids)

		running_loss += loss.item() * images.size(0)
		preds = logits.argmax(dim=1)
		correct += (preds == class_ids).sum().item()
		total += images.size(0)

	return {
		"loss": running_loss / max(total, 1),
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

	images_shown = 0
	cols = 4
	rows = int(np.ceil(num_samples / cols))
	fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
	axes = np.array(axes).reshape(-1)

	for images, _masks, class_ids in dataloader:
		images = images.to(device)
		class_ids = class_ids.to(device)
		logits = to_classification_logits(model(images), num_classes=len(class_names))
		preds = logits.argmax(dim=1)

		for i in range(images.size(0)):
			if images_shown >= num_samples:
				break

			img = images[i].detach().cpu().permute(1, 2, 0).numpy()
			gt = int(class_ids[i].item())
			pred = int(preds[i].item())

			ax = axes[images_shown]
			ax.imshow(np.clip(img, 0.0, 1.0))
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
	val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

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

	model = YOLO11_v2().to(device)

	# from torchinfo import summary

	# summary(model, input_size=(16, 3, 224, 224))

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

	num_epochs = 50
	best_val_acc = 0.0
	best_ckpt_path = os.path.join(output_dir, "best_yolov11_v2_gesture.pt")

	for epoch in range(1, num_epochs + 1):
		train_metrics = run_one_epoch(
			model,
			train_dataloader,
			criterion,
			optimizer,
			device,
			num_classes=len(GESTURE_NAMES),
		)
		val_metrics = evaluate(
			model,
			val_dataloader,
			criterion,
			device,
			num_classes=len(GESTURE_NAMES),
		)

		print(
			f"Epoch [{epoch:02d}/{num_epochs}] | "
			f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.4f} | "
			f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.4f}"
		)

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






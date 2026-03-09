import os
import json
import math
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import your custom modules
from dataloader import HandGestureDataset_v2, SegAugment_v2, detection_collate_fn
from model import HandGestureMultiTask
from loss import YOLODetectionLoss, generate_anchors, decode_predictions, cxcywh_to_xyxy


def compute_macro_f1(confusion_matrix: np.ndarray) -> float:
    f1_scores = []
    for class_idx in range(confusion_matrix.shape[0]):
        tp = float(confusion_matrix[class_idx, class_idx])
        fp = float(confusion_matrix[:, class_idx].sum() - tp)
        fn = float(confusion_matrix[class_idx, :].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def box_iou_diagonal(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1 + area2 - inter + eps
    return inter / union


def update_detection_metrics(
    det_outputs: dict,
    gt_bboxes_cxcywh: torch.Tensor,
) -> tuple[float, int]:
    pred_boxes = det_outputs["boxes"].transpose(1, 2)   # (B, 6300, 4)
    pred_scores = det_outputs["scores"].transpose(1, 2) # (B, 6300, 10)

    anchors, strides = generate_anchors(det_outputs["feats"])
    decoded_boxes = decode_predictions(pred_boxes, anchors, strides)
    gt_xyxy = cxcywh_to_xyxy(gt_bboxes_cxcywh)

    bsz, _, num_classes = pred_scores.shape
    probs = pred_scores.sigmoid()
    best_flat_idx = probs.reshape(bsz, -1).argmax(dim=1)
    best_anchor_idx = best_flat_idx // num_classes
    pred_top_boxes = decoded_boxes[torch.arange(bsz, device=decoded_boxes.device), best_anchor_idx]
    ious = box_iou_diagonal(pred_top_boxes, gt_xyxy)

    # Match detector metric definition used in visual.py (IoU threshold only).
    correct_det_iou50 = int((ious >= 0.5).sum().item())

    return float(ious.sum().item()), correct_det_iou50


def init_epoch_stats(num_classes: int) -> dict:
    return {
        "loss_sum": 0.0,
        "num_batches": 0,
        "num_samples": 0,
        "cls_correct": 0,
        "det_iou_sum": 0.0,
        "det_iou50_correct": 0,
        "confusion": np.zeros((num_classes, num_classes), dtype=np.int64),
    }


def finalize_epoch_stats(stats: dict) -> dict[str, float]:
    num_samples = max(stats["num_samples"], 1)
    num_batches = max(stats["num_batches"], 1)

    avg_loss = stats["loss_sum"] / num_batches
    top1_acc = stats["cls_correct"] / num_samples
    det_acc_iou50 = stats["det_iou50_correct"] / num_samples
    mean_bbox_iou = stats["det_iou_sum"] / num_samples
    macro_f1 = compute_macro_f1(stats["confusion"])

    return {
        "loss": float(avg_loss),
        "det_acc_iou50": float(det_acc_iou50),
        "mean_bbox_iou": float(mean_bbox_iou),
        "top1_acc": float(top1_acc),
        "macro_f1": float(macro_f1),
    }


def save_training_records(history: dict, output_dir: str) -> None:
    history_path = os.path.join(output_dir, "metrics_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def save_metric_plots(history: dict, output_dir: str) -> None:
    metrics = [
        ("loss", "Loss"),
        ("det_acc_iou50", "Detection Accuracy @0.5 IoU"),
        ("mean_bbox_iou", "Mean Bounding-Box IoU"),
        ("top1_acc", "Overall Top-1 Accuracy"),
        ("macro_f1", "Macro F1 (10 classes)"),
    ]

    epochs = range(1, len(history["train"]["loss"]) + 1)

    # Overview with all requested metrics in one artifact.
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), dpi=140)
    axes = axes.flatten()
    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx]
        ax.plot(epochs, history["train"][key], label="Train", linewidth=2)
        ax.plot(epochs, history["val"][key], label="Val", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metrics_overview.png"))
    plt.close(fig)

    # Dedicated per-metric plots for easier reporting.
    for key, title in metrics:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
        ax.plot(epochs, history["train"][key], label="Train", linewidth=2)
        ax.plot(epochs, history["val"][key], label="Val", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"plot_{key}.png"))
        plt.close(fig)


def log_message(message: str, log_path: str | None = None) -> None:
    print(message)
    if log_path is None:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def train():
    # ==========================================
    # 1. Configuration & Hyperparameters
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = "dataset/dataset_v1/train"
    val_dir = "dataset/dataset_v1/val" 
    num_classes = 10
    batch_size = 16
    epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-2

    # Save directory for checkpoints
    output_dir = "C_model/outputs/s1/test1"
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training_log.txt")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Run start\n")

    log_message(f"Using device: {device}", log_path)

    # ==========================================
    # 2. Dataset & DataLoader Setup
    # ==========================================
    input_size = (480, 640)
    
    # Training dataset with augmentations
    train_transform = SegAugment_v2(out_size=input_size)
    train_dataset = HandGestureDataset_v2(
        root_dir=train_dir, 
        transform=train_transform,
        resize_shape=list(input_size)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, drop_last=True,
        collate_fn=detection_collate_fn,
    )

    # Validation dataset WITHOUT augmentations (transform=None)
    val_dataset = HandGestureDataset_v2(
        root_dir=val_dir, 
        transform=None,  # Only native resizing applied
        resize_shape=list(input_size)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, drop_last=False,
        collate_fn=detection_collate_fn,
    )

    # ==========================================
    # 3. Model & Loss & Optimizer Setup
    # ==========================================
    model = HandGestureMultiTask(num_classes=num_classes, reg_max=1).to(device)
    
    # Prior bias trick for stable initialization
    prior_prob = 0.01
    bias_value = -math.log((1 - prior_prob) / prior_prob)
    for branch in model.detect.cls_branches:
        nn_conv = branch[-1] # The final Conv2d layer
        torch.nn.init.constant_(nn_conv.bias, bias_value)

    criterion = YOLODetectionLoss(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    # ==========================================
    # 4. Main Training & Validation Loop
    # ==========================================
    log_message("Starting training...", log_path)
    best_val_loss = float('inf')
    history = {
        "train": {"loss": [], "det_acc_iou50": [], "mean_bbox_iou": [], "top1_acc": [], "macro_f1": []},
        "val": {"loss": [], "det_acc_iou50": [], "mean_bbox_iou": [], "top1_acc": [], "macro_f1": []},
    }

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_stats = init_epoch_stats(num_classes)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, masks, class_ids, bboxes in pbar:
            images = images.to(device)
            bboxes = bboxes.to(device)       
            class_ids = class_ids.to(device) 

            preds = model(images)
            loss, loss_metrics = criterion(preds["det"], bboxes, class_ids)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            cls_pred = preds["cls"].argmax(dim=1)
            train_stats["cls_correct"] += int((cls_pred == class_ids).sum().item())
            train_stats["num_samples"] += int(images.shape[0])
            train_stats["loss_sum"] += loss.item()
            train_stats["num_batches"] += 1

            for true_id, pred_id in zip(class_ids.detach().cpu().tolist(), cls_pred.detach().cpu().tolist()):
                train_stats["confusion"][int(true_id), int(pred_id)] += 1

            det_iou_sum, det_iou50_correct = update_detection_metrics(preds["det"], bboxes)
            train_stats["det_iou_sum"] += det_iou_sum
            train_stats["det_iou50_correct"] += det_iou50_correct

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "Cls": f"{loss_metrics['loss_cls']:.4f}", 
                "Box": f"{loss_metrics['loss_box']:.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.6f}"
            })

        train_metrics = finalize_epoch_stats(train_stats)

        # --- Validation Phase ---
        model.eval()
        val_stats = init_epoch_stats(num_classes)

        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ")
            for images, masks, class_ids, bboxes in vbar:
                images = images.to(device)
                bboxes = bboxes.to(device)
                class_ids = class_ids.to(device)

                preds = model(images)
                loss, loss_metrics = criterion(preds["det"], bboxes, class_ids)

                cls_pred = preds["cls"].argmax(dim=1)
                val_stats["cls_correct"] += int((cls_pred == class_ids).sum().item())
                val_stats["num_samples"] += int(images.shape[0])
                val_stats["loss_sum"] += loss.item()
                val_stats["num_batches"] += 1

                for true_id, pred_id in zip(class_ids.detach().cpu().tolist(), cls_pred.detach().cpu().tolist()):
                    val_stats["confusion"][int(true_id), int(pred_id)] += 1

                det_iou_sum, det_iou50_correct = update_detection_metrics(preds["det"], bboxes)
                val_stats["det_iou_sum"] += det_iou_sum
                val_stats["det_iou50_correct"] += det_iou50_correct
                
                vbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        val_metrics = finalize_epoch_stats(val_stats)

        for key in history["train"]:
            history["train"][key].append(train_metrics[key])
            history["val"][key].append(val_metrics[key])

        save_training_records(history, output_dir)
        save_metric_plots(history, output_dir)
        
        log_message(
            f"End of Epoch {epoch+1} | "
            f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Train Det@0.5: {train_metrics['det_acc_iou50'] * 100:.2f}% | "
            f"Val Det@0.5: {val_metrics['det_acc_iou50'] * 100:.2f}% | "
            f"Train IoU: {train_metrics['mean_bbox_iou']:.4f} | "
            f"Val IoU: {val_metrics['mean_bbox_iou']:.4f} | "
            f"Train Top1: {train_metrics['top1_acc'] * 100:.2f}% | "
            f"Val Top1: {val_metrics['top1_acc'] * 100:.2f}% | "
            f"Train F1: {train_metrics['macro_f1']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f}",
            log_path,
        )

        # --- Save Checkpoints ---
        # Save last model
        torch.save(model.state_dict(), f"{output_dir}/last_model.pth")
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), f"{output_dir}/best_model.pth")
            log_message(f"--> Saved new BEST model to {output_dir}/best_model.pth! (Val Loss: {best_val_loss:.4f})", log_path)

    log_message("Training finished.", log_path)

if __name__ == "__main__":
    train()
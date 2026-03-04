
import argparse
import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import HandGestureDataset_v2, SegAugment_v2
from model import HandGestureMultiTask

# ---------------------------------------------------------------------------
# Config & Setup (Remains the same)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    train_dataset_path: str = "dataset/dataset_v1/train"
    val_dataset_path: str = "dataset/dataset_v1/val"
    output_dir: str = "outputs/stage_2/train_2"
    epochs: int = 20
    batch_size: int = 32
    num_workers: int = 0
    lr: float = 5e-4
    weight_decay: float = 1e-3
    val_split: float = 0.1
    seed: int = 42
    num_classes: int = 10
    scale: str = "n"
    end2end: bool = True

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def multitask_collate_fn(batch):
    """Collate that normalizes shapes/dtypes for multi-task training.

    `HandGestureDataset_test` returns `class_id` as a 1-element tensor.
    Default PyTorch collation makes that shape (B, 1), but CrossEntropyLoss
    expects a 1D target tensor of shape (B,) with dtype long.
    """
    images, masks, class_ids, bboxes = zip(*batch)
    images = torch.stack(images, dim=0).float()
    masks = torch.stack(masks, dim=0)

    class_ids = torch.as_tensor(
        [int(torch.as_tensor(c).view(-1)[0].item()) for c in class_ids],
        dtype=torch.long,
    )
    bboxes = torch.stack(
        [torch.as_tensor(b, dtype=torch.float32).view(-1)[:4] for b in bboxes],
        dim=0,
    )
    return images, masks, class_ids, bboxes

def build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    # FIX 5: Enable augmentation for the training set
    # NOTE: SegAugment expects out_size=(H, W). Your saved tensors are (H=480, W=640).
    train_aug = SegAugment_v2(out_size=(256, 256))
    train_dataset = HandGestureDataset_v2(root_dir=cfg.train_dataset_path, transform=train_aug, resize_shape=(256, 256))
    val_dataset = HandGestureDataset_v2(root_dir=cfg.val_dataset_path, transform=None, resize_shape=(256, 256))

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=multitask_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=multitask_collate_fn,
    )
    return train_loader, val_loader

# ---------------------------------------------------------------------------
# NEW: Segmentation Metric Helper
# ---------------------------------------------------------------------------
def compute_dice(pred_logits, targets, threshold=0.5, eps=1e-6):
    """Calculates the Dice Coefficient for binary segmentation."""
    # Apply sigmoid and threshold to get binary predictions (0 or 1)
    preds = (pred_logits.sigmoid() > threshold).float()
    
    # Calculate intersection and union
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    
    # Compute dice score per image, then average over the batch
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean().item()

def bce_dice_loss(pred_logits, targets, bce_weight=0.5, dice_weight=0.5, eps=1e-6):
    """Combines BCE and Dice Loss for highly detailed segmentation boundaries."""
    # 1. Standard BCE
    bce = F.binary_cross_entropy_with_logits(pred_logits, targets)
    
    # 2. Differentiable Dice Loss (Using raw sigmoid probabilities, NOT thresholded predictions)
    preds = pred_logits.sigmoid()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2. * intersection + eps) / (union + eps)).mean()
    
    return (bce_weight * bce) + (dice_weight * dice_loss)


def _bytes_to_mib(x: int) -> float:
    return x / (1024 ** 2)


def build_model_and_profile_gpu_usage(device: torch.device) -> HandGestureMultiTask:
    """Prints GPU memory usage for model load + one-image inference.

    Metrics printed:
      1) Model load memory usage (allocated delta).
      2) One image inference incremental peak above loaded model.
      3) Total peak usage from baseline through load+inference.
    """
    if device.type != "cuda":
        print("GPU memory profiling skipped (CUDA not available).")
        model = HandGestureMultiTask(yolo_weights_path="outputs/stage_1/train_2/best.pt", num_classes=10).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model total parameters: {total_params:,}")
        return model

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    baseline_alloc = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    # 1) Load model onto GPU
    checkpoint_path = "outputs/stage_1/train_2/best.pt"
    check_point = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"]
    model = HandGestureMultiTask(yolo_weights_path=checkpoint_path, num_classes=10).to(device)
    model.load_state_dict(check_point)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model total parameters: {total_params:,}")
    torch.cuda.synchronize(device)
    after_model_alloc = torch.cuda.memory_allocated(device)
    model_load_usage = max(0, after_model_alloc - baseline_alloc)

    # 2) One-image inference usage
    dummy_image = torch.randn(1, 3, 256, 256, device=device)
    model.eval()
    with torch.no_grad():
        _ = model(dummy_image)
    torch.cuda.synchronize(device)

    # 3) Total peak usage across model load + inference
    total_peak_alloc = torch.cuda.max_memory_allocated(device)

    one_image_usage = max(0, total_peak_alloc - after_model_alloc)
    total_peak_usage = max(0, total_peak_alloc - baseline_alloc)

    print("GPU Memory Usage (MiB):")
    print(f"1) Model load usage: {_bytes_to_mib(model_load_usage):.2f} MiB")
    print(f"2) One image input usage (incremental peak): {_bytes_to_mib(one_image_usage):.2f} MiB")
    print(f"3) Total peak usage (load + 1-image inference): {_bytes_to_mib(total_peak_usage):.2f} MiB")

    model.train()
    return model

# ---------------------------------------------------------------------------
# Main training driver
# ---------------------------------------------------------------------------

def train() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 1. Initialize and Freeze
    model = build_model_and_profile_gpu_usage(device)
    # model.freeze_base_model()

    # 2. Optimizer strictly for the new heads
    import itertools
    trainable_params = itertools.chain(model.cls_head.parameters(), model.seg_conv.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Define Loss Functions
    criterion_cls = torch.nn.CrossEntropyLoss()
    # criterion_seg = torch.nn.BCEWithLogitsLoss() 

    best_train_loss = float("inf")
    best_val_loss = float("inf")
    

    for epoch in range(1, cfg.epochs + 1):
        # Tracking variables for training
        epoch_train_loss = 0.0
        train_cls_correct = 0
        train_cls_total = 0
        train_dice_sum = 0.0

        model.train()
        for images, masks, class_ids, bboxes in train_loader:
            images = images.to(device)
            masks = masks.to(device).float() # Shape: (B, 1, H, W)
            class_ids = class_ids.to(device)

            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss_cls = criterion_cls(outputs["cls"], class_ids) 
            # loss_seg = criterion_seg(outputs["seg"], masks)
            loss_seg = bce_dice_loss(outputs["seg"], masks)
            total_train_loss = loss_cls + loss_seg
            
            total_train_loss.backward()
            optimizer.step()
            
            epoch_train_loss += total_train_loss.item()

            # --- Calculate Training Metrics ---
            # Classification
            preds = torch.argmax(outputs["cls"], dim=1)
            train_cls_correct += (preds == class_ids).sum().item()
            train_cls_total += class_ids.size(0)
            
            # Segmentation
            train_dice_sum += compute_dice(outputs["seg"], masks)

        # Average training metrics
        train_acc = train_cls_correct / train_cls_total
        train_mean_dice = train_dice_sum / len(train_loader)
        
        if epoch_train_loss < best_train_loss:
            best_train_loss = epoch_train_loss
        torch.save(model.state_dict(), f"{cfg.output_dir}/last.pt")

        # Tracking variables for validation
        epoch_val_loss = 0.0
        val_cls_correct = 0
        val_cls_total = 0
        val_dice_sum = 0.0

        model.eval()
        with torch.no_grad():
            for images, masks, class_ids, bboxes in val_loader:
                images = images.to(device)
                masks = masks.to(device).float() 
                class_ids = class_ids.to(device)
                
                outputs = model(images)
                
                loss_cls = criterion_cls(outputs["cls"], class_ids) 
                # loss_seg = criterion_seg(outputs["seg"], masks)
                loss_seg = bce_dice_loss(outputs["seg"], masks)

                
                total_val_loss = loss_cls + loss_seg
                epoch_val_loss += total_val_loss.item()

                # --- Calculate Validation Metrics ---
                preds = torch.argmax(outputs["cls"], dim=1)
                val_cls_correct += (preds == class_ids).sum().item()
                val_cls_total += class_ids.size(0)
                val_dice_sum += compute_dice(outputs["seg"], masks)
            
            # Average validation metrics
            val_acc = val_cls_correct / val_cls_total
            val_mean_dice = val_dice_sum / len(val_loader)

            

        # Clean, easy-to-read logging
        print(f"Epoch {epoch:02d}/{cfg.epochs} | "
              f"Mean Loss (Train: {epoch_train_loss/len(train_loader):.3f} | Val: {epoch_val_loss/len(val_loader):.3f}) || "
              f"Acc (Train: {train_acc:.3f} | Val: {val_acc:.3f}) || "
              f"Dice (Train: {train_mean_dice:.3f} | Val: {val_mean_dice:.3f})")
        

        if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(model.state_dict(), f"{cfg.output_dir}/best.pt")
                print(f"New best model saved to {cfg.output_dir}/best.pt with Mean Val Loss: {best_val_loss/len(val_loader):.3f}")

if __name__ == "__main__":
    train()
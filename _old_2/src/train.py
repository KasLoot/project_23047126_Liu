import math
import os
import argparse
import torch
import torch.nn as nn
from model import HandGestureModel, ModelConfig
from torch.utils.data import DataLoader, Subset
from dataloader import HandGestureDataset, SegAugment
from utils import compute_detection_loss, decode_predictions, evaluate_batch, plot_confusion_matrix, plot_validation_samples
import random
import torch.nn.functional as F
import torchinfo
import matplotlib.pyplot as plt
import numpy as np




def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def empty_cache():
    torch.cuda.empty_cache()




def get_model(checkpoint_path: str | None = None, model_config: ModelConfig | None = None) -> HandGestureModel:
    if model_config is None:
        model_config = ModelConfig()
    model = HandGestureModel(model_config=model_config)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))
    return model


class Train_S1_Config:
    num_epochs = 80
    batch_size = 32
    learning_rate = 5e-4
    weight_decay = 1e-4

    train_dataset_dir = "./dataset/dataset_v1/train"
    train_image_tensor_dir = os.path.join(train_dataset_dir, "image_tensors")
    train_mask_tensor_dir = os.path.join(train_dataset_dir, "mask_tensors")
    train_image_info_json = os.path.join(train_dataset_dir, "image_info.json")

    val_dataset_dir = "./dataset/dataset_v1/val"
    val_image_tensor_dir = os.path.join(val_dataset_dir, "image_tensors")
    val_mask_tensor_dir = os.path.join(val_dataset_dir, "mask_tensors")
    val_image_info_json = os.path.join(val_dataset_dir, "image_info.json")


    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Train_S2_Config:
    num_epochs = 60
    batch_size = 16
    backbone_learning_rate = 1e-4
    heads_learning_rate = 3e-4
    weight_decay = 1e-4

    cls_loss_weight = 1.0
    det_loss_weight = 1.0
    seg_loss_weight = 1.0

    train_dataset_dir = "./dataset/dataset_v1/train"
    train_image_tensor_dir = os.path.join(train_dataset_dir, "image_tensors")
    train_mask_tensor_dir = os.path.join(train_dataset_dir, "mask_tensors")
    train_image_info_json = os.path.join(train_dataset_dir, "image_info.json")

    val_dataset_dir = "./dataset/dataset_v1/val"
    val_image_tensor_dir = os.path.join(val_dataset_dir, "image_tensors")
    val_mask_tensor_dir = os.path.join(val_dataset_dir, "mask_tensors")
    val_image_info_json = os.path.join(val_dataset_dir, "image_info.json")

    init_checkpoint_path = "outputs/stage_1_training_2/stage1_best.pt"

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def detection_collate_fn(batch):
    images, masks, class_ids, bboxes = zip(*batch)
    images = torch.stack(images, dim=0).float()
    masks = torch.stack(masks, dim=0)
    class_ids = torch.as_tensor(class_ids, dtype=torch.long)
    bboxes = torch.stack([b.float() for b in bboxes], dim=0)
    return images, masks, class_ids, bboxes


class WarmupCosineScheduler:
    """Linear warmup then cosine annealing (per-step)."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self._step <= self.warmup_steps:
                lr = base_lr * self._step / max(self.warmup_steps, 1)
            else:
                progress = (self._step - self.warmup_steps) / max(
                    self.total_steps - self.warmup_steps, 1
                )
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            pg["lr"] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def compute_dice_score(pred_logits, target_masks, threshold: float = 0.5, eps: float = 1e-6) -> float:
    preds = (torch.sigmoid(pred_logits) > threshold).float()
    targets = (target_masks > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return float(dice.mean().item())


def bce_dice_loss(pred_logits, target_masks, bce_weight: float = 0.5, dice_weight: float = 0.5, eps: float = 1e-6):
    targets = (target_masks > 0.5).float()
    bce = F.binary_cross_entropy_with_logits(pred_logits, targets)
    probs = torch.sigmoid(pred_logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + eps) / (union + eps)).mean()
    return bce_weight * bce + dice_weight * dice_loss


def plot_train_val_histories(train_loss_history, val_loss_history, iou50_history, output_dir):
    epochs = list(range(1, len(train_loss_history) + 1))
    if len(epochs) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss_history, label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_loss_history, label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train/Validation Loss History")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, iou50_history, label="IoU@50", color="tab:green", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("IoU@50 Accuracy History")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_histories.png"))
    plt.close(fig)



def train_stage_1():
    train_stage_1_config = Train_S1_Config()
    print(f"Using device: {train_stage_1_config.device}")

    output_dir = "outputs/stage_1_training_2"
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Initialize Model ---
    model = get_model(model_config=ModelConfig())
    model.to(torch.bfloat16).to(train_stage_1_config.device)

    torchinfo.summary(model, input_size=(1, 3, 640, 480), device=train_stage_1_config.device, dtypes=[torch.bfloat16])

    # --- 2. Freeze all heads EXCEPT the Detection Head ---
    # for param in model.backbone.parameters(): param.requires_grad = True
    for param in model.classifier_head.parameters(): param.requires_grad = False
    for param in model.segmentation_head.parameters(): param.requires_grad = False
    
    # Verify what is trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 1 Training (Backbone + Detection Head)\n Total Trainable Params: {trainable_params:,}")

    # --- 3. Dataloaders ---
    train_dataset = HandGestureDataset(
        image_tensor_dir=train_stage_1_config.train_image_tensor_dir,
        mask_tensor_dir=train_stage_1_config.train_mask_tensor_dir,
        image_info_json=train_stage_1_config.train_image_info_json,
        transform=SegAugment()
    )
    val_dataset = HandGestureDataset(
        image_tensor_dir=train_stage_1_config.val_image_tensor_dir,
        mask_tensor_dir=train_stage_1_config.val_mask_tensor_dir,
        image_info_json=train_stage_1_config.val_image_info_json
    )
    
    # Using your existing custom collate function from train_1.py if applicable
    train_loader = DataLoader(train_dataset, batch_size=train_stage_1_config.batch_size, shuffle=True, collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_stage_1_config.batch_size, shuffle=False, collate_fn=detection_collate_fn)

    # --- 4. Optimizer ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=train_stage_1_config.learning_rate, 
        weight_decay=train_stage_1_config.weight_decay
    )
    epochs = train_stage_1_config.num_epochs
    total_steps = epochs * len(train_loader)
    warmup_steps = 3 * len(train_loader) # 3 epochs of warmup
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    # --- 5. Training Loop ---
    best_val_loss = float('inf')

    train_loss_history = []
    val_loss_history = []
    iou50_history = []


    for epoch in range(train_stage_1_config.num_epochs):
        model.train()
        train_loss = 0.0
        
        # --- TRAINING LOOP ---
        for images, masks, class_ids, bboxes in train_loader:

            images = images.to(torch.bfloat16).to(train_stage_1_config.device)
            class_ids = class_ids.to(train_stage_1_config.device)
            bboxes = bboxes.to(train_stage_1_config.device)


            optimizer.zero_grad()
            cls_logits, bbox_preds, bbox_cls, seg_map = model(images)
            
            input_h, input_w = images.shape[-2], images.shape[-1]
            loss, c_loss, b_loss = compute_detection_loss(bbox_preds, bbox_cls, bboxes, class_ids, input_h, input_w)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        # --- VALIDATION LOOP ---
        model.eval()
        val_loss = 0.0
        all_y_true, all_y_pred = [], []
        vis_samples = []
        
        with torch.no_grad():
            for images, masks, class_ids, bboxes in val_loader:
                images = images.to(torch.bfloat16).to(train_stage_1_config.device)
                class_ids = class_ids.to(train_stage_1_config.device)
                bboxes = bboxes.to(train_stage_1_config.device)
                
                input_h, input_w = images.shape[-2], images.shape[-1]
                cls_logits, bbox_preds, bbox_cls, seg_map = model(images)
                
                # 1. Compute Validation Loss
                loss, c_loss, b_loss = compute_detection_loss(bbox_preds, bbox_cls, bboxes, class_ids, input_h, input_w)
                val_loss += loss.item()
                
                # 2. Decode Predictions for Metrics
                decoded_preds = decode_predictions(bbox_preds, bbox_cls, input_h, input_w, conf_thresh=0.05)
                
                # 3. Match Preds to GT for Confusion Matrix
                y_true, y_pred = evaluate_batch(decoded_preds, bboxes, class_ids, input_h, input_w)
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
                
                # 4. Save samples for visualization (collect up to 10)
                if len(vis_samples) < 10:
                    for bi in range(images.shape[0]):
                        if len(vis_samples) < 10:
                            vis_samples.append((images[bi].cpu(), bboxes[bi].cpu(), class_ids[bi].cpu(), decoded_preds[bi].detach().cpu()))

        # --- CALCULATE EPOCH METRICS ---
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate IoU@50 Accuracy (Ignoring BG classes for pure accuracy)
        correct = sum(1 for t, p in zip(all_y_true, all_y_pred) if t == p and p != 10)
        total_gt = len(all_y_true)
        iou50_acc = correct / total_gt if total_gt > 0 else 0.0
        
        print(f"Epoch [{epoch+1}/{train_stage_1_config.num_epochs}]: Learning Rate: {scheduler.get_lr():.6f} |Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | IoU@50 Acc: {iou50_acc:.4f}")

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        iou50_history.append(iou50_acc)
        
        # --- SAVE PLOTS & METRICS ---
        with open(f"{output_dir}/val_metrics.txt", "a") as f:
            f.write(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, IoU@50: {iou50_acc:.4f}\n")
            
        plot_confusion_matrix(all_y_true, all_y_pred, num_classes=10, output_dir=output_dir)
        
        # Shuffle samples to get different visual results across epochs, then plot
        random.shuffle(vis_samples)
        plot_validation_samples(vis_samples, output_dir=output_dir)
        plot_train_val_histories(train_loss_history, val_loss_history, iou50_history, output_dir)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{output_dir}/stage1_best.pt")
            print(f"--> Saved new best model to {output_dir}/stage1_best.pt")
    
    torch.save(model.state_dict(), f"{output_dir}/stage1_last.pt")


def train_stage_2():
    train_stage_2_config = Train_S2_Config()
    print(f"Using device: {train_stage_2_config.device}")

    output_dir = "outputs/stage_2_training"
    os.makedirs(output_dir, exist_ok=True)

    init_checkpoint_path = train_stage_2_config.init_checkpoint_path
    if init_checkpoint_path is not None and not os.path.exists(init_checkpoint_path):
        init_checkpoint_path = None

    model = get_model(checkpoint_path=init_checkpoint_path, model_config=ModelConfig())
    model.to(torch.bfloat16).to(train_stage_2_config.device)

    torchinfo.summary(model, input_size=(1, 3, 640, 480), device=train_stage_2_config.device, dtypes=[torch.bfloat16])

    for param in model.backbone.parameters():
        param.requires_grad = True
    for param in model.classifier_head.parameters():
        param.requires_grad = True
    for param in model.detection_head.parameters():
        param.requires_grad = True
    for param in model.segmentation_head.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 2 Training (Backbone/Neck + Classifier + Detection + Segmentation)\n Total Trainable Params: {trainable_params:,}")

    train_dataset = HandGestureDataset(
        image_tensor_dir=train_stage_2_config.train_image_tensor_dir,
        mask_tensor_dir=train_stage_2_config.train_mask_tensor_dir,
        image_info_json=train_stage_2_config.train_image_info_json,
        transform=SegAugment()
    )
    val_dataset = HandGestureDataset(
        image_tensor_dir=train_stage_2_config.val_image_tensor_dir,
        mask_tensor_dir=train_stage_2_config.val_mask_tensor_dir,
        image_info_json=train_stage_2_config.val_image_info_json
    )

    train_loader = DataLoader(train_dataset, batch_size=train_stage_2_config.batch_size, shuffle=True, collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_stage_2_config.batch_size, shuffle=False, collate_fn=detection_collate_fn)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone.parameters(),
                "lr": train_stage_2_config.backbone_learning_rate,
            },
            {
                "params": model.classifier_head.parameters(),
                "lr": train_stage_2_config.heads_learning_rate,
            },
            {
                "params": model.detection_head.parameters(),
                "lr": train_stage_2_config.heads_learning_rate,
            },
            {
                "params": model.segmentation_head.parameters(),
                "lr": train_stage_2_config.heads_learning_rate,
            },
        ],
        weight_decay=train_stage_2_config.weight_decay,
    )

    epochs = train_stage_2_config.num_epochs
    total_steps = epochs * len(train_loader)
    warmup_steps = 3 * len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    cls_criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    train_loss_history = []
    val_loss_history = []
    iou50_history = []

    for epoch in range(train_stage_2_config.num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks, class_ids, bboxes in train_loader:
            images = images.to(torch.bfloat16).to(train_stage_2_config.device)
            masks = masks.to(train_stage_2_config.device)
            class_ids = class_ids.to(train_stage_2_config.device)
            bboxes = bboxes.to(train_stage_2_config.device)

            optimizer.zero_grad()
            cls_logits, bbox_preds, bbox_cls, seg_map = model(images)

            input_h, input_w = images.shape[-2], images.shape[-1]

            det_loss, det_cls_loss, det_box_loss = compute_detection_loss(
                bbox_preds, bbox_cls, bboxes, class_ids, input_h, input_w
            )
            cls_loss = cls_criterion(cls_logits.float(), class_ids)

            if seg_map.shape[-2:] != masks.shape[-2:]:
                masks_resized = F.interpolate(masks.float(), size=seg_map.shape[-2:], mode="nearest")
            else:
                masks_resized = masks.float()
            seg_loss = bce_dice_loss(seg_map.float(), masks_resized)

            loss = (
                train_stage_2_config.det_loss_weight * det_loss
                + train_stage_2_config.cls_loss_weight * cls_loss
                + train_stage_2_config.seg_loss_weight * seg_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_cls_correct = 0
        val_cls_total = 0
        val_dice_accum = 0.0
        val_dice_steps = 0
        all_y_true, all_y_pred = [], []
        vis_samples = []

        with torch.no_grad():
            for images, masks, class_ids, bboxes in val_loader:
                images = images.to(torch.bfloat16).to(train_stage_2_config.device)
                masks = masks.to(train_stage_2_config.device)
                class_ids = class_ids.to(train_stage_2_config.device)
                bboxes = bboxes.to(train_stage_2_config.device)

                input_h, input_w = images.shape[-2], images.shape[-1]
                cls_logits, bbox_preds, bbox_cls, seg_map = model(images)

                det_loss, det_cls_loss, det_box_loss = compute_detection_loss(
                    bbox_preds, bbox_cls, bboxes, class_ids, input_h, input_w
                )
                cls_loss = cls_criterion(cls_logits.float(), class_ids)
                if seg_map.shape[-2:] != masks.shape[-2:]:
                    masks_resized = F.interpolate(masks.float(), size=seg_map.shape[-2:], mode="nearest")
                else:
                    masks_resized = masks.float()
                seg_loss = bce_dice_loss(seg_map.float(), masks_resized)

                loss = (
                    train_stage_2_config.det_loss_weight * det_loss
                    + train_stage_2_config.cls_loss_weight * cls_loss
                    + train_stage_2_config.seg_loss_weight * seg_loss
                )
                val_loss += loss.item()

                pred_cls_ids = torch.argmax(cls_logits, dim=1)
                val_cls_correct += (pred_cls_ids == class_ids).sum().item()
                val_cls_total += class_ids.size(0)

                val_dice_accum += compute_dice_score(seg_map.float(), masks_resized)
                val_dice_steps += 1

                decoded_preds = decode_predictions(bbox_preds, bbox_cls, input_h, input_w, conf_thresh=0.05)
                y_true, y_pred = evaluate_batch(decoded_preds, bboxes, class_ids, input_h, input_w)
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)

                if len(vis_samples) < 10:
                    for bi in range(images.shape[0]):
                        if len(vis_samples) < 10:
                            vis_samples.append((images[bi].cpu(), bboxes[bi].cpu(), class_ids[bi].cpu(), decoded_preds[bi].detach().cpu()))

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        correct = sum(1 for t, p in zip(all_y_true, all_y_pred) if t == p and p != 10)
        total_gt = len(all_y_true)
        iou50_acc = correct / total_gt if total_gt > 0 else 0.0
        cls_acc = val_cls_correct / max(val_cls_total, 1)
        seg_dice = val_dice_accum / max(val_dice_steps, 1)

        backbone_lr = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[1]["lr"]
        print(
            f"Epoch [{epoch+1}/{train_stage_2_config.num_epochs}] "
            f"| backbone_lr: {backbone_lr:.6f} "
            f"| head_lr: {head_lr:.6f} "
            f"| train_loss: {avg_train_loss:.4f} "
            f"| val_loss: {avg_val_loss:.4f} "
            f"| det_iou50: {iou50_acc:.4f} "
            f"| cls_acc: {cls_acc:.4f} "
            f"| seg_dice: {seg_dice:.4f}"
        )

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        iou50_history.append(iou50_acc)

        with open(f"{output_dir}/val_metrics.txt", "a") as f:
            f.write(
                f"Epoch {epoch+1}: "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Det IoU@50: {iou50_acc:.4f}, "
                f"Cls Acc: {cls_acc:.4f}, "
                f"Seg Dice: {seg_dice:.4f}\n"
            )

        plot_confusion_matrix(all_y_true, all_y_pred, num_classes=10, output_dir=output_dir)
        random.shuffle(vis_samples)
        plot_validation_samples(vis_samples, output_dir=output_dir)
        plot_train_val_histories(train_loss_history, val_loss_history, iou50_history, output_dir)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{output_dir}/stage2_best.pt")
            print(f"--> Saved new best model to {output_dir}/stage2_best.pt")

    torch.save(model.state_dict(), f"{output_dir}/stage2_last.pt")
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2], help="Training stage to run")
    args = parser.parse_args()

    empty_cache()
    set_seed(42)
    if args.stage == 1:
        train_stage_1()
    else:
        train_stage_2()
    

    

    


    
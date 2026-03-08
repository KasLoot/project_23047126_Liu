import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import your custom modules
from dataloader import HandGestureDataset_v2, SegAugment_v2, detection_collate_fn, CLASS_ID_TO_NAME
from model import HandGestureMultiTask


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


def format_confusion_matrix(confusion_matrix: np.ndarray) -> str:
    class_names = [CLASS_ID_TO_NAME[idx] for idx in range(confusion_matrix.shape[0])]
    cell_width = max(max(len(name) for name in class_names), len("true\\pred"), 5) + 2

    header = "true\\pred".ljust(cell_width) + "".join(name.rjust(cell_width) for name in class_names)
    rows = [header]
    for row_idx, row_name in enumerate(class_names):
        row_values = "".join(str(int(value)).rjust(cell_width) for value in confusion_matrix[row_idx])
        rows.append(row_name.ljust(cell_width) + row_values)

    return "\n".join(rows)


def summarize_segmentation_metrics(pred_logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    pred_masks = (pred_logits.sigmoid() > threshold).float()
    targets = (targets > 0.5).float()

    hand_intersection = float((pred_masks * targets).sum().item())
    hand_union = float(((pred_masks + targets) > 0).sum().item())

    pred_background = 1.0 - pred_masks
    target_background = 1.0 - targets
    background_intersection = float((pred_background * target_background).sum().item())
    background_union = float(((pred_background + target_background) > 0).sum().item())

    pred_hand_pixels = float(pred_masks.sum().item())
    gt_hand_pixels = float(targets.sum().item())

    return {
        "hand_intersection": hand_intersection,
        "hand_union": hand_union,
        "background_intersection": background_intersection,
        "background_union": background_union,
        "pred_hand_pixels": pred_hand_pixels,
        "gt_hand_pixels": gt_hand_pixels,
    }


def finalize_segmentation_metrics(metric_sums: dict[str, float]) -> tuple[float, float, float, float]:
    hand_iou = metric_sums["hand_intersection"] / metric_sums["hand_union"] if metric_sums["hand_union"] > 0 else 0.0
    background_iou = (
        metric_sums["background_intersection"] / metric_sums["background_union"]
        if metric_sums["background_union"] > 0
        else 0.0
    )
    mean_iou = (hand_iou + background_iou) / 2.0
    dice = (
        (2.0 * metric_sums["hand_intersection"]) /
        (metric_sums["pred_hand_pixels"] + metric_sums["gt_hand_pixels"])
        if (metric_sums["pred_hand_pixels"] + metric_sums["gt_hand_pixels"]) > 0
        else 0.0
    )
    return mean_iou, dice, hand_iou, background_iou


def set_stage2_train_mode(model: HandGestureMultiTask) -> None:
    model.train()
    model.backbone.eval()
    model.neck.eval()
    model.detect.eval()



def train():
    # ==========================================
    # 1. Configuration & Hyperparameters
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dir = "dataset/dataset_v1/train"
    val_dir = "dataset/dataset_v1/val" 
    num_classes = 10
    batch_size = 32
    epochs = 20
    learning_rate = 5e-3
    weight_decay = 1e-2

    # Save directory for checkpoints
    os.makedirs("weights", exist_ok=True)

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
        persistent_workers=True,
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
        persistent_workers=True,
    )

    # ==========================================
    # 3. Model & Loss & Optimizer Setup
    # ==========================================
    model = HandGestureMultiTask(num_classes=num_classes, reg_max=1)
    checkpoint_path = "weights/best_model.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"Loaded weights from {checkpoint_path} for Stage 2 training.")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting Stage 2 training from scratch.")
    model.freeze_for_s2_training()  # Freeze backbone and detection head for Stage 2 training
    torch.set_float32_matmul_precision('high')
    model = model.to(device)
    


    cls_loss_fn = torch.nn.CrossEntropyLoss()
    seg_loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_val_selection = (-1.0, -1.0, float("-inf"))
    for epoch in range(epochs):
        set_stage2_train_mode(model)
        total_loss = 0.0
        train_cls_correct = 0
        train_cls_total = 0
        train_seg_sums = {
            "hand_intersection": 0.0,
            "hand_union": 0.0,
            "background_intersection": 0.0,
            "background_union": 0.0,
            "pred_hand_pixels": 0.0,
            "gt_hand_pixels": 0.0,
        }

        for images, masks, class_ids, gt_bboxes in train_loader:
            images = images.to(device)
            masks = masks.to(device).float()
            class_ids = class_ids.to(device)

            optimizer.zero_grad()

            outputs = model(images, tasks=("cls", "seg"))
            cls_outputs = outputs["cls"]  # (B, 10)
            seg_outputs = outputs["seg"]  # (B, 1, H, W)

            cls_loss = cls_loss_fn(cls_outputs, class_ids)
            seg_loss = seg_loss_fn(seg_outputs, masks)
            loss = cls_loss + seg_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            train_preds = cls_outputs.argmax(dim=1)
            train_cls_correct += (train_preds == class_ids).sum().item()
            train_cls_total += class_ids.size(0)

            train_batch_seg = summarize_segmentation_metrics(seg_outputs, masks)
            for key, value in train_batch_seg.items():
                train_seg_sums[key] += value

        avg_train_loss = total_loss / len(train_loader)
        train_acc = train_cls_correct / train_cls_total if train_cls_total > 0 else 0.0
        train_miou, train_dice, train_hand_iou, train_background_iou = finalize_segmentation_metrics(train_seg_sums)

        model.eval()
        val_total_loss = 0.0
        val_cls_correct = 0
        val_cls_total = 0
        val_confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        val_seg_sums = {
            "hand_intersection": 0.0,
            "hand_union": 0.0,
            "background_intersection": 0.0,
            "background_union": 0.0,
            "pred_hand_pixels": 0.0,
            "gt_hand_pixels": 0.0,
        }

        with torch.no_grad():
            for images, masks, class_ids, gt_bboxes in val_loader:
                images = images.to(device)
                masks = masks.to(device).float()
                class_ids = class_ids.to(device)

                outputs = model(images, tasks=("cls", "seg"))
                cls_outputs = outputs["cls"]
                seg_outputs = outputs["seg"]

                cls_loss = cls_loss_fn(cls_outputs, class_ids)
                seg_loss = seg_loss_fn(seg_outputs, masks)
                loss = cls_loss + seg_loss
                val_total_loss += loss.item()

                val_preds = cls_outputs.argmax(dim=1)
                val_cls_correct += (val_preds == class_ids).sum().item()
                val_cls_total += class_ids.size(0)

                for true_id, pred_id in zip(class_ids.detach().cpu().tolist(), val_preds.detach().cpu().tolist()):
                    val_confusion_matrix[int(true_id), int(pred_id)] += 1

                val_batch_seg = summarize_segmentation_metrics(seg_outputs, masks)
                for key, value in val_batch_seg.items():
                    val_seg_sums[key] += value

        avg_val_loss = val_total_loss / len(val_loader)
        val_acc = val_cls_correct / val_cls_total if val_cls_total > 0 else 0.0
        val_macro_f1 = compute_macro_f1(val_confusion_matrix)
        val_miou, val_dice, val_hand_iou, val_background_iou = finalize_segmentation_metrics(val_seg_sums)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f}"
        )
        print(
            f"Train Seg mIoU: {train_miou:.4f} (hand={train_hand_iou:.4f}, background={train_background_iou:.4f}) | "
            f"Train Dice: {train_dice:.4f}"
        )
        print(
            f"Val Seg mIoU: {val_miou:.4f} (hand={val_hand_iou:.4f}, background={val_background_iou:.4f}) | "
            f"Val Dice: {val_dice:.4f}"
        )
        print("Validation Confusion Matrix (rows=true, cols=pred):")
        print(format_confusion_matrix(val_confusion_matrix))

        torch.save(model.state_dict(), "weights/s2_last_model.pth")
        current_selection = (val_macro_f1, val_dice, -avg_val_loss)
        if current_selection > best_val_selection:
            best_val_selection = current_selection
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "weights/s2_best_model.pth")
            print(
                "Saved new best Stage 2 model to weights/s2_best_model.pth "
                f"(macro_f1={val_macro_f1:.4f}, dice={val_dice:.4f}, val_loss={best_val_loss:.4f})"
            )


if __name__ == "__main__":
    train()
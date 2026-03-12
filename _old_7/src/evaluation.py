import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
from torchvision.ops import box_iou

# Import from your provided files
from model import RGB_V1, RGB_V2, RGB_V3
from dataloader import HandGestureDataset_v2, detection_collate_fn, CLASS_ID_TO_NAME

# ==========================================
# 1. Helper Functions for Evaluation
# ==========================================

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    x1 = boxes[..., 0] - boxes[..., 2] / 2
    y1 = boxes[..., 1] - boxes[..., 3] / 2
    x2 = boxes[..., 0] + boxes[..., 2] / 2
    y2 = boxes[..., 1] + boxes[..., 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def make_anchors(feats, strides=(8, 16, 32)):
    """Generates grid anchor points [6300, 2] and strides [6300, 1] for decoding."""
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(w, device=feats[i].device, dtype=feats[i].dtype) + 0.5
        sy = torch.arange(h, device=feats[i].device, dtype=feats[i].dtype) + 0.5
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, device=feats[i].device, dtype=feats[i].dtype))
        
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def calculate_seg_metrics(pred_logits, gt_masks, smooth=1e-6):
    """Calculates Mean IoU and Dice Coefficient for Segmentation."""
    pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
    gt_masks = gt_masks.float()
    
    # Calculate across batch, spatial dimensions
    intersection = (pred_bin * gt_masks).sum(dim=(1, 2, 3))
    sum_pred = pred_bin.sum(dim=(1, 2, 3))
    sum_gt = gt_masks.sum(dim=(1, 2, 3))
    union = sum_pred + sum_gt - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    dice = (2. * intersection + smooth) / (sum_pred + sum_gt + smooth)
    
    return iou, dice


def save_confusion_matrix(targets, preds, save_path, class_names):
    """Generates and saves a confusion matrix image."""
    cm = confusion_matrix(targets, preds, labels=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Gesture')
    plt.ylabel('Actual Gesture')
    plt.title('Gesture Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# 2. Main Evaluation Loop
# ==========================================

def evaluate_split(model, dataloader, device, split_name, save_dir):
    model.eval()
    
    # Trackers for Detection
    all_det_ious = []
    
    # Trackers for Segmentation
    all_seg_ious = []
    all_seg_dices = []
    
    # Trackers for Classification
    all_cls_preds = []
    all_cls_targets = []
    
    pbar = tqdm(dataloader, desc=f"Evaluating {split_name}")
    
    with torch.no_grad():
        for images, masks, class_ids, bboxes in pbar:
            images = images.to(device)
            masks = masks.to(device)
            class_ids = class_ids.to(device)
            bboxes = bboxes.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # ---------------------------
            # 1. Detection Evaluation
            # ---------------------------
            preds_det = outputs["det"]
            pred_boxes = preds_det["boxes"].transpose(1, 2)    # [B, 6300, 4]
            pred_scores = preds_det["scores"].transpose(1, 2)  # [B, 6300, 10]
            feats = preds_det["feats"]
            
            anchors, stride_tensor = make_anchors(feats)
            
            # Decode boxes
            dxdy = pred_boxes[..., :2]
            twth = pred_boxes[..., 2:]
            pred_cxcy = (anchors.unsqueeze(0) + dxdy) * stride_tensor.unsqueeze(0)
            pred_wh = torch.exp(twth.clamp(max=10)) * stride_tensor.unsqueeze(0)
            decoded_boxes_cxcywh = torch.cat([pred_cxcy, pred_wh], dim=-1)
            decoded_boxes_xyxy = cxcywh_to_xyxy(decoded_boxes_cxcywh) # [B, 6300, 4]
            
            # Get max confidence box for each image in batch (Anchor-Free Object selection)
            pred_scores_sigmoid = torch.sigmoid(pred_scores)
            max_scores_per_box, _ = torch.max(pred_scores_sigmoid, dim=2) # [B, 6300]
            best_box_indices = torch.argmax(max_scores_per_box, dim=1)    # [B]
            
            batch_indices = torch.arange(images.size(0))
            best_pred_boxes = decoded_boxes_xyxy[batch_indices, best_box_indices] # [B, 4]
            gt_boxes_xyxy = cxcywh_to_xyxy(bboxes) # [B, 4]
            
            # Compute pairwise IoU, grab diagonal for matched (image-to-image) IoU
            ious = torch.diag(box_iou(best_pred_boxes, gt_boxes_xyxy)) # [B]
            all_det_ious.extend(ious.cpu().tolist())
            
            # ---------------------------
            # 2. Segmentation Evaluation
            # ---------------------------
            seg_iou, seg_dice = calculate_seg_metrics(outputs["seg"], masks)
            all_seg_ious.extend(seg_iou.cpu().tolist())
            all_seg_dices.extend(seg_dice.cpu().tolist())
            
            # ---------------------------
            # 3. Classification Evaluation
            # ---------------------------
            cls_logits = outputs["cls"] # [B, 10]
            cls_preds = torch.argmax(cls_logits, dim=1) # [B]
            
            all_cls_preds.extend(cls_preds.cpu().tolist())
            all_cls_targets.extend(class_ids.cpu().tolist())

    # --- Compute Final Metrics ---
    
    # Detection
    mean_bbox_iou = np.mean(all_det_ious)
    acc_05_iou = np.mean([1 if iou >= 0.5 else 0 for iou in all_det_ious])
    
    # Segmentation
    mean_seg_iou = np.mean(all_seg_ious)
    mean_seg_dice = np.mean(all_seg_dices)
    
    # Classification
    all_cls_targets = np.array(all_cls_targets)
    all_cls_preds = np.array(all_cls_preds)
    
    top1_acc = np.mean(all_cls_preds == all_cls_targets)
    macro_f1 = f1_score(all_cls_targets, all_cls_preds, average='macro')
    
    # Save Confusion Matrix
    class_names = [CLASS_ID_TO_NAME[i] for i in range(10)]
    cm_path = os.path.join(save_dir, f"confusion_matrix_{split_name}.png")
    save_confusion_matrix(all_cls_targets, all_cls_preds, cm_path, class_names)
    
    print(f"\n======================================")
    print(f"   EVALUATION RESULTS: {split_name.upper()} SET")
    print(f"======================================")
    print(f"1. Detection Metrics:")
    print(f"   - Mean Bounding-Box IoU: {mean_bbox_iou:.4f}")
    print(f"   - Detection Acc @ 0.5:   {acc_05_iou:.4f}")
    print(f"\n2. Segmentation Metrics:")
    print(f"   - Mean Hand IoU:         {mean_seg_iou:.4f}")
    print(f"   - Mean Dice Coefficient: {mean_seg_dice:.4f}")
    print(f"\n3. Classification Metrics:")
    print(f"   - Top-1 Accuracy:        {top1_acc:.4f}")
    print(f"   - Macro-averaged F1:     {macro_f1:.4f}")
    print(f"   - Confusion Matrix:      Saved -> {cm_path}")
    print(f"======================================\n")
    
    return {
        "det_mIoU": mean_bbox_iou, "det_acc50": acc_05_iou,
        "seg_mIoU": mean_seg_iou, "seg_dice": mean_seg_dice,
        "cls_top1": top1_acc, "cls_f1": macro_f1
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device}")
    
    # Load Model
    if args.model_version.upper() == "RGB_V1":
        model = RGB_V1(num_classes=10, reg_max=1)
    elif args.model_version.upper() == "RGB_V2":
        model = RGB_V2(num_classes=10, reg_max=1)
    elif args.model_version.upper() == "RGB_V3":
        model = RGB_V3(num_classes=10, reg_max=1)
    else:
        raise ValueError(f"Unknown model version: {args.model_version}")

    weights_path = os.path.join(args.save_dir, args.model_version, args.weights)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Could not find weights at {weights_path}!")
        
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    print(f"[*] Successfully loaded weights: {weights_path}")
    
    # Create save directory for images
    eval_save_dir = os.path.join(args.save_dir, args.model_version, "eval_results")
    os.makedirs(eval_save_dir, exist_ok=True)
    
    # Dataloaders
    val_dataset = HandGestureDataset_v2(root_dir=args.val_dir, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=detection_collate_fn, num_workers=4)
    
    # Evaluate Validation Set
    evaluate_split(model, val_loader, device, "Validation", eval_save_dir)
    
    # Evaluate Test Set (if provided)
    if args.test_dir and os.path.exists(args.test_dir):
        test_dataset = HandGestureDataset_v2(root_dir=args.test_dir, transform=None)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=detection_collate_fn, num_workers=4)
        evaluate_split(model, test_loader, device, "Test", eval_save_dir)
    else:
        print("[!] No test_dir provided or path invalid. Skipping test set evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Multi-Task Model")
    parser.add_argument("--val_dir", type=str, default="dataset/dataset_v1/val", help="Path to validation dataset folder")
    parser.add_argument("--test_dir", type=str, default="dataset/dataset_v1/test", help="Path to test dataset folder (Optional)")
    parser.add_argument("--model_version", type=str, default="RGB_V2", choices=["RGB_V1", "RGB_V2", "RGB_V3"], help="Which model to use")
    parser.add_argument("--save_dir", type=str, default="weights", help="Base directory where model weights are stored")
    parser.add_argument("--weights", type=str, default="best_model_stage2.pth", help="Name of the weights file to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    
    args = parser.parse_args()
    main(args)
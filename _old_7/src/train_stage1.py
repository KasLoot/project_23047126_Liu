import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.ops import generalized_box_iou
import random
import numpy as np


# Import from your provided files
from model import RGB_V1, RGB_V2, RGB_V3
from dataloader import HandGestureDataset_v2, SegAugment_v2, detection_collate_fn

# ==========================================
# 1. Detection Loss Implementation
# ==========================================

def set_seed(seed):
    """Utility function to set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)

class DetectionLoss(nn.Module):
    """
    Anchor-free loss for YOLO-style outputs.
    Matches the single ground truth box to the top-K closest grid cells.
    Computes Focal Loss for classification and GIoU Loss for regression.
    """
    def __init__(self, num_classes=10, strides=(8, 16, 32), topk=5):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.topk = topk  # Number of positive grid cells to assign per ground truth box

    def make_anchors(self, feats):
        """Generates grid anchor points [6300, 2] and strides [6300, 1] for all feature levels."""
        anchor_points, stride_tensor = [], []
        for i, stride in enumerate(self.strides):
            _, _, h, w = feats[i].shape
            # Create grid coordinates
            sx = torch.arange(w, device=feats[i].device, dtype=feats[i].dtype) + 0.5
            sy = torch.arange(h, device=feats[i].device, dtype=feats[i].dtype) + 0.5
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, device=feats[i].device, dtype=feats[i].dtype))
            
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def forward(self, preds_det, targets_bbox, targets_cls):
        """
        preds_det: dict with "boxes" [B, 4, 6300], "scores" [B, 10, 6300], "feats"
        targets_bbox: [B, 4] in absolute pixel format (cx, cy, w, h)
        targets_cls: [B] class IDs
        """
        # Transpose predictions to [B, 6300, 4] and [B, 6300, 10]
        pred_boxes = preds_det["boxes"].transpose(1, 2)
        pred_scores = preds_det["scores"].transpose(1, 2)
        feats = preds_det["feats"]
        
        B = pred_boxes.shape[0]
        anchors, stride_tensor = self.make_anchors(feats) # [6300, 2], [6300, 1]
        
        # 1. Decode Predicted Boxes
        # We assume the model predicts dx, dy (offset) and tw, th (log scale)
        dxdy = pred_boxes[..., :2]
        twth = pred_boxes[..., 2:]
        
        pred_cxcy = (anchors.unsqueeze(0) + dxdy) * stride_tensor.unsqueeze(0)
        # Clamp twth to prevent exponent explosion/NaNs
        pred_wh = torch.exp(twth.clamp(max=10)) * stride_tensor.unsqueeze(0)
        decoded_boxes = torch.cat([pred_cxcy, pred_wh], dim=-1) # [B, 6300, 4] (cx, cy, w, h)
        
        # 2. Target Assignment (Top-K closest centers)
        loss_box = 0.
        target_scores = torch.zeros_like(pred_scores)
        
        gt_cxcy = targets_bbox[:, :2] # [B, 2]
        
        # Distance from all anchors to GT centers: shape [B, 6300]
        anchor_centers = anchors.unsqueeze(0) * stride_tensor.unsqueeze(0)
        dist = torch.norm(anchor_centers - gt_cxcy.unsqueeze(1), dim=-1)
        
        # Find the Top-K closest anchors for each image in the batch
        _, topk_idx = torch.topk(dist, self.topk, dim=1, largest=False) # [B, K]
        
        for b in range(B):
            pos_idx = topk_idx[b]
            cls_id = targets_cls[b]
            
            # Classification target: 1.0 for the correct class at positive anchor points
            target_scores[b, pos_idx, cls_id] = 1.0
            
            # Regression Target: GIoU loss for positive anchors only
            pos_pred_boxes = decoded_boxes[b, pos_idx] # [K, 4]
            pos_gt_boxes = targets_bbox[b].unsqueeze(0).expand(self.topk, 4) # [K, 4]
            
            pred_xyxy = cxcywh_to_xyxy(pos_pred_boxes)
            gt_xyxy = cxcywh_to_xyxy(pos_gt_boxes)
            
            # Compute Generalized IoU
            iou = generalized_box_iou(pred_xyxy, gt_xyxy) # [K, K]
            loss_box += (1.0 - iou.diag()).mean()
            
        loss_box = loss_box / B
        
        # 3. Focal Loss for Classification
        bce = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = 0.25 * (1 - pt)**2 * bce
        loss_cls = focal_loss.sum() / B  # Average across batch, sum across anchors/classes
        
        # Weighting: Regression is typically harder to learn early on, give it a slightly higher weight
        total_loss = 5.0 * loss_box + 1.0 * loss_cls
        
        return total_loss, {"loss_box": loss_box.item(), "loss_cls": loss_cls.item()}


# ==========================================
# 2. Freezing Logic & Setup
# ==========================================

def prepare_stage1_model(model):
    """Freezes classification and segmentation heads."""
    print("Freezing classification and segmentation heads for Stage 1...")
    
    # Freeze Cls Head
    for param in model.cls_head.parameters():
        param.requires_grad = False
        
    # Freeze Seg Head
    for param in model.seg_head.parameters():
        param.requires_grad = False
        
    # Ensure backbone, neck, and detect head are explicitly unfrozen
    for component in [model.backbone, model.neck, model.detect]:
        for param in component.parameters():
            param.requires_grad = True
            
    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable Parameters: {trainable:,} | Frozen Parameters: {frozen:,}")
    return model


# ==========================================
# 3. Main Training Loop
# ==========================================

def train_stage1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = os.path.join(args.save_dir, args.model_version)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Model
    print(f"Initializing {args.model_version}...")
    if args.model_version.upper() == "RGB_V1":
        model = RGB_V1(num_classes=10, reg_max=1)
    elif args.model_version.upper() == "RGB_V2":
        model = RGB_V2(num_classes=10, reg_max=1)
    elif args.model_version.upper() == "RGB_V3":
        model = RGB_V3(num_classes=10, reg_max=1)
    else:
        raise ValueError(f"Unknown model version: {args.model_version}")
    model = prepare_stage1_model(model)
    model.to(device)

    # Dataloaders
    print("Loading datasets...")
    train_dataset = HandGestureDataset_v2(
        root_dir=args.train_dir, 
        transform=SegAugment_v2(out_size=(480, 640))
    )
    val_dataset = HandGestureDataset_v2(
        root_dir=args.val_dir, 
        transform=None # No augmentation for validation
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=detection_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=detection_collate_fn, num_workers=4)

    # Optimizer and Loss
    criterion = DetectionLoss(num_classes=10, topk=5)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Outputs
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        # -- TRAIN --
        model.train()
        train_loss, train_box, train_cls = 0.0, 0.0, 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for images, masks, class_ids, bboxes in train_pbar:
            images = images.to(device)
            class_ids = class_ids.to(device)
            bboxes = bboxes.to(device)

            optimizer.zero_grad()
            
            # Forward pass: since seg/cls are frozen, we only care about "det"
            outputs = model(images)
            det_outputs = outputs["det"]
            
            # Compute loss
            loss, loss_dict = criterion(det_outputs, bboxes, class_ids)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # Gradient clipping to stabilize training
            optimizer.step()
            
            # Tracking
            train_loss += loss.item()
            train_box += loss_dict["loss_box"]
            train_cls += loss_dict["loss_cls"]
            
            train_pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Box": f"{loss_dict['loss_box']:.4f}", "Cls": f"{loss_dict['loss_cls']:.4f}"})

        scheduler.step()

        # -- VALIDATION --
        model.eval()
        val_loss, val_box, val_cls = 0.0, 0.0, 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]  ")
        with torch.no_grad():
            for images, masks, class_ids, bboxes in val_pbar:
                images = images.to(device)
                class_ids = class_ids.to(device)
                bboxes = bboxes.to(device)
                
                outputs = model(images)
                loss, loss_dict = criterion(outputs["det"], bboxes, class_ids)
                
                val_loss += loss.item()
                val_box += loss_dict["loss_box"]
                val_cls += loss_dict["loss_cls"]
                
                val_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\n=> Epoch {epoch} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} (Box: {train_box/len(train_loader):.4f}, Cls: {train_cls/len(train_loader):.4f})")
        print(f"   Val Loss:   {avg_val_loss:.4f} (Box: {val_box/len(val_loader):.4f}, Cls: {val_cls/len(val_loader):.4f})")

        # -- SAVING MODELS --
        last_model_path = os.path.join(args.save_dir, args.model_version, "last_model_stage1.pth")
        torch.save(model.state_dict(), last_model_path)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(args.save_dir, args.model_version, "best_model_stage1.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"   [*] Best model saved! (Val Loss: {best_val_loss:.4f})")
            
        print("-" * 50)


if __name__ == "__main__":
    set_seed(42)  # For reproducibility
    parser = argparse.ArgumentParser(description="Stage 1 Training: Detection Backbone & Neck")
    parser.add_argument("--train_dir", type=str, required=True, default="dataset/dataset_v1/train", help="Path to training dataset folder")
    parser.add_argument("--val_dir", type=str, required=True, default="dataset/dataset_v1/val", help="Path to validation dataset folder")
    parser.add_argument("--model_version", type=str, default="RGB_V3", choices=["RGB_V1", "RGB_V2", "RGB_V3"], help="Which model to use")
    parser.add_argument("--save_dir", type=str, default="weights", help="Directory to save model weights")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for Stage 1")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    
    args = parser.parse_args()
    train_stage1(args)
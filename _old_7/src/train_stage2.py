import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.ops import generalized_box_iou

# Import from your provided files
from model import RGB_V1, RGB_V2, RGB_V3
from dataloader import HandGestureDataset_v2, SegAugment_v2, detection_collate_fn

# ==========================================
# 1. Loss Implementations
# ==========================================

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)

class DetectionLoss(nn.Module):
    """Anchor-free loss for YOLO-style outputs (from Stage 1)."""
    def __init__(self, num_classes=10, strides=(8, 16, 32), topk=5):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.topk = topk

    def make_anchors(self, feats):
        anchor_points, stride_tensor = [], []
        for i, stride in enumerate(self.strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(w, device=feats[i].device, dtype=feats[i].dtype) + 0.5
            sy = torch.arange(h, device=feats[i].device, dtype=feats[i].dtype) + 0.5
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, device=feats[i].device, dtype=feats[i].dtype))
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def forward(self, preds_det, targets_bbox, targets_cls):
        pred_boxes = preds_det["boxes"].transpose(1, 2)
        pred_scores = preds_det["scores"].transpose(1, 2)
        feats = preds_det["feats"]
        
        B = pred_boxes.shape[0]
        anchors, stride_tensor = self.make_anchors(feats)
        
        dxdy = pred_boxes[..., :2]
        twth = pred_boxes[..., 2:]
        pred_cxcy = (anchors.unsqueeze(0) + dxdy) * stride_tensor.unsqueeze(0)
        pred_wh = torch.exp(twth.clamp(max=10)) * stride_tensor.unsqueeze(0)
        decoded_boxes = torch.cat([pred_cxcy, pred_wh], dim=-1)
        
        loss_box = 0.
        target_scores = torch.zeros_like(pred_scores)
        gt_cxcy = targets_bbox[:, :2]
        
        anchor_centers = anchors.unsqueeze(0) * stride_tensor.unsqueeze(0)
        dist = torch.norm(anchor_centers - gt_cxcy.unsqueeze(1), dim=-1)
        _, topk_idx = torch.topk(dist, self.topk, dim=1, largest=False)
        
        for b in range(B):
            pos_idx = topk_idx[b]
            cls_id = targets_cls[b]
            target_scores[b, pos_idx, cls_id] = 1.0
            
            pos_pred_boxes = decoded_boxes[b, pos_idx]
            pos_gt_boxes = targets_bbox[b].unsqueeze(0).expand(self.topk, 4)
            
            pred_xyxy = cxcywh_to_xyxy(pos_pred_boxes)
            gt_xyxy = cxcywh_to_xyxy(pos_gt_boxes)
            
            iou = generalized_box_iou(pred_xyxy, gt_xyxy)
            loss_box += (1.0 - iou.diag()).mean()
            
        loss_box = loss_box / B
        
        bce = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = 0.25 * (1 - pt)**2 * bce
        loss_cls = focal_loss.sum() / B 
        
        total_loss = 2.0 * loss_box + 1.0 * loss_cls
        return total_loss, {"loss_box": loss_box.item(), "loss_cls": loss_cls.item()}

class SegmentationLoss(nn.Module):
    """Combines BCE and Dice Loss for robust segmentation of small objects (hands)."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1e-5):
        # inputs: [B, 1, H, W] logits
        # targets: [B, 1, H, W] binary mask
        
        bce_loss = self.bce(inputs, targets)
        
        # Calculate Dice Loss
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        dice_loss = 1.0 - dice_score
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

# ==========================================
# 2. Main Training Loop
# ==========================================

def train_stage2(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # Load Stage 1 Pre-trained Weights
    stage_1_weights_path = os.path.join(args.save_dir, args.model_version, args.stage1_weights)
    if args.stage1_weights and os.path.exists(stage_1_weights_path):
        model.load_state_dict(torch.load(stage_1_weights_path, map_location=device, weights_only=True))
        print(f"[*] Successfully loaded Stage 1 weights from {stage_1_weights_path}")
    else:
        print("[!] Warning: No Stage 1 weights found. Training from scratch!")

    model.to(device)

    # Ensure everything is unfrozen for Stage 2
    for param in model.parameters():
        param.requires_grad = True

    # Dataloaders
    print("Loading datasets...")
    train_dataset = HandGestureDataset_v2(root_dir=args.train_dir, transform=SegAugment_v2(out_size=(480, 640)))
    val_dataset = HandGestureDataset_v2(root_dir=args.val_dir, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=detection_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=detection_collate_fn, num_workers=4)

    # ------------------------------------------
    # Differential Learning Rates Setup
    # ------------------------------------------
    base_lr = args.lr
    pretrained_lr = base_lr * args.lr_decay # Default: 1e-4 for pre-trained, 1e-3 for new heads
    
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': pretrained_lr},
        {'params': model.neck.parameters(), 'lr': pretrained_lr},
        {'params': model.detect.parameters(), 'lr': pretrained_lr},
        {'params': model.cls_head.parameters(), 'lr': base_lr},
        {'params': model.seg_head.parameters(), 'lr': base_lr}
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss Functions
    det_criterion = DetectionLoss(num_classes=10, topk=5)
    cls_criterion = nn.CrossEntropyLoss()
    seg_criterion = SegmentationLoss()

    # Outputs
    os.makedirs(os.path.join(args.save_dir, args.model_version), exist_ok=True)
    best_val_loss = float('inf')

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        # -- TRAIN --
        model.train()
        train_loss_total, t_det, t_cls, t_seg = 0.0, 0.0, 0.0, 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for images, masks, class_ids, bboxes in train_pbar:
            images = images.to(device)
            masks = masks.to(device) # Shape: [B, 1, 480, 640]
            class_ids = class_ids.to(device) # Shape: [B]
            bboxes = bboxes.to(device)

            optimizer.zero_grad()
            
            # Forward pass: Full Multi-Task inference
            outputs = model(images)
            
            # 1. Detection Loss
            loss_det, _ = det_criterion(outputs["det"], bboxes, class_ids)
            
            # 2. Classification Loss (Global image class)
            loss_cls = cls_criterion(outputs["cls"], class_ids)
            
            # 3. Segmentation Loss
            loss_seg = seg_criterion(outputs["seg"], masks)
            
            # ------------------------------------------
            # Loss Weighting
            # ------------------------------------------
            loss = (args.w_det * loss_det) + (args.w_cls * loss_cls) + (args.w_seg * loss_seg)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            # Tracking
            train_loss_total += loss.item()
            t_det += loss_det.item()
            t_cls += loss_cls.item()
            t_seg += loss_seg.item()
            
            train_pbar.set_postfix({
                "Ttl": f"{loss.item():.3f}", 
                "Det": f"{loss_det.item():.3f}", 
                "Cls": f"{loss_cls.item():.3f}",
                "Seg": f"{loss_seg.item():.3f}"
            })

        scheduler.step()

        # -- VALIDATION --
        model.eval()
        val_loss_total, v_det, v_cls, v_seg = 0.0, 0.0, 0.0, 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]  ")
        with torch.no_grad():
            for images, masks, class_ids, bboxes in val_pbar:
                images = images.to(device)
                masks = masks.to(device)
                class_ids = class_ids.to(device)
                bboxes = bboxes.to(device)
                
                outputs = model(images)
                
                loss_det, _ = det_criterion(outputs["det"], bboxes, class_ids)
                loss_cls = cls_criterion(outputs["cls"], class_ids)
                loss_seg = seg_criterion(outputs["seg"], masks)
                
                loss = (args.w_det * loss_det) + (args.w_cls * loss_cls) + (args.w_seg * loss_seg)
                
                val_loss_total += loss.item()
                v_det += loss_det.item()
                v_cls += loss_cls.item()
                v_seg += loss_seg.item()
                
                val_pbar.set_postfix({"Ttl": f"{loss.item():.3f}"})

        # Calculate averages
        avg_train_loss = train_loss_total / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)
        
        print(f"\n=> Epoch {epoch} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Det: {t_det/len(train_loader):.4f} | Cls: {t_cls/len(train_loader):.4f} | Seg: {t_seg/len(train_loader):.4f}")
        print(f"   Val Loss:   {avg_val_loss:.4f} | Det: {v_det/len(val_loader):.4f} | Cls: {v_cls/len(val_loader):.4f} | Seg: {v_seg/len(val_loader):.4f}")

        # -- SAVING MODELS --
        last_model_path = os.path.join(args.save_dir, args.model_version, "last_model_stage2.pth")
        torch.save(model.state_dict(), last_model_path)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(args.save_dir, args.model_version, "best_model_stage2.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"   [*] Best model saved! (Total Val Loss: {best_val_loss:.4f})")
            
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Training: Full Multi-Task Fine-tuning")
    parser.add_argument("--train_dir", type=str, default="dataset/dataset_v1/train", help="Path to training dataset folder")
    parser.add_argument("--val_dir", type=str, default="dataset/dataset_v1/val", help="Path to validation dataset folder")
    parser.add_argument("--model_version", type=str, default="RGB_V2", choices=["RGB_V1", "RGB_V2", "RGB_V3"], help="Which model to use")
    parser.add_argument("--stage1_weights", type=str, default="best_model_stage1.pth", help="Path to Stage 1 pretrained weights")
    parser.add_argument("--save_dir", type=str, default="weights", help="Directory to save model weights")
    
    # Training Parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for Stage 2")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    
    # Learning Rates
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate for NEW heads")
    parser.add_argument("--lr_decay", type=float, default=0.1, help="Multiplier for pretrained modules (e.g. 0.1 * 1e-3 = 1e-4)")
    
    # Loss Weights
    parser.add_argument("--w_det", type=float, default=1.0, help="Weight multiplier for Detection Loss")
    parser.add_argument("--w_cls", type=float, default=1.0, help="Weight multiplier for Classification Loss")
    parser.add_argument("--w_seg", type=float, default=5.0, help="Weight multiplier for Segmentation Loss")
    
    args = parser.parse_args()
    train_stage2(args)
import math
import os
import argparse
import torch
import torch.nn as nn
from model import HandGestureModel_v4, ModelConfig
from torch.utils.data import DataLoader, Subset
# Note: Ensure dataloader and utils are correctly imported in your environment
from dataloader import HandGestureDataset, SegAugment 
import random
import torch.nn.functional as F
import torchinfo
import numpy as np

from utils import WarmupCosineScheduler

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def empty_cache():
    torch.cuda.empty_cache()

class Stage_1_Config:
    epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    lr = 5e-4
    weight_decay = 1e-4
    output_dir = 'outputs/stage_1/train_1'

# ==========================================
# 1. ANCHOR-FREE ASSIGNMENT & LOSS FUNCTIONS
# ==========================================

def generate_grid_points(image_size, strides=[8, 16, 32], device='cpu'):
    H, W = image_size
    all_points = []
    for stride in strides:
        h_grid, w_grid = H // stride, W // stride
        shift_y = torch.arange(0, h_grid, device=device) * stride + (stride // 2)
        shift_x = torch.arange(0, w_grid, device=device) * stride + (stride // 2)
        y, x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        points = torch.stack([x.flatten(), y.flatten()], dim=-1)
        all_points.append(points)
    return torch.cat(all_points, dim=0)

def assign_targets(grid_points, gt_boxes, gt_classes, num_classes=10, center_sample_ratio=0.5):
    batch_size, num_objs, _ = gt_boxes.shape
    num_points = grid_points.shape[0]

    pts = grid_points.view(1, 1, num_points, 2) 
    boxes = gt_boxes.unsqueeze(2)               

    x, y = pts[..., 0], pts[..., 1]
    xmin, ymin = boxes[..., 0], boxes[..., 1]
    xmax, ymax = boxes[..., 2], boxes[..., 3]

    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    w, h = xmax - xmin, ymax - ymin

    c_xmin, c_xmax = cx - (w * center_sample_ratio / 2.0), cx + (w * center_sample_ratio / 2.0)
    c_ymin, c_ymax = cy - (h * center_sample_ratio / 2.0), cy + (h * center_sample_ratio / 2.0)

    inside_original = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    inside_center = (x >= c_xmin) & (x <= c_xmax) & (y >= c_ymin) & (y <= c_ymax)
    is_inside = inside_original & inside_center

    areas = w * h
    point_areas = areas.expand(-1, -1, num_points).clone()
    point_areas[~is_inside] = float('inf')

    min_areas, matched_obj_idx = torch.min(point_areas, dim=1)
    is_positive = min_areas < float('inf') 

    target_cls = torch.zeros(batch_size, num_classes, num_points, device=gt_boxes.device)
    for b in range(batch_size):
        pos_point_indices = torch.arange(num_points, device=gt_boxes.device)[is_positive[b]]
        obj_indices = matched_obj_idx[b][is_positive[b]]
        pos_classes = gt_classes[b, obj_indices].long()
        target_cls[b, pos_classes, pos_point_indices] = 1.0

    return target_cls, is_positive, matched_obj_idx

class DetectionLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def cxcywh_to_xyxy(self, boxes):
        """Converts [cx, cy, w, h] to [xmin, ymin, xmax, ymax] with safety constraints."""
        cx = boxes[..., 0]
        cy = boxes[..., 1]
        
        # FIX: Ensure width and height are strictly positive!
        # F.softplus allows negative raw logits to safely approach near-zero
        w = F.softplus(boxes[..., 2])
        h = F.softplus(boxes[..., 3])
        
        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2
        
        return torch.stack([xmin, ymin, xmax, ymax], dim=-1)

    def calculate_giou(self, pred_boxes, gt_boxes):
        inter_xmin = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
        inter_ymin = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
        inter_xmax = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
        inter_ymax = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])
        inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(inter_ymax - inter_ymin, min=0)

        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        union_area = pred_area + gt_area - inter_area + 1e-6

        iou = inter_area / (union_area + 1e-6)
        

        enclose_xmin = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
        enclose_ymin = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
        enclose_xmax = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
        enclose_ymax = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
        enclose_area = torch.clamp(enclose_xmax - enclose_xmin, min=0) * torch.clamp(enclose_ymax - enclose_ymin, min=0) + 1e-6
        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)

        return 1.0 - giou

    def forward(self, preds, gt_boxes_xyxy, grid_points, target_cls, is_positive, matched_obj_idx):
        pred_bboxes, pred_cls, pred_center = preds
        batch_size = pred_bboxes.shape[0]

        # 1. Classification Loss (Focal)
        bce_loss = self.bce(pred_cls, target_cls)
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        loss_cls = focal_loss.sum() / max(1, is_positive.sum())

        loss_reg = torch.tensor(0.0, device=pred_bboxes.device)
        loss_center = torch.tensor(0.0, device=pred_bboxes.device)

        if is_positive.sum() > 0:
            # 2. Extract Positives
            flat_pred_bboxes = pred_bboxes.permute(0, 2, 1).reshape(-1, 4)[is_positive.flatten()]
            flat_pred_center = pred_center.permute(0, 2, 1).reshape(-1)[is_positive.flatten()]
            
            flat_gt_boxes = []
            flat_grid_points = []
            for b in range(batch_size):
                pos_idx = is_positive[b]
                gt_idx = matched_obj_idx[b][pos_idx]
                flat_gt_boxes.append(gt_boxes_xyxy[b, gt_idx])
                flat_grid_points.append(grid_points[pos_idx])
                
            flat_gt_boxes = torch.cat(flat_gt_boxes, dim=0)
            flat_grid_points = torch.cat(flat_grid_points, dim=0)

            # 3. Centerness Loss
            x, y = flat_grid_points[:, 0], flat_grid_points[:, 1]
            l = x - flat_gt_boxes[:, 0]
            t = y - flat_gt_boxes[:, 1]
            r = flat_gt_boxes[:, 2] - x
            b = flat_gt_boxes[:, 3] - y
            
            target_centerness = torch.sqrt(
                (torch.min(l, r) / torch.max(l, r).clamp(min=1e-6)) * (torch.min(t, b) / torch.max(t, b).clamp(min=1e-6))
            )
            loss_center = self.bce(flat_pred_center, target_centerness).mean()

            # 4. Regression Loss (GIoU weighted by target centerness)
            # 4.1. Flatten predictions as usual
            flat_pred_bboxes = pred_bboxes.permute(0, 2, 1).reshape(-1, 4)[is_positive.flatten()]
            
            # 4.2. Extract raw network outputs
            pred_raw_cx = flat_pred_bboxes[:, 0]
            pred_raw_cy = flat_pred_bboxes[:, 1]
            pred_raw_w = flat_pred_bboxes[:, 2]
            pred_raw_h = flat_pred_bboxes[:, 3]
            
            # 4.3. CONVERT TO ABSOLUTE PIXELS USING GRID OFFSETS
            # Add the feature map's grid (x,y) location to the predicted center offsets
            pred_cx = flat_grid_points[:, 0] + pred_raw_cx 
            pred_cy = flat_grid_points[:, 1] + pred_raw_cy 
            
            # Use exponential to allow the network to predict small values (e.g., 4.5) 
            # that smoothly scale up to large pixel widths (e.g., exp(4.5) = 90 pixels)
            pred_w = torch.exp(pred_raw_w) 
            pred_h = torch.exp(pred_raw_h)
            
            # 4. Re-stack into standard [cx, cy, w, h] format
            decoded_pred_bboxes = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)
            
            # Now pass decoded_pred_bboxes to your cxcywh_to_xyxy function and GIoU!
            pred_xyxy = self.cxcywh_to_xyxy(decoded_pred_bboxes)
            giou = self.calculate_giou(pred_xyxy, flat_gt_boxes)
            loss_reg = (giou * target_centerness).mean()

        # Weighted Sum
        lambda_cls, lambda_reg, lambda_center = 1.0, 2.0, 1.0
        total_loss = (lambda_cls * loss_cls) + (lambda_reg * loss_reg) + (lambda_center * loss_center)

        return total_loss, loss_cls, loss_reg, loss_center


# ==========================================
# 2. TRAINING LOOP
# ==========================================

def train_stage_1(config: Stage_1_Config):
    os.makedirs(config.output_dir, exist_ok=True)

    # --- Dataset & Dataloader ---
    train_dataset = HandGestureDataset(root_dir='dataset/dataset_v1/train', transform=SegAugment())
    val_dataset = HandGestureDataset(root_dir='dataset/dataset_v1/val', transform=None)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # --- Model ---
    model_config = ModelConfig()
    model = HandGestureModel_v4(model_config).to(config.device)
    
    # Freeze heads not in Stage 1
    for param in model.classifier_head.parameters(): param.requires_grad = False
    for param in model.segmentation_head.parameters(): param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 1 Training (Backbone + Detection Head)\n Total Trainable Params: {trainable_params:,}")

    # --- Optimizer, Scheduler, & Loss ---
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=int(0.1 * total_steps), total_steps=total_steps)
    criterion = DetectionLoss().to(config.device)

    # Grid points will be generated on the first batch
    grid_points = None

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for i, (image_tensor, mask_tensor, class_id, bbox) in enumerate(train_loader):
            image_tensor = image_tensor.to(config.device)
            class_id = class_id.to(config.device)
            bbox = bbox.to(config.device)

            # Ensure ground truth is [Batch, Num_Objs, ...]
            if bbox.ndim == 2:
                bbox = bbox.unsqueeze(1)
            if class_id.ndim == 1:
                class_id = class_id.unsqueeze(1)

            # Generate grid points dynamically based on the image size
            if grid_points is None:
                H, W = image_tensor.shape[2], image_tensor.shape[3]
                grid_points = generate_grid_points((H, W), device=config.device)

            optimizer.zero_grad()
            
            # Forward Pass (Note: Ensure HandGestureModel_v4 returns centerness)
            cls_logits, bbox_preds, bbox_cls, centerness, seg_map = model(image_tensor)

            # 1. Convert GT boxes from [cx, cy, w, h] to [xmin, ymin, xmax, ymax] for target assignment
            gt_boxes_xyxy = criterion.cxcywh_to_xyxy(bbox)

            # 2. Label Assignment
            target_cls, is_positive, matched_obj_idx = assign_targets(
                grid_points=grid_points,
                gt_boxes=gt_boxes_xyxy,
                gt_classes=class_id,
                num_classes=model_config.num_classes,
                center_sample_ratio=0.5
            )

            # 3. Calculate Loss
            preds = (bbox_preds, bbox_cls, centerness)
            loss, loss_cls, loss_reg, loss_center = criterion(
                preds, gt_boxes_xyxy, grid_points, target_cls, is_positive, matched_obj_idx
            )

            # Backpropagation
            loss.backward()
            # Gradient clipping is highly recommended for anchor-free object detectors
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch [{epoch}/{config.epochs}] Batch {i}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f} (Cls: {loss_cls.item():.4f}, "
                      f"Reg: {loss_reg.item():.4f}, Center: {loss_center.item():.4f})")

        train_losses.append(epoch_loss / len(train_loader))

                
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_image_tensor, val_mask_tensor, val_class_id, val_bbox in val_loader:
                val_image_tensor = val_image_tensor.to(config.device)
                val_class_id = val_class_id.to(config.device)
                val_bbox = val_bbox.to(config.device)

                # Ensure ground truth is [Batch, Num_Objs, ...]
                if val_bbox.ndim == 2:
                    val_bbox = val_bbox.unsqueeze(1)
                if val_class_id.ndim == 1:
                    val_class_id = val_class_id.unsqueeze(1)
                # Generate grid points dynamically based on the image size
                if grid_points is None:
                    H, W = val_image_tensor.shape[2], val_image_tensor.shape[3]
                    grid_points = generate_grid_points((H, W), device=config.device)
                
                val_cls_logits, val_bbox_preds, val_bbox_cls, val_centerness, val_seg_map = model(val_image_tensor)

                # 1. Convert GT boxes from [cx, cy, w, h] to [xmin, ymin, xmax, ymax] for target assignment
                gt_boxes_xyxy = criterion.cxcywh_to_xyxy(bbox)

                # 2. Label Assignment
                target_cls, is_positive, matched_obj_idx = assign_targets(
                    grid_points=grid_points,
                    gt_boxes=gt_boxes_xyxy,
                    gt_classes=class_id,
                    num_classes=model_config.num_classes,
                    center_sample_ratio=0.5
                )

                # 3. Calculate Loss
                preds = (bbox_preds, bbox_cls, centerness)
                loss, loss_cls, loss_reg, loss_center = criterion(
                    preds, gt_boxes_xyxy, grid_points, target_cls, is_positive, matched_obj_idx
                )
                val_loss += loss.item()

            val_losses.append(val_loss / len(val_loader))

            if val_loss / len(val_loader) < best_val_loss:
                best_val_loss = val_loss / len(val_loader)
                torch.save(model.state_dict(), os.path.join(config.output_dir, f"stage_1_best.pt"))
                print(f"New best model saved at epoch {epoch} with val loss {best_val_loss:.4f}")
                

        print(f"==== Epoch {epoch} Complete | Train Avg Loss: {epoch_loss / len(train_loader):.4f} | Val Avg Loss: {val_loss / len(val_loader):.4f} ====")
        torch.save(model.state_dict(), os.path.join(config.output_dir, f"stage_1_last.pt"))

if __name__ == "__main__":
    set_seed(42)
    empty_cache()
    config = Stage_1_Config()
    train_stage_1(config)
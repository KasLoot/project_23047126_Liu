import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import confusion_matrix
import math
import torch.nn.functional as F


def _xywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    hw, hh = w * 0.5, h * 0.5
    return torch.stack((cx - hw, cy - hh, cx + hw, cy + hh), -1)

def _bbox_iou_xyxy(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Element-wise IoU between (N,4) tensors in xyxy format."""
    ix1 = torch.maximum(pred[:, 0], tgt[:, 0])
    iy1 = torch.maximum(pred[:, 1], tgt[:, 1])
    ix2 = torch.minimum(pred[:, 2], tgt[:, 2])
    iy2 = torch.minimum(pred[:, 3], tgt[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    a1 = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    a2 = (tgt[:, 2] - tgt[:, 0]).clamp(min=0) * (tgt[:, 3] - tgt[:, 1]).clamp(min=0)
    return inter / (a1 + a2 - inter).clamp(min=1e-6)

def _focal_bce(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Sigmoid focal loss normalized by positive anchors."""
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * ((1 - p_t) ** gamma) * ce
    
    num_positives = torch.clamp(targets.sum(), min=1.0)
    return loss.sum() / num_positives

def _ciou_loss(pred_xyxy: torch.Tensor, tgt_xyxy: torch.Tensor) -> torch.Tensor:
    """Complete-IoU loss (mean-reduced). Inputs are (N, 4) xyxy tensors."""
    eps = 1e-7

    # Intersection
    ix1 = torch.maximum(pred_xyxy[:, 0], tgt_xyxy[:, 0])
    iy1 = torch.maximum(pred_xyxy[:, 1], tgt_xyxy[:, 1])
    ix2 = torch.minimum(pred_xyxy[:, 2], tgt_xyxy[:, 2])
    iy2 = torch.minimum(pred_xyxy[:, 3], tgt_xyxy[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    # Areas
    pw = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0)
    ph = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0)
    tw = (tgt_xyxy[:, 2] - tgt_xyxy[:, 0]).clamp(min=0)
    th = (tgt_xyxy[:, 3] - tgt_xyxy[:, 1]).clamp(min=0)
    area_pred = pw * ph
    area_tgt = tw * th
    union = area_pred + area_tgt - inter
    iou = inter / union.clamp(min=eps)

    # Enclosing box
    ex1 = torch.minimum(pred_xyxy[:, 0], tgt_xyxy[:, 0])
    ey1 = torch.minimum(pred_xyxy[:, 1], tgt_xyxy[:, 1])
    ex2 = torch.maximum(pred_xyxy[:, 2], tgt_xyxy[:, 2])
    ey2 = torch.maximum(pred_xyxy[:, 3], tgt_xyxy[:, 3])
    c_diag_sq = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + eps

    # Center distance
    pcx = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) * 0.5
    pcy = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) * 0.5
    tcx = (tgt_xyxy[:, 0] + tgt_xyxy[:, 2]) * 0.5
    tcy = (tgt_xyxy[:, 1] + tgt_xyxy[:, 3]) * 0.5
    rho_sq = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    # Aspect ratio consistency
    v = (4.0 / (math.pi ** 2)) * (torch.atan(tw / th.clamp(min=eps)) - torch.atan(pw / ph.clamp(min=eps))) ** 2
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    ciou = iou - rho_sq / c_diag_sq - alpha * v
    return (1.0 - ciou).mean()

def generate_anchors(input_h: int, input_w: int, strides: list[int], device: torch.device):
    """Generates anchor centers and strides dynamically based on input resolution."""
    centers_all, strides_all = [], []
    for stride in strides:
        h, w = input_h // stride, input_w // stride
        yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        cx = (xx + 0.5) * stride
        cy = (yy + 0.5) * stride
        centers = torch.stack((cx, cy), dim=-1).reshape(-1, 2)
        stride_tensor = torch.full((h * w, 2), fill_value=float(stride), device=device)
        centers_all.append(centers)
        strides_all.append(stride_tensor)
    return torch.cat(centers_all, 0), torch.cat(strides_all, 0)

def compute_detection_loss(bbox_preds, bbox_cls, bboxes_xywh, class_ids, input_h, input_w, num_classes=10):
    device = bbox_preds.device
    bsz = bbox_preds.shape[0]
    
    if bboxes_xywh.max() <= 1.0:
        bboxes_xywh = bboxes_xywh.clone()
        bboxes_xywh[:, [0, 2]] *= input_w
        bboxes_xywh[:, [1, 3]] *= input_h

    # EDITED: Switched to the [32, 16, 8] strides
    strides_list = [32, 16, 8] 
    centers, strides = generate_anchors(input_h, input_w, strides_list, device)
    num_anchors = centers.shape[0] # Down to ~6,300

    pred_boxes_all = bbox_preds.permute(0, 2, 1) 
    pred_scores_all = bbox_cls.permute(0, 2, 1) 

    gt_xywh = bboxes_xywh.clone()
    bboxes_xyxy = _xywh_to_xyxy(gt_xywh)
    bboxes_xyxy[:, [0, 2]] = bboxes_xyxy[:, [0, 2]].clamp(0, float(input_w - 1))
    bboxes_xyxy[:, [1, 3]] = bboxes_xyxy[:, [1, 3]].clamp(0, float(input_h - 1))

    topk = 13
    cls_target = torch.zeros((bsz, num_classes, num_anchors), dtype=pred_scores_all.dtype, device=device)
    pos_mask = torch.zeros((bsz, num_anchors), dtype=torch.bool, device=device)

    for bi in range(bsz):
        gt_cx, gt_cy = gt_xywh[bi, 0], gt_xywh[bi, 1]
        d2 = (centers[:, 0] - gt_cx) ** 2 + (centers[:, 1] - gt_cy) ** 2
        
        _, topk_idx = d2.topk(min(topk, num_anchors), largest=False)
        pos_mask[bi, topk_idx] = True
        cls_id = int(class_ids[bi].item())
        cls_target[bi, cls_id, topk_idx] = 1.0

    cls_loss = _focal_bce(bbox_cls, cls_target)

    pred_xyxy_list, tgt_xyxy_list = [], []
    for bi in range(bsz):
        pos_idx = pos_mask[bi].nonzero(as_tuple=True)[0]
        if pos_idx.numel() == 0:
            continue

        raw = pred_boxes_all[bi, pos_idx]
        anchor_c = centers[pos_idx]
        anchor_s = strides[pos_idx]

        pred_xy = anchor_c + torch.tanh(raw[:, :2]) * anchor_s
        pred_wh = torch.exp(raw[:, 2:].clamp(-4.0, 4.0)) * anchor_s
        pred_xywh_i = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_xyxy_i = _xywh_to_xyxy(pred_xywh_i)
        
        pred_xyxy_i[:, [0, 2]] = pred_xyxy_i[:, [0, 2]].clamp(0, float(input_w - 1))
        pred_xyxy_i[:, [1, 3]] = pred_xyxy_i[:, [1, 3]].clamp(0, float(input_h - 1))

        tgt_xyxy_i = bboxes_xyxy[bi].unsqueeze(0).expand_as(pred_xyxy_i)
        
        pred_xyxy_list.append(pred_xyxy_i)
        tgt_xyxy_list.append(tgt_xyxy_i)

    if len(pred_xyxy_list) > 0:
        box_loss = _ciou_loss(torch.cat(pred_xyxy_list, dim=0), torch.cat(tgt_xyxy_list, dim=0))
    else:
        box_loss = torch.tensor(0.0, device=device)

    total_loss = (1.0 * cls_loss) + (2.0 * box_loss)
    return total_loss, cls_loss.detach().item(), box_loss.detach().item()


def decode_predictions(bbox_preds, bbox_cls, input_h, input_w, conf_thresh=0.3):
    device = bbox_preds.device
    bsz = bbox_preds.shape[0]
    
    # EDITED: Switched to the [32, 16, 8] strides
    strides_list = [32, 16, 8]
    centers, strides = generate_anchors(input_h, input_w, strides_list, device)
    
    pred_boxes = bbox_preds.permute(0, 2, 1)
    pred_scores = bbox_cls.sigmoid().permute(0, 2, 1)
    
    max_scores, cls_idx = pred_scores.max(dim=-1)
    
    batch_results = []
    for bi in range(bsz):
        mask = max_scores[bi] > conf_thresh
        if not mask.any():
            batch_results.append(torch.empty((0, 6), device=device))
            continue
            
        b_scores = max_scores[bi][mask]
        b_cls = cls_idx[bi][mask]
        b_raw_boxes = pred_boxes[bi][mask]
        b_centers = centers[mask]
        b_strides = strides[mask]
        
        pred_xy = b_centers + torch.tanh(b_raw_boxes[:, :2]) * b_strides
        pred_wh = torch.exp(b_raw_boxes[:, 2:].clamp(-4.0, 4.0)) * b_strides
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        
        pred_xyxy = _xywh_to_xyxy(pred_xywh)
        pred_xyxy[:, [0, 2]] = pred_xyxy[:, [0, 2]].clamp(0, float(input_w - 1))
        pred_xyxy[:, [1, 3]] = pred_xyxy[:, [1, 3]].clamp(0, float(input_h - 1))
        
        result = torch.cat([pred_xyxy, b_scores.unsqueeze(1), b_cls.unsqueeze(1).float()], dim=1)
        batch_results.append(result)
        
    return batch_results

def evaluate_batch(decoded_preds, gt_xywh, gt_classes, input_h, input_w, iou_thresh=0.5):
    y_true, y_pred = [], []
    
    if gt_xywh.max() <= 1.0:
        gt_xywh = gt_xywh.clone()
        gt_xywh[:, [0, 2]] *= input_w
        gt_xywh[:, [1, 3]] *= input_h
        
    gt_xyxy = _xywh_to_xyxy(gt_xywh)
    gt_xyxy[:, [0, 2]] = gt_xyxy[:, [0, 2]].clamp(0, float(input_w - 1))
    gt_xyxy[:, [1, 3]] = gt_xyxy[:, [1, 3]].clamp(0, float(input_h - 1))
    
    for bi in range(len(decoded_preds)):
        preds = decoded_preds[bi]
        gt_box = gt_xyxy[bi].unsqueeze(0)
        gt_cls = int(gt_classes[bi].item())
        
        if len(preds) == 0:
            y_true.append(gt_cls)
            y_pred.append(10)
            continue
            
        gt_box_expanded = gt_box.expand(preds.shape[0], -1)
        ious = _bbox_iou_xyxy(preds[:, :4], gt_box_expanded)
        max_iou_val, max_iou_idx = ious.max(dim=0)
        
        if max_iou_val > iou_thresh:
            pred_cls = int(preds[max_iou_idx, 5].item())
            y_true.append(gt_cls)
            y_pred.append(pred_cls)
        else:
            y_true.append(gt_cls)
            y_pred.append(10) 
            
    return y_true, y_pred

def plot_validation_samples(samples, output_dir):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, (img, gt_box, gt_cls, preds) in enumerate(samples[:10]):
        ax = axes[i]
        img_np = img.float().permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)
        
        cx, cy, w, h = gt_box.float().cpu().numpy()
        
        if w <= 1.0 and h <= 1.0:
            input_h, input_w = img.shape[-2], img.shape[-1]
            cx *= input_w; w *= input_w
            cy *= input_h; h *= input_h

        x1, y1 = cx - w/2, cy - h/2
        rect_gt = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='g', facecolor='none', label=f"GT: {int(gt_cls)}")
        ax.add_patch(rect_gt)
        
        if len(preds) > 0:
            best_pred = preds[preds[:, 4].argmax()].float().cpu().numpy()
            px1, py1, px2, py2, pconf, pcls = best_pred
            rect_pred = patches.Rectangle((px1, py1), px2-px1, py2-py1, linewidth=2, edgecolor='r', facecolor='none', linestyle='--', label=f"Pred: {int(pcls)} ({pconf:.2f})")
            ax.add_patch(rect_pred)
            
        ax.axis('off')
        ax.legend(loc='upper left', fontsize=8)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "val_samples.png"))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, num_classes, output_dir):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes + 1)))
    plt.figure(figsize=(10, 8))
    
    class_names = [str(i) for i in range(num_classes)] + ["BG"]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Validation Confusion Matrix (IoU > 0.5)')
    
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
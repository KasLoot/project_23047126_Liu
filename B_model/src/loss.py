import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from model import HandGestureMultiTask


# ==========================================
# 1. Utilities
# ==========================================

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Converts [cx, cy, w, h] to [x_min, y_min, x_max, y_max]."""
    x1 = boxes[..., 0] - boxes[..., 2] / 2
    y1 = boxes[..., 1] - boxes[..., 3] / 2
    x2 = boxes[..., 0] + boxes[..., 2] / 2
    y2 = boxes[..., 1] + boxes[..., 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def generate_anchors(neck_features: list[torch.Tensor], strides: list[int] = [8, 16, 32]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamically generates (x, y) center coordinates for all 6300 grid cells.
    Returns:
        anchors: (6300, 2) tensor of absolute [x, y] center coordinates.
        anchor_strides: (6300,) tensor of the stride corresponding to each anchor.
    """
    device = neck_features[0].device
    anchors = []
    anchor_strides = []

    for feat, stride in zip(neck_features, strides):
        _, _, h, w = feat.shape
        
        # Create grid coordinates (e.g., 0, 1, 2, ..., w-1)
        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        
        # Move coordinates to the center of the cell and scale by stride
        # Example: Cell (0,0) with stride 8 has its center at pixel (4, 4)
        center_x = (grid_x.flatten() + 0.5) * stride
        center_y = (grid_y.flatten() + 0.5) * stride
        
        anchors.append(torch.stack([center_x, center_y], dim=-1))
        anchor_strides.append(torch.full((h * w,), stride, device=device))

    return torch.cat(anchors, dim=0), torch.cat(anchor_strides, dim=0)


def decode_predictions(pred_boxes: torch.Tensor, anchors: torch.Tensor, strides: torch.Tensor) -> torch.Tensor:
    """
    Converts raw neural network outputs into absolute (x1, y1, x2, y2) bounding boxes.
    Assumes pred_boxes are (left, top, right, bottom) distances from the anchor.
    """
    # Enforce positive distances using ReLU, then scale by the feature map stride
    distances = F.relu(pred_boxes) * strides.view(1, -1, 1)
    
    # LTRB to XYXY coordinates
    x1 = anchors[:, 0].unsqueeze(0) - distances[..., 0]
    y1 = anchors[:, 1].unsqueeze(0) - distances[..., 1]
    x2 = anchors[:, 0].unsqueeze(0) + distances[..., 2]
    y2 = anchors[:, 1].unsqueeze(0) + distances[..., 3]
    
    return torch.stack([x1, y1, x2, y2], dim=-1)


# ==========================================
# 2. Task Aligned Assigner
# ==========================================

class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk: int = 10, alpha: float = 0.5, beta: float = 6.0):
        super().__init__()
        self.topk = topk    # Number of positive anchors to select per Ground Truth
        self.alpha = alpha  # Weight of classification score in alignment metric
        self.beta = beta    # Weight of IoU in alignment metric

    @torch.no_grad()
    def forward(self, pred_scores: torch.Tensor, pred_bboxes_xyxy: torch.Tensor, 
                anchors: torch.Tensor, gt_labels: torch.Tensor, gt_bboxes_xyxy: torch.Tensor):
        """
        Assigns the 6300 predictions to exactly 1 ground truth target per image.
        """
        batch_size, num_anchors, num_classes = pred_scores.shape
        device = pred_scores.device

        # Output target tensors
        target_bboxes = torch.zeros_like(pred_bboxes_xyxy)
        target_scores = torch.zeros_like(pred_scores)
        fg_mask = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=device)

        # Loop over each image in the batch
        for b in range(batch_size):
            # For this simplified dataset, we assume 1 Ground Truth object per image.
            gt_box = gt_bboxes_xyxy[b] # Shape: (1, 4)
            gt_cls = int(gt_labels[b].item())

            # Skip if the GT box is degenerate (e.g., width or height is 0)
            if (gt_box[0, 2] <= gt_box[0, 0]) or (gt_box[0, 3] <= gt_box[0, 1]):
                continue

            # --- Step 1: Center Prior (Spatial Matching) ---
            # Find all anchors whose centers fall physically inside the Ground Truth Box
            cx, cy = anchors[:, 0], anchors[:, 1]
            is_in_gt = (cx >= gt_box[0, 0]) & (cx <= gt_box[0, 2]) & (cy >= gt_box[0, 1]) & (cy <= gt_box[0, 3])
            valid_indices = torch.where(is_in_gt)[0]

            # If no anchors fall inside (e.g., extremely small object), fallback to the absolute closest center
            if len(valid_indices) == 0:
                distances = (cx - (gt_box[0, 0] + gt_box[0, 2])/2)**2 + (cy - (gt_box[0, 1] + gt_box[0, 3])/2)**2
                valid_indices = torch.argmin(distances).unsqueeze(0)

            # --- Step 2: Alignment Metric ---
            # Subset only the predictions that fell inside the GT box
            valid_pred_boxes = pred_bboxes_xyxy[b, valid_indices]        # (Num_Valid, 4)
            
            # Use Sigmoid to get probability of the true class
            valid_pred_scores = pred_scores[b, valid_indices, gt_cls].sigmoid() # (Num_Valid,)

            # Calculate actual IoU between these valid predictions and the GT box
            ious = ops.box_iou(valid_pred_boxes, gt_box).squeeze(-1)    # (Num_Valid,)
            ious = torch.clamp(ious, min=1e-9)

            # Metric = (Class Score ^ alpha) * (IoU ^ beta)
            alignment_metrics = (valid_pred_scores ** self.alpha) * (ious ** self.beta)

            # --- Step 3: Top-K Selection ---
            # Pick the top 10 best aligned predictions to act as our Positives
            k = min(self.topk, len(valid_indices))
            _, topk_idx = torch.topk(alignment_metrics, k)
            
            final_pos_indices = valid_indices[topk_idx]
            final_ious = ious[topk_idx]

            # --- Step 4: Populate Targets ---
            fg_mask[b, final_pos_indices] = True
            target_bboxes[b, final_pos_indices] = gt_box[0]
            
            # Soft Labels: We set the target probability to the IoU score, NOT 1.0. 
            # This teaches the network to predict higher confidence for better boxes!
            target_scores[b, final_pos_indices, gt_cls] = final_ious

        return target_bboxes, target_scores, fg_mask


# ==========================================
# 3. Complete Loss Module
# ==========================================

class YOLODetectionLoss(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.assigner = TaskAlignedAssigner(topk=10)
        
        # Loss multipliers
        self.cls_weight = 1.0
        self.box_weight = 2.5

    def _normalize_targets(
        self,
        gt_bboxes_cxcywh: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gt_bboxes_cxcywh.ndim == 2:
            gt_bboxes_cxcywh = gt_bboxes_cxcywh.unsqueeze(1)
        elif gt_bboxes_cxcywh.ndim != 3:
            raise ValueError(
                f"Expected gt_bboxes_cxcywh with shape (B, 4) or (B, 1, 4), got {tuple(gt_bboxes_cxcywh.shape)}"
            )

        if gt_bboxes_cxcywh.shape[-1] != 4:
            raise ValueError(
                f"Expected last bbox dimension to be 4, got {tuple(gt_bboxes_cxcywh.shape)}"
            )

        gt_labels = gt_labels.view(-1)
        if gt_labels.shape[0] != gt_bboxes_cxcywh.shape[0]:
            raise ValueError(
                "Ground-truth labels and bounding boxes must have the same batch dimension"
            )

        return gt_bboxes_cxcywh, gt_labels

    def forward(self, preds_dict: dict, gt_bboxes_cxcywh: torch.Tensor, gt_labels: torch.Tensor):
        """
        Inputs:
            preds_dict: The dict returned by model(x)["det"]
            gt_bboxes_cxcywh: Ground truth boxes shaped (B, 4) or (B, 1, 4)
            gt_labels: Ground truth classes shaped (B,) or (B, 1)
        """
        # 1. Extract and reshape model predictions
        # Transpose from (B, C, 6300) to (B, 6300, C) for easier calculations
        pred_boxes = preds_dict["boxes"].transpose(1, 2)   # (B, 6300, 4)
        pred_scores = preds_dict["scores"].transpose(1, 2) # (B, 6300, 10)
        neck_features = preds_dict["feats"]

        gt_bboxes_cxcywh, gt_labels = self._normalize_targets(gt_bboxes_cxcywh, gt_labels)

        # 2. Convert Ground Truth to xyxy format
        gt_bboxes_xyxy = cxcywh_to_xyxy(gt_bboxes_cxcywh)

        # 3. Generate Anchors and Decode predicted boxes
        anchors, strides = generate_anchors(neck_features)
        pred_bboxes_xyxy = decode_predictions(pred_boxes, anchors, strides)

        # 4. Perform Label Assignment
        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores, pred_bboxes_xyxy, anchors, gt_labels, gt_bboxes_xyxy
        )

        # 5. Compute Classification Loss (Focal Loss on ALL 6300 anchors)
        loss_cls = ops.sigmoid_focal_loss(
            pred_scores, target_scores, alpha=0.25, gamma=2.0, reduction="sum"
        )

        # 6. Compute Bounding Box Loss (CIoU strictly on the POSITIVE anchors)
        pos_pred_bboxes = pred_bboxes_xyxy[fg_mask]
        pos_target_bboxes = target_bboxes[fg_mask]

        if pos_pred_bboxes.shape[0] > 0:
            loss_box = ops.complete_box_iou_loss(pos_pred_bboxes, pos_target_bboxes, reduction="sum")
        else:
            loss_box = torch.tensor(0.0, device=pred_scores.device)

        # Normalize losses by the number of positive anchors found in this batch
        num_positives = max(1.0, fg_mask.sum().item())
        loss_cls = loss_cls / num_positives
        loss_box = loss_box / num_positives

        # Final weighted sum
        total_loss = (loss_cls * self.cls_weight) + (loss_box * self.box_weight)

        return total_loss, {"loss_cls": loss_cls.item(), "loss_box": loss_box.item()}


if __name__ == "__main__":
    dummy_x = torch.randn(2, 3, 480, 640)
    model = HandGestureMultiTask(num_classes=10)
    with torch.no_grad():
        preds = model(dummy_x)

    loss = YOLODetectionLoss(num_classes=10)
    gt_boxes = torch.tensor([
        [[240.0, 320.0, 80.0, 80.0]],
        [[120.0, 160.0, 60.0, 60.0]],
    ])
    gt_labels = torch.tensor([[3], [7]])
    total_loss, loss_dict = loss(preds["det"], gt_boxes, gt_labels)
    print(f"Total Loss: {total_loss.item():.4f}, Breakdown: {loss_dict}")
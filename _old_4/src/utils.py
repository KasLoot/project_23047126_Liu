import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import confusion_matrix
import math
import torch.nn.functional as F





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
    


def generate_grid_points(image_size=(640, 480), strides=[8, 16, 32]):
    """
    Generates the (x, y) center coordinates for all 6300 anchor-free predictions.
    image_size: (Height, Width)
    """
    H, W = image_size
    all_points = []
    
    for stride in strides:
        h_grid, w_grid = H // stride, W // stride
        
        # Shift by stride // 2 to get the exact center pixel of the receptive field
        shift_y = torch.arange(0, h_grid) * stride + (stride // 2)
        shift_x = torch.arange(0, w_grid) * stride + (stride // 2)
        
        # Create a meshgrid. indexing='ij' means y goes along rows, x along columns
        y, x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        
        # Flatten and stack to get a list of (x, y) coordinates
        points = torch.stack([x.flatten(), y.flatten()], dim=-1)
        all_points.append(points)
        
    # Concatenate all scales together. Shape becomes (6300, 2)
    return torch.cat(all_points, dim=0)



def assign_targets(grid_points, gt_boxes, gt_classes, num_classes=10, center_sample_ratio=0.5):
    """
    grid_points: (6300, 2) tensor of (x, y) coordinates
    gt_boxes: (batch_size, num_objs, 4) tensor of [xmin, ymin, xmax, ymax]
    gt_classes: (batch_size, num_objs) integer class labels (0 to 9)
    center_sample_ratio: Float. 0.5 means the point must fall within the inner 50% of the box.
    """
    batch_size, num_objs, _ = gt_boxes.shape
    num_points = grid_points.shape[0]

    # 1. Expand dimensions for broadcasting
    pts = grid_points.view(1, 1, num_points, 2) 
    boxes = gt_boxes.unsqueeze(2)               

    # 2. Extract coordinates and calculate standard boundaries
    x, y = pts[..., 0], pts[..., 1]
    xmin, ymin = boxes[..., 0], boxes[..., 1]
    xmax, ymax = boxes[..., 2], boxes[..., 3]

    # Calculate Box Centers and Dimensions
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin

    # 3. Define the Center Sampling Sub-Box
    # Shrink the acceptable area based on the center_sample_ratio
    c_xmin = cx - (w * center_sample_ratio / 2.0)
    c_xmax = cx + (w * center_sample_ratio / 2.0)
    c_ymin = cy - (h * center_sample_ratio / 2.0)
    c_ymax = cy + (h * center_sample_ratio / 2.0)

    # 4. The Point-in-Box Check (Must be inside the Center Sub-Box)
    # Note: We also check the original box just in case the ratio is set > 1.0
    inside_original_x = (x >= xmin) & (x <= xmax)
    inside_original_y = (y >= ymin) & (y <= ymax)
    inside_center_x = (x >= c_xmin) & (x <= c_xmax)
    inside_center_y = (y >= c_ymin) & (y <= c_ymax)
    
    # A point is only valid if it satisfies all conditions
    is_inside = inside_original_x & inside_original_y & inside_center_x & inside_center_y

    # 5. Handle overlaps (Tie-breaker: choose smallest box area)
    areas = w * h
    point_areas = areas.expand(-1, -1, num_points).clone()
    point_areas[~is_inside] = float('inf')

    min_areas, matched_obj_idx = torch.min(point_areas, dim=1)
    is_positive = min_areas < float('inf') 

    # 6. Build the final target tensor for Focal Loss
    target_cls = torch.zeros(batch_size, num_classes, num_points, device=gt_boxes.device)
    
    for b in range(batch_size):
        pos_point_indices = torch.arange(num_points)[is_positive[b]]
        obj_indices = matched_obj_idx[b][is_positive[b]]
        pos_classes = gt_classes[b, obj_indices].long()
        target_cls[b, pos_classes, pos_point_indices] = 1.0

    return target_cls, is_positive, matched_obj_idx


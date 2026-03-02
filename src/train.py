from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataloader import HandGestureDataset, SegAugment
from model import YOLO26MultiTask


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    train_dataset_path: str = "dataset/dataset_v1/train"
    val_dataset_path: str = "dataset/dataset_v1/val"
    output_dir: str = "output/stage_1/train_1"
    epochs: int = 80
    batch_size: int = 16
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.1
    seed: int = 42
    num_classes: int = 10
    scale: str = "n"
    end2end: bool = True
    reg_max: int = 1
    cls_loss_weight: float = 1.0
    box_loss_weight: float = 2.0
    one2one_loss_weight: float = 1.0
    # Number of positive anchors per GT for the one2many branch.
    topk_many: int = 13
    # Number of positive anchors per GT for the one2one branch.
    topk_one: int = 1
    warmup_epochs: int = 5
    # Focal loss parameters
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def detection_collate_fn(batch):
    images, masks, class_ids, bboxes = zip(*batch)
    images = torch.stack(images, dim=0).float()
    masks = torch.stack(masks, dim=0)
    class_ids = torch.as_tensor(class_ids, dtype=torch.long)
    bboxes = torch.stack([b.float() for b in bboxes], dim=0)
    return images, masks, class_ids, bboxes


# ---------------------------------------------------------------------------
# Anchor helpers
# ---------------------------------------------------------------------------

def _build_anchor_meta(
    feats: list[torch.Tensor], input_h: int, input_w: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return anchor centres (A, 2) and strides (A, 2) for all feature levels."""
    centers_all, strides_all = [], []
    for feat in feats:
        _, _, h, w = feat.shape
        stride_y = input_h / float(h)
        stride_x = input_w / float(w)
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        cx = (xx + 0.5) * stride_x
        cy = (yy + 0.5) * stride_y
        centers = torch.stack((cx, cy), dim=-1).reshape(-1, 2)
        strides = torch.full((h * w, 2), fill_value=0.0, device=device)
        strides[:, 0] = stride_x
        strides[:, 1] = stride_y
        centers_all.append(centers)
        strides_all.append(strides)
    return torch.cat(centers_all, 0), torch.cat(strides_all, 0)


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------

def _xyxy_to_xywh(b: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = b.unbind(-1)
    return torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5,
                        (x2 - x1).clamp(min=1.0), (y2 - y1).clamp(min=1.0)), -1)


def _xywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    hw, hh = w * 0.5, h * 0.5
    return torch.stack((cx - hw, cy - hh, cx + hw, cy + hh), -1)


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

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


def _ciou_loss(pred_xyxy: torch.Tensor, tgt_xyxy: torch.Tensor) -> torch.Tensor:
    """Complete-IoU loss (mean-reduced).  Inputs are (N, 4) xyxy tensors."""
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
    v = (4.0 / (math.pi ** 2)) * (
        torch.atan(tw / th.clamp(min=eps)) - torch.atan(pw / ph.clamp(min=eps))
    ) ** 2
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    ciou = iou - rho_sq / c_diag_sq - alpha * v
    return (1.0 - ciou).mean()


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def _focal_bce(
    logits: torch.Tensor, targets: torch.Tensor,
    alpha: float = 0.25, gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid focal loss normalized by positive anchors."""
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * ((1 - p_t) ** gamma) * ce
    
    # FIX: Sum the loss and divide by the number of positive targets
    num_positives = torch.clamp(targets.sum(), min=1.0)
    return loss.sum() / num_positives


# ---------------------------------------------------------------------------
# Branch loss (core fix)
# ---------------------------------------------------------------------------

def _branch_loss(
    branch_out: dict[str, torch.Tensor],
    class_ids: torch.Tensor,        # (B,)
    bboxes_xywh: torch.Tensor,      # (B, 4)
    input_h: int,
    input_w: int,
    num_classes: int,
    cls_loss_weight: float,
    box_loss_weight: float,
    topk: int = 13,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    boxes = branch_out["boxes"]   # (B, 4, A)
    scores = branch_out["scores"] # (B, C, A)
    feats = branch_out["feats"]

    bsz, _, num_anchors = boxes.shape
    device = boxes.device

    centers, strides = _build_anchor_meta(feats, input_h, input_w, device)
    assert centers.shape[0] == num_anchors

    # --- FIX STARTS HERE ---
    # 1. We already have xywh, so clone it directly for anchor assignment
    gt_xywh = bboxes_xywh.clone()

    # 2. Convert to xyxy specifically for the clamping and later CIoU calculations
    bboxes_xyxy = _xywh_to_xyxy(gt_xywh)
    
    # Clamp GT boxes inside image using the corner coordinates
    bboxes_xyxy[:, [0, 2]] = bboxes_xyxy[:, [0, 2]].clamp(0, float(input_w - 1))
    bboxes_xyxy[:, [1, 3]] = bboxes_xyxy[:, [1, 3]].clamp(0, float(input_h - 1))
    # --- FIX ENDS HERE ---

    # ------------------------------------------------------------------
    # Top-k positive anchor assignment (instead of single anchor)
    # ------------------------------------------------------------------
    cls_target = torch.zeros((bsz, num_classes, num_anchors), dtype=scores.dtype, device=device)
    pos_mask = torch.zeros((bsz, num_anchors), dtype=torch.bool, device=device)

    for bi in range(bsz):
        gt_cx, gt_cy = gt_xywh[bi, 0], gt_xywh[bi, 1]
        d2 = (centers[:, 0] - gt_cx) ** 2 + (centers[:, 1] - gt_cy) ** 2
        # Assign top-k nearest anchors as positive
        k = min(topk, num_anchors)
        _, topk_idx = d2.topk(k, largest=False)
        pos_mask[bi, topk_idx] = True
        cls_id = int(class_ids[bi].item())
        cls_target[bi, cls_id, topk_idx] = 1.0

    # ------------------------------------------------------------------
    # FIX 2: Focal loss for classification
    # ------------------------------------------------------------------
    cls_loss = _focal_bce(scores, cls_target, alpha=focal_alpha, gamma=focal_gamma)

    # ------------------------------------------------------------------
    # Decode predicted boxes at ALL positive anchors
    # ------------------------------------------------------------------
    pred_boxes_all = boxes.permute(0, 2, 1)  # (B, A, 4)
    pred_scores_all = scores.permute(0, 2, 1)  # (B, A, C)

    # Collect positive-anchor predictions and their targets for box loss.
    pred_xyxy_list = []
    tgt_xyxy_list = []
    pred_cls_correct = 0
    pred_cls_total = 0
    iou_list = []

    for bi in range(bsz):
        pos_idx = pos_mask[bi].nonzero(as_tuple=True)[0]  # (K,)
        if pos_idx.numel() == 0:
            continue

        raw = pred_boxes_all[bi, pos_idx]  # (K, 4)
        anchor_c = centers[pos_idx]        # (K, 2)
        anchor_s = strides[pos_idx]        # (K, 2)

        # Decode raw → absolute pixel xywh
        pred_xy = anchor_c + torch.tanh(raw[:, :2]) * anchor_s
        pred_wh = torch.exp(raw[:, 2:].clamp(-4.0, 4.0)) * anchor_s
        pred_xywh_i = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_xyxy_i = _xywh_to_xyxy(pred_xywh_i)
        pred_xyxy_i[:, [0, 2]] = pred_xyxy_i[:, [0, 2]].clamp(0, float(input_w - 1))
        pred_xyxy_i[:, [1, 3]] = pred_xyxy_i[:, [1, 3]].clamp(0, float(input_h - 1))

        # Target box is the same for all positive anchors of this image.
        tgt_xyxy_i = bboxes_xyxy[bi].unsqueeze(0).expand_as(pred_xyxy_i)

        pred_xyxy_list.append(pred_xyxy_i)
        tgt_xyxy_list.append(tgt_xyxy_i)

        # Classification accuracy (use the first/best positive anchor for metric)
        best_idx = pos_idx[0]
        pred_cls_label = pred_scores_all[bi, best_idx].argmax()
        pred_cls_correct += (pred_cls_label == class_ids[bi]).float().item()
        pred_cls_total += 1

        # IoU for the best anchor
        iou_i = _bbox_iou_xyxy(
            pred_xyxy_i[:1], bboxes_xyxy[bi].unsqueeze(0)
        )
        iou_list.append(iou_i.item())

    # ------------------------------------------------------------------
    # FIX 3: CIoU loss instead of absolute-pixel smooth_l1
    # ------------------------------------------------------------------
    if len(pred_xyxy_list) > 0:
        all_pred = torch.cat(pred_xyxy_list, dim=0)
        all_tgt = torch.cat(tgt_xyxy_list, dim=0)
        box_loss = _ciou_loss(all_pred, all_tgt)
    else:
        box_loss = torch.tensor(0.0, device=device)

    cls_acc = pred_cls_correct / max(pred_cls_total, 1)
    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0
    iou50_acc = float(np.mean([1.0 if v >= 0.5 else 0.0 for v in iou_list])) if iou_list else 0.0

    total = cls_loss_weight * cls_loss + box_loss_weight * box_loss

    logs = {
        "cls_loss": float(cls_loss.detach().item()),
        "box_loss": float(box_loss.detach().item()),
        "cls_acc": cls_acc,
        "mean_iou": mean_iou,
        "iou50_acc": iou50_acc,
        "total": float(total.detach().item()),
    }
    return total, logs


# ---------------------------------------------------------------------------
# Overall detection loss
# ---------------------------------------------------------------------------

def compute_detection_loss(
    model_out,
    class_ids: torch.Tensor,
    bboxes_xywh: torch.Tensor,
    input_h: int,
    input_w: int,
    cfg: TrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    if isinstance(model_out, tuple):
        _, model_out = model_out

    if not isinstance(model_out, dict):
        raise TypeError(f"Unexpected model output type: {type(model_out)}")

    if "one2many" in model_out and "one2one" in model_out:
        loss_many, logs_many = _branch_loss(
            model_out["one2many"], class_ids, bboxes_xywh,
            input_h, input_w, cfg.num_classes,
            cfg.cls_loss_weight, cfg.box_loss_weight,
            topk=cfg.topk_many,
            focal_alpha=cfg.focal_alpha, focal_gamma=cfg.focal_gamma,
        )
        loss_one, logs_one = _branch_loss(
            model_out["one2one"], class_ids, bboxes_xywh,
            input_h, input_w, cfg.num_classes,
            cfg.cls_loss_weight, cfg.box_loss_weight,
            topk=cfg.topk_one,
            focal_alpha=cfg.focal_alpha, focal_gamma=cfg.focal_gamma,
        )
        total = loss_many + cfg.one2one_loss_weight * loss_one
        logs = {
            "loss": float(total.detach().item()),
            "one2many_total": logs_many["total"],
            "one2many_cls": logs_many["cls_loss"],
            "one2many_box": logs_many["box_loss"],
            "one2many_cls_acc": logs_many["cls_acc"],
            "one2many_mean_iou": logs_many["mean_iou"],
            "one2many_iou50_acc": logs_many["iou50_acc"],
            "one2one_total": logs_one["total"],
            "one2one_cls": logs_one["cls_loss"],
            "one2one_box": logs_one["box_loss"],
            "one2one_cls_acc": logs_one["cls_acc"],
            "one2one_mean_iou": logs_one["mean_iou"],
            "one2one_iou50_acc": logs_one["iou50_acc"],
            "acc": logs_many["cls_acc"],
        }
        return total, logs

    loss_single, logs_single = _branch_loss(
        model_out, class_ids, bboxes_xywh,
        input_h, input_w, cfg.num_classes,
        cfg.cls_loss_weight, cfg.box_loss_weight,
        topk=cfg.topk_many,
        focal_alpha=cfg.focal_alpha, focal_gamma=cfg.focal_gamma,
    )
    logs = {
        "loss": float(loss_single.detach().item()),
        "one2many_total": logs_single["total"],
        "one2many_cls": logs_single["cls_loss"],
        "one2many_box": logs_single["box_loss"],
        "one2many_cls_acc": logs_single["cls_acc"],
        "one2many_mean_iou": logs_single["mean_iou"],
        "one2many_iou50_acc": logs_single["iou50_acc"],
        "acc": logs_single["cls_acc"],
    }
    return loss_single, logs


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: YOLO26MultiTask,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: TrainConfig,
    train: bool,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train() if train else model.eval()

    running = {
        "loss": 0.0,
        "one2many_total": 0.0, "one2many_cls": 0.0, "one2many_box": 0.0,
        "one2many_cls_acc": 0.0, "one2many_mean_iou": 0.0, "one2many_iou50_acc": 0.0,
        "one2one_total": 0.0, "one2one_cls": 0.0, "one2one_box": 0.0,
        "one2one_cls_acc": 0.0, "one2one_mean_iou": 0.0, "one2one_iou50_acc": 0.0,
        "acc": 0.0,
    }
    num_batches = 0
    use_amp = scaler is not None

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for images, _masks, class_ids, bboxes in loader:
            images = images.to(device)
            class_ids = class_ids.to(device)
            bboxes = bboxes.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    detection_preds = outputs.get("det", None)
                    loss, logs = compute_detection_loss(
                        detection_preds, class_ids, bboxes,
                        input_h=images.shape[-2], input_w=images.shape[-1],
                        cfg=cfg,
                    )
            else:
                outputs = model(images)
                detection_preds = outputs.get("det", None)
                loss, logs = compute_detection_loss(
                    detection_preds, class_ids, bboxes,
                    input_h=images.shape[-2], input_w=images.shape[-1],
                    cfg=cfg,
                )

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()

            num_batches += 1
            for k in running:
                if k in logs:
                    running[k] += logs[k]

    if num_batches == 0:
        raise RuntimeError("No batches produced. Check dataset path and split.")

    for k in running:
        running[k] /= num_batches
    return running


# ---------------------------------------------------------------------------
# Warmup + cosine LR scheduler
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    # FIX 5: Enable augmentation for the training set
    train_aug = SegAugment(out_size=(640, 480))
    train_dataset = HandGestureDataset(root_dir=cfg.train_dataset_path, transform=train_aug)
    val_dataset = HandGestureDataset(root_dir=cfg.val_dataset_path, transform=None)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=detection_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=detection_collate_fn
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main training driver
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = YOLO26MultiTask().to(device)

    # FIX 4: Proper weight initialisation
    _init_weights(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        train_logs = run_epoch(model, train_loader, optimizer, device, cfg, train=True, scaler=scaler)

        # Step the scheduler once per epoch (for simplicity — you could also do per-step)
        for _ in range(len(train_loader)):
            scheduler.step()

        if len(val_loader) > 0:
            val_logs = run_epoch(model, val_loader, optimizer, device, cfg, train=False, scaler=scaler)
            val_loss = val_logs["loss"]
        else:
            val_logs = None
            val_loss = train_logs["loss"]

        lr = scheduler.get_lr()
        if val_logs is not None:
            print(
                f"Epoch {epoch:03d}/{cfg.epochs} | "
                f"lr={lr:.3e} | "
                f"train_loss={train_logs['loss']:.4f} | "
                f"val_loss={val_logs['loss']:.4f} | "
                f"train_acc={train_logs['acc']:.4f} | "
                f"val_acc={val_logs['acc']:.4f} | "
                f"train_iou50={train_logs['one2many_iou50_acc']:.4f} | "
                f"val_iou50={val_logs['one2many_iou50_acc']:.4f} | "
                f"train_miou={train_logs['one2many_mean_iou']:.4f} | "
                f"val_miou={val_logs['one2many_mean_iou']:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:03d}/{cfg.epochs} | "
                f"lr={lr:.3e} | "
                f"train_loss={train_logs['loss']:.4f} | "
                f"train_acc={train_logs['acc']:.4f} | "
                f"train_iou50={train_logs['one2many_iou50_acc']:.4f}"
            )

        last_ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg.__dict__,
            "train_logs": train_logs,
            "val_logs": val_logs,
        }
        torch.save(last_ckpt, os.path.join(cfg.output_dir, "last.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(last_ckpt, os.path.join(cfg.output_dir, "best.pt"))

    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")


def _init_weights(model: nn.Module) -> None:
    """Kaiming init for Conv layers, proper bias init for detection heads."""
    import math
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # Bias-init trick: initialize classification head bias so the initial
    # sigmoid output is ~(1/num_classes)*prior, preventing early explosion.
    prior = 0.01
    if hasattr(model, "detect"):
        for cv3 in [model.detect.cv3]:
            for seq in cv3:
                # last layer is nn.Conv2d
                if hasattr(seq[-1], "bias") and seq[-1].bias is not None:
                    nn.init.constant_(seq[-1].bias, -math.log((1 - prior) / prior))
        if hasattr(model.detect, "one2one_cv3"):
            for seq in model.detect.one2one_cv3:
                if hasattr(seq[-1], "bias") and seq[-1].bias is not None:
                    nn.init.constant_(seq[-1].bias, -math.log((1 - prior) / prior))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train YOLO26 from-scratch on HandGestureDataset")
    parser.add_argument("--output-dir", type=str, default="output/stage_1/train_1")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--scale", type=str, default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--end2end", action="store_true", default=True)
    parser.add_argument("--no-end2end", action="store_false", dest="end2end")
    parser.add_argument("--reg-max", type=int, default=1)
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)
    parser.add_argument("--box-loss-weight", type=float, default=2.0)
    parser.add_argument("--one2one-loss-weight", type=float, default=1.0)
    parser.add_argument("--topk-many", type=int, default=13)
    parser.add_argument("--topk-one", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    args = parser.parse_args()
    return TrainConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        seed=args.seed,
        num_classes=args.num_classes,
        scale=args.scale,
        end2end=args.end2end,
        reg_max=args.reg_max,
        cls_loss_weight=args.cls_loss_weight,
        box_loss_weight=args.box_loss_weight,
        one2one_loss_weight=args.one2one_loss_weight,
        topk_many=args.topk_many,
        topk_one=args.topk_one,
        warmup_epochs=args.warmup_epochs,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )


if __name__ == "__main__":
    import torch.nn as nn
    cfg = parse_args()
    train(cfg)
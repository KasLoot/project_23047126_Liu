from __future__ import annotations

import argparse
import copy
import os
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader

from model import HandGestureMultiTask
from training_utils import (
    LossWeights,
    average_log_dict,
    compute_multitask_loss,
    create_dataset,
    multitask_collate_fn,
    set_seed,
)


@dataclass
class TrainConfig:
    train_dataset_path: str = "/workspace/project_23047126_Liu/dataset/dataset_v1/train"
    val_dataset_path: str = "/workspace/project_23047126_Liu/dataset/dataset_v1/val"
    output_dir: str = "/workspace/project_23047126_Liu/outputs/train_3"
    epochs: int = 50
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 5e-4
    weight_decay: float = 1e-2
    min_lr_ratio: float = 0.05
    warmup_epochs: int = 3
    seed: int = 42
    input_h: int = 256
    input_w: int = 256
    num_classes: int = 10
    ema_decay: float = 0.999


class ModelEMA:
    def __init__(self, model: HandGestureMultiTask, decay: float) -> None:
        self.decay = float(decay)
        self.num_updates = 0
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    def current_decay(self) -> float:
        self.num_updates += 1
        # Keep EMA close to the live model early on so validation does not lag behind
        # by multiple epochs on small training runs.
        warmup_decay = (1.0 + self.num_updates) / (10.0 + self.num_updates)
        return min(self.decay, warmup_decay)

    @torch.no_grad()
    def update(self, model: HandGestureMultiTask) -> None:
        decay = self.current_decay()
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if not torch.is_floating_point(ema_value):
                ema_value.copy_(model_value)
                continue
            ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)


def build_optimizer(model: HandGestureMultiTask, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def run_epoch(
    model: HandGestureMultiTask,
    loader: DataLoader,
    device: torch.device,
    weights: LossWeights,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None = None,
    ema: ModelEMA | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    amp_enabled = device.type == "cuda"

    logs = []
    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        for images, masks, class_ids, bboxes in loader:
            images = images.to(device)
            masks = masks.to(device)
            class_ids = class_ids.to(device)
            bboxes = bboxes.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(images, tasks=("classification", "detection", "segmentation"))
                loss, batch_log = compute_multitask_loss(
                    outputs=outputs,
                    class_ids=class_ids,
                    masks=masks,
                    bboxes=bboxes,
                    image_h=images.shape[-2],
                    image_w=images.shape[-1],
                    weights=weights,
                )

            if is_train:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                if ema is not None:
                    ema.update(model)

            logs.append(batch_log)

    return average_log_dict(logs)


def parse_args() -> tuple[TrainConfig, LossWeights]:
    parser = argparse.ArgumentParser(description="Train the hand-gesture multi-task model")
    parser.add_argument("--train_dataset", type=str, default=TrainConfig.train_dataset_path)
    parser.add_argument("--val_dataset", type=str, default=TrainConfig.val_dataset_path)
    parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--num_workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--min_lr_ratio", type=float, default=TrainConfig.min_lr_ratio)
    parser.add_argument("--warmup_epochs", type=int, default=TrainConfig.warmup_epochs)
    parser.add_argument("--ema_decay", type=float, default=TrainConfig.ema_decay)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--input_h", type=int, default=TrainConfig.input_h)
    parser.add_argument("--input_w", type=int, default=TrainConfig.input_w)

    parser.add_argument("--w_cls", type=float, default=1.0)
    parser.add_argument("--w_det_bbox", type=float, default=2.0)
    parser.add_argument("--w_det_obj", type=float, default=0.0)
    parser.add_argument("--w_det_cls", type=float, default=0.5)
    parser.add_argument("--w_seg_bce", type=float, default=1.0)
    parser.add_argument("--w_seg_dice", type=float, default=1.0)

    args = parser.parse_args()

    cfg = TrainConfig(
        train_dataset_path=args.train_dataset,
        val_dataset_path=args.val_dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        min_lr_ratio=args.min_lr_ratio,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
        input_h=args.input_h,
        input_w=args.input_w,
        ema_decay=args.ema_decay,
    )

    weights = LossWeights(
        cls=args.w_cls,
        det_bbox=args.w_det_bbox,
        det_obj=args.w_det_obj,
        det_cls=args.w_det_cls,
        seg_bce=args.w_seg_bce,
        seg_dice=args.w_seg_dice,
    )

    return cfg, weights


def main() -> None:
    cfg, weights = parse_args()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    resize_shape = (cfg.input_h, cfg.input_w)
    train_dataset = create_dataset(cfg.train_dataset_path, resize_shape=resize_shape, augment=True)
    val_dataset = create_dataset(cfg.val_dataset_path, resize_shape=resize_shape, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=multitask_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=multitask_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandGestureMultiTask(num_classes=cfg.num_classes, seg_out_channels=1).to(device)
    optimizer = build_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_epochs = min(max(cfg.warmup_epochs, 0), max(cfg.epochs - 1, 0))
    cosine_epochs = max(cfg.epochs - warmup_epochs, 1)
    min_lr = cfg.lr * cfg.min_lr_ratio
    if warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=cosine_epochs,
                    eta_min=min_lr,
                ),
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(cfg.epochs, 1),
            eta_min=min_lr,
        )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    ema = ModelEMA(model, decay=cfg.ema_decay)

    best_val_loss = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_logs = run_epoch(
            model,
            train_loader,
            device=device,
            weights=weights,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
        )
        val_logs = run_epoch(model, val_loader, device=device, weights=weights, optimizer=None)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_logs['loss']:.4f} val_loss={val_logs['loss']:.4f} | "
            f"train_cls_acc={train_logs['cls_acc']:.4f} val_cls_acc={val_logs['cls_acc']:.4f} | "
            f"train_det_iou={train_logs['det_iou']:.4f} val_det_iou={val_logs['det_iou']:.4f} | "
            f"train_seg_dice={train_logs['seg_dice']:.4f} val_seg_dice={val_logs['seg_dice']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema_model": ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_logs": train_logs,
            "val_logs": val_logs,
            "best_checkpoint_source": "model",
            "ema_num_updates": ema.num_updates,
            "train_config": asdict(cfg),
            "loss_weights": asdict(weights),
        }

        last_path = os.path.join(cfg.output_dir, "last.pt")
        torch.save(ckpt, last_path)

        if val_logs["loss"] < best_val_loss:
            best_val_loss = val_logs["loss"]
            best_path = os.path.join(cfg.output_dir, "best.pt")
            torch.save(ckpt, best_path)
            print(f"Saved new best checkpoint: {best_path}")

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

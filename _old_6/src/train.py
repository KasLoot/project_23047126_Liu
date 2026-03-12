import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import CLASS_ID_TO_NAME, HandGestureDataset_v2, SegAugment_v2, detection_collate_fn
from model import RGB_V1, RGB_V2
from utils import (
    YOLODetectionLoss,
    build_logger,
    compute_macro_f1,
    ensure_dir,
    ensure_project_dirs,
    finalize_epoch_stats,
    finalize_segmentation_metrics,
    init_epoch_stats,
    save_json,
    set_seed,
    summarize_segmentation_metrics,
    update_detection_metrics,
)


def _build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=detection_collate_fn,
        persistent_workers=(num_workers > 0),
    )


def _build_model(model_name: str, num_classes: int, reg_max: int = 1):
    if model_name == "rgb_v1":
        return RGB_V1(num_classes=num_classes, reg_max=reg_max)
    return RGB_V2(num_classes=num_classes, reg_max=reg_max)


def _checkpoint_paths(model_name: str, weights_dir: str, run_name: str) -> dict[str, str]:
    s1_dir = ensure_dir(os.path.join(weights_dir, model_name, "s1", run_name))
    s2_dir = ensure_dir(os.path.join(weights_dir, model_name, "s2", run_name))
    return {
        "s1_best": os.path.join(s1_dir, "best_model.pth"),
        "s1_last": os.path.join(s1_dir, "last_model.pth"),
        "s2_best": os.path.join(s2_dir, "best_model.pth"),
        "s2_last": os.path.join(s2_dir, "last_model.pth"),
    }


def _build_stage2_scheduler(
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    steps_per_epoch: int,
) -> tuple[optim.lr_scheduler.LRScheduler | None, bool]:
    if args.stage2_scheduler == "none":
        return None, False
    if args.stage2_scheduler == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
        )
        return scheduler, True
    if args.stage2_scheduler == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(args.stage2_step_size, 1),
            gamma=args.stage2_gamma,
        )
        return scheduler, False

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.min_lr,
    )
    return scheduler, False


def _save_stage1_plots(history: dict, out_dir: str) -> None:
    metrics = [
        ("loss", "Loss", "loss_curve.png"),
        ("det_acc_iou50", "Detection Accuracy @0.5 IoU", "det_acc_iou50.png"),
        ("mean_bbox_iou", "Mean Bounding-Box IoU", "mean_bbox_iou.png"),
        ("top1_acc", "Overall Top-1 Accuracy", "cls_top1_accuracy.png"),
        ("macro_f1", "Macro F1 (10 classes)", "cls_macro_f1.png"),
    ]

    epochs = range(1, len(history["train"]["loss"]) + 1)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), dpi=140)
    axes = axes.flatten()
    for idx, (key, title, _) in enumerate(metrics):
        ax = axes[idx]
        ax.plot(epochs, history["train"][key], label="Train", linewidth=2)
        ax.plot(epochs, history["val"][key], label="Val", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "metrics_overview.png"))
    plt.close(fig)

    for key, title, filename in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history["train"][key], label="Train", linewidth=2)
        plt.plot(epochs, history["val"][key], label="Val", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=150)
        plt.close()


def _save_stage2_plots(history: dict[str, list[float]], output_dir: str) -> None:
    epochs = np.arange(1, len(history["train_acc"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_hand_iou"], label="Train Hand IoU")
    plt.plot(epochs, history["train_background_iou"], label="Train Background IoU")
    plt.plot(epochs, history["val_hand_iou"], label="Val Hand IoU")
    plt.plot(epochs, history["val_background_iou"], label="Val Background IoU")
    plt.plot(epochs, history["train_miou"], label="Train Mean IoU", linestyle="--")
    plt.plot(epochs, history["val_miou"], label="Val Mean IoU", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Segmentation IoU (Hand vs Background)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "seg_iou_hand_vs_background.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_dice"], label="Train Dice")
    plt.plot(epochs, history["val_dice"], label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Segmentation Dice Coefficient")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "seg_dice.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_acc"], label="Train Top-1 Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Top-1 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Classification Top-1 Accuracy")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cls_top1_accuracy.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_macro_f1"], label="Train Macro-F1")
    plt.plot(epochs, history["val_macro_f1"], label="Val Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Classification Macro-F1 (10 Classes)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cls_macro_f1.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_det_acc_iou50"], label="Train Det@0.5")
    plt.plot(epochs, history["val_det_acc_iou50"], label="Val Det@0.5")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Detection Accuracy @0.5 IoU")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "det_acc_iou50.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_mean_bbox_iou"], label="Train Mean BBox IoU")
    plt.plot(epochs, history["val_mean_bbox_iou"], label="Val Mean BBox IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Mean Bounding-Box IoU")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_bbox_iou.png"), dpi=150)
    plt.close()


def _train_stage1(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_dir, results_dir = ensure_project_dirs(args.weights_dir, args.results_dir)
    checkpoint_paths = _checkpoint_paths(args.model, weights_dir, args.run_name)
    out_dir = ensure_dir(os.path.join(results_dir, args.model, "train", "s1", args.run_name))
    log, close_log = build_logger(os.path.join(out_dir, "training_log.txt"), mode="w")

    try:
        log(f"Using device: {device}")
        input_size = (args.image_h, args.image_w)
        train_transform = SegAugment_v2(out_size=input_size)

        train_dataset = HandGestureDataset_v2(
            root_dir=args.train_dir,
            transform=train_transform,
            resize_shape=list(input_size),
        )
        val_dataset = HandGestureDataset_v2(
            root_dir=args.val_dir,
            transform=None,
            resize_shape=list(input_size),
        )

        train_drop_last = len(train_dataset) >= args.batch_size
        train_loader = _build_loader(train_dataset, args.batch_size, True, args.num_workers, train_drop_last)
        val_loader = _build_loader(val_dataset, args.batch_size, False, args.num_workers, False)

        if len(train_loader) == 0:
            raise ValueError("Training loader is empty. Reduce batch size or check dataset path.")

        model = _build_model(args.model, num_classes=args.num_classes, reg_max=1).to(device)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for branch in model.detect.cls_branches:
            torch.nn.init.constant_(branch[-1].bias, bias_value)

        criterion = YOLODetectionLoss(num_classes=args.num_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
        )

        history = {
            "train": {"loss": [], "det_acc_iou50": [], "mean_bbox_iou": [], "top1_acc": [], "macro_f1": []},
            "val": {"loss": [], "det_acc_iou50": [], "mean_bbox_iou": [], "top1_acc": [], "macro_f1": []},
        }
        best_val_loss = float("inf")

        for epoch in range(args.epochs):
            model.train()
            train_stats = init_epoch_stats(args.num_classes)
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
            for images, masks, class_ids, bboxes in pbar:
                images = images.to(device)
                bboxes = bboxes.to(device)
                class_ids = class_ids.to(device)

                preds = model(images)
                loss, loss_metrics = criterion(preds["det"], bboxes, class_ids)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                scheduler.step()

                cls_pred = preds["cls"].argmax(dim=1)
                train_stats["cls_correct"] += int((cls_pred == class_ids).sum().item())
                train_stats["num_samples"] += int(images.shape[0])
                train_stats["loss_sum"] += loss.item()
                train_stats["num_batches"] += 1
                for true_id, pred_id in zip(class_ids.detach().cpu().tolist(), cls_pred.detach().cpu().tolist()):
                    train_stats["confusion"][int(true_id), int(pred_id)] += 1

                det_iou_sum, det_iou50_correct = update_detection_metrics(preds["det"], bboxes)
                train_stats["det_iou_sum"] += det_iou_sum
                train_stats["det_iou50_correct"] += det_iou50_correct

                pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Cls": f"{loss_metrics['loss_cls']:.4f}",
                        "Box": f"{loss_metrics['loss_box']:.4f}",
                        "LR": f"{scheduler.get_last_lr()[0]:.6f}",
                    }
                )

            train_metrics = finalize_epoch_stats(train_stats)

            model.eval()
            val_stats = init_epoch_stats(args.num_classes)
            with torch.no_grad():
                vbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]")
                for images, masks, class_ids, bboxes in vbar:
                    images = images.to(device)
                    bboxes = bboxes.to(device)
                    class_ids = class_ids.to(device)

                    preds = model(images)
                    loss, _ = criterion(preds["det"], bboxes, class_ids)

                    cls_pred = preds["cls"].argmax(dim=1)
                    val_stats["cls_correct"] += int((cls_pred == class_ids).sum().item())
                    val_stats["num_samples"] += int(images.shape[0])
                    val_stats["loss_sum"] += loss.item()
                    val_stats["num_batches"] += 1
                    for true_id, pred_id in zip(class_ids.detach().cpu().tolist(), cls_pred.detach().cpu().tolist()):
                        val_stats["confusion"][int(true_id), int(pred_id)] += 1

                    det_iou_sum, det_iou50_correct = update_detection_metrics(preds["det"], bboxes)
                    val_stats["det_iou_sum"] += det_iou_sum
                    val_stats["det_iou50_correct"] += det_iou50_correct
                    vbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            val_metrics = finalize_epoch_stats(val_stats)

            for key in history["train"]:
                history["train"][key].append(train_metrics[key])
                history["val"][key].append(val_metrics[key])

            save_json(history, os.path.join(out_dir, "metrics_history.json"))
            _save_stage1_plots(history, out_dir)

            log(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss={train_metrics['loss']:.4f} Val Loss={val_metrics['loss']:.4f} | "
                f"Train Det@0.5={train_metrics['det_acc_iou50'] * 100:.2f}% Val Det@0.5={val_metrics['det_acc_iou50'] * 100:.2f}% | "
                f"Train IoU={train_metrics['mean_bbox_iou']:.4f} Val IoU={val_metrics['mean_bbox_iou']:.4f} | "
                # f"Train Top1={train_metrics['top1_acc'] * 100:.2f}% Val Top1={val_metrics['top1_acc'] * 100:.2f}% | "
                # f"Train F1={train_metrics['macro_f1']:.4f} Val F1={val_metrics['macro_f1']:.4f}"
            )

            torch.save(model.state_dict(), checkpoint_paths["s1_last"])
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(model.state_dict(), checkpoint_paths["s1_best"])
                log(f"Saved new best Stage-1 model (Val Loss: {best_val_loss:.4f})")
    finally:
        close_log()


def _train_stage2(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_dir, results_dir = ensure_project_dirs(args.weights_dir, args.results_dir)
    checkpoint_paths = _checkpoint_paths(args.model, weights_dir, args.run_name)
    out_dir = ensure_dir(os.path.join(results_dir, args.model, "train", "s2", args.run_name))
    log, close_log = build_logger(os.path.join(out_dir, "training_log.txt"), mode="w")

    try:
        log(f"Using device: {device}")
        input_size = (args.image_h, args.image_w)

        train_dataset = HandGestureDataset_v2(
            root_dir=args.train_dir,
            transform=SegAugment_v2(out_size=input_size),
            resize_shape=list(input_size),
        )
        val_dataset = HandGestureDataset_v2(
            root_dir=args.val_dir,
            transform=None,
            resize_shape=list(input_size),
        )
        train_drop_last = len(train_dataset) >= args.batch_size
        train_loader = _build_loader(train_dataset, args.batch_size, True, args.num_workers, train_drop_last)
        val_loader = _build_loader(val_dataset, args.batch_size, False, args.num_workers, False)

        if len(train_loader) == 0:
            raise ValueError("Training loader is empty. Reduce batch size or check dataset path.")

        model = _build_model(args.model, num_classes=args.num_classes, reg_max=1)
        stage1_weights_path = args.stage1_weights
        if stage1_weights_path is None:
            stage1_weights_path = checkpoint_paths["s1_best"]
            if not os.path.exists(stage1_weights_path):
                legacy_path = os.path.join(weights_dir, f"{args.model}_s1_best_model.pth")
                if os.path.exists(legacy_path):
                    stage1_weights_path = legacy_path
        if os.path.exists(stage1_weights_path):
            model.load_state_dict(torch.load(stage1_weights_path, map_location=device, weights_only=True))
            log(f"Loaded Stage-1 weights from {stage1_weights_path}")
        else:
            log(f"No Stage-1 checkpoint found at {stage1_weights_path}; starting Stage-2 from scratch.")

        # Freeze backbone, neck, and detection head; only train classification and segmentation heads
        # for param in model.backbone.parameters():
        #     param.requires_grad = False
        # for param in model.neck.parameters():
        #     param.requires_grad = False
        # for param in model.detect.parameters():
        #     param.requires_grad = False
        # print("Frozen backbone, neck, and detection head parameters.")

        model = model.to(device)
        cls_loss_fn = torch.nn.CrossEntropyLoss()
        seg_loss_fn = torch.nn.BCEWithLogitsLoss()
        det_criterion = YOLODetectionLoss(num_classes=args.num_classes).to(device)

        # Optimize ALL parameters
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler, scheduler_step_per_batch = _build_stage2_scheduler(optimizer, args, len(train_loader))
        log(f"Stage-2 scheduler: {args.stage2_scheduler}")

        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "train_miou": [], "val_miou": [],
            "train_hand_iou": [], "train_background_iou": [],
            "val_hand_iou": [], "val_background_iou": [],
            "train_dice": [], "val_dice": [],
            "train_acc": [], "val_acc": [],
            "train_macro_f1": [], "val_macro_f1": [],
            "train_det_acc_iou50": [], "val_det_acc_iou50": [],
            "train_mean_bbox_iou": [], "val_mean_bbox_iou": [],
        }

        best_val_loss = float("inf")
        class_names = [CLASS_ID_TO_NAME[i] for i in range(args.num_classes)]

        for epoch in range(args.epochs):
            model.train()

            total_loss = 0.0
            train_cls_correct = 0
            train_cls_total = 0
            train_det_iou_sum = 0.0
            train_det_iou50_correct = 0

            train_confusion_matrix = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
            train_seg_sums = {
                "hand_intersection": 0.0, "hand_union": 0.0,
                "background_intersection": 0.0, "background_union": 0.0,
                "pred_hand_pixels": 0.0, "gt_hand_pixels": 0.0,
            }

            for images, masks, class_ids, gt_bboxes in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]"):
                images = images.to(device)
                masks = masks.to(device).float()
                class_ids = class_ids.to(device)
                gt_bboxes = gt_bboxes.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                
                # Calculate joint loss
                cls_loss = cls_loss_fn(outputs["cls"], class_ids)
                seg_loss = seg_loss_fn(outputs["seg"], masks)
                det_loss, _ = det_criterion(outputs["det"], gt_bboxes, class_ids)
                
                loss = cls_loss + 2.5*seg_loss + 0.5*det_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                if scheduler is not None and scheduler_step_per_batch:
                    scheduler.step()
                total_loss += loss.item()

                # Classification Metrics
                train_preds = outputs["cls"].argmax(dim=1)
                train_cls_correct += (train_preds == class_ids).sum().item()
                train_cls_total += class_ids.size(0)
                for true_id, pred_id in zip(class_ids.detach().cpu().tolist(), train_preds.detach().cpu().tolist()):
                    train_confusion_matrix[int(true_id), int(pred_id)] += 1

                # Detection Metrics
                batch_det_iou_sum, batch_det_iou50_correct = update_detection_metrics(outputs["det"], gt_bboxes)
                train_det_iou_sum += batch_det_iou_sum
                train_det_iou50_correct += batch_det_iou50_correct

                # Segmentation Metrics
                train_batch_seg = summarize_segmentation_metrics(outputs["seg"], masks)
                for key, value in train_batch_seg.items():
                    train_seg_sums[key] += value

            avg_train_loss = total_loss / max(len(train_loader), 1)
            train_acc = train_cls_correct / train_cls_total if train_cls_total > 0 else 0.0
            train_macro_f1 = compute_macro_f1(train_confusion_matrix)
            train_det_acc_iou50 = train_det_iou50_correct / train_cls_total if train_cls_total > 0 else 0.0
            train_mean_bbox_iou = train_det_iou_sum / train_cls_total if train_cls_total > 0 else 0.0
            train_miou, train_dice, train_hand_iou, train_background_iou = finalize_segmentation_metrics(train_seg_sums)

            if scheduler is not None and not scheduler_step_per_batch:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            model.eval()
            val_total_loss = 0.0
            val_cls_correct = 0
            val_cls_total = 0
            val_det_iou_sum = 0.0
            val_det_iou50_correct = 0

            val_confusion_matrix = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
            val_seg_sums = {
                "hand_intersection": 0.0, "hand_union": 0.0,
                "background_intersection": 0.0, "background_union": 0.0,
                "pred_hand_pixels": 0.0, "gt_hand_pixels": 0.0,
            }

            with torch.no_grad():
                for images, masks, class_ids, gt_bboxes in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]"):
                    images = images.to(device)
                    masks = masks.to(device).float()
                    class_ids = class_ids.to(device)
                    gt_bboxes = gt_bboxes.to(device)

                    outputs = model(images)
                    
                    # Calculate joint loss
                    cls_loss = cls_loss_fn(outputs["cls"], class_ids)
                    seg_loss = seg_loss_fn(outputs["seg"], masks)
                    det_loss, _ = det_criterion(outputs["det"], gt_bboxes, class_ids)
                    
                    loss = cls_loss + 2.5 * seg_loss + 0.5 * det_loss
                    val_total_loss += loss.item()

                    # Classification Metrics
                    val_preds = outputs["cls"].argmax(dim=1)
                    val_cls_correct += (val_preds == class_ids).sum().item()
                    val_cls_total += class_ids.size(0)
                    for true_id, pred_id in zip(class_ids.detach().cpu().tolist(), val_preds.detach().cpu().tolist()):
                        val_confusion_matrix[int(true_id), int(pred_id)] += 1

                    # Detection Metrics
                    batch_det_iou_sum, batch_det_iou50_correct = update_detection_metrics(outputs["det"], gt_bboxes)
                    val_det_iou_sum += batch_det_iou_sum
                    val_det_iou50_correct += batch_det_iou50_correct

                    # Segmentation Metrics
                    val_batch_seg = summarize_segmentation_metrics(outputs["seg"], masks)
                    for key, value in val_batch_seg.items():
                        val_seg_sums[key] += value

            avg_val_loss = val_total_loss / max(len(val_loader), 1)
            val_acc = val_cls_correct / val_cls_total if val_cls_total > 0 else 0.0
            val_macro_f1 = compute_macro_f1(val_confusion_matrix)
            val_det_acc_iou50 = val_det_iou50_correct / val_cls_total if val_cls_total > 0 else 0.0
            val_mean_bbox_iou = val_det_iou_sum / val_cls_total if val_cls_total > 0 else 0.0
            val_miou, val_dice, val_hand_iou, val_background_iou = finalize_segmentation_metrics(val_seg_sums)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["train_hand_iou"].append(train_hand_iou)
            history["train_background_iou"].append(train_background_iou)
            history["val_hand_iou"].append(val_hand_iou)
            history["val_background_iou"].append(val_background_iou)
            history["train_miou"].append(train_miou)
            history["val_miou"].append(val_miou)
            history["train_dice"].append(train_dice)
            history["val_dice"].append(val_dice)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["train_macro_f1"].append(train_macro_f1)
            history["val_macro_f1"].append(val_macro_f1)
            history["train_det_acc_iou50"].append(train_det_acc_iou50)
            history["val_det_acc_iou50"].append(val_det_acc_iou50)
            history["train_mean_bbox_iou"].append(train_mean_bbox_iou)
            history["val_mean_bbox_iou"].append(val_mean_bbox_iou)

            _save_stage2_plots(history, out_dir)
            save_json(history, os.path.join(out_dir, "metrics_history.json"))

            log(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss={avg_train_loss:.4f} Val Loss={avg_val_loss:.4f} | "
                f"Train ClsAcc={train_acc:.4f} Val ClsAcc={val_acc:.4f} | "
                f"Train Det@0.5={train_det_acc_iou50:.4f} Val Det@0.5={val_det_acc_iou50:.4f} | "
                # f"LR={current_lr:.6e}"
            )
            log(
                f"Train Seg mIoU={train_miou:.4f} (hand={train_hand_iou:.4f}, background={train_background_iou:.4f}) | "
                f"Train Dice={train_dice:.4f}"
            )
            log(
                f"Val Seg mIoU={val_miou:.4f} (hand={val_hand_iou:.4f}, background={val_background_iou:.4f}) | "
                f"Val Dice={val_dice:.4f}"
            )

            torch.save(model.state_dict(), checkpoint_paths["s2_last"])
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), checkpoint_paths["s2_best"])
                log(
                    "Saved new best Stage-2 model "
                    f"(macro_f1={val_macro_f1:.4f}, dice={val_dice:.4f}, val_loss={avg_val_loss:.4f})"
                )

        log("")
        log("Class Names: " + ", ".join(class_names))
    finally:
        close_log()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified training script for Stage-1 and Stage-2")
    parser.add_argument("--stage", type=str, choices=["s1", "s2"], required=True, help="Training stage")
    parser.add_argument("--model", type=str, choices=["rgb_v1", "rgb_v2"], required=True, help="Model architecture")
    parser.add_argument("--train_dir", type=str, default="dataset/dataset_v1/train")
    parser.add_argument("--val_dir", type=str, default="dataset/dataset_v1/val")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=None, help="If omitted: s1=16, s2=32")
    parser.add_argument("--epochs", type=int, default=None, help="If omitted: s1=50, s2=20")
    parser.add_argument("--lr", type=float, default=None, help="If omitted: s1=1e-3, s2=5e-3")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="default", help="Run identifier used for model/stage output folders")
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--stage1_weights", type=str, default=None, help="Optional Stage-1 checkpoint path for Stage-2")
    parser.add_argument(
        "--stage2_scheduler",
        type=str,
        choices=["none", "cosine", "steplr", "onecycle"],
        default="cosine",
        help="Learning-rate scheduler for Stage-2 training",
    )
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR for cosine scheduler")
    parser.add_argument("--stage2_step_size", type=int, default=5, help="Step size for StepLR in Stage-2")
    parser.add_argument("--stage2_gamma", type=float, default=0.5, help="Gamma for StepLR in Stage-2")

    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = 16 if args.stage == "s1" else 32
    if args.epochs is None:
        args.epochs = 50 if args.stage == "s1" else 20
    if args.lr is None:
        args.lr = 1e-3 if args.stage == "s1" else 5e-3
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.stage == "s1":
        _train_stage1(args)
    else:
        _train_stage2(args)


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from model import HandGestureMultiTask
from training_utils import (
    CLASS_ID_TO_NAME,
    LossWeights,
    average_log_dict,
    compute_multitask_loss,
    create_dataset,
    multitask_collate_fn,
)


def resolve_checkpoint_source(ckpt: dict, checkpoint_source: str) -> str:
    if checkpoint_source == "model":
        if "model" not in ckpt:
            raise KeyError("Checkpoint does not contain a model state_dict")
        return "model"

    if checkpoint_source == "ema":
        if "ema_model" not in ckpt:
            raise KeyError("Checkpoint does not contain an ema_model state_dict")
        return "ema_model"

    preferred = ckpt.get("best_checkpoint_source")
    if preferred in ("model", "ema_model") and preferred in ckpt:
        return preferred

    if "model" in ckpt:
        return "model"
    if "ema_model" in ckpt:
        return "ema_model"

    raise KeyError("Checkpoint does not contain a supported model state_dict")


def load_checkpoint(
    model: HandGestureMultiTask,
    checkpoint_path: str,
    device: torch.device,
    checkpoint_source: str,
) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict):
        state_key = resolve_checkpoint_source(ckpt, checkpoint_source)
        state = ckpt[state_key]
    else:
        state = ckpt
        state_key = "model"
    model.load_state_dict(state, strict=True)
    if not isinstance(ckpt, dict):
        return {"_loaded_checkpoint_source": state_key}

    ckpt = dict(ckpt)
    ckpt["_loaded_checkpoint_source"] = state_key
    return ckpt


@torch.no_grad()
def evaluate(
    model: HandGestureMultiTask,
    loader: DataLoader,
    device: torch.device,
    weights: LossWeights,
) -> tuple[dict[str, float], dict[int, float]]:
    model.eval()
    logs = []

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for images, masks, class_ids, bboxes in loader:
        images = images.to(device)
        masks = masks.to(device)
        class_ids = class_ids.to(device)
        bboxes = bboxes.to(device)

        outputs = model(images, tasks=("classification", "detection", "segmentation"))
        _, batch_log = compute_multitask_loss(
            outputs=outputs,
            class_ids=class_ids,
            masks=masks,
            bboxes=bboxes,
            image_h=images.shape[-2],
            image_w=images.shape[-1],
            weights=weights,
        )
        logs.append(batch_log)

        pred_ids = outputs["cls"].argmax(dim=1)
        for gt, pred in zip(class_ids.tolist(), pred_ids.tolist()):
            class_total[int(gt)] += 1
            if int(gt) == int(pred):
                class_correct[int(gt)] += 1

    avg_logs = average_log_dict(logs)
    per_class_acc = {
        cid: (class_correct[cid] / class_total[cid] if class_total[cid] > 0 else 0.0)
        for cid in sorted(class_total.keys())
    }
    return avg_logs, per_class_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the hand-gesture multi-task model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--checkpoint_source",
        type=str,
        choices=("auto", "model", "ema"),
        default="auto",
        help="Which weights to evaluate from a training checkpoint. 'auto' uses checkpoint metadata when available and otherwise defaults to the live model weights.",
    )
    parser.add_argument("--dataset", type=str, default="/workspace/project_23047126_Liu/dataset/dataset_v1/test")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=256)

    parser.add_argument("--w_cls", type=float, default=1.0)
    parser.add_argument("--w_det_bbox", type=float, default=1.0)
    parser.add_argument("--w_det_obj", type=float, default=0.5)
    parser.add_argument("--w_det_cls", type=float, default=0.5)
    parser.add_argument("--w_seg_bce", type=float, default=1.0)
    parser.add_argument("--w_seg_dice", type=float, default=1.0)

    args = parser.parse_args()

    weights = LossWeights(
        cls=args.w_cls,
        det_bbox=args.w_det_bbox,
        det_obj=args.w_det_obj,
        det_cls=args.w_det_cls,
        seg_bce=args.w_seg_bce,
        seg_dice=args.w_seg_dice,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandGestureMultiTask(num_classes=10, seg_out_channels=1).to(device)
    ckpt = load_checkpoint(model, args.checkpoint, device, checkpoint_source=args.checkpoint_source)

    if "loss_weights" in ckpt:
        # If checkpoint has training weights, use those unless user explicitly overrides.
        ckpt_w = ckpt["loss_weights"]
        weights = LossWeights(
            cls=float(ckpt_w.get("cls", weights.cls)),
            det_bbox=float(ckpt_w.get("det_bbox", weights.det_bbox)),
            det_obj=float(ckpt_w.get("det_obj", weights.det_obj)),
            det_cls=float(ckpt_w.get("det_cls", weights.det_cls)),
            seg_bce=float(ckpt_w.get("seg_bce", weights.seg_bce)),
            seg_dice=float(ckpt_w.get("seg_dice", weights.seg_dice)),
        )

    dataset = create_dataset(args.dataset, resize_shape=(args.input_h, args.input_w), augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=multitask_collate_fn,
    )

    print(f"Evaluating dataset: {args.dataset}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Loaded checkpoint source: {ckpt.get('_loaded_checkpoint_source', 'model')}")

    avg_logs, per_class_acc = evaluate(model, loader, device=device, weights=weights)

    print("\n=== Aggregate metrics ===")
    for k in sorted(avg_logs.keys()):
        print(f"{k}: {avg_logs[k]:.6f}")

    print("\n=== Per-class classification accuracy (cls head) ===")
    for cid, acc in per_class_acc.items():
        cname = CLASS_ID_TO_NAME.get(cid, str(cid))
        print(f"{cid:2d} ({cname:8s}): {acc:.4f}")

    if isinstance(ckpt, dict) and "val_logs" in ckpt:
        print("\nCheckpoint stored validation logs:")
        for k, v in ckpt["val_logs"].items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()

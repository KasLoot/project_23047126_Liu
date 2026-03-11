import argparse
import heapq
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.ops as ops
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import (
    CLASS_ID_TO_NAME,
    HandGestureDataset_v2,
    _to_numpy_image_chw,
    _to_numpy_mask,
    detection_collate_fn,
)
from model import RGB_Base, RGB_Dynamic, RGB_Attention
from utils import (
    build_logger,
    cxcywh_to_xyxy,
    decode_predictions,
    ensure_dir,
    ensure_project_dirs,
    generate_anchors,
)


def _build_model(model_name: str, num_classes: int, reg_max: int = 1):
    if model_name == "rgb_base":
        return RGB_Base(num_classes=num_classes, reg_max=reg_max)
    elif model_name == "rgb_dynamic":
        return RGB_Dynamic(num_classes=num_classes, reg_max=reg_max)
    elif model_name == "rgb_attention":
        return RGB_Attention(num_classes=num_classes, reg_max=reg_max)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")


def _default_weights_path(model_name: str, weights_dir: str, stage: str, run_name: str) -> str:
    structured_path = os.path.join(weights_dir, model_name, stage, run_name, "best_model.pth")
    if os.path.exists(structured_path):
        return structured_path
    return os.path.join(weights_dir, f"{model_name}_{stage}_best_model.pth")


def _save_demo_prediction_figure(sample: dict, save_path: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(sample["img_np"])

    pred_mask_overlay = np.ma.masked_where(sample["pred_mask_np"] < 0.5, sample["pred_mask_np"])
    ax.imshow(pred_mask_overlay, cmap="cool", alpha=0.30, interpolation="nearest")
    if sample["gt_mask_np"].max() > 0:
        ax.contour(sample["gt_mask_np"], levels=[0.5], colors=["yellow"], linewidths=1.5)

    gt_box_np = sample["gt_box_np"]
    pr_box_np = sample["pr_box_np"]
    gt_rect = patches.Rectangle(
        (gt_box_np[0], gt_box_np[1]),
        gt_box_np[2] - gt_box_np[0],
        gt_box_np[3] - gt_box_np[1],
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
        label="Ground Truth",
    )
    ax.add_patch(gt_rect)

    det_pred_name = sample["det_pred_name"]
    pred_conf = sample["pred_conf"]
    pr_rect = patches.Rectangle(
        (pr_box_np[0], pr_box_np[1]),
        pr_box_np[2] - pr_box_np[0],
        pr_box_np[3] - pr_box_np[1],
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
        label=f"Det: {det_pred_name} ({pred_conf:.2f})",
    )
    ax.add_patch(pr_rect)

    true_name = sample["true_name"]
    pred_name = sample["pred_name"]
    pred_class_conf = sample["pred_class_conf"]
    iou = sample["iou"]
    seg_hand_iou = sample["seg_hand_iou"]
    seg_background_iou = sample["seg_background_iou"]
    seg_miou = sample["seg_miou"]
    seg_dice = sample["seg_dice"]

    ax.text(
        0.02,
        0.98,
        (
            f"Cls True: {true_name}\n"
            f"Cls Pred: {pred_name} ({pred_class_conf:.2f})\n"
            f"Det Pred: {det_pred_name} ({pred_conf:.2f})\n"
            f"Det IoU: {iou:.2f}\n"
            f"Seg mIoU: {seg_miou:.2f} (h={seg_hand_iou:.2f}, b={seg_background_iou:.2f})\n"
            f"Seg Dice: {seg_dice:.2f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.75, "pad": 6},
    )

    ax.set_title(
        f"True: {true_name} | Cls: {pred_name} | Det: {det_pred_name} | "
        f"IoU: {iou:.2f} | Seg mIoU: {seg_miou:.2f} | Dice: {seg_dice:.2f}"
    )
    ax.axis("off")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


@torch.no_grad()
def visualise(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, results_dir = ensure_project_dirs(args.weights_dir, args.results_dir)
    out_dir = ensure_dir(os.path.join(results_dir, args.model, "visualise", args.stage, args.run_name))
    log, close_log = build_logger(os.path.join(out_dir, "visualise_log.txt"), mode="w")
    weights_path = args.weights or _default_weights_path(args.model, args.weights_dir, args.stage, args.run_name)

    try:
        model = _build_model(args.model, num_classes=args.num_classes, reg_max=1).to(device)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Could not find weights at {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.eval()

        log(f"Model: {args.model}")
        log(f"Stage: {args.stage}")
        log(f"Loaded weights: {weights_path}")

        input_size = (args.image_h, args.image_w)
        dataset = HandGestureDataset_v2(root_dir=args.data_dir, transform=None, resize_shape=list(input_size))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=detection_collate_fn,
            persistent_workers=(args.num_workers > 0),
        )

        best_visualizations = []
        candidate_id = 0
        total_samples = 0

        for images, masks, class_ids, gt_bboxes in tqdm(loader, desc="Collecting visualisations"):
            images = images.to(device)
            masks = masks.to(device).float()
            gt_bboxes = gt_bboxes.to(device)
            class_ids = class_ids.to(device)

            preds = model(images)
            cls_probs = preds["cls"].softmax(dim=1)
            pred_class_ids = cls_probs.argmax(dim=1)
            pred_class_conf = cls_probs.gather(1, pred_class_ids.unsqueeze(1)).squeeze(1)

            seg_logits = preds["seg"]
            pred_masks = (seg_logits.sigmoid() > 0.5).float()
            target_masks = (masks > 0.5).float()

            pred_boxes = preds["det"]["boxes"].transpose(1, 2)
            pred_scores = preds["det"]["scores"].transpose(1, 2)
            anchors, strides = generate_anchors(preds["det"]["feats"])
            decoded_boxes = decode_predictions(pred_boxes, anchors, strides)
            gt_xyxy = cxcywh_to_xyxy(gt_bboxes)

            batch_size = images.shape[0]
            for b in range(batch_size):
                total_samples += 1
                true_class_id = int(class_ids[b].item())
                predicted_class_id = int(pred_class_ids[b].item())

                probs = pred_scores[b].sigmoid()
                max_conf, max_idx = probs.max(dim=0)
                best_cls = max_conf.argmax()
                best_anchor_idx = max_idx[best_cls]
                detector_class_id = int(best_cls.item())
                pred_conf = max_conf[best_cls].item()
                pred_box = decoded_boxes[b, best_anchor_idx]

                iou = ops.box_iou(pred_box.unsqueeze(0), gt_xyxy[b].unsqueeze(0)).item()

                sample_pred_mask = pred_masks[b]
                sample_target_mask = target_masks[b]
                sample_hand_intersection = float((sample_pred_mask * sample_target_mask).sum().item())
                sample_hand_union = float(((sample_pred_mask + sample_target_mask) > 0).sum().item())
                sample_hand_iou = sample_hand_intersection / sample_hand_union if sample_hand_union > 0 else 0.0

                sample_pred_background = 1.0 - sample_pred_mask
                sample_target_background = 1.0 - sample_target_mask
                sample_background_intersection = float((sample_pred_background * sample_target_background).sum().item())
                sample_background_union = float(((sample_pred_background + sample_target_background) > 0).sum().item())
                sample_background_iou = sample_background_intersection / sample_background_union if sample_background_union > 0 else 0.0

                sample_seg_miou = (sample_hand_iou + sample_background_iou) / 2.0
                sample_dice_denom = float(sample_pred_mask.sum().item() + sample_target_mask.sum().item())
                sample_seg_dice = (2.0 * sample_hand_intersection / sample_dice_denom) if sample_dice_denom > 0 else 0.0

                quality_key = (
                    int(predicted_class_id == true_class_id),
                    int(detector_class_id == true_class_id),
                    float(sample_seg_miou),
                    float(sample_seg_dice),
                    float(iou),
                    float(pred_class_conf[b].item()),
                    float(pred_conf),
                )

                should_store = len(best_visualizations) < args.num_samples
                if not should_store and len(best_visualizations) > 0:
                    should_store = quality_key > best_visualizations[0][0]

                if should_store:
                    true_name = CLASS_ID_TO_NAME.get(true_class_id, "Unknown")
                    pred_name = CLASS_ID_TO_NAME.get(predicted_class_id, "Unknown")
                    det_pred_name = CLASS_ID_TO_NAME.get(detector_class_id, "Unknown")

                    payload = {
                        "sample_index": total_samples,
                        "img_np": _to_numpy_image_chw(images[b]),
                        "gt_mask_np": _to_numpy_mask(target_masks[b]),
                        "pred_mask_np": _to_numpy_mask(pred_masks[b]),
                        "gt_box_np": gt_xyxy[b].cpu().numpy(),
                        "pr_box_np": pred_box.cpu().numpy(),
                        "true_name": true_name,
                        "pred_name": pred_name,
                        "det_pred_name": det_pred_name,
                        "pred_class_conf": float(pred_class_conf[b].item()),
                        "pred_conf": float(pred_conf),
                        "iou": float(iou),
                        "seg_hand_iou": float(sample_hand_iou),
                        "seg_background_iou": float(sample_background_iou),
                        "seg_miou": float(sample_seg_miou),
                        "seg_dice": float(sample_seg_dice),
                    }
                    item = (quality_key, candidate_id, payload)
                    if len(best_visualizations) < args.num_samples:
                        heapq.heappush(best_visualizations, item)
                    else:
                        heapq.heapreplace(best_visualizations, item)
                    candidate_id += 1

        ranked = sorted(best_visualizations, key=lambda item: (item[0], item[1]), reverse=True)
        for rank, (_, _, payload) in enumerate(ranked, start=1):
            save_path = os.path.join(out_dir, f"sample_{rank:02d}.png")
            _save_demo_prediction_figure(payload, save_path)
            log(
                f"Rank {rank:02d} | Sample {payload['sample_index']:04d} | "
                f"Cls true={payload['true_name']} pred={payload['pred_name']} ({payload['pred_class_conf']:.2f}) | "
                f"Det pred={payload['det_pred_name']} ({payload['pred_conf']:.2f}) | "
                f"IoU={payload['iou']:.2f} | Seg mIoU={payload['seg_miou']:.2f} | Dice={payload['seg_dice']:.2f}"
            )

        log(f"Saved {len(ranked)} visualisations to {out_dir}")
    finally:
        close_log()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save qualitative prediction visualisations")
    parser.add_argument("--model", type=str, choices=["rgb_base", "rgb_dynamic", "rgb_attention"], default="rgb_base")
    parser.add_argument("--stage", type=str, choices=["s1", "s2"], default="s2")
    parser.add_argument("--weights", type=str, default=None, help="Optional checkpoint path override")
    parser.add_argument("--data_dir", type=str, default="dataset/dataset_v1/test")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=640)
    parser.add_argument("--run_name", type=str, default="default", help="Training run identifier for model/stage outputs")
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--results_dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    visualise(parse_args())
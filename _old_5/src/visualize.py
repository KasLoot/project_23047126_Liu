from __future__ import annotations

import argparse
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from torch.nn import functional as F
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from model import HandGestureMultiTask
from training_utils import (
    CLASS_ID_TO_NAME,
    bbox_cxcywh_to_xyxy,
    create_dataset,
    decode_det_bbox_to_pixels,
    multitask_collate_fn,
)


def _unwrap_dataset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def _resolve_dataset_records(dataset) -> list[dict]:
    if isinstance(dataset, Subset):
        parent_records = _resolve_dataset_records(dataset.dataset)
        return [parent_records[int(index)] for index in dataset.indices]

    image_info = getattr(dataset, "image_info", None)
    if image_info is None:
        raise TypeError("Visualization requires a dataset exposing image_info metadata.")
    return list(image_info)


def _normalize_image_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    image_tensor = image_tensor.detach().cpu().float()
    if image_tensor.numel() > 0 and image_tensor.max() > 1.5:
        image_tensor = image_tensor / 255.0
    return image_tensor.clamp(0.0, 1.0)


def _normalize_mask_tensor(mask_tensor: torch.Tensor) -> torch.Tensor:
    mask_tensor = mask_tensor.detach().cpu().float()
    if mask_tensor.numel() > 0 and mask_tensor.max() > 1.5:
        mask_tensor = mask_tensor / 255.0
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    return (mask_tensor > 0.5).float()


def _load_original_sample(dataset, record: dict) -> tuple[torch.Tensor, torch.Tensor]:
    base_dataset = _unwrap_dataset(dataset)

    image_tensor_dir = getattr(base_dataset, "image_tensor_dir", None)
    mask_tensor_dir = getattr(base_dataset, "mask_tensor_dir", None)
    if image_tensor_dir and mask_tensor_dir and os.path.isdir(image_tensor_dir) and os.path.isdir(mask_tensor_dir):
        image_name = record["new_image_name"].replace(".png", ".pt")
        mask_name = record["new_mask_name"].replace(".png", ".pt")
        image_tensor = torch.load(os.path.join(image_tensor_dir, image_name), weights_only=True)
        mask_tensor = torch.load(os.path.join(mask_tensor_dir, mask_name), weights_only=True)
        return _normalize_image_tensor(image_tensor), _normalize_mask_tensor(mask_tensor)

    image_dir = getattr(base_dataset, "image_dir", os.path.join(base_dataset.root_dir, "images"))
    mask_dir = getattr(base_dataset, "mask_dir", os.path.join(base_dataset.root_dir, "masks"))
    image_path = os.path.join(image_dir, record["new_image_name"])
    mask_path = os.path.join(mask_dir, record["new_mask_name"])

    image_tensor = TF.to_tensor(Image.open(image_path).convert("RGB")).float()
    mask_tensor = TF.to_tensor(Image.open(mask_path).convert("L")).float()
    return _normalize_image_tensor(image_tensor), _normalize_mask_tensor(mask_tensor)


def _record_bbox_xyxy(record: dict, fallback_mask: torch.Tensor) -> torch.Tensor:
    bbox = record.get("bbox")
    if bbox is not None:
        bbox_tensor = torch.as_tensor(bbox, dtype=torch.float32).view(1, 4)
        return bbox_cxcywh_to_xyxy(bbox_tensor)[0]

    mask_2d = fallback_mask[0] if fallback_mask.ndim == 3 else fallback_mask
    active = mask_2d > 0.5
    if not torch.any(active):
        return torch.zeros(4, dtype=torch.float32)

    ys, xs = torch.where(active)
    return torch.tensor(
        [
            float(xs.min().item()),
            float(ys.min().item()),
            float(xs.max().item()),
            float(ys.max().item()),
        ],
        dtype=torch.float32,
    )


def _resize_binary_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return TF.resize(mask.unsqueeze(0), size=size, interpolation=InterpolationMode.NEAREST).squeeze(0)


def _scale_xyxy_box(box: torch.Tensor, scale_x: float, scale_y: float) -> torch.Tensor:
    scaled = box.clone().float()
    scaled[0::2] *= scale_x
    scaled[1::2] *= scale_y
    return scaled


def _draw_overlay_panel(
    ax,
    image_np,
    gt_mask,
    pred_mask,
    gt_box,
    pred_box,
    title: str,
    add_legend: bool = False,
) -> None:
    ax.imshow(image_np)
    ax.imshow(gt_mask, cmap="Reds", alpha=0.30, vmin=0, vmax=1)
    ax.imshow(pred_mask, cmap="Blues", alpha=0.30, vmin=0, vmax=1)
    ax.add_patch(
        Rectangle(
            (gt_box[0], gt_box[1]),
            max(0.0, gt_box[2] - gt_box[0]),
            max(0.0, gt_box[3] - gt_box[1]),
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
    )
    ax.add_patch(
        Rectangle(
            (pred_box[0], pred_box[1]),
            max(0.0, pred_box[2] - pred_box[0]),
            max(0.0, pred_box[3] - pred_box[1]),
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
        )
    )
    ax.set_title(title)
    ax.axis("off")

    if add_legend:
        ax.legend(
            handles=[
                Patch(facecolor="red", edgecolor="none", alpha=0.30, label="GT mask"),
                Patch(facecolor="blue", edgecolor="none", alpha=0.30, label="Pred mask"),
                Line2D([0], [0], color="lime", lw=2, label="GT box"),
                Line2D([0], [0], color="yellow", lw=2, label="Pred box"),
            ],
            loc="lower left",
            fontsize=8,
            framealpha=0.9,
        )


@torch.no_grad()
def save_visualizations(
    model: HandGestureMultiTask,
    loader: DataLoader,
    device: torch.device,
    out_dir: str,
    max_samples: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    ordered_records = _resolve_dataset_records(loader.dataset)
    saved = 0
    seen = 0
    for images, masks, class_ids, bboxes in loader:
        images = images.to(device)
        masks = masks.to(device)
        class_ids = class_ids.to(device)
        bboxes = bboxes.to(device)

        outputs = model(images, tasks=("classification", "detection", "segmentation"))
        cls_pred = outputs["cls"].argmax(dim=1)
        det = outputs["det"]
        seg_prob = outputs["seg"].sigmoid()

        det_bbox_pred = decode_det_bbox_to_pixels(det["bbox"], images.shape[-2], images.shape[-1])
        det_bbox_xyxy = bbox_cxcywh_to_xyxy(det_bbox_pred)
        gt_bbox_xyxy = bbox_cxcywh_to_xyxy(bboxes)

        det_cls_pred = det["class_logits"].argmax(dim=1)
        det_obj = det["objectness"].sigmoid().view(-1)

        bs = images.shape[0]
        for i in range(bs):
            if saved >= max_samples:
                return

            record = ordered_records[seen]
            seen += 1

            image_np = images[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            gt_mask = masks[i, 0].detach().cpu().numpy()
            pred_mask_t = (seg_prob[i, 0].detach().cpu() >= 0.5).float()
            pred_mask = pred_mask_t.numpy()

            gt_box = gt_bbox_xyxy[i].detach().cpu()
            pd_box = det_bbox_xyxy[i].detach().cpu()

            original_image_t, original_mask_t = _load_original_sample(loader.dataset, record)
            orig_h, orig_w = original_image_t.shape[-2:]
            input_h, input_w = image_np.shape[:2]
            scale_x = orig_w / float(input_w)
            scale_y = orig_h / float(input_h)

            original_image_np = original_image_t.permute(1, 2, 0).numpy()
            original_gt_mask = original_mask_t[0].numpy()
            original_pred_mask = _resize_binary_mask(pred_mask_t.unsqueeze(0), (orig_h, orig_w))[0].numpy()
            original_gt_box = _record_bbox_xyxy(record, original_mask_t).numpy()
            original_pd_box = _scale_xyxy_box(pd_box, scale_x=scale_x, scale_y=scale_y).numpy()

            fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=120)

            # Panel 1: input image
            axes[0].imshow(image_np)
            axes[0].set_title("Input image")
            axes[0].axis("off")

            gt_cls = int(class_ids[i].item())
            p_cls = int(cls_pred[i].item())
            p_det_cls = int(det_cls_pred[i].item())
            p_obj = float(det_obj[i].item())
            _draw_overlay_panel(
                axes[1],
                image_np=image_np,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                gt_box=gt_box.numpy(),
                pred_box=pd_box.numpy(),
                title=(
                    f"Combined @ model input | GT cls={gt_cls} ({CLASS_ID_TO_NAME.get(gt_cls, 'unknown')})\n"
                    f"Pred cls={p_cls} ({CLASS_ID_TO_NAME.get(p_cls, 'unknown')}), "
                    f"det_cls={p_det_cls} ({CLASS_ID_TO_NAME.get(p_det_cls, 'unknown')}), obj={p_obj:.2f}"
                ),
                add_legend=True,
            )
            _draw_overlay_panel(
                axes[2],
                image_np=original_image_np,
                gt_mask=original_gt_mask,
                pred_mask=original_pred_mask,
                gt_box=original_gt_box,
                pred_box=original_pd_box,
                title="Combined @ original resolution",
            )

            fig.tight_layout()
            out_path = os.path.join(out_dir, f"sample_{saved:04d}.png")
            fig.savefig(out_path)
            plt.close(fig)
            saved += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize model predictions against GT")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="/workspace/project_23047126_Liu/dataset/dataset_v1/test")
    parser.add_argument("--out_dir", type=str, default="/workspace/project_23047126_Liu/outputs/visualizations")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = create_dataset(args.dataset, resize_shape=(args.input_h, args.input_w), augment=False)

    # Sample a subset randomly for faster visual checks.
    if args.num_samples < len(dataset):
        indices = random.sample(range(len(dataset)), k=args.num_samples)
        subset = torch.utils.data.Subset(dataset, indices)
    else:
        subset = dataset

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=multitask_collate_fn,
    )

    model = HandGestureMultiTask(num_classes=10, seg_out_channels=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    save_visualizations(
        model=model,
        loader=loader,
        device=device,
        out_dir=args.out_dir,
        max_samples=args.num_samples,
    )

    print(f"Saved visualization images to: {args.out_dir}")


if __name__ == "__main__":
    main()

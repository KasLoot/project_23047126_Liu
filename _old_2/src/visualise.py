import argparse
import json
import math
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from model import HandGestureModel, ModelConfig
from utils import decode_predictions


IMG_EXTS = {".pt", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

GESTURE_CLASSES = {
    0: "call",
    1: "dislike",
    2: "like",
    3: "ok",
    4: "one",
    5: "palm",
    6: "peace",
    7: "rock",
    8: "stop",
    9: "three",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize model predictions in canvases of 10 images.")
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Dataset split root (contains images/ or image_tensors/, masks/ or mask_tensors/, and image_info.json).",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--output-dir", type=str, default="outputs/visualisations", help="Directory to save canvas images.")
    parser.add_argument(
        "--image-info-json",
        type=str,
        default=None,
        help="Optional override for image_info.json (auto-detected from --image-dir by default).",
    )
    parser.add_argument(
        "--mask-tensor-dir",
        type=str,
        default=None,
        help="Optional override for GT mask directory (auto-detected from --image-dir by default).",
    )
    parser.add_argument("--conf-thresh", type=float, default=0.3, help="Confidence threshold for decoded detections.")
    parser.add_argument("--device", type=str, default=None, help="Device override: cuda|mps|cpu")
    return parser.parse_args()


def class_name(class_id: int | None) -> str:
    if class_id is None:
        return "N/A"
    return GESTURE_CLASSES.get(class_id, str(class_id))


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> HandGestureModel:
    model = HandGestureModel(ModelConfig())
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def resolve_input_image_dir(dataset_dir: Path) -> Path:
    image_tensors_dir = dataset_dir / "image_tensors"
    images_dir = dataset_dir / "images"
    if image_tensors_dir.exists() and image_tensors_dir.is_dir():
        return image_tensors_dir
    if images_dir.exists() and images_dir.is_dir():
        return images_dir
    return dataset_dir


def resolve_mask_dir(dataset_dir: Path, override_mask_dir: Path | None) -> Path | None:
    if override_mask_dir is not None:
        return override_mask_dir

    mask_tensors_dir = dataset_dir / "mask_tensors"
    masks_dir = dataset_dir / "masks"
    if mask_tensors_dir.exists() and mask_tensors_dir.is_dir():
        return mask_tensors_dir
    if masks_dir.exists() and masks_dir.is_dir():
        return masks_dir
    return None


def resolve_image_info_json(dataset_dir: Path, override_json: Path | None) -> Path | None:
    if override_json is not None:
        return override_json
    candidate = dataset_dir / "image_info.json"
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def list_inputs(image_dir: Path) -> list[Path]:
    files = [p for p in sorted(image_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return files


def image_to_tensor(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".pt":
        img = torch.load(path)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        if img.ndim != 3:
            raise ValueError(f"Expected image tensor with shape [C,H,W] or [H,W], got {tuple(img.shape)} at {path}")
        img = img.float()
        if img.max() > 1.5:
            img = img / 255.0
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img

    pil = Image.open(path).convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def normalize_for_display(image_chw: torch.Tensor) -> np.ndarray:
    x = image_chw.detach().cpu().float()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    x = x.permute(1, 2, 0).numpy()
    x = np.nan_to_num(x)
    if x.max() > 1.0 or x.min() < 0.0:
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        else:
            x = np.zeros_like(x)
    return np.clip(x, 0.0, 1.0)


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float):
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx, cy, w, h


def xywh_to_xyxy(box_xywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box_xywh.tolist()
    return torch.tensor([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=torch.float32)


def bbox_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    union = area1 + area2 - inter
    if union <= 1e-6:
        return 0.0
    return inter / union


def seg_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    inter = torch.logical_and(pred_bool, gt_bool).sum().item()
    union = torch.logical_or(pred_bool, gt_bool).sum().item()
    if union == 0:
        return 1.0
    return inter / union


def load_gt_lookup(image_info_json: Path | None) -> dict[str, dict]:
    if image_info_json is None:
        return {}
    with open(image_info_json, "r") as f:
        items = json.load(f)

    lookup = {}
    for row in items:
        name = row.get("new_image_name", "")
        stem = Path(name).stem
        lookup[stem] = row
    return lookup


def get_gt_mask(gt_row: dict | None, mask_dir: Path | None) -> torch.Tensor | None:
    if gt_row is None or mask_dir is None:
        return None
    mask_name = gt_row.get("new_mask_name", None)
    if mask_name is None:
        return None
    stem = Path(mask_name).stem

    pt_path = mask_dir / f"{stem}.pt"
    if pt_path.exists():
        mask = torch.load(pt_path).float()
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        return (mask > 0.5).to(torch.uint8)

    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        img_path = mask_dir / f"{stem}{ext}"
        if img_path.exists():
            arr = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
            return torch.from_numpy((arr > 127).astype(np.uint8))

    return None


def prepare_gt_bbox_xyxy(gt_row: dict | None, width: int, height: int) -> torch.Tensor | None:
    if gt_row is None or "bbox" not in gt_row:
        return None

    gt_xywh = torch.tensor(gt_row["bbox"], dtype=torch.float32)
    if gt_xywh.max() <= 1.0:
        gt_xywh[[0, 2]] *= float(width)
        gt_xywh[[1, 3]] *= float(height)
    gt_xyxy = xywh_to_xyxy(gt_xywh)
    gt_xyxy[0::2] = gt_xyxy[0::2].clamp(0, float(width - 1))
    gt_xyxy[1::2] = gt_xyxy[1::2].clamp(0, float(height - 1))
    return gt_xyxy


def draw_canvas(page_items: list[dict], page_idx: int, output_dir: Path):
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(24, 11))
    axes = axes.flatten()

    for i in range(rows * cols):
        ax = axes[i]
        if i >= len(page_items):
            ax.axis("off")
            continue

        item = page_items[i]
        image_np = item["image_np"]
        ax.imshow(image_np)

        pred_seg = item["pred_seg"]
        if pred_seg is not None:
            seg_overlay = np.ma.masked_where(pred_seg == 0, pred_seg)
            ax.imshow(seg_overlay, alpha=0.35, cmap="spring")

        pred_box_xyxy = item["pred_box_xyxy"]
        pred_det_cls = item["det_class"]
        if pred_box_xyxy is not None:
            x1, y1, x2, y2 = pred_box_xyxy
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            ax.text(
                x1,
                max(0.0, y1 - 3.0),
                f"bbox_cls: {class_name(pred_det_cls)}",
                color="white",
                fontsize=8,
                bbox=dict(facecolor="red", edgecolor="none", alpha=0.7, pad=1),
            )

        ax.set_title(item["name"], fontsize=10)
        ax.axis("off")

        pred_cls = item["cls_pred"]
        pred_xywh = item["pred_box_xywh"]
        if pred_xywh is None:
            pred_xywh_txt = "[N/A, N/A, N/A, N/A]"
        else:
            pred_xywh_txt = f"[{pred_xywh[0]:.1f}, {pred_xywh[1]:.1f}, {pred_xywh[2]:.1f}, {pred_xywh[3]:.1f}]"

        bbox_iou_val = item["bbox_iou"]
        seg_iou_val = item["seg_iou"]
        bbox_iou_txt = "N/A" if bbox_iou_val is None else f"{bbox_iou_val:.4f}"
        seg_iou_txt = "N/A" if seg_iou_val is None else f"{seg_iou_val:.4f}"

        info_line = (
            f"Classification: {class_name(pred_cls)}\n"
            f"Detection: bbox: {pred_xywh_txt}, class={class_name(pred_det_cls)}\nbbox_iou={bbox_iou_txt}\n"
            f"Segmentation: seg_iou={seg_iou_txt}"
        )
        ax.text(
            0.5,
            -0.18,
            info_line,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="black",
        )

    plt.tight_layout()
    out_file = output_dir / f"canvas_{page_idx:03d}.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    dataset_dir = Path(args.image_dir)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_info_json_override = Path(args.image_info_json) if args.image_info_json else None
    mask_dir_override = Path(args.mask_tensor_dir) if args.mask_tensor_dir else None

    image_dir = resolve_input_image_dir(dataset_dir)
    image_info_json = resolve_image_info_json(dataset_dir, image_info_json_override)
    mask_dir = resolve_mask_dir(dataset_dir, mask_dir_override)

    device = resolve_device(args.device)
    model = load_model(checkpoint, device)

    gt_lookup = load_gt_lookup(image_info_json)
    input_paths = list_inputs(image_dir)
    if len(input_paths) == 0:
        raise RuntimeError(
            f"No supported images found under {image_dir}. Expected files directly in that folder, "
            "or pass a dataset split folder containing image_tensors/ or images/."
        )

    results = []
    with torch.no_grad():
        for path in tqdm(input_paths, desc="Processing images"):
            image = image_to_tensor(path)
            h, w = int(image.shape[1]), int(image.shape[2])
            image_batch = image.unsqueeze(0).to(device)

            cls_logits, bbox_preds, bbox_cls, seg_logits = model(image_batch)
            cls_pred = int(torch.argmax(cls_logits, dim=1).item())

            decoded = decode_predictions(
                bbox_preds,
                bbox_cls,
                input_h=h,
                input_w=w,
                conf_thresh=args.conf_thresh,
            )[0]

            pred_box_xyxy = None
            pred_box_xywh = None
            det_class = None
            if decoded.shape[0] > 0:
                best_idx = int(torch.argmax(decoded[:, 4]).item())
                best = decoded[best_idx].detach().cpu().float()
                x1, y1, x2, y2, conf, det_class_t = best.tolist()
                pred_box_xyxy = [x1, y1, x2, y2]
                pred_box_xywh = list(xyxy_to_xywh(x1, y1, x2, y2))
                det_class = int(det_class_t)

            pred_seg = torch.sigmoid(seg_logits[0, 0]).detach().cpu()
            pred_seg_bin = (pred_seg > 0.5).to(torch.uint8)

            stem = path.stem
            gt_row = gt_lookup.get(stem, None)
            gt_bbox_xyxy = prepare_gt_bbox_xyxy(gt_row, w, h)
            gt_mask = get_gt_mask(gt_row, mask_dir)

            bbox_iou_val = None
            if pred_box_xyxy is not None and gt_bbox_xyxy is not None:
                bbox_iou_val = bbox_iou_xyxy(torch.tensor(pred_box_xyxy), gt_bbox_xyxy)

            seg_iou_val = None
            if gt_mask is not None:
                if gt_mask.shape != pred_seg_bin.shape:
                    gt_mask = F.interpolate(
                        gt_mask.unsqueeze(0).unsqueeze(0).float(),
                        size=pred_seg_bin.shape,
                        mode="nearest",
                    ).squeeze().to(torch.uint8)
                seg_iou_val = seg_iou(pred_seg_bin, gt_mask)

            results.append(
                {
                    "name": path.name,
                    "image_np": normalize_for_display(image),
                    "pred_seg": pred_seg_bin.numpy(),
                    "pred_box_xyxy": pred_box_xyxy,
                    "pred_box_xywh": pred_box_xywh,
                    "cls_pred": cls_pred,
                    "det_class": det_class,
                    "bbox_iou": bbox_iou_val,
                    "seg_iou": seg_iou_val,
                }
            )

    page_size = 10
    num_pages = math.ceil(len(results) / page_size)
    for page_idx in tqdm(range(num_pages), desc="Creating canvases"):
        start = page_idx * page_size
        end = min(start + page_size, len(results))
        draw_canvas(results[start:end], page_idx, output_dir)

    print(f"Saved {num_pages} canvas image(s) to: {output_dir}")


if __name__ == "__main__":
    main()

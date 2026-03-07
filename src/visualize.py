from __future__ import annotations

import argparse
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader

from model import HandGestureMultiTask
from training_utils import (
    CLASS_ID_TO_NAME,
    bbox_cxcywh_to_xyxy,
    create_dataset,
    decode_det_bbox_to_pixels,
    multitask_collate_fn,
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

    saved = 0
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

            image_np = images[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            gt_mask = masks[i, 0].detach().cpu().numpy()
            pred_mask = (seg_prob[i, 0].detach().cpu().numpy() >= 0.5).astype(float)

            gt_box = gt_bbox_xyxy[i].detach().cpu().numpy()
            pd_box = det_bbox_xyxy[i].detach().cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=120)

            # Panel 1: input image
            axes[0].imshow(image_np)
            axes[0].set_title("Input image")
            axes[0].axis("off")

            # Panel 2: GT mask + GT box
            axes[1].imshow(image_np)
            axes[1].imshow(gt_mask, cmap="Reds", alpha=0.35, vmin=0, vmax=1)
            axes[1].add_patch(
                Rectangle(
                    (gt_box[0], gt_box[1]),
                    max(0.0, gt_box[2] - gt_box[0]),
                    max(0.0, gt_box[3] - gt_box[1]),
                    linewidth=2,
                    edgecolor="lime",
                    facecolor="none",
                )
            )
            gt_cls = int(class_ids[i].item())
            axes[1].set_title(f"GT | cls={gt_cls} ({CLASS_ID_TO_NAME.get(gt_cls, 'unknown')})")
            axes[1].axis("off")

            # Panel 3: predicted mask + predicted box
            axes[2].imshow(image_np)
            axes[2].imshow(pred_mask, cmap="Blues", alpha=0.35, vmin=0, vmax=1)
            axes[2].add_patch(
                Rectangle(
                    (pd_box[0], pd_box[1]),
                    max(0.0, pd_box[2] - pd_box[0]),
                    max(0.0, pd_box[3] - pd_box[1]),
                    linewidth=2,
                    edgecolor="yellow",
                    facecolor="none",
                )
            )
            p_cls = int(cls_pred[i].item())
            p_det_cls = int(det_cls_pred[i].item())
            p_obj = float(det_obj[i].item())
            axes[2].set_title(
                f"Pred | cls={p_cls} ({CLASS_ID_TO_NAME.get(p_cls, 'unknown')})\n"
                f"det_cls={p_det_cls} ({CLASS_ID_TO_NAME.get(p_det_cls, 'unknown')}), obj={p_obj:.2f}"
            )
            axes[2].axis("off")

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

import PIL.Image as Image
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import math
from typing import Optional, Tuple


CLASS_ID_TO_NAME = {
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

def mask_to_bbox(mask):
    # convert mask to [x_center, y_center, width, height] (absolute pixel coords)
    # Supports mask shaped (H,W) or (C,H,W) by collapsing channels.
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)

    if mask.ndim == 3:
        # Accept (1,H,W) or (C,H,W); treat any channel as foreground.
        mask_2d = mask.max(dim=0).values
    elif mask.ndim == 2:
        mask_2d = mask
    else:
        raise ValueError(f"Expected mask of shape (H,W) or (C,H,W), got {tuple(mask.shape)}")

    coords = torch.where(mask_2d > 0)
    if coords[0].numel() == 0:
        # Keep training robust: return a degenerate bbox instead of exiting the whole process.
        return torch.zeros(4, dtype=torch.float32, device=mask.device)

    y_coords, x_coords = coords[0], coords[1]
    x_min, x_max = torch.min(x_coords), torch.max(x_coords)
    y_min, y_max = torch.min(y_coords), torch.max(y_coords)

    x_center = (x_min + x_max).to(torch.float32) / 2.0
    y_center = (y_min + y_max).to(torch.float32) / 2.0
    width = (x_max - x_min).to(torch.float32)
    height = (y_max - y_min).to(torch.float32)

    return torch.stack([x_center, y_center, width, height]).to(device=mask.device)


class SegAugment_v2:
    """Geometric + appearance augmentations applied identically to image, mask & bounding box."""

    def __init__(self, out_size=(480, 640)):
        self.out_size = out_size


    def __call__(self, image, mask, bbox, like_or_dislike):
        # bbox is expected to be (1, 4) or (4,) in absolute pixel coords: [cx, cy, w, h]
        if image.ndim != 3:
            raise ValueError(f"Expected image of shape (C,H,W), got {tuple(image.shape)}")
        _, h, w = image.shape

        if isinstance(bbox, torch.Tensor):
            if bbox.ndim == 2 and bbox.shape[0] == 1 and bbox.shape[1] == 4:
                bbox_1d = bbox[0]
            elif bbox.ndim == 1 and bbox.shape[0] == 4:
                bbox_1d = bbox
            else:
                raise ValueError(f"Expected bbox of shape (1,4) or (4,), got {tuple(bbox.shape)}")
        else:
            bbox_1d = torch.tensor(bbox, dtype=torch.float32)
        
        # 1. Convert cx, cy, w, h to corners to track them safely
        cx, cy, bw, bh = bbox_1d[0], bbox_1d[1], bbox_1d[2], bbox_1d[3]
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

        # --- Geometric Transforms ---

        if random.random() < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
            x1, x2 = w - x2, w - x1
        
        if like_or_dislike == False and random.random() < 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)
            y1, y2 = h - y2, h - y1

        angle_limit = 12.0 if like_or_dislike else 20.0
        angle = random.uniform(-angle_limit, angle_limit)
        translate_x = random.uniform(-0.06, 0.06) * w
        translate_y = random.uniform(-0.06, 0.06) * h
        scale = random.uniform(0.92, 1.08)

        image = F.affine(
            image,
            angle=angle,
            translate=[int(round(translate_x)), int(round(translate_y))],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = F.affine(
            mask,
            angle=angle,
            translate=[int(round(translate_x)), int(round(translate_y))],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
        )

        target_h, target_w = int(self.out_size[0]), int(self.out_size[1])
        image = F.resize(image, (target_h, target_w), interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, (target_h, target_w), interpolation=InterpolationMode.NEAREST)

        bbox_out = mask_to_bbox(mask)

        # --- Appearance Transforms (Image Only) ---
        if random.random() < 0.6:
            image = F.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))
        if random.random() < 0.6:
            image = F.adjust_contrast(image, contrast_factor=random.uniform(0.8, 1.2))
        if random.random() < 0.35:
            image = F.adjust_saturation(image, saturation_factor=random.uniform(0.85, 1.15))
        if random.random() < 0.2:
            image = F.adjust_hue(image, hue_factor=random.uniform(-0.03, 0.03))

        if random.random() < 0.15:
            noise = torch.randn_like(image) * 0.02
            image = (image + noise).clamp(0.0, 1.0)

        # Keep dataset's expected bbox shape (1, 4)
        return image, mask, bbox_out.unsqueeze(0)


class HandGestureDataset_v2(Dataset):
    def __init__(self, root_dir, transform=None, resize_shape: list | None=None):
        self.image_tensor_dir = os.path.join(root_dir, "image_tensors")
        self.mask_tensor_dir = os.path.join(root_dir, "mask_tensors")
        self.image_info_json = os.path.join(root_dir, "image_info.json")
        self.resize_shape = [480, 640] if resize_shape is None else resize_shape
        print(f"Initialized HandGestureDataset_v2 with root_dir={root_dir}, resize_shape={self.resize_shape}")
        self.transform = transform
        with open(self.image_info_json, "r") as f:
            self.image_info = json.load(f)
    
    def resize(self, image, mask, out_size):
        target_h, target_w = int(out_size[0]), int(out_size[1])
        image = F.resize(image, (target_h, target_w), interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, (target_h, target_w), interpolation=InterpolationMode.NEAREST)
        return image, mask

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        image_info = self.image_info[idx]
        image_tensor_path = os.path.join(self.image_tensor_dir, image_info["new_image_name"].replace(".png", ".pt"))
        mask_tensor_path = os.path.join(self.mask_tensor_dir, image_info["new_mask_name"].replace(".png", ".pt"))
        if os.path.exists(image_tensor_path) and os.path.exists(mask_tensor_path):
            image_tensor = torch.load(image_tensor_path, weights_only=True).float()
            mask_tensor = torch.load(mask_tensor_path, weights_only=True).float()
        else:
            image_path = image_tensor_path.replace(".pt", ".png").replace("image_tensors", "images")
            mask_path = mask_tensor_path.replace(".pt", ".png").replace("mask_tensors", "masks")
            with Image.open(image_path) as image_file:
                image_tensor = F.pil_to_tensor(image_file.convert("RGB")).float() / 255.0
            with Image.open(mask_path) as mask_file:
                mask_tensor = F.pil_to_tensor(mask_file.convert("L")).float() / 255.0

        # Defensive normalization in case any tensors were saved as 0..255.
        if image_tensor.numel() > 0 and image_tensor.max() > 1.5:
            image_tensor = image_tensor / 255.0
        if mask_tensor.numel() > 0 and mask_tensor.max() > 1.5:
            mask_tensor = mask_tensor / 255.0

        # Ensure mask is binary (dataset is a hand/no-hand mask).
        mask_tensor = (mask_tensor > 0.5).float()
        class_id = torch.tensor([image_info["class_id"]])
        bbox = torch.tensor([image_info["bbox"]], dtype=torch.float32)
        if class_id.item() in [1, 2]:  # dislike or like
            like_or_dislike = True
        else:
            like_or_dislike = False
        
        if self.resize_shape is not None:
            image_tensor, mask_tensor = self.resize(image_tensor, mask_tensor, self.resize_shape)
            bbox = mask_to_bbox(mask_tensor)
        
        if self.transform:
            image_tensor, mask_tensor, bbox = self.transform(image_tensor, mask_tensor, bbox, like_or_dislike)

        

        return image_tensor, mask_tensor, class_id, bbox


def detection_collate_fn(batch):
    images, masks, class_ids, bboxes = zip(*batch)
    images = torch.stack(images, dim=0).float()
    masks = torch.stack(masks, dim=0)

    # Normalize per-sample targets to avoid shape mismatch between
    # augmented train set (bbox shape: [4]) and non-aug val set (bbox shape: [1, 4]).
    class_ids = torch.as_tensor(
        [int(torch.as_tensor(c).view(-1)[0].item()) for c in class_ids],
        dtype=torch.long,
    )
    bboxes = torch.stack(
        [torch.as_tensor(b, dtype=torch.float32).view(-1)[:4] for b in bboxes],
        dim=0,
    )
    return images, masks, class_ids, bboxes



def _to_numpy_image_chw(image_tensor: torch.Tensor):
    """Convert a torch image tensor (C,H,W) to a numpy array (H,W,3) in [0,1]."""
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(image_tensor)}")
    if image_tensor.ndim != 3:
        raise ValueError(f"Expected image tensor of shape (C,H,W), got {tuple(image_tensor.shape)}")

    image = image_tensor.detach().cpu().float()
    image = torch.clamp(image, 0.0, 1.0)

    c, _, _ = image.shape
    if c == 1:
        image = image.repeat(3, 1, 1)
    elif c >= 3:
        image = image[:3]
    else:
        raise ValueError(f"Unsupported channel count: {c}")

    return image.permute(1, 2, 0).numpy()


def _to_numpy_mask(mask_tensor: torch.Tensor):
    """Convert a torch mask tensor to numpy (H,W) in {0,1}."""
    if not isinstance(mask_tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(mask_tensor)}")

    mask = mask_tensor.detach().cpu().float()
    if mask.ndim == 3:
        # Accept (1,H,W) or (C,H,W); treat any channel as foreground
        mask = mask.max(dim=0).values
    elif mask.ndim != 2:
        raise ValueError(f"Expected mask tensor of shape (H,W) or (C,H,W), got {tuple(mask.shape)}")

    mask = (mask > 0.5).to(dtype=torch.float32)
    return mask.numpy()


def _bbox_cxcywh_to_xyxy(bbox: torch.Tensor):
    """Accept bbox as (1,4) or (4,) tensor in absolute pixel coords; returns (x1,y1,x2,y2)."""
    if not isinstance(bbox, torch.Tensor):
        bbox = torch.tensor(bbox, dtype=torch.float32)

    if bbox.ndim == 2 and bbox.shape == (1, 4):
        bbox_1d = bbox[0]
    elif bbox.ndim == 1 and bbox.shape[0] == 4:
        bbox_1d = bbox
    else:
        raise ValueError(f"Expected bbox of shape (1,4) or (4,), got {tuple(bbox.shape)}")

    cx, cy, bw, bh = [float(x) for x in bbox_1d.detach().cpu().float()]
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return x1, y1, x2, y2


def visualize_augmented_samples(
    root_dir: str,
    out_dir: str = "results/visualise/augmented_data",
    num_samples: int = 8,
    seed: int = 0,
    out_size: Tuple[int, int] = (256, 256),
    indices: Optional[list] = None,
):
    """Save a few augmented (image, mask, bbox) visualizations to disk.

    Expects `root_dir` to be compatible with `HandGestureDataset_v2` (i.e. contains
    `image_tensors/`, `mask_tensors/`, and `image_info.json`).
    """
    import numpy as np
    import matplotlib

    # Use a non-interactive backend so this works over SSH/headless.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    os.makedirs(out_dir, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)

    dataset = HandGestureDataset_v2(root_dir=root_dir, resize_shape=out_size, transform=SegAugment_v2(out_size=out_size))
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty: {root_dir}")

    if indices is None:
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        sample_count = min(num_samples, len(dataset))
        indices = random.sample(range(len(dataset)), k=sample_count)
    else:
        indices = [int(i) for i in indices]

    for n, idx in enumerate(indices):
        image_tensor, mask_tensor, class_id, bbox = dataset[idx]

        class_id_int = int(class_id.item())
        class_name = CLASS_ID_TO_NAME.get(class_id_int, "unknown")

        image_np = _to_numpy_image_chw(image_tensor)
        mask_np = _to_numpy_mask(mask_tensor)
        x1, y1, x2, y2 = _bbox_cxcywh_to_xyxy(bbox)

        h, w = mask_np.shape
        # Clamp bbox for safety (visualization only)
        x1 = float(np.clip(x1, 0, w - 1))
        x2 = float(np.clip(x2, 0, w - 1))
        y1 = float(np.clip(y1, 0, h - 1))
        y2 = float(np.clip(y2, 0, h - 1))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

        # Panel 1: image + bbox
        ax = axes[0]
        ax.imshow(image_np)
        ax.add_patch(
            Rectangle(
                (x1, y1),
                max(0.0, x2 - x1),
                max(0.0, y2 - y1),
                fill=False,
                linewidth=2,
                edgecolor="lime",
            )
        )
        ax.set_title(f"Aug image + bbox | idx={idx} | class={class_id_int} ({class_name})")
        ax.axis("off")

        # Panel 2: image + mask overlay + bbox
        ax = axes[1]
        ax.imshow(image_np)
        ax.imshow(mask_np, cmap="Reds", alpha=0.35, vmin=0.0, vmax=1.0)
        ax.add_patch(
            Rectangle(
                (x1, y1),
                max(0.0, x2 - x1),
                max(0.0, y2 - y1),
                fill=False,
                linewidth=2,
                edgecolor="lime",
            )
        )
        ax.set_title("Aug image + mask + bbox")
        ax.axis("off")

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"aug_sample_{n:02d}_idx_{idx}.png")
        fig.savefig(out_path)
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize augmented dataset samples")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Dataset folder containing image_tensors/, mask_tensors/, image_info.json",
    )
    parser.add_argument("--out_dir", type=str, default="results/visualise/augmented_data", help="Where to save PNGs")
    parser.add_argument("--num_samples", type=int, default=8, help="How many samples to save")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--out_h",
        type=int,
        default=256,
        help="Augmentation output height (matches SegAugment out_size)",
    )
    parser.add_argument(
        "--out_w",
        type=int,
        default=256,
        help="Augmentation output width (matches SegAugment out_size)",
    )
    args = parser.parse_args()

    visualize_augmented_samples(
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        out_size=(args.out_h, args.out_w),
    )
    
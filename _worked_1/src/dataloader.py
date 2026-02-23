import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import PIL.ImageDraw as ImageDraw
import numpy as np
import json

import math

import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


class SegAugment:
    """Geometric + appearance augmentations applied identically to image, mask & bounding box."""

    def __init__(self, out_size=(640, 480)):
        self.out_size = out_size

    def __call__(self, image, mask, bbox):
        # bbox is expected to be [cx, cy, w, h] as a 1D tensor
        _, h, w = image.shape
        
        # 1. Convert cx, cy, w, h to corners to track them safely
        cx, cy, bw, bh = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

        # --- Geometric Transforms ---
        
        # # Horizontal flip
        # if random.random() < 0.5:
        #     image = F.hflip(image)
        #     mask = F.hflip(mask)
        #     x1, x2 = w - x2, w - x1

        # # Vertical flip
        # if random.random() < 0.3:
        #     image = F.vflip(image)
        #     mask = F.vflip(mask)
        #     y1, y2 = h - y2, h - y1

        # Rotation
        angle = random.uniform(-10, 10)
        if angle != 0:
            image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            
            # Rotate box corners around the image center
            angle_rad = math.radians(-angle) # -angle because torchvision rotates CCW
            icx, icy = w / 2.0, h / 2.0 # Image center
            
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            new_corners = []
            for px, py in corners:
                # Shift to origin
                ox, oy = px - icx, py - icy
                # Apply rotation
                rx = ox * math.cos(angle_rad) - oy * math.sin(angle_rad)
                ry = ox * math.sin(angle_rad) + oy * math.cos(angle_rad)
                # Shift back
                new_corners.append((rx + icx, ry + icy))
                
            xs = [c[0] for c in new_corners]
            ys = [c[1] for c in new_corners]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

        # Scale/Resize
        target_w, target_h = self.out_size[0], self.out_size[1]
        scale_x = target_w / w
        scale_y = target_h / h
        
        image = F.resize(image, (target_h, target_w), interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, (target_h, target_w), interpolation=InterpolationMode.NEAREST)
        
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

        # Clamp the box strictly to the new image boundaries
        x1 = max(0.0, min(x1, target_w - 1.0))
        x2 = max(0.0, min(x2, target_w - 1.0))
        y1 = max(0.0, min(y1, target_h - 1.0))
        y2 = max(0.0, min(y2, target_h - 1.0))

        # Convert back to cx, cy, w, h
        new_cx = (x1 + x2) / 2.0
        new_cy = (y1 + y2) / 2.0
        new_bw = x2 - x1
        new_bh = y2 - y1
        
        bbox = torch.tensor([new_cx, new_cy, new_bw, new_bh], dtype=bbox.dtype, device=bbox.device)

        # --- Appearance Transforms (Image Only) ---
        if random.random() < 0.5:
            image = F.adjust_brightness(image, brightness_factor=random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            image = F.adjust_contrast(image, contrast_factor=random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            image = F.adjust_saturation(image, saturation_factor=random.uniform(0.8, 1.2))
        if random.random() < 0.3:
            image = F.adjust_hue(image, hue_factor=random.uniform(-0.05, 0.05))

        return image, mask, bbox


class HandGestureDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.images_tensor_dir = os.path.join(dataset_path, "image_tensors")
        self.masks_tensor_dir = os.path.join(dataset_path, "mask_tensors")

        image_files = [f for f in os.listdir(self.images_tensor_dir) if f.endswith(".pt")]
        mask_files = [f for f in os.listdir(self.masks_tensor_dir) if f.endswith(".pt")]
        self.image_indices = sorted(int(os.path.splitext(f)[0]) for f in image_files)
        self.mask_indices = sorted(int(os.path.splitext(f)[0]) for f in mask_files)

        self.num_samples = len(self.image_indices)

        self.annotation_json_path = os.path.join(dataset_path, "annotations", "all_annotations.json")
        self.annotation_json = json.load(open(self.annotation_json_path, "r"))

        assert self.num_samples == len(self.mask_indices), \
            "Number of image tensors and mask tensors must be the same"
        assert self.image_indices == self.mask_indices, \
            "Image and mask tensor indices are not aligned"
        assert self.num_samples == len(self.annotation_json), \
            "num_samples must match the number of annotations in the JSON file"

        print(f"Loaded {self.num_samples} samples from {dataset_path}")
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample_idx = self.image_indices[idx]
        image_path = os.path.join(self.images_tensor_dir, f"{sample_idx}.pt")
        mask_path = os.path.join(self.masks_tensor_dir, f"{sample_idx}.pt")

        image_tensor = torch.load(image_path, map_location="cpu")
        mask_tensor = torch.load(mask_path, map_location="cpu")

        class_id = self.annotation_json[sample_idx]["gesture_id"]
        bbox = torch.tensor(self.annotation_json[sample_idx]["box"])  # [x_min, y_min, x_max, y_max]

        if self.transform is not None:
            image_tensor, mask_tensor, bbox = self.transform(image_tensor, mask_tensor, bbox)

        return image_tensor, mask_tensor, class_id, bbox


# ---------------------------------------------------------------------------
# Utility: check resolutions
# ---------------------------------------------------------------------------

def check_image_mask_resolutions(dataset_path, show_examples=10):
    images_dir = os.path.join(dataset_path, "images")
    masks_dir = os.path.join(dataset_path, "masks")

    image_files = sorted(
        [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    )
    mask_files = sorted(
        [f for f in os.listdir(masks_dir) if os.path.isfile(os.path.join(masks_dir, f))]
    )

    image_map = {os.path.splitext(f)[0]: f for f in image_files}
    mask_map = {os.path.splitext(f)[0]: f for f in mask_files}

    image_keys = set(image_map.keys())
    mask_keys = set(mask_map.keys())
    common_keys = sorted(image_keys & mask_keys)
    missing_masks = sorted(image_keys - mask_keys)
    missing_images = sorted(mask_keys - image_keys)

    image_resolutions = set()
    mask_resolutions = set()
    pair_mismatches = []

    for key in common_keys:
        image_path = os.path.join(images_dir, image_map[key])
        mask_path = os.path.join(masks_dir, mask_map[key])

        with Image.open(image_path) as image:
            image_size = image.size
        with Image.open(mask_path) as mask:
            mask_size = mask.size

        image_resolutions.add(image_size)
        mask_resolutions.add(mask_size)

        if image_size != mask_size:
            pair_mismatches.append((image_map[key], mask_map[key], image_size, mask_size))

    print("=== Resolution Check Summary ===")
    print(f"Total image files: {len(image_files)}")
    print(f"Total mask files:  {len(mask_files)}")
    print(f"Paired samples:    {len(common_keys)}")

    if missing_masks:
        print(f"Missing masks for {len(missing_masks)} images")
        print("Examples:", missing_masks[:show_examples])
    if missing_images:
        print(f"Missing images for {len(missing_images)} masks")
        print("Examples:", missing_images[:show_examples])

    print("\nImage resolutions found (w, h):")
    for w, h in sorted(image_resolutions):
        print(f"  - {w}x{h}")

    print("Mask resolutions found (w, h):")
    for w, h in sorted(mask_resolutions):
        print(f"  - {w}x{h}")

    if pair_mismatches:
        print(f"\nFound {len(pair_mismatches)} image/mask resolution mismatches.")
        print("Examples:")
        for image_name, mask_name, image_size, mask_size in pair_mismatches[:show_examples]:
            print(
                f"  - {image_name} ({image_size[0]}x{image_size[1]}) vs "
                f"{mask_name} ({mask_size[0]}x{mask_size[1]})"
            )
    else:
        print("\nAll paired images and masks have matching resolutions.")

    if len(image_resolutions) == 1 and len(mask_resolutions) == 1 and not pair_mismatches:
        only_w, only_h = next(iter(image_resolutions))
        print(f"Dataset resolution is consistent (w, h): {only_w}x{only_h}")
    else:
        print("Dataset does not have a single consistent resolution across all files.")



def xywh_to_xyxy(box):
    x_center, y_center, width, height = box
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    return [x_min, y_min, x_max, y_max]



if __name__ == "__main__":
    dataset_path = "/home/yuxin/Object_Detection/project_23047126_Liu/dataset/processed_full_dataset"
    check_image_mask_resolutions(dataset_path)
    dataset = HandGestureDataset(dataset_path=dataset_path, transform=None)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
    for images, masks, class_ids, bboxes in dataloader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch masks shape: {masks.shape}")
        print(f"Batch class IDs: {class_ids}")
        print(f"Batch bounding boxes: {bboxes}")

        # Denormalize and convert first image/mask in batch to PIL for visualization
        img_tensor = images[0].permute(1, 2, 0).numpy()  # CxHxW -> HxWxC
        img_tensor = (img_tensor * 255).astype(np.uint8)
        mask_tensor = masks[0].squeeze(0).numpy()  # 1xHxW -> HxW
        if mask_tensor.max() <= 1.0:
            # Binary/normalized masks need scaling for human-visible display.
            mask_tensor = (mask_tensor * 255.0).astype(np.uint8)
        else:
            mask_tensor = np.clip(mask_tensor, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_tensor)
        mask_pil = Image.fromarray(mask_tensor, mode="L")
        
        # draw bounding box on image
        bbox = bboxes[0].numpy().astype(int)
        img_pil_with_box = img_pil.copy()
        draw = ImageDraw.Draw(img_pil_with_box)
        bbox = xywh_to_xyxy(bbox)
        draw.rectangle(bbox, outline="red", width=2)

        img_pil.show()
        mask_pil.show()
        img_pil_with_box.show()
        break
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import math



class SegAugment:
    """Geometric + appearance augmentations applied identically to image, mask & bounding box."""

    def __init__(self, out_size=(640, 480)):
        self.out_size = out_size

    def __call__(self, image, mask, bbox):
        # bbox is expected to be [cx, cy, w, h] as a 1D tensor
        _, w, h = image.shape

        bbox = bbox[0]
        
        # 1. Convert cx, cy, w, h to corners to track them safely
        cx, cy, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
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
        
        image = F.resize(image, (target_w, target_h), interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, (target_w, target_h), interpolation=InterpolationMode.NEAREST)
        
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
    def __init__(self, root_dir, transform=None):
        self.image_tensor_dir = os.path.join(root_dir, "image_tensors")
        self.mask_tensor_dir = os.path.join(root_dir, "mask_tensors")
        self.image_info_json = os.path.join(root_dir, "image_info.json")
        self.transform = transform
        with open(self.image_info_json, "r") as f:
            self.image_info = json.load(f)

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        image_info = self.image_info[idx]
        image_tensor_path = os.path.join(self.image_tensor_dir, image_info["new_image_name"].replace(".png", ".pt"))
        mask_tensor_path = os.path.join(self.mask_tensor_dir, image_info["new_mask_name"].replace(".png", ".pt"))
        image_tensor = torch.load(image_tensor_path)
        mask_tensor = torch.load(mask_tensor_path)
        class_id = torch.tensor([image_info["class_id"]])
        bbox = torch.tensor([image_info["bbox"]], dtype=torch.float32)
        if self.transform:
            image_tensor, mask_tensor, bbox = self.transform(image_tensor, mask_tensor, bbox)

        return image_tensor, mask_tensor, class_id, bbox
    

# class HandGestureDataset_Test(Dataset):
#     def __init__(self, image_tensor_dir, mask_tensor_dir, image_info_json):
#         self.image_tensor_dir = image_tensor_dir
#         self.mask_tensor_dir = mask_tensor_dir
#         with open(image_info_json, "r") as f:
#             self.image_info = json.load(f)

#     def __len__(self):
#         return len(self.image_info)

#     def __getitem__(self, idx):
#         image_info = self.image_info[idx]
#         image_tensor_path = os.path.join(self.image_tensor_dir, image_info["new_image_name"].replace(".png", ".pt"))
#         mask_tensor_path = os.path.join(self.mask_tensor_dir, image_info["new_mask_name"].replace(".png", ".pt"))
#         image_tensor = torch.load(image_tensor_path)
#         mask_tensor = torch.load(mask_tensor_path)
#         class_id = image_info["class_id"]
#         bbox = image_info["bbox"]
#         return image_tensor, mask_tensor, class_id, bbox
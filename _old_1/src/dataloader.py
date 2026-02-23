import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import json


import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


class SegAugment:
    def __init__(self, out_size=(640, 480)):
        self.out_size = out_size

    def __call__(self, image, mask):
        # --- same geometric transform for both ---
        # if random.random() < 0.5:
        #     image = F.hflip(image)
        #     mask = F.hflip(mask)

        # if random.random() < 0.5:
        #     image = F.vflip(image)
        #     mask = F.vflip(mask)

        angle = random.uniform(-10, 10)
        image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
        mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        image = F.resize(image, self.out_size, interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, self.out_size, interpolation=InterpolationMode.NEAREST)

        # --- image-only appearance transforms ---
        if random.random() < 0.5:
            image = F.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            image = F.adjust_contrast(image, contrast_factor=random.uniform(0.8, 1.2))


        return image, mask


# class HandGestureDataset(Dataset):
#     def __init__(self, dataset_path, transform=None):
#         self.images_dir = os.path.join(dataset_path, "images")
#         self.masks_dir = os.path.join(dataset_path, "masks")
#         self.annotation_json_path = os.path.join(dataset_path, "annotations", "all_annotations.json")
#         self.annotation_json = json.load(open(self.annotation_json_path, "r"))
#         self.image_files = sorted(os.listdir(self.images_dir))
#         self.mask_files = sorted(os.listdir(self.masks_dir))
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         image_path = os.path.join(self.images_dir, self.image_files[idx])
#         mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

#         # get the image file name without extension to match with annotation
#         image_name = int(os.path.splitext(self.image_files[idx])[0])
        

#         image = Image.open(image_path).convert("RGB")
#         mask = Image.open(mask_path).convert("L")

#         # if self.transform is not None:
#         #     image, mask = self.transform(image, mask)
#         # else:

#         image = F.to_tensor(image)
#         # normalize to [0,1] and convert to tensor
#         image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#         mask = torch.from_numpy(np.array(mask, dtype=np.int64))
#         class_id = self.annotation_json[image_name]["gesture_id"]
#         """class_id = 0: call
#            class_id = 1: dislike
#            class_id = 2: like
#            class_id = 3: ok
#            class_id = 4: one
#            class_id = 5: palm
#            class_id = 6: peace
#            class_id = 7: rock
#            class_id = 8: stop
#            class_id = 9: three
#         """

#         # print(f"Loaded sample {idx}: image file {image_path}, mask file {mask_path}, class_id {class_id}, image name {image_name}, idx {idx}")

#         return image, mask, class_id


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
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor, class_id, bbox














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
            image_size = image.size  # (W, H)
        with Image.open(mask_path) as mask:
            mask_size = mask.size  # (W, H)

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






# processed_dataset_path = "/home/yuxin/Object_Detection/project_23047126_Liu/dataset/processed_full_dataset"
# image_dir = os.path.join(processed_dataset_path, "images")
# mask_dir = os.path.join(processed_dataset_path, "masks")



# dataset = HandGestureDataset(dataset_path=processed_dataset_path, transform=None)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)


# check_image_mask_resolutions(processed_dataset_path)

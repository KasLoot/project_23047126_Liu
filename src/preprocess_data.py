import torch
import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as F
from tqdm import tqdm
import shutil


def varify_dir_list(dir_list):
    for dir_path in dir_list:
        if not os.path.exists(dir_path):
            print(f"Directory {dir_path} does not exist.")
            return False
    return True



def mask_to_bbox(mask):
    # convert mask to [x_center, y_center, width, height]
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        print("Warning: No positive pixels found in the mask. Returning default bbox [0, 0, 0, 0].")
        exit(1)
    x_min, x_max = np.min(coords[1]), np.max(coords[1])
    y_min, y_max = np.min(coords[0]), np.max(coords[0])
    x_center = float((x_min + x_max) / 2)
    y_center = float((y_min + y_max) / 2)
    width = float(x_max - x_min)
    height = float(y_max - y_min)
    return [x_center, y_center, width, height]



def gether_images_and_masks(dataset_path, output_dir):
    output_image_dir = os.path.join(output_dir, "images")
    output_mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    image_name_index = 0
    all_image_info = []

    for student_dir in tqdm(os.listdir(dataset_path), desc=f"Processing"):
        student_dir_path = os.path.join(dataset_path, student_dir)
        if os.path.isdir(student_dir_path):
            call_mask_dir = [os.path.join(student_dir_path, "G01_call", f"clip0{clipid}", "annotation") for clipid in range(1, 6)]
            dislike_mask_dir = [os.path.join(student_dir_path, "G02_dislike", f"clip0{clipid}", "annotation") for clipid in range(1, 6)]
            like_mask_dir = [os.path.join(student_dir_path, "G03_like", f"clip0{clipid}", "annotation") for clipid in range(1, 6)]
            ok_mask_dir = [os.path.join(student_dir_path, "G04_ok", f"clip0{clipid}", "annotation") for clipid in range(1, 6)]
            one_mask_dir = [os.path.join(student_dir_path, "G05_one", f"clip0{clipid}", "annotation") for clipid in range(1, 6)]
            palm_mask_dir = [os.path.join(student_dir_path, "G06_palm", f"clip0{clipid}", "annotation") for clipid in range(1, 6)]
            peace_mask_dir = [os.path.join(student_dir_path, "G07_peace", f"clip0{clipid}", "annotation") for clipid in range(1, 6)]
            rock_mask_dir = [os.path.join(student_dir_path, "G08_rock", f"clip0{clipid}", "annotation") for clipid in range(1, 6)]
            stop_mask_dir = [os.path.join(student_dir_path, "G09_stop", f"clip0{clipid}", "annotation") for clipid in range(1, 6)]
            three_mask_dir = [os.path.join(student_dir_path, "G10_three", f"clip0{clipid}", "annotation") for clipid in range(1, 6)]

            all_mask_dirs = call_mask_dir + dislike_mask_dir + like_mask_dir + ok_mask_dir + one_mask_dir + palm_mask_dir + peace_mask_dir + rock_mask_dir + stop_mask_dir + three_mask_dir
            assert len(all_mask_dirs) == 50, f"Expected 50 mask directories, but got {len(all_mask_dirs)} for student {student_dir}"
            assert varify_dir_list(all_mask_dirs), f"One or more mask directories do not exist for student {student_dir}"

            for mask_dir in all_mask_dirs:
                image_dir = mask_dir.replace("annotation", "rgb")
                for file in os.listdir(mask_dir):
                    if file.endswith(".png"):
                        mask_path = os.path.join(mask_dir, file)
                        image_path = os.path.join(image_dir, file)
                        if not os.path.exists(image_path) and not os.path.exists(mask_path):
                            print(f"Image {image_path} does not exist for mask {mask_path}. Skipping.")
                            continue
                        new_image_name = f"{image_name_index}.png"
                        new_mask_name = f"{image_name_index}.png"
                        shutil.copy(image_path, os.path.join(output_image_dir, new_image_name))
                        shutil.copy(mask_path, os.path.join(output_mask_dir, new_mask_name))
                        

                        class_name = mask_dir.split("/")[-3].split("_")[1]
                        class_id = int(mask_dir.split("/")[-3].split("_")[0][1:]) - 1
                        bbox = mask_to_bbox(np.array(Image.open(mask_path).convert("L")))

                        image_info = {
                            "name_index": image_name_index,
                            "old_image_path": image_path,
                            "old_mask_path": mask_path,
                            "new_image_path": os.path.join(output_image_dir, new_image_name),
                            "new_mask_path": os.path.join(output_mask_dir, new_mask_name),
                            "new_image_name": new_image_name,
                            "new_mask_name": new_mask_name,
                            "class_name": class_name,
                            "class_id": class_id,
                            "bbox": bbox
                        }
                        all_image_info.append(image_info)

                        image_name_index += 1
        
    with open(os.path.join(output_image_dir.replace("images", ""), "image_info.json"), "w") as f:
        json.dump(all_image_info, f, indent=4)


if __name__ == "__main__":
    original_dataset_path = "dataset/rgb_only_filtered"

    train_dataset_path = "dataset/dataset_v1/train"

    gether_images_and_masks(original_dataset_path, train_dataset_path)
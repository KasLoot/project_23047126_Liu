import torch
import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as F
from tqdm import tqdm
import shutil
import random


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
            # assert varify_dir_list(all_mask_dirs), f"One or more mask directories do not exist for student {student_dir}"
            
            try:
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
            except Exception as e:
                print(f"Error processing student {student_dir}: {e}")
                continue
        
    with open(os.path.join(output_image_dir.replace("images", ""), "image_info.json"), "w") as f:
        json.dump(all_image_info, f, indent=4)

def image_to_tensor(dataset_path):
    images_dir = os.path.join(dataset_path, "images")
    masks_dir = os.path.join(dataset_path, "masks")
    os.makedirs(os.path.join(dataset_path, "image_tensors"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "mask_tensors"), exist_ok=True)

    for file in tqdm(os.listdir(images_dir), desc=f"Processing images in {dataset_path}"):
        if file.endswith(".png"):
            image_path = os.path.join(images_dir, file)
            mask_path = os.path.join(masks_dir, file)
            image_tensor = F.to_tensor(Image.open(image_path).convert("RGB"))
            mask_tensor = F.to_tensor(Image.open(mask_path).convert("L"))
            torch.save(image_tensor, os.path.join(dataset_path, "image_tensors", file.replace(".png", ".pt")))
            torch.save(mask_tensor, os.path.join(dataset_path, "mask_tensors", file.replace(".png", ".pt")))


def process_test_dataset(test_dataset_path, output_dir):
    output_image_dir = os.path.join(output_dir, "images")
    output_mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    image_name_index = 0
    all_image_info = []

    for gesture_dir in tqdm(os.listdir(test_dataset_path), desc=f"Processing test dataset"):
        gesture_dir_path = os.path.join(test_dataset_path, gesture_dir)
        if os.path.isdir(gesture_dir_path):
            for clip_dir in os.listdir(gesture_dir_path):
                clip_dir_path = os.path.join(gesture_dir_path, clip_dir)
                if os.path.isdir(clip_dir_path):
                    mask_dir = os.path.join(clip_dir_path, "annotation")
                    image_dir = os.path.join(clip_dir_path, "rgb")
                    if not os.path.exists(mask_dir) or not os.path.exists(image_dir):
                        print(f"Mask directory {mask_dir} or image directory {image_dir} does not exist. Skipping.")
                        continue
                    try:
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
                                

                                class_name = gesture_dir.split("_")[1]
                                class_id = int(gesture_dir.split("_")[0][1:]) - 1
                                
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
                    except Exception as e:
                        print(f"Error processing gesture {gesture_dir} in test dataset: {e}")
                        continue
    with open(os.path.join(output_image_dir.replace("images", ""), "image_info.json"), "w") as f:
        json.dump(all_image_info, f, indent=4)




def balance_data_distribution(dataset_path):
    """
    Balances the data distribution in a given folder by undersampling majority classes.
    It removes image, mask, and tensor files of the removed samples and updates image_info.json.
    """
    image_info_path = os.path.join(dataset_path, "image_info.json")
    if not os.path.exists(image_info_path):
        print(f"Error: 'image_info.json' not found in {dataset_path}")
        return

    with open(image_info_path, "r") as f:
        all_image_info = json.load(f)

    # Get the set of existing image filenames from the images directory
    images_dir = os.path.join(dataset_path, "images")
    try:
        existing_images = set(os.listdir(images_dir))
    except FileNotFoundError:
        print(f"Error: 'images' directory not found in {dataset_path}")
        return

    # Filter all_image_info to only include entries with existing images
    original_count = len(all_image_info)
    all_image_info = [info for info in all_image_info if info["new_image_name"] in existing_images]
    filtered_count = len(all_image_info)
    
    if original_count > filtered_count:
        print(f"Filtered out {original_count - filtered_count} entries from image_info.json that did not have corresponding image files.")

    # Group images by class
    class_groups = {}
    for item in all_image_info:
        class_id = item["class_id"]
        if class_id not in class_groups:
            class_groups[class_id] = []
        class_groups[class_id].append(item)

    # Find the minimum number of samples in any class
    if not class_groups:
        print("No classes found in the dataset.")
        return
    min_samples = min(len(samples) for samples in class_groups.values())
    print(f"Balancing dataset to {min_samples} samples per class.")

    new_all_image_info = []
    removed_files_count = 0
    files_to_remove = []

    # Undersample classes with more samples than min_samples
    for class_id, samples in class_groups.items():
        if len(samples) > min_samples:
            samples_to_keep = random.sample(samples, min_samples)
            samples_to_remove = [s for s in samples if s not in samples_to_keep]
            
            for sample in samples_to_remove:
                files_to_remove.append(os.path.join(dataset_path, "images", sample["new_image_name"]))
                files_to_remove.append(os.path.join(dataset_path, "masks", sample["new_mask_name"]))
                tensor_name = sample["new_image_name"].replace(".png", ".pt")
                files_to_remove.append(os.path.join(dataset_path, "image_tensors", tensor_name))
                files_to_remove.append(os.path.join(dataset_path, "mask_tensors", tensor_name))
            
            new_all_image_info.extend(samples_to_keep)
        else:
            new_all_image_info.extend(samples)

    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            removed_files_count += 1
        except FileNotFoundError:
            print(f"Warning: Could not find a file to delete: {file_path}")

    print(f"Removed {removed_files_count} files to balance the dataset.")

    # Update the image_info.json file
    with open(image_info_path, "w") as f:
        json.dump(new_all_image_info, f, indent=4)

    # Print final class distribution after balancing
    final_distribution = {}
    for item in new_all_image_info:
        class_id = item["class_id"]
        final_distribution[class_id] = final_distribution.get(class_id, 0) + 1

    print("Final data distribution (class_id: count):")
    for class_id in sorted(final_distribution):
        print(f"  {class_id}: {final_distribution[class_id]}")
    
    print(f"Successfully balanced dataset at {dataset_path} and updated image_info.json.")


if __name__ == "__main__":
    # original_train_dataset_path = "dataset/rgb_only_filtered_train"

    # train_dataset_path = "dataset/dataset_v1/train"

    # gether_images_and_masks(original_train_dataset_path, train_dataset_path)

    # original_val_dataset_path = "dataset/rgb_only_filtered_val"

    # val_dataset_path = "dataset/dataset_v1/val"

    # gether_images_and_masks(original_val_dataset_path, val_dataset_path)

    # train_dataset_path = "dataset/dataset_v1/train"
    # image_to_tensor(train_dataset_path)
    # val_dataset_path = "dataset/dataset_v1/val"
    # image_to_tensor(val_dataset_path)

    # test_tensor_path = "dataset/dataset_v1/train/image_tensors/0.pt"
    # test_tensor = torch.load(test_tensor_path)
    # # print test_tensor value range
    # print(f"Test tensor value range: min={test_tensor.min().item()}, max={test_tensor.max().item()}")
    # print(f"Test tensor shape: {test_tensor.shape}")

    # test_dataset_path = "dataset/test_data"
    # output_test_dir = "dataset/dataset_v1/test"
    # process_test_dataset(test_dataset_path, output_test_dir)

    # image_to_tensor(output_test_dir)

    balance_data_distribution("dataset/dataset_v1/val")
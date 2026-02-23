import os
import shutil
import json

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import numpy as np
import torch

from tqdm import tqdm


def combine_dataset():
    dataset_path = "/home/yuxin/Object_Detection/project_23047126_Liu/dataset/dataset_full/rgb_only"

    post_processed_dataset_path= "/home/yuxin/Object_Detection/project_23047126_Liu/dataset/processed_full_dataset"
    post_processed_image_path = os.path.join(post_processed_dataset_path, "images")
    post_processed_mask_path = os.path.join(post_processed_dataset_path, "masks")

    os.makedirs(post_processed_image_path, exist_ok=True)
    os.makedirs(post_processed_mask_path, exist_ok=True)

    index_counter = 0
    name_counter = 0

    error_directories = []
    copy_errors = []

    all_annotations = []
    annotation_json_path = os.path.join(post_processed_dataset_path, "annotations")

    for dirs in os.listdir(dataset_path):
        student_dir_path = os.path.join(dataset_path, dirs)
        if os.path.isdir(student_dir_path):
            print(f"\nProcessing student directory: {student_dir_path}")
            for gesture_dir in os.listdir(student_dir_path):
                gesture_dir_path = os.path.join(student_dir_path, gesture_dir)
                if os.path.isdir(gesture_dir_path):
                    if gesture_dir == "G01_call":
                        gesture = "call"
                        gesture_id = 0
                    elif gesture_dir == "G02_dislike":
                        gesture = "dislike"
                        gesture_id = 1
                    elif gesture_dir == "G03_like":
                        gesture = "like"
                        gesture_id = 2
                    elif gesture_dir == "G04_ok":
                        gesture = "ok"
                        gesture_id = 3
                    elif gesture_dir == "G05_one":
                        gesture = "one"
                        gesture_id = 4
                    elif gesture_dir == "G06_palm":
                        gesture = "palm"
                        gesture_id = 5
                    elif gesture_dir == "G07_peace":
                        gesture = "peace"
                        gesture_id = 6
                    elif gesture_dir == "G08_rock":
                        gesture = "rock"
                        gesture_id = 7
                    elif gesture_dir == "G09_stop":
                        gesture = "stop"
                        gesture_id = 8
                    elif gesture_dir == "G10_three":
                        gesture = "three"
                        gesture_id = 9

                    for clip_dir in os.listdir(gesture_dir_path):
                        clip_dir_path = os.path.join(gesture_dir_path, clip_dir)
                        if os.path.isdir(clip_dir_path):
                            # print(f"clip_dir_path: {clip_dir_path}")
                            clip_annotation_path = os.path.join(clip_dir_path, "annotation")
                            # print(f"clip_annotation_path: {clip_annotation_path}")
                            if os.path.isdir(clip_annotation_path):
                                for annotation_file in os.listdir(clip_annotation_path):
                                    if annotation_file.endswith(".png"):
                                        annotation_file_path = os.path.join(clip_annotation_path, annotation_file)
                                        # print(f"annotation_file: {annotation_file}")
                                        # print(f"annotation_file_path: {annotation_file_path}")
                                        image_file_path = annotation_file_path.replace("annotation", "rgb")
                                        # print(f"image_file_path: {image_file_path}")
                                        # check if the corresponding image file exists
                                        if os.path.exists(image_file_path):
                                            # print(f"Found matching image file: {image_file_path}")
                                            # copy and rename the annotation and image files to the post-processed dataset directories
                                            new_image_file_name = f"{name_counter}.png"
                                            new_mask_file_name = f"{name_counter}.png"
                                            new_image_file_path = os.path.join(post_processed_image_path, new_image_file_name)
                                            new_mask_file_path = os.path.join(post_processed_mask_path, new_mask_file_name)
                                            try:
                                                shutil.copy2(image_file_path, new_image_file_path)
                                                shutil.copy2(annotation_file_path, new_mask_file_path)
                                                # print(f"Copied {image_file_path} to {new_image_file_path}")
                                                # print(f"Copied {annotation_file_path} to {new_mask_file_path}")
                                                annotation_json = {
                                                    "name_index": name_counter,
                                                    "original_image_file": os.path.join("rgb_only", dirs, gesture_dir, clip_dir, "rgb", annotation_file),
                                                    "original_annotation_file": os.path.join("rgb_only", dirs, gesture_dir, clip_dir, "rgb", annotation_file),
                                                    "new_image_file": new_image_file_name,
                                                    "new_mask_file": new_mask_file_name,
                                                    "gesture": gesture,
                                                    "gesture_id": gesture_id
                                                }
                                                all_annotations.append(annotation_json)
                                            
                                                # with open(os.path.join(annotation_json_path, f"{name_counter}.json"), "w") as f:
                                                #     json.dump(annotation_json, f, indent=4)
                                                name_counter += 1
                                                index_counter += 1
                                                # if name_counter == 500:
                                                #     print(annotation_file_path)
                                                #     exit(0)
                                            except OSError as e:
                                                print(f"Error copying files for {annotation_file_path}: {e}")
                                                copy_errors.append((annotation_file_path, str(e)))
                                                index_counter += 1
                                        else:
                                            print(f"Warning: No matching image file found for {annotation_file_path}")
                                            error_directories.append(annotation_file_path)
                                            index_counter += 1
                                            continue

    # Save all annotations to a single JSON file
    with open(os.path.join(annotation_json_path, "all_annotations.json"), "w") as f:
        json.dump(all_annotations, f, indent=4)                                   
                                        
    print(f"\nTotal number of processed images and masks: {name_counter}")
    print(f"Total number of processed images: {index_counter}")
    print(f"Error directories: {error_directories}")
    print(f"Copy errors: {copy_errors}")

    desired_total_images = 31 * 100

    image_num = 0
    for file in os.listdir(post_processed_image_path):
        if file.endswith(".png"):
            image_num += 1

    mask_num = 0
    for file in os.listdir(post_processed_mask_path):
        if file.endswith(".png"):
            mask_num += 1

    print(f"Number of images in post-processed images directory: {image_num}")
    print(f"Number of masks in post-processed masks directory: {mask_num}")
    # ...existing code...


def image_to_tensor_dataset():
    images_path = "/home/yuxin/Object_Detection/project_23047126_Liu/dataset/processed_full_dataset/images"
    masks_path = "/home/yuxin/Object_Detection/project_23047126_Liu/dataset/processed_full_dataset/masks"
    all_in_one_tensor_path = "/home/yuxin/Object_Detection/project_23047126_Liu/dataset/processed_full_dataset/all_in_one_tensors"
    os.makedirs(all_in_one_tensor_path, exist_ok=True)

    image_tensor_path = "/home/yuxin/Object_Detection/project_23047126_Liu/dataset/processed_full_dataset/image_tensors"
    mask_tensor_path = "/home/yuxin/Object_Detection/project_23047126_Liu/dataset/processed_full_dataset/mask_tensors"
    os.makedirs(image_tensor_path, exist_ok=True)
    os.makedirs(mask_tensor_path, exist_ok=True)

    annotation_json_path = "/home/yuxin/Object_Detection/project_23047126_Liu/dataset/processed_full_dataset/annotations/all_annotations.json"
    annotation_json = json.load(open(annotation_json_path, "r"))

    # image_tensor_all = []
    # mask_tensor_all = []
    # class_id_all = []

    for image_object in tqdm(annotation_json, desc="Processing images"):
        image_name = image_object["new_image_file"]
        image_index = os.path.splitext(image_name)[0]
        image_file_path = os.path.join(images_path, image_name)
        # mask_file_path = os.path.join(masks_path, image_name)

        if os.path.exists(image_file_path):
            image = Image.open(image_file_path).convert("RGB")
            # convert image to tensor and normalize
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            torch.save(image_tensor, os.path.join(image_tensor_path, f"{image_index}.pt"))
                # image_tensor_all.append(image_tensor)

            
    # image_tensor_all = torch.stack(image_tensor_all)
    # torch.save(image_tensor_all, os.path.join(all_in_one_tensor_path, "image_tensors.pt"))
    # print(f"Saved image tensors to {os.path.join(all_in_one_tensor_path, 'image_tensors.pt')}")


    for image_object in tqdm(annotation_json, desc="Processing masks"):
        image_name = image_object["new_image_file"]
        image_index = os.path.splitext(image_name)[0]
        # image_file_path = os.path.join(images_path, image_name)
        mask_file_path = os.path.join(masks_path, image_name)

        if os.path.exists(mask_file_path):
            mask = Image.open(mask_file_path).convert("L")
            # convert image to tensor and normalize
            # t = torch.from_numpy(np.array(mask))
            # print("dtype:", t.dtype)
            # print("min/max:", t.min().item(), t.max().item())
            # print("unique count:", torch.unique(t).numel())
            # print("is binary (0/255):", torch.all((t == 0) | (t == 255)).item())
            mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0
            torch.save(mask_tensor, os.path.join(mask_tensor_path, f"{image_index}.pt"))
            # mask_tensor_all.append(mask_tensor)


    
    # mask_tensor_all = torch.stack(mask_tensor_all)
    # torch.save(mask_tensor_all, os.path.join(all_in_one_tensor_path, "mask_tensors.pt"))
    # print(f"Saved mask tensors to {os.path.join(all_in_one_tensor_path, 'mask_tensors.pt')}")


def _visualize_bounding_boxes(annotation_json, sample_count=30):
    """Save side-by-side image/mask previews with the computed bounding box drawn."""
    dataset_root = "/home/yuxin/Object_Detection/project_23047126_Liu/dataset/processed_full_dataset"
    images_path = os.path.join(dataset_root, "images")
    masks_path = os.path.join(dataset_root, "masks")
    vis_path = os.path.join(dataset_root, "bbox_visualizations")
    os.makedirs(vis_path, exist_ok=True)

    items_with_box = [item for item in annotation_json if "box" in item]
    items_with_box = items_with_box[:sample_count]

    saved_count = 0
    skipped_count = 0

    for item in tqdm(items_with_box, desc="Saving bbox visualizations"):
        image_name = item.get("new_image_file")
        mask_name = item.get("new_mask_file", image_name)
        box = item.get("box", [0.0, 0.0, 0.0, 0.0])

        if not image_name or not mask_name:
            skipped_count += 1
            continue

        image_file_path = os.path.join(images_path, image_name)
        mask_file_path = os.path.join(masks_path, mask_name)

        if not os.path.exists(image_file_path) or not os.path.exists(mask_file_path):
            skipped_count += 1
            continue

        image = Image.open(image_file_path).convert("RGB")
        mask = Image.open(mask_file_path).convert("L").convert("RGB")

        xc, yc, bw, bh = box
        x_min = int(round(xc - bw / 2.0))
        y_min = int(round(yc - bh / 2.0))
        x_max = int(round(xc + bw / 2.0))
        y_max = int(round(yc + bh / 2.0))

        # Keep coordinates within valid image range
        img_w, img_h = image.size
        x_min = max(0, min(img_w - 1, x_min))
        y_min = max(0, min(img_h - 1, y_min))
        x_max = max(0, min(img_w - 1, x_max))
        y_max = max(0, min(img_h - 1, y_max))

        image_draw = ImageDraw.Draw(image)
        mask_draw = ImageDraw.Draw(mask)
        image_draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0), width=2)
        mask_draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0), width=2)

        # Side-by-side preview: left=image, right=mask
        preview = Image.new("RGB", (img_w * 2, img_h))
        preview.paste(image, (0, 0))
        preview.paste(mask, (img_w, 0))

        stem = os.path.splitext(image_name)[0]
        preview.save(os.path.join(vis_path, f"{stem}_bbox_check.png"))
        saved_count += 1

    print(f"Saved {saved_count} visualization images to: {vis_path}")
    print(f"Skipped {skipped_count} samples during visualization")


def get_bouding_box_from_mask(visualize=True, visualization_samples=30):
    masks_path = "/home/yuxin/Object_Detection/project_23047126_Liu/dataset/processed_full_dataset/masks"

    annotation_json_path = "/home/yuxin/Object_Detection/project_23047126_Liu/dataset/processed_full_dataset/annotations/all_annotations.json"
    with open(annotation_json_path, "r") as f:
        annotation_json = json.load(f)

    # Build index lookup for fast matching: mask_index -> annotation object
    annotation_by_index = {}
    for image_object in annotation_json:
        if "new_mask_file" in image_object:
            key = os.path.splitext(image_object["new_mask_file"])[0]
            annotation_by_index[key] = image_object
        elif "new_image_file" in image_object:
            key = os.path.splitext(image_object["new_image_file"])[0]
            annotation_by_index[key] = image_object
        else:
            key = str(image_object.get("name_index", ""))
            if key:
                annotation_by_index[key] = image_object

    mask_files = [
        file_name for file_name in os.listdir(masks_path)
        if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    ]

    matched_count = 0
    missing_annotation_count = 0
    empty_mask_count = 0

    for mask_file_name in tqdm(mask_files, desc="Extracting bounding boxes from masks"):
        # 1) mask_index from file name (without extension)
        mask_index = os.path.splitext(mask_file_name)[0]
        mask_file_path = os.path.join(masks_path, mask_file_name)

        if mask_index not in annotation_by_index:
            missing_annotation_count += 1
            continue

        mask = Image.open(mask_file_path).convert("L")
        mask_array = np.array(mask)

        # Foreground pixels are non-zero
        ys, xs = np.where(mask_array > 0)

        if len(xs) == 0 or len(ys) == 0:
            # Keep a valid placeholder when mask is empty
            xc, yc, bw, bh = 0.0, 0.0, 0.0, 0.0
            empty_mask_count += 1
        else:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            # 2) centre and width/height in pixel units
            bw = float(x_max - x_min + 1)
            bh = float(y_max - y_min + 1)
            xc = float((x_min + x_max) / 2.0)
            yc = float((y_min + y_max) / 2.0)

        # 3) write to matching annotation object
        annotation_by_index[mask_index]["box"] = [xc, yc, bw, bh]
        matched_count += 1

    with open(annotation_json_path, "w") as f:
        json.dump(annotation_json, f, indent=4)

    print(f"Updated box for {matched_count} masks.")
    print(f"Masks without matching annotation: {missing_annotation_count}")
    print(f"Empty masks: {empty_mask_count}")

    if visualize:
        _visualize_bounding_boxes(annotation_json, sample_count=visualization_samples)







if __name__ == "__main__":
    # combine_dataset()
    # image_to_tensor_dataset()
    get_bouding_box_from_mask()
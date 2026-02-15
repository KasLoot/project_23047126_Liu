import os
import shutil
import json



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






if __name__ == "__main__":
    combine_dataset()

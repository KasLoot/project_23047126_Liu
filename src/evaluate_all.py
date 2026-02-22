import argparse
import os
import glob
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
import torchvision 

# IMPORTANT: Import the MultiTask model
from model_1 import YOLO26MultiTask

global_seed = 10

random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)

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

# ---------------------------------------------------------------------------
# Helper Functions 
# ---------------------------------------------------------------------------

def _build_anchor_meta(feats, input_h, input_w, device):
    centers_all, strides_all = [], []
    for feat in feats:
        _, _, h, w = feat.shape
        stride_y = input_h / float(h)
        stride_x = input_w / float(w)
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        cx = (xx + 0.5) * stride_x
        cy = (yy + 0.5) * stride_y
        centers = torch.stack((cx, cy), dim=-1).reshape(-1, 2)
        strides = torch.full((h * w, 2), fill_value=0.0, device=device)
        strides[:, 0] = stride_x
        strides[:, 1] = stride_y
        centers_all.append(centers)
        strides_all.append(strides)
    return torch.cat(centers_all, 0), torch.cat(strides_all, 0)

def _xywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    hw, hh = w * 0.5, h * 0.5
    return torch.stack((cx - hw, cy - hh, cx + hw, cy + hh), -1)

# ---------------------------------------------------------------------------
# Main Inference Logic
# ---------------------------------------------------------------------------

def run_folder_inference(
    folder_path,
    weights_path,
    conf_threshold=0.4,
    num_images=10,
    image_patterns=None,
    canvas_path="outputs/inference/multitask_canvas_result.jpg",
    canvas_title="YOLO26 MultiTask Inference Results\n(Classification + Detection + Segmentation)",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading MultiTask weights from {weights_path}...")
    
    # 1. Initialize MultiTask Model
    num_classes = 10
    model = YOLO26MultiTask(num_classes=num_classes, end2end=True).to(device)
    
    # Load checkpoint
    ckpt = torch.load(weights_path, map_location=device)
    # Handle both direct state_dict and nested 'model' dict saving formats
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    if image_patterns is None:
        image_patterns = ["*.png"]

    all_images = []
    for pattern in image_patterns:
        all_images.extend(glob.glob(os.path.join(folder_path, pattern)))

    if not all_images:
        print(f"No matching images found in {folder_path} for patterns: {image_patterns}")
        return

    selected_images = random.sample(all_images, min(len(all_images), num_images))
    print(f"Randomly selected {len(selected_images)} images for inference.")

    processed_images = []

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    for img_path in selected_images:
        original_img = Image.open(img_path).convert("RGB")
        img_resized = original_img.resize((640, 480), Image.Resampling.BILINEAR)
        img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 2. Omni-Model Forward Pass
            outputs = model(img_tensor)
            
            # Extract the three branches
            det_out = outputs["det"]
            cls_out = outputs["cls"]
            seg_out = outputs["seg"]

            # --- A. Process Classification ---
            cls_probs = cls_out[0].softmax(dim=-1)
            global_conf, global_cls_id = cls_probs.max(dim=-1)
            global_label = f"Global Cls: {GESTURE_CLASSES[global_cls_id.item()]} ({global_conf.item():.2f})"

            # --- B. Process Segmentation ---
            # Apply sigmoid and threshold to create a binary mask
            mask = (seg_out.sigmoid() > 0.5).squeeze().cpu().numpy() # Shape: (480, 640)
            
            # --- C. Process Detection ---
            _, aux = det_out
            boxes_raw = aux["one2one"]["boxes"]
            scores_raw = aux["one2one"]["scores"]
            feats = aux["one2one"]["feats"]
            
            input_h, input_w = img_tensor.shape[-2:]
            centers, strides = _build_anchor_meta(feats, input_h, input_w, device)

            boxes_raw = boxes_raw.permute(0, 2, 1)[0]
            pred_xy = centers + torch.tanh(boxes_raw[:, :2]) * strides
            pred_wh = torch.exp(boxes_raw[:, 2:].clamp(-4.0, 4.0)) * strides
            decoded_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
            decoded_xyxy = _xywh_to_xyxy(decoded_xywh)
            
            # Apply Logit Shift for cleaner thresholds
            logit_shift = 2.0 
            scores = (scores_raw + logit_shift).sigmoid().permute(0, 2, 1)[0]
            max_scores, class_preds = scores.max(dim=-1)
            
            mask_det = max_scores > conf_threshold
            final_boxes = decoded_xyxy[mask_det]
            final_scores = max_scores[mask_det]
            final_classes = class_preds[mask_det]

        if len(final_boxes) > 0:
            keep_indices = torchvision.ops.nms(final_boxes, final_scores, iou_threshold=0.45)
            final_boxes = final_boxes[keep_indices]
            final_scores = final_scores[keep_indices]
            final_classes = final_classes[keep_indices]
            
        # ---------------------------------------------------------------------------
        # Composite Drawing
        # ---------------------------------------------------------------------------
        
        # 1. Add Segmentation Overlay (Cyan, semi-transparent)
        img_rgba = img_resized.convert("RGBA")
        mask_rgba = np.zeros((480, 640, 4), dtype=np.uint8)
        mask_rgba[mask > 0] = [0, 255, 255, 120] # R, G, B, Alpha
        mask_pil = Image.fromarray(mask_rgba, mode="RGBA")
        
        img_composited = Image.alpha_composite(img_rgba, mask_pil)
        
        # 2. Draw Bounding Boxes
        draw = ImageDraw.Draw(img_composited)
        detections_text_list = []
        
        for box, score, cls_id in zip(final_boxes, final_scores, final_classes):
            x1, y1, x2, y2 = box.cpu().tolist()
            conf = score.item()
            c_id = cls_id.item()
            
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
            label = f"{GESTURE_CLASSES[c_id]}: {conf:.2f}"
            
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill="lime")
            draw.text((x1 + 2, y1 - text_h - 4), label, fill="black", font=font)

            detections_text_list.append(f"Det: {label}")

        # 3. Format Combined Text
        det_text = global_label + "\n"
        if detections_text_list:
            det_text += "\n".join(detections_text_list)
        else:
            det_text += "Det: None"

        filename = os.path.basename(img_path)
        # Convert back to RGB for matplotlib
        processed_images.append((filename, img_composited.convert("RGB"), det_text))
        print(f"Processed {filename}: Found {len(final_boxes)} objects.")

    # ---------------------------------------------------------------------------
    # Canvas Generation
    # ---------------------------------------------------------------------------
    print("Generating canvas...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 11))
    fig.suptitle(canvas_title, fontsize=18, fontweight='bold')
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < len(processed_images):
            img_name, img, det_text = processed_images[idx]
            ax.imshow(img)
            ax.set_title(img_name, fontsize=10)
            
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            ax.set_xlabel(det_text, fontsize=11, fontweight='bold', color='darkblue', labelpad=8)
        else:
            ax.axis('off')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(canvas_path), exist_ok=True)
    plt.savefig(canvas_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Canvas saved successfully to {canvas_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Path to folder containing primary inference images", default="dataset/processed_full_dataset/images")
    parser.add_argument(
        "--test_folder",
        type=str,
        default="dataset/processed_full_dataset/test_images",
        help="Path to test image folder (JPEG set, e.g. 1600x1200)",
    )
    parser.add_argument("--weights", type=str, default="outputs/yolo26_multi/YOLO26MultiTask_best.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--num_images", type=int, default=10, help="Number of images sampled for each canvas")
    args = parser.parse_args()
    
    # Existing / primary run (keeps previous behavior)
    run_folder_inference(
        folder_path=args.folder,
        weights_path=args.weights,
        conf_threshold=args.conf,
        num_images=args.num_images,
        image_patterns=["*.png"],
        canvas_path="outputs/inference/multitask_canvas_result.jpg",
        canvas_title="YOLO26 MultiTask Inference Results\n(Classification + Detection + Segmentation)",
    )

    # Additional test-set run for JPEG images (e.g., 1600x1200)
    run_folder_inference(
        folder_path=args.test_folder,
        weights_path=args.weights,
        conf_threshold=args.conf,
        num_images=args.num_images,
        image_patterns=["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png"],
        canvas_path="outputs/inference/multitask_canvas_testset_result.jpg",
        canvas_title="YOLO26 MultiTask Test-Set Inference Results\n(Classification + Detection + Segmentation)",
    )
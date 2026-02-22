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

from model_1 import YOLO26

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
    """Rebuilds anchor centers and strides to decode raw network outputs."""
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
    """Converts [cx, cy, w, h] to [x_min, y_min, x_max, y_max]."""
    cx, cy, w, h = b.unbind(-1)
    hw, hh = w * 0.5, h * 0.5
    return torch.stack((cx - hw, cy - hh, cx + hw, cy + hh), -1)

# ---------------------------------------------------------------------------
# Main Inference Logic
# ---------------------------------------------------------------------------

def run_folder_inference(folder_path, weights_path, conf_threshold=0.4, num_images=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading weights from {weights_path}...")
    
    # Load checkpoint
    ckpt = torch.load(weights_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    
    # Initialize model using saved config once
    num_classes = cfg.get("num_classes", 10)
    model = YOLO26(
        nc=num_classes,
        scale=cfg.get("scale", "n"),
        end2end=cfg.get("end2end", True),
        reg_max=cfg.get("reg_max", 1)
    ).to(device)
    
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Find and sample images
    all_images = glob.glob(os.path.join(folder_path, "*.png"))
    if not all_images:
        print(f"No .png images found in {folder_path}!")
        return

    selected_images = random.sample(all_images, min(len(all_images), num_images))
    print(f"Randomly selected {len(selected_images)} images for inference.")

    processed_images = []

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    # Process each image
    for img_path in selected_images:
        original_img = Image.open(img_path).convert("RGB")
        img_resized = original_img.resize((640, 480), Image.Resampling.BILINEAR)
        img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, aux = model(img_tensor)
            
            boxes_raw = aux["one2one"]["boxes"]
            scores_raw = aux["one2one"]["scores"]
            feats = aux["one2one"]["feats"]
            
            input_h, input_w = img_tensor.shape[-2:]
            centers, strides = _build_anchor_meta(feats, input_h, input_w, device)

            # Decode boxes
            boxes_raw = boxes_raw.permute(0, 2, 1)[0]
            pred_xy = centers + torch.tanh(boxes_raw[:, :2]) * strides
            pred_wh = torch.exp(boxes_raw[:, 2:].clamp(-4.0, 4.0)) * strides
            decoded_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
            decoded_xyxy = _xywh_to_xyxy(decoded_xywh)
            
            # Process scores
            logit_shift = 2.0 
            scores = (scores_raw + logit_shift).sigmoid().permute(0, 2, 1)[0]
            max_scores, class_preds = scores.max(dim=-1)
            
            # Filter by confidence
            mask = max_scores > conf_threshold
            final_boxes = decoded_xyxy[mask]
            final_scores = max_scores[mask]
            final_classes = class_preds[mask]

        # Apply NMS
        if len(final_boxes) > 0:
            keep_indices = torchvision.ops.nms(final_boxes, final_scores, iou_threshold=0.45)
            final_boxes = final_boxes[keep_indices]
            final_scores = final_scores[keep_indices]
            final_classes = final_classes[keep_indices]
            
        # Draw bounding boxes and collect text for canvas
        draw = ImageDraw.Draw(img_resized)
        detections_text_list = []
        
        for box, score, cls_id in zip(final_boxes, final_scores, final_classes):
            x1, y1, x2, y2 = box.cpu().tolist()
            conf = score.item()
            c_id = cls_id.item()
            
            # Draw on the image
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
            label = f"{GESTURE_CLASSES[c_id]}: {conf:.2f}"
            
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill="lime")
            draw.text((x1 + 2, y1 - text_h - 4), label, fill="black", font=font)

            # Collect for text underneath
            detections_text_list.append(label)

        # Format detection text for the subplot label
        if detections_text_list:
            det_text = "\n".join(detections_text_list)
        else:
            det_text = "No detections"

        # Store the drawn image and the text string for the canvas
        filename = os.path.basename(img_path)
        processed_images.append((filename, img_resized, det_text))
        print(f"Processed {filename}: Found {len(final_boxes)} objects.")

    # ---------------------------------------------------------------------------
    # Canvas Generation
    # ---------------------------------------------------------------------------
    print("Generating canvas...")
    # Slightly increased height (10 instead of 8) to accommodate the text underneath
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    fig.suptitle("YOLO26 Batch Inference Results", fontsize=18, fontweight='bold')
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < len(processed_images):
            img_name, img, det_text = processed_images[idx]
            ax.imshow(img)
            ax.set_title(img_name, fontsize=10)
            
            # Turn off ticks and plot outlines, but keep the axis active to show xlabel
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Add text underneath the image
            # 'darkred' makes it stand out, but you can change it to 'black' if preferred
            ax.set_xlabel(det_text, fontsize=11, fontweight='bold', color='darkred', labelpad=8)
        else:
            # For empty subplots, simply turn them entirely off
            ax.axis('off')
        
    plt.tight_layout()
    os.makedirs("outputs/inference", exist_ok=True)
    canvas_path = "outputs/inference/canvas_result.jpg"
    plt.savefig(canvas_path, dpi=300, bbox_inches='tight')
    print(f"Canvas saved successfully to {canvas_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing .png images")
    parser.add_argument("--weights", type=str, default="outputs/yolo26_scratch/best.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()
    
    run_folder_inference(args.folder, args.weights, args.conf)
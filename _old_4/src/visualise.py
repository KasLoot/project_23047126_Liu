import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.ops import nms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from model import HandGestureModel_v4, ModelConfig
from tqdm import tqdm

# ==========================================
# 1. INFERENCE AND NMS LOGIC
# ==========================================
def process_predictions(bbox_preds, bbox_cls, centerness, conf_threshold=0.4, iou_threshold=0.4):
    """Filters 6300 raw predictions down to the final valid detections."""
    # 1. Apply Sigmoid to get probabilities
    cls_probs = torch.sigmoid(bbox_cls[0])      # [10, 6300]
    center_probs = torch.sigmoid(centerness[0]) # [1, 6300]
    
    # 2. Get the highest class probability for each prediction
    max_cls_probs, max_cls_idx = torch.max(cls_probs, dim=0) # [6300]
    
    # 3. Calculate Final Score: Class Confidence * Centerness
    final_scores = max_cls_probs * center_probs.squeeze(0)   # [6300]
    
    # 4. Filter by confidence threshold
    keep_mask = final_scores > conf_threshold
    
    filtered_boxes = bbox_preds[0, :, keep_mask].T # [N, 4]
    filtered_scores = final_scores[keep_mask]      # [N]
    filtered_classes = max_cls_idx[keep_mask]      # [N]
    
    if len(filtered_boxes) == 0:
        return [], [], []

    # 5. Convert [cx, cy, raw_w, raw_h] to [xmin, ymin, xmax, ymax] safely
    cx = filtered_boxes[:, 0]
    cy = filtered_boxes[:, 1]
    w = F.softplus(filtered_boxes[:, 2]) # Ensure strictly positive width
    h = F.softplus(filtered_boxes[:, 3]) # Ensure strictly positive height
    
    xyxy_boxes = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)

    # 6. Apply Non-Maximum Suppression (NMS)
    nms_indices = nms(xyxy_boxes, filtered_scores, iou_threshold)
    
    final_boxes = xyxy_boxes[nms_indices].cpu().numpy()
    final_scores = filtered_scores[nms_indices].cpu().numpy()
    final_classes = filtered_classes[nms_indices].cpu().numpy()
    
    return final_boxes, final_scores, final_classes

# ==========================================
# 2. METRICS (IoU)
# ==========================================
def calculate_box_iou(box1, box2):
    """Calculates IoU between two [xmin, ymin, xmax, ymax] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / (union_area + 1e-6)

def calculate_mask_iou(pred_mask, gt_mask):
    """Calculates IoU for boolean segmentation masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / (union + 1e-6)

# ==========================================
# 3. VISUALIZATION ENGINE
# ==========================================
def visualize_predictions(root_folder, model_path=None, save_dir="outputs/visualise"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    config = ModelConfig()
    model = HandGestureModel_v4(config).to(device)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image_dir = os.path.join(root_folder, "images")
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    transform = T.Compose([
        T.Resize((480, 640)), # Ensure standard size
        T.ToTensor()
    ])

    # Process in chunks of 5
    for chunk_start in tqdm(range(0, len(image_files), 5), desc="Visualizing Batches"):
        chunk_files = image_files[chunk_start:chunk_start+5]
        
        # Build Canvas: 2 rows (Images on top, Histograms on bottom), 5 columns
        fig, axes = plt.subplots(2, 5, figsize=(25, 12))
        fig.suptitle(f"Inference Batch {chunk_start//5 + 1}", fontsize=20)
        
        for idx, img_name in enumerate(chunk_files):
            img_path = os.path.join(image_dir, img_name)
            raw_img = Image.open(img_path).convert("RGB")
            img_tensor = transform(raw_img).unsqueeze(0).to(device)

            # --- 1. Forward Pass ---
            with torch.no_grad():
                cls_logits, bbox_preds, bbox_cls, centerness, seg_map = model(img_tensor)
            
            # --- 2. Post-Process Detections ---
            boxes, scores, classes = process_predictions(bbox_preds, bbox_cls, centerness)
            
            # --- 3. Post-Process Segmentation ---
            # Sigmoid and threshold at 0.5 for binary mask
            seg_prob = torch.sigmoid(seg_map[0, 0]).cpu().numpy()
            pred_mask = seg_prob > 0.5

            # --- 4. Get Ground Truth (MOCK - Replace with your dataset logic) ---
            # gt_box = [xmin, ymin, xmax, ymax], gt_mask = boolean numpy array
            gt_box = [100, 100, 300, 300] 
            gt_mask = np.zeros((480, 640), dtype=bool) 
            
            det_iou = calculate_box_iou(boxes[0], gt_box) if len(boxes) > 0 else 0.0
            seg_iou = calculate_mask_iou(pred_mask, gt_mask)

            # --- 5. Plot Top Row: Image + BBox + Mask ---
            ax_img = axes[0, idx]
            ax_img.imshow(raw_img.resize((640, 480)))
            
            # Overlay Mask (Alpha blending)
            mask_overlay = np.zeros((480, 640, 4))
            mask_overlay[pred_mask] = [0, 1, 0, 0.4] # Green with 40% opacity
            ax_img.imshow(mask_overlay)

            # Draw Bounding Boxes
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax_img.add_patch(rect)
                ax_img.text(xmin, ymin - 10, f"Class {cls}: {score:.2f}", 
                            color='white', backgroundcolor='red', fontsize=10, weight='bold')
            
            ax_img.axis('off')
            ax_img.set_title(f"{img_name}\nDet IoU: {det_iou:.2f} | Seg IoU: {seg_iou:.2f}", fontsize=12)

            # --- 6. Plot Bottom Row: Class Histogram ---
            ax_hist = axes[1, idx]
            class_counts = [list(classes).count(c) for c in range(config.num_classes)]
            
            ax_hist.bar(range(config.num_classes), class_counts, color='royalblue')
            ax_hist.set_xticks(range(config.num_classes))
            ax_hist.set_xlabel("Gesture Class ID")
            ax_hist.set_ylabel("Detections")
            ax_hist.set_title("Class Distribution")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"inference_batch_{chunk_start//5 + 1}.png"))
        plt.close()

if __name__ == "__main__":
    # Point this to your dataset directory
    ROOT_DIR = "dataset/dataset_v1/val" 
    
    # Optional: Path to your saved weights from training
    MODEL_WEIGHTS = "outputs/stage_1/train_1/stage_1_best.pt" 

    save_dir = "outputs/visualise"
    os.makedirs(save_dir, exist_ok=True)
    
    visualize_predictions(ROOT_DIR, model_path=MODEL_WEIGHTS, save_dir=save_dir)
import os
import time
import numpy as np
import torch
import torchvision.ops as ops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchinfo

from dataloader import HandGestureDataset_v2, CLASS_ID_TO_NAME, _to_numpy_image_chw, _to_numpy_mask, detection_collate_fn
from model import HandGestureMultiTask
from loss import generate_anchors, decode_predictions, cxcywh_to_xyxy


def _compute_macro_f1(confusion_matrix: np.ndarray) -> float:
    f1_scores = []
    for class_idx in range(confusion_matrix.shape[0]):
        tp = float(confusion_matrix[class_idx, class_idx])
        fp = float(confusion_matrix[:, class_idx].sum() - tp)
        fn = float(confusion_matrix[class_idx, :].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def _format_confusion_matrix(confusion_matrix: np.ndarray) -> str:
    class_names = [CLASS_ID_TO_NAME[idx] for idx in range(confusion_matrix.shape[0])]
    cell_width = max(max(len(name) for name in class_names), len("true\\pred"), 5) + 2

    header = "true\\pred".ljust(cell_width) + "".join(name.rjust(cell_width) for name in class_names)
    rows = [header]
    for row_idx, row_name in enumerate(class_names):
        row_values = "".join(str(int(value)).rjust(cell_width) for value in confusion_matrix[row_idx])
        rows.append(row_name.ljust(cell_width) + row_values)
    return "\n".join(rows)


def _save_confusion_matrix_figure(confusion_matrix: np.ndarray, out_path: str) -> None:
    class_names = [CLASS_ID_TO_NAME[idx] for idx in range(confusion_matrix.shape[0])]

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(confusion_matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Classification Confusion Matrix")

    max_value = int(confusion_matrix.max()) if confusion_matrix.size > 0 else 0
    threshold = max_value / 2.0 if max_value > 0 else 0.0
    for row_idx in range(confusion_matrix.shape[0]):
        for col_idx in range(confusion_matrix.shape[1]):
            value = int(confusion_matrix[row_idx, col_idx])
            text_color = "white" if value > threshold else "black"
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=text_color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

@torch.no_grad()
def evaluate_and_visualize(weights_path: str = "C_model/outputs/s2/test1/best_model.pth", 
                           val_dir: str = "dataset/dataset_v1/test",
                           out_dir: str = "C_model/outputs/eval_results",
                           num_visualize: int = 10,
                           num_speed_warmup_batches: int = 5):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Setup Model & Dataset
    input_size = (480, 640)
    model = HandGestureMultiTask(num_classes=10, reg_max=1).to(device)
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        raise FileNotFoundError(f"Could not find weights at {weights_path}")
    
    model.eval()
    torchinfo.summary(model, input_size=(1, 3, input_size[0], input_size[1]), device=device)

    val_dataset = HandGestureDataset_v2(root_dir=val_dir, transform=None, resize_shape=list(input_size))
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=detection_collate_fn,
    )

    # Metrics trackers
    total_samples = 0
    correct_classes = 0
    correct_detector_classes = 0
    total_iou = 0.0
    correct_detections_iou50 = 0
    confusion_matrix = np.zeros((10, 10), dtype=np.int64)

    hand_intersection = 0.0
    hand_union = 0.0
    background_intersection = 0.0
    background_union = 0.0
    total_pred_hand_pixels = 0.0
    total_gt_hand_pixels = 0.0

    speed_warmup_batches = max(0, int(num_speed_warmup_batches))
    measured_forward_time = 0.0
    measured_forward_batches = 0
    measured_forward_samples = 0

    visualized = 0

    print(f"Starting Evaluation on {val_dir}...")
    for batch_idx, (images, masks, class_ids, gt_bboxes) in enumerate(tqdm(val_loader)):
        images = images.to(device)
        masks = masks.to(device).float()
        gt_bboxes = gt_bboxes.to(device)
        class_ids = class_ids.to(device)
        
        # Forward pass
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_start = time.perf_counter()
        preds = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_elapsed = time.perf_counter() - forward_start
        if batch_idx >= speed_warmup_batches:
            measured_forward_time += forward_elapsed
            measured_forward_batches += 1
            measured_forward_samples += images.shape[0]

        cls_logits = preds["cls"]
        cls_probs = cls_logits.softmax(dim=1)
        pred_class_ids = cls_probs.argmax(dim=1)
        pred_class_conf = cls_probs.gather(1, pred_class_ids.unsqueeze(1)).squeeze(1)

        seg_logits = preds["seg"]
        pred_masks = (seg_logits.sigmoid() > 0.5).float()
        target_masks = (masks > 0.5).float()

        hand_intersection += float((pred_masks * target_masks).sum().item())
        hand_union += float(((pred_masks + target_masks) > 0).sum().item())

        pred_background = 1.0 - pred_masks
        target_background = 1.0 - target_masks
        background_intersection += float((pred_background * target_background).sum().item())
        background_union += float(((pred_background + target_background) > 0).sum().item())

        total_pred_hand_pixels += float(pred_masks.sum().item())
        total_gt_hand_pixels += float(target_masks.sum().item())
        
        # Parse outputs: Transpose to (B, 6300, 4) and (B, 6300, 10)
        pred_boxes = preds["det"]["boxes"].transpose(1, 2)
        pred_scores = preds["det"]["scores"].transpose(1, 2)
        
        # Decode boxes from network output to absolute xyxy coordinates
        anchors, strides = generate_anchors(preds["det"]["feats"])
        decoded_boxes = decode_predictions(pred_boxes, anchors, strides) # (B, 6300, 4) in xyxy
        
        # Convert Ground Truth to xyxy
        gt_xyxy = cxcywh_to_xyxy(gt_bboxes) # (B, 4)

        batch_size = images.shape[0]
        for b in range(batch_size):
            total_samples += 1

            true_class_id = int(class_ids[b].item())
            predicted_class_id = int(pred_class_ids[b].item())
            confusion_matrix[true_class_id, predicted_class_id] += 1
            
            # --- Top-1 Prediction Logic ---
            # Get the probability for all classes across all 6300 anchors
            probs = pred_scores[b].sigmoid() # (6300, 10)
            
            # Find the absolute highest confidence score in the entire grid
            max_conf, max_idx = probs.max(dim=0) # Get max per class
            best_cls = max_conf.argmax()         # The predicted class
            best_anchor_idx = max_idx[best_cls]  # The index of the bounding box for that class
            detector_class_id = int(best_cls.item())
            
            pred_conf = max_conf[best_cls].item()
            pred_box = decoded_boxes[b, best_anchor_idx]
            
            # --- Metrics ---
            if predicted_class_id == true_class_id:
                correct_classes += 1
            if detector_class_id == true_class_id:
                correct_detector_classes += 1
                
            iou = ops.box_iou(pred_box.unsqueeze(0), gt_xyxy[b].unsqueeze(0)).item()
            total_iou += iou
            if iou >= 0.5:
                correct_detections_iou50 += 1

            # --- Visualization ---
            if visualized < num_visualize:
                # Convert tensors to CPU for plotting
                img_np = _to_numpy_image_chw(images[b])
                gt_mask_np = _to_numpy_mask(target_masks[b])
                pred_mask_np = _to_numpy_mask(pred_masks[b])
                gt_box_np = gt_xyxy[b].cpu().numpy()
                pr_box_np = pred_box.cpu().numpy()
                
                true_name = CLASS_ID_TO_NAME.get(true_class_id, "Unknown")
                pred_name = CLASS_ID_TO_NAME.get(predicted_class_id, "Unknown")
                det_pred_name = CLASS_ID_TO_NAME.get(detector_class_id, "Unknown")

                print(
                    f"Sample {total_samples:04d} | "
                    f"Cls true={true_name} pred={pred_name} ({pred_class_conf[b].item():.2f}) | "
                    f"Det pred={det_pred_name} ({pred_conf:.2f})"
                )

                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.imshow(img_np)

                pred_mask_overlay = np.ma.masked_where(pred_mask_np < 0.5, pred_mask_np)
                ax.imshow(pred_mask_overlay, cmap="cool", alpha=0.30, interpolation="nearest")
                if gt_mask_np.max() > 0:
                    ax.contour(gt_mask_np, levels=[0.5], colors=["yellow"], linewidths=1.5)
                
                # Draw Ground Truth Box (Green)
                gt_rect = patches.Rectangle((gt_box_np[0], gt_box_np[1]), 
                                            gt_box_np[2] - gt_box_np[0], gt_box_np[3] - gt_box_np[1], 
                                            linewidth=2, edgecolor='lime', facecolor='none', label='Ground Truth')
                ax.add_patch(gt_rect)
                
                # Draw Predicted Box (Red)
                pr_rect = patches.Rectangle((pr_box_np[0], pr_box_np[1]), 
                                            pr_box_np[2] - pr_box_np[0], pr_box_np[3] - pr_box_np[1], 
                                            linewidth=2, edgecolor='red', facecolor='none', linestyle='--', 
                                            label=f'Det: {det_pred_name} ({pred_conf:.2f})')
                ax.add_patch(pr_rect)

                ax.text(
                    0.02,
                    0.98,
                    (
                        f"Cls True: {true_name}\n"
                        f"Cls Pred: {pred_name} ({pred_class_conf[b].item():.2f})\n"
                        f"Det Pred: {det_pred_name} ({pred_conf:.2f})\n"
                        f"Det IoU: {iou:.2f}"
                    ),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=10,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.75, "pad": 6},
                )
                
                ax.set_title(f"True: {true_name} | Cls: {pred_name} | Det: {det_pred_name} | IoU: {iou:.2f}")
                ax.axis('off')
                ax.legend(loc='upper right')
                
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"eval_sample_{visualized}.png"))
                plt.close(fig)
                visualized += 1

    # Print Final Statistics
    acc = (correct_classes / total_samples) * 100 if total_samples > 0 else 0.0
    det_class_acc = (correct_detector_classes / total_samples) * 100 if total_samples > 0 else 0.0
    macro_f1 = _compute_macro_f1(confusion_matrix)
    mean_iou = total_iou / total_samples if total_samples > 0 else 0.0
    det_acc_iou50 = (correct_detections_iou50 / total_samples) * 100 if total_samples > 0 else 0.0

    hand_iou = hand_intersection / hand_union if hand_union > 0 else 0.0
    background_iou = background_intersection / background_union if background_union > 0 else 0.0
    seg_mean_iou = (hand_iou + background_iou) / 2.0
    dice = (2.0 * hand_intersection) / (total_pred_hand_pixels + total_gt_hand_pixels) if (total_pred_hand_pixels + total_gt_hand_pixels) > 0 else 0.0
    avg_batch_latency_ms = (measured_forward_time / measured_forward_batches) * 1000.0 if measured_forward_batches > 0 else 0.0
    avg_image_latency_ms = (measured_forward_time / measured_forward_samples) * 1000.0 if measured_forward_samples > 0 else 0.0
    throughput_fps = measured_forward_samples / measured_forward_time if measured_forward_time > 0 else 0.0

    confusion_matrix_path = os.path.join(out_dir, "classification_confusion_matrix.png")
    _save_confusion_matrix_figure(confusion_matrix, confusion_matrix_path)
    
    print("-" * 40)
    print("EVALUATION RESULTS")
    print("-" * 40)
    print(f"Dataset Path          : {val_dir}")
    print(f"Total Samples Tested : {total_samples}")
    print(f"Image Classifier Top-1: {acc:.2f}%")
    print(f"Classification Macro F1: {macro_f1:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(_format_confusion_matrix(confusion_matrix))

    print(f"Detector Class Top-1 : {det_class_acc:.2f}%")
    print(f"Detection Acc@IoU0.5 : {det_acc_iou50:.2f}%")
    print(f"Mean Bounding Box IoU: {mean_iou:.4f}")

    print(f"Segmentation mIoU    : {seg_mean_iou:.4f} (hand={hand_iou:.4f}, background={background_iou:.4f})")
    print(f"Segmentation Dice    : {dice:.4f}")
    print(
        f"Inference Speed      : {throughput_fps:.2f} img/s | "
        f"{avg_image_latency_ms:.2f} ms/img | {avg_batch_latency_ms:.2f} ms/batch"
    )
    print(
        f"Speed Measurement    : forward-only over {measured_forward_batches} batches "
        f"({measured_forward_samples} images), warmup_batches={speed_warmup_batches}"
    )
    print(f"Confusion matrix saved to '{confusion_matrix_path}'")
    print(f"Visualizations saved to '{out_dir}/'")

if __name__ == "__main__":
    evaluate_and_visualize()
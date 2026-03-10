import os
import time
import heapq
import numpy as np
import torch
import torchvision.ops as ops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchinfo

from dataloader import HandGestureDataset_v2, CLASS_ID_TO_NAME, _to_numpy_image_chw, _to_numpy_mask, detection_collate_fn
from model import RGB_V1, RGB_V2
from loss import generate_anchors, decode_predictions, cxcywh_to_xyxy


def _build_eval_logger(log_path: str):
    log_file = open(log_path, "w", encoding="utf-8")

    def _log(message: str = "") -> None:
        print(message)
        log_file.write(f"{message}\n")
        log_file.flush()

    return _log, log_file


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


def _save_demo_prediction_figure(sample: dict, save_path: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(sample["img_np"])

    pred_mask_overlay = np.ma.masked_where(sample["pred_mask_np"] < 0.5, sample["pred_mask_np"])
    ax.imshow(pred_mask_overlay, cmap="cool", alpha=0.30, interpolation="nearest")
    if sample["gt_mask_np"].max() > 0:
        ax.contour(sample["gt_mask_np"], levels=[0.5], colors=["yellow"], linewidths=1.5)

    gt_box_np = sample["gt_box_np"]
    pr_box_np = sample["pr_box_np"]
    gt_rect = patches.Rectangle(
        (gt_box_np[0], gt_box_np[1]),
        gt_box_np[2] - gt_box_np[0],
        gt_box_np[3] - gt_box_np[1],
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
        label="Ground Truth",
    )
    ax.add_patch(gt_rect)

    det_pred_name = sample["det_pred_name"]
    pred_conf = sample["pred_conf"]
    pr_rect = patches.Rectangle(
        (pr_box_np[0], pr_box_np[1]),
        pr_box_np[2] - pr_box_np[0],
        pr_box_np[3] - pr_box_np[1],
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
        label=f"Det: {det_pred_name} ({pred_conf:.2f})",
    )
    ax.add_patch(pr_rect)

    true_name = sample["true_name"]
    pred_name = sample["pred_name"]
    pred_class_conf = sample["pred_class_conf"]
    iou = sample["iou"]
    seg_hand_iou = sample["seg_hand_iou"]
    seg_background_iou = sample["seg_background_iou"]
    seg_miou = sample["seg_miou"]
    seg_dice = sample["seg_dice"]

    ax.text(
        0.02,
        0.98,
        (
            f"Cls True: {true_name}\n"
            f"Cls Pred: {pred_name} ({pred_class_conf:.2f})\n"
            f"Det Pred: {det_pred_name} ({pred_conf:.2f})\n"
            f"Det IoU: {iou:.2f}\n"
            f"Seg mIoU: {seg_miou:.2f} (h={seg_hand_iou:.2f}, b={seg_background_iou:.2f})\n"
            f"Seg Dice: {seg_dice:.2f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.75, "pad": 6},
    )

    ax.set_title(
        f"True: {true_name} | Cls: {pred_name} | Det: {det_pred_name} | "
        f"IoU: {iou:.2f} | Seg mIoU: {seg_miou:.2f} | Dice: {seg_dice:.2f}"
    )
    ax.axis("off")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

@torch.no_grad()
def evaluate_and_visualize(weights_path: str = "_combined_worked/outputs/RGB_V1/s2/t1/best_model.pth", 
                           val_dir: str = "dataset/dataset_v1/test",
                           num_visualize: int = 10,
                           num_speed_warmup_batches: int = 5):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir: str = weights_path.replace("best_model.pth", "eval_visualizations")

    os.makedirs(out_dir, exist_ok=True)
    eval_log_path = os.path.join(out_dir, "eval.txt")
    log, log_file = _build_eval_logger(eval_log_path)
    
    try:
        # 1. Setup Model & Dataset
        input_size = (480, 640)
        model = RGB_V1(num_classes=10, reg_max=1).to(device)
        
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
            log(f"Loaded weights from {weights_path}")
        else:
            raise FileNotFoundError(f"Could not find weights at {weights_path}")
        
        model.eval()
        model_summary = torchinfo.summary(model, input_size=(1, 3, input_size[0], input_size[1]), device=device, verbose=0)
        log(str(model_summary))

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

        best_visualizations = []
        visualization_candidate_id = 0

        log(f"Starting Evaluation on {val_dir}...")
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

                sample_pred_mask = pred_masks[b]
                sample_target_mask = target_masks[b]
                sample_hand_intersection = float((sample_pred_mask * sample_target_mask).sum().item())
                sample_hand_union = float(((sample_pred_mask + sample_target_mask) > 0).sum().item())
                sample_hand_iou = sample_hand_intersection / sample_hand_union if sample_hand_union > 0 else 0.0

                sample_pred_background = 1.0 - sample_pred_mask
                sample_target_background = 1.0 - sample_target_mask
                sample_background_intersection = float((sample_pred_background * sample_target_background).sum().item())
                sample_background_union = float(((sample_pred_background + sample_target_background) > 0).sum().item())
                sample_background_iou = sample_background_intersection / sample_background_union if sample_background_union > 0 else 0.0

                sample_seg_miou = (sample_hand_iou + sample_background_iou) / 2.0
                sample_dice_denom = float(sample_pred_mask.sum().item() + sample_target_mask.sum().item())
                sample_seg_dice = (2.0 * sample_hand_intersection / sample_dice_denom) if sample_dice_denom > 0 else 0.0

                # --- Visualization Candidate Ranking ---
                if num_visualize > 0:
                    pred_cls_conf_value = float(pred_class_conf[b].item())
                    quality_key = (
                        int(predicted_class_id == true_class_id),
                        int(detector_class_id == true_class_id),
                        float(sample_seg_miou),
                        float(sample_seg_dice),
                        float(iou),
                        float(pred_cls_conf_value),
                        float(pred_conf),
                    )

                    should_store = len(best_visualizations) < num_visualize
                    if not should_store and len(best_visualizations) > 0:
                        should_store = quality_key > best_visualizations[0][0]

                    if should_store:
                        true_name = CLASS_ID_TO_NAME.get(true_class_id, "Unknown")
                        pred_name = CLASS_ID_TO_NAME.get(predicted_class_id, "Unknown")
                        det_pred_name = CLASS_ID_TO_NAME.get(detector_class_id, "Unknown")

                        sample_payload = {
                            "sample_index": total_samples,
                            "img_np": _to_numpy_image_chw(images[b]),
                            "gt_mask_np": _to_numpy_mask(target_masks[b]),
                            "pred_mask_np": _to_numpy_mask(pred_masks[b]),
                            "gt_box_np": gt_xyxy[b].cpu().numpy(),
                            "pr_box_np": pred_box.cpu().numpy(),
                            "true_name": true_name,
                            "pred_name": pred_name,
                            "det_pred_name": det_pred_name,
                            "pred_class_conf": pred_cls_conf_value,
                            "pred_conf": float(pred_conf),
                            "iou": float(iou),
                            "seg_hand_iou": float(sample_hand_iou),
                            "seg_background_iou": float(sample_background_iou),
                            "seg_miou": float(sample_seg_miou),
                            "seg_dice": float(sample_seg_dice),
                        }
                        heap_item = (quality_key, visualization_candidate_id, sample_payload)
                        if len(best_visualizations) < num_visualize:
                            heapq.heappush(best_visualizations, heap_item)
                        else:
                            heapq.heapreplace(best_visualizations, heap_item)
                        visualization_candidate_id += 1

        sorted_visualizations = sorted(best_visualizations, key=lambda item: (item[0], item[1]), reverse=True)
        for rank, (_, _, sample_payload) in enumerate(sorted_visualizations, start=1):
            save_path = os.path.join(out_dir, f"eval_sample_{rank - 1}.png")
            _save_demo_prediction_figure(sample_payload, save_path)
            log(
                f"Demo Rank {rank:02d} | Source Sample {sample_payload['sample_index']:04d} | "
                f"Cls true={sample_payload['true_name']} pred={sample_payload['pred_name']} ({sample_payload['pred_class_conf']:.2f}) | "
                f"Det pred={sample_payload['det_pred_name']} ({sample_payload['pred_conf']:.2f}) | "
                f"IoU={sample_payload['iou']:.2f} | "
                f"Seg mIoU={sample_payload['seg_miou']:.2f} (h={sample_payload['seg_hand_iou']:.2f}, b={sample_payload['seg_background_iou']:.2f}) | "
                f"Seg Dice={sample_payload['seg_dice']:.2f}"
            )

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
        
        log("-" * 40)
        log("EVALUATION RESULTS")
        log("-" * 40)
        log(f"Dataset Path          : {val_dir}")
        log(f"Total Samples Tested : {total_samples}")
        log(f"Image Classifier Top-1: {acc:.2f}%")
        log(f"Classification Macro F1: {macro_f1:.4f}")
        log("Confusion Matrix (rows=true, cols=pred):")
        log(_format_confusion_matrix(confusion_matrix))

        log(f"Detector Class Top-1 : {det_class_acc:.2f}%")
        log(f"Detection Acc@IoU0.5 : {det_acc_iou50:.2f}%")
        log(f"Mean Bounding Box IoU: {mean_iou:.4f}")

        log(f"Segmentation mIoU    : {seg_mean_iou:.4f} (hand={hand_iou:.4f}, background={background_iou:.4f})")
        log(f"Segmentation Dice    : {dice:.4f}")
        log(
            f"Inference Speed      : {throughput_fps:.2f} img/s | "
            f"{avg_image_latency_ms:.2f} ms/img | {avg_batch_latency_ms:.2f} ms/batch"
        )
        log(
            f"Speed Measurement    : forward-only over {measured_forward_batches} batches "
            f"({measured_forward_samples} images), warmup_batches={speed_warmup_batches}"
        )
        log(f"Confusion matrix saved to '{confusion_matrix_path}'")
        log(f"Visualizations saved to '{out_dir}/'")
        log(f"Evaluation log saved to '{eval_log_path}'")
    finally:
        log_file.close()

if __name__ == "__main__":
    evaluate_and_visualize()
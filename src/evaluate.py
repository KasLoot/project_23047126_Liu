import argparse
import os
import time

import numpy as np
import torch
import torchvision.ops as ops
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import CLASS_ID_TO_NAME, HandGestureDataset_v2, detection_collate_fn
from model import RGB_V1, RGB_V2, RGB_V3, RGB_V4
from utils import (
    build_logger,
    compute_macro_f1,
    cxcywh_to_xyxy,
    decode_predictions,
    ensure_dir,
    ensure_project_dirs,
    format_confusion_matrix,
    generate_anchors,
    save_confusion_matrix_figure,
    save_json,
)


def _build_model(model_name: str, num_classes: int, reg_max: int = 1):
    if model_name == "rgb_v1":
        return RGB_V1(num_classes=num_classes, reg_max=reg_max)
    elif model_name == "rgb_v2":
        return RGB_V2(num_classes=num_classes, reg_max=reg_max)
    elif model_name == "rgb_v3":
        return RGB_V3(num_classes=num_classes, reg_max=reg_max)
    elif model_name == "rgb_v4":
        return RGB_V4(num_classes=num_classes, reg_max=reg_max)




def _default_weights_path(model_name: str, weights_dir: str, stage: str, run_name: str) -> str:
    structured_path = os.path.join(weights_dir, model_name, stage, run_name, "best_model.pth")
    if os.path.exists(structured_path):
        return structured_path
    return os.path.join(weights_dir, f"{model_name}_{stage}_best_model.pth")


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, results_dir = ensure_project_dirs(args.weights_dir, args.results_dir)
    out_dir = ensure_dir(os.path.join(results_dir, args.model, "eval", args.stage, args.run_name))
    log, close_log = build_logger(os.path.join(out_dir, "eval.txt"), mode="w")
    weights_path = args.weights or _default_weights_path(args.model, args.weights_dir, args.stage, args.run_name)

    try:
        model = _build_model(args.model, num_classes=args.num_classes, reg_max=1).to(device)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Could not find weights at {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.eval()

        input_size = (args.image_h, args.image_w)
        dataset = HandGestureDataset_v2(root_dir=args.data_dir, transform=None, resize_shape=list(input_size))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=detection_collate_fn,
            persistent_workers=(args.num_workers > 0),
        )

        total_samples = 0
        correct_classes = 0
        correct_detector_classes = 0
        total_iou = 0.0
        correct_detections_iou50 = 0
        confusion_matrix = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)

        hand_intersection = 0.0
        hand_union = 0.0
        background_intersection = 0.0
        background_union = 0.0
        total_pred_hand_pixels = 0.0
        total_gt_hand_pixels = 0.0

        speed_warmup_batches = max(0, int(args.warmup_batches))
        measured_forward_time = 0.0
        measured_forward_batches = 0
        measured_forward_samples = 0

        log(f"Using device: {device}")
        log(f"Model: {args.model}")
        log(f"Stage: {args.stage}")
        log(f"Loaded weights: {weights_path}")
        log(f"Evaluating: {args.data_dir}")

        for batch_idx, (images, masks, class_ids, gt_bboxes) in enumerate(tqdm(loader, desc="Evaluating")):
            images = images.to(device)
            masks = masks.to(device).float()
            gt_bboxes = gt_bboxes.to(device)
            class_ids = class_ids.to(device)

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

            pred_boxes = preds["det"]["boxes"].transpose(1, 2)
            pred_scores = preds["det"]["scores"].transpose(1, 2)
            anchors, strides = generate_anchors(preds["det"]["feats"])
            decoded_boxes = decode_predictions(pred_boxes, anchors, strides)
            gt_xyxy = cxcywh_to_xyxy(gt_bboxes)

            batch_size = images.shape[0]
            for b in range(batch_size):
                total_samples += 1

                true_class_id = int(class_ids[b].item())
                predicted_class_id = int(pred_class_ids[b].item())
                confusion_matrix[true_class_id, predicted_class_id] += 1

                probs = pred_scores[b].sigmoid()
                max_conf, max_idx = probs.max(dim=0)
                best_cls = max_conf.argmax()
                best_anchor_idx = max_idx[best_cls]
                detector_class_id = int(best_cls.item())
                pred_box = decoded_boxes[b, best_anchor_idx]

                if predicted_class_id == true_class_id:
                    correct_classes += 1
                if detector_class_id == true_class_id:
                    correct_detector_classes += 1

                iou = ops.box_iou(pred_box.unsqueeze(0), gt_xyxy[b].unsqueeze(0)).item()
                total_iou += iou
                if iou >= 0.5:
                    correct_detections_iou50 += 1

        acc = (correct_classes / total_samples) * 100 if total_samples > 0 else 0.0
        det_class_acc = (correct_detector_classes / total_samples) * 100 if total_samples > 0 else 0.0
        macro_f1 = compute_macro_f1(confusion_matrix)
        mean_iou = total_iou / total_samples if total_samples > 0 else 0.0
        det_acc_iou50 = (correct_detections_iou50 / total_samples) * 100 if total_samples > 0 else 0.0

        hand_iou = hand_intersection / hand_union if hand_union > 0 else 0.0
        background_iou = background_intersection / background_union if background_union > 0 else 0.0
        seg_mean_iou = (hand_iou + background_iou) / 2.0
        dice = (
            (2.0 * hand_intersection) / (total_pred_hand_pixels + total_gt_hand_pixels)
            if (total_pred_hand_pixels + total_gt_hand_pixels) > 0
            else 0.0
        )
        avg_batch_latency_ms = (measured_forward_time / measured_forward_batches) * 1000.0 if measured_forward_batches > 0 else 0.0
        avg_image_latency_ms = (measured_forward_time / measured_forward_samples) * 1000.0 if measured_forward_samples > 0 else 0.0
        throughput_fps = measured_forward_samples / measured_forward_time if measured_forward_time > 0 else 0.0

        class_names = [CLASS_ID_TO_NAME[idx] for idx in range(args.num_classes)]
        confusion_matrix_path = os.path.join(out_dir, "classification_confusion_matrix.png")
        save_confusion_matrix_figure(confusion_matrix, class_names, confusion_matrix_path)

        metrics = {
            "model": args.model,
            "dataset": args.data_dir,
            "weights": weights_path,
            "total_samples": int(total_samples),
            "image_classifier_top1_percent": float(acc),
            "classification_macro_f1": float(macro_f1),
            "detector_class_top1_percent": float(det_class_acc),
            "detection_acc_iou50_percent": float(det_acc_iou50),
            "mean_bbox_iou": float(mean_iou),
            "segmentation_miou": float(seg_mean_iou),
            "segmentation_hand_iou": float(hand_iou),
            "segmentation_background_iou": float(background_iou),
            "segmentation_dice": float(dice),
            "speed_throughput_fps": float(throughput_fps),
            "speed_avg_image_latency_ms": float(avg_image_latency_ms),
            "speed_avg_batch_latency_ms": float(avg_batch_latency_ms),
            "speed_warmup_batches": int(speed_warmup_batches),
            "speed_measured_batches": int(measured_forward_batches),
            "speed_measured_images": int(measured_forward_samples),
        }
        save_json(metrics, os.path.join(out_dir, "metrics.json"))

        log("-" * 40)
        log("EVALUATION RESULTS")
        log("-" * 40)
        log(f"Total Samples Tested : {total_samples}")
        log(f"Image Classifier Top-1: {acc:.2f}%")
        log(f"Classification Macro F1: {macro_f1:.4f}")
        log("Confusion Matrix (rows=true, cols=pred):")
        log(format_confusion_matrix(confusion_matrix, class_names))
        log(f"Detector Class Top-1 : {det_class_acc:.2f}%")
        log(f"Detection Acc@IoU0.5 : {det_acc_iou50:.2f}%")
        log(f"Mean Bounding Box IoU: {mean_iou:.4f}")
        log(f"Segmentation mIoU    : {seg_mean_iou:.4f} (hand={hand_iou:.4f}, background={background_iou:.4f})")
        log(f"Segmentation Dice    : {dice:.4f}")
        log(
            f"Inference Speed      : {throughput_fps:.2f} img/s | "
            f"{avg_image_latency_ms:.2f} ms/img | {avg_batch_latency_ms:.2f} ms/batch"
        )
        log(f"Confusion matrix saved to {confusion_matrix_path}")
        log(f"Metrics JSON saved to {os.path.join(out_dir, 'metrics.json')}")
    finally:
        close_log()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", type=str, choices=["rgb_v1", "rgb_v2", "rgb_v3", "rgb_v4"], default="rgb_v2")
    parser.add_argument("--stage", type=str, choices=["s1", "s2"], default="s2")
    parser.add_argument("--weights", type=str, default=None, help="Optional checkpoint path override")
    parser.add_argument("--data_dir", type=str, default="dataset/dataset_v1/test")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=640)
    parser.add_argument("--warmup_batches", type=int, default=5)
    parser.add_argument("--run_name", type=str, default="default", help="Training run identifier for model/stage outputs")
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--results_dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
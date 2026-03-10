import json
import os
import random
import shutil
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from torchvision.transforms import functional as tvf
from tqdm import tqdm


CLASS_ID_TO_NAME = {
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


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ensure_project_dirs(weights_dir: str = "weights", results_dir: str = "results") -> tuple[str, str]:
    return ensure_dir(weights_dir), ensure_dir(results_dir)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(payload: dict, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_logger(log_path: str, mode: str = "w") -> tuple[Callable[[str], None], Callable[[], None]]:
    ensure_dir(os.path.dirname(log_path) or ".")
    log_file = open(log_path, mode, encoding="utf-8")

    def log(message: str = "") -> None:
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    def close() -> None:
        log_file.close()

    return log, close


def compute_macro_f1(confusion_matrix: np.ndarray) -> float:
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


def format_confusion_matrix(confusion_matrix: np.ndarray, class_names: list[str]) -> str:
    cell_width = max(max(len(name) for name in class_names), len("true\\pred"), 5) + 2
    header = "true\\pred".ljust(cell_width) + "".join(name.rjust(cell_width) for name in class_names)
    rows = [header]
    for row_idx, row_name in enumerate(class_names):
        row_values = "".join(str(int(value)).rjust(cell_width) for value in confusion_matrix[row_idx])
        rows.append(row_name.ljust(cell_width) + row_values)
    return "\n".join(rows)


def save_confusion_matrix_figure(confusion_matrix: np.ndarray, class_names: list[str], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
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


def summarize_segmentation_metrics(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    pred_masks = (pred_logits.sigmoid() > threshold).float()
    targets = (targets > 0.5).float()

    hand_intersection = float((pred_masks * targets).sum().item())
    hand_union = float(((pred_masks + targets) > 0).sum().item())

    pred_background = 1.0 - pred_masks
    target_background = 1.0 - targets
    background_intersection = float((pred_background * target_background).sum().item())
    background_union = float(((pred_background + target_background) > 0).sum().item())

    pred_hand_pixels = float(pred_masks.sum().item())
    gt_hand_pixels = float(targets.sum().item())

    return {
        "hand_intersection": hand_intersection,
        "hand_union": hand_union,
        "background_intersection": background_intersection,
        "background_union": background_union,
        "pred_hand_pixels": pred_hand_pixels,
        "gt_hand_pixels": gt_hand_pixels,
    }


def finalize_segmentation_metrics(metric_sums: dict[str, float]) -> tuple[float, float, float, float]:
    hand_iou = metric_sums["hand_intersection"] / metric_sums["hand_union"] if metric_sums["hand_union"] > 0 else 0.0
    background_iou = (
        metric_sums["background_intersection"] / metric_sums["background_union"]
        if metric_sums["background_union"] > 0
        else 0.0
    )
    mean_iou = (hand_iou + background_iou) / 2.0
    dice = (
        (2.0 * metric_sums["hand_intersection"]) / (metric_sums["pred_hand_pixels"] + metric_sums["gt_hand_pixels"])
        if (metric_sums["pred_hand_pixels"] + metric_sums["gt_hand_pixels"]) > 0
        else 0.0
    )
    return mean_iou, dice, hand_iou, background_iou


def box_iou_diagonal(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1 + area2 - inter + eps
    return inter / union


def init_epoch_stats(num_classes: int) -> dict:
    return {
        "loss_sum": 0.0,
        "num_batches": 0,
        "num_samples": 0,
        "cls_correct": 0,
        "det_iou_sum": 0.0,
        "det_iou50_correct": 0,
        "confusion": np.zeros((num_classes, num_classes), dtype=np.int64),
    }


def finalize_epoch_stats(stats: dict) -> dict[str, float]:
    num_samples = max(stats["num_samples"], 1)
    num_batches = max(stats["num_batches"], 1)

    avg_loss = stats["loss_sum"] / num_batches
    top1_acc = stats["cls_correct"] / num_samples
    det_acc_iou50 = stats["det_iou50_correct"] / num_samples
    mean_bbox_iou = stats["det_iou_sum"] / num_samples
    macro_f1 = compute_macro_f1(stats["confusion"])

    return {
        "loss": float(avg_loss),
        "det_acc_iou50": float(det_acc_iou50),
        "mean_bbox_iou": float(mean_bbox_iou),
        "top1_acc": float(top1_acc),
        "macro_f1": float(macro_f1),
    }


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x1 = boxes[..., 0] - boxes[..., 2] / 2
    y1 = boxes[..., 1] - boxes[..., 3] / 2
    x2 = boxes[..., 0] + boxes[..., 2] / 2
    y2 = boxes[..., 1] + boxes[..., 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def generate_anchors(neck_features: list[torch.Tensor], strides: list[int] = [8, 16, 32]) -> tuple[torch.Tensor, torch.Tensor]:
    device = neck_features[0].device
    anchors = []
    anchor_strides = []

    for feat, stride in zip(neck_features, strides):
        _, _, h, w = feat.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")

        center_x = (grid_x.flatten() + 0.5) * stride
        center_y = (grid_y.flatten() + 0.5) * stride

        anchors.append(torch.stack([center_x, center_y], dim=-1))
        anchor_strides.append(torch.full((h * w,), stride, device=device))

    return torch.cat(anchors, dim=0), torch.cat(anchor_strides, dim=0)


def decode_predictions(pred_boxes: torch.Tensor, anchors: torch.Tensor, strides: torch.Tensor) -> torch.Tensor:
    distances = F.relu(pred_boxes) * strides.view(1, -1, 1)

    x1 = anchors[:, 0].unsqueeze(0) - distances[..., 0]
    y1 = anchors[:, 1].unsqueeze(0) - distances[..., 1]
    x2 = anchors[:, 0].unsqueeze(0) + distances[..., 2]
    y2 = anchors[:, 1].unsqueeze(0) + distances[..., 3]

    return torch.stack([x1, y1, x2, y2], dim=-1)


def update_detection_metrics(det_outputs: dict, gt_bboxes_cxcywh: torch.Tensor) -> tuple[float, int]:
    pred_boxes = det_outputs["boxes"].transpose(1, 2)
    pred_scores = det_outputs["scores"].transpose(1, 2)

    anchors, strides = generate_anchors(det_outputs["feats"])
    decoded_boxes = decode_predictions(pred_boxes, anchors, strides)
    gt_xyxy = cxcywh_to_xyxy(gt_bboxes_cxcywh)

    bsz, _, num_classes = pred_scores.shape
    probs = pred_scores.sigmoid()
    best_flat_idx = probs.reshape(bsz, -1).argmax(dim=1)
    best_anchor_idx = best_flat_idx // num_classes
    pred_top_boxes = decoded_boxes[torch.arange(bsz, device=decoded_boxes.device), best_anchor_idx]
    ious = box_iou_diagonal(pred_top_boxes, gt_xyxy)
    correct_det_iou50 = int((ious >= 0.5).sum().item())

    return float(ious.sum().item()), correct_det_iou50


class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk: int = 10, alpha: float = 0.5, beta: float = 6.0):
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def forward(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes_xyxy: torch.Tensor,
        anchors: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes_xyxy: torch.Tensor,
    ):
        batch_size, num_anchors, _ = pred_scores.shape
        device = pred_scores.device

        target_bboxes = torch.zeros_like(pred_bboxes_xyxy)
        target_scores = torch.zeros_like(pred_scores)
        fg_mask = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=device)

        for b in range(batch_size):
            gt_box = gt_bboxes_xyxy[b]
            gt_cls = int(gt_labels[b].item())

            if (gt_box[0, 2] <= gt_box[0, 0]) or (gt_box[0, 3] <= gt_box[0, 1]):
                continue

            cx, cy = anchors[:, 0], anchors[:, 1]
            is_in_gt = (cx >= gt_box[0, 0]) & (cx <= gt_box[0, 2]) & (cy >= gt_box[0, 1]) & (cy <= gt_box[0, 3])
            valid_indices = torch.where(is_in_gt)[0]

            if len(valid_indices) == 0:
                distances = (cx - (gt_box[0, 0] + gt_box[0, 2]) / 2) ** 2 + (cy - (gt_box[0, 1] + gt_box[0, 3]) / 2) ** 2
                valid_indices = torch.argmin(distances).unsqueeze(0)

            valid_pred_boxes = pred_bboxes_xyxy[b, valid_indices]
            valid_pred_scores = pred_scores[b, valid_indices, gt_cls].sigmoid()

            ious = ops.box_iou(valid_pred_boxes, gt_box).squeeze(-1)
            ious = torch.clamp(ious, min=1e-9)
            alignment_metrics = (valid_pred_scores ** self.alpha) * (ious ** self.beta)

            k = min(self.topk, len(valid_indices))
            _, topk_idx = torch.topk(alignment_metrics, k)

            final_pos_indices = valid_indices[topk_idx]
            final_ious = ious[topk_idx]

            fg_mask[b, final_pos_indices] = True
            target_bboxes[b, final_pos_indices] = gt_box[0]
            target_scores[b, final_pos_indices, gt_cls] = final_ious

        return target_bboxes, target_scores, fg_mask


def custom_ciou_loss(boxes1: torch.Tensor, boxes2: torch.Tensor, reduction: str = "none", eps: float = 1e-7) -> torch.Tensor:
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.unbind(-1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw**2 + ch**2 + eps

    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

    v = (4 / (torch.pi**2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))

    loss = 1 - iou + (rho2 / c2) + (alpha * v)

    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


class YOLODetectionLoss(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.assigner = TaskAlignedAssigner(topk=10)
        self.cls_weight = 1.0
        self.box_weight = 2.5

    def _normalize_targets(self, gt_bboxes_cxcywh: torch.Tensor, gt_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if gt_bboxes_cxcywh.ndim == 2:
            gt_bboxes_cxcywh = gt_bboxes_cxcywh.unsqueeze(1)
        elif gt_bboxes_cxcywh.ndim != 3:
            raise ValueError(
                f"Expected gt_bboxes_cxcywh with shape (B, 4) or (B, 1, 4), got {tuple(gt_bboxes_cxcywh.shape)}"
            )

        if gt_bboxes_cxcywh.shape[-1] != 4:
            raise ValueError(f"Expected last bbox dimension to be 4, got {tuple(gt_bboxes_cxcywh.shape)}")

        gt_labels = gt_labels.view(-1)
        if gt_labels.shape[0] != gt_bboxes_cxcywh.shape[0]:
            raise ValueError("Ground-truth labels and bounding boxes must have the same batch dimension")

        return gt_bboxes_cxcywh, gt_labels

    def forward(self, preds_dict: dict, gt_bboxes_cxcywh: torch.Tensor, gt_labels: torch.Tensor):
        pred_boxes = preds_dict["boxes"].transpose(1, 2)
        pred_scores = preds_dict["scores"].transpose(1, 2)
        neck_features = preds_dict["feats"]

        gt_bboxes_cxcywh, gt_labels = self._normalize_targets(gt_bboxes_cxcywh, gt_labels)
        gt_bboxes_xyxy = cxcywh_to_xyxy(gt_bboxes_cxcywh)

        anchors, strides = generate_anchors(neck_features)
        pred_bboxes_xyxy = decode_predictions(pred_boxes, anchors, strides)

        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores,
            pred_bboxes_xyxy,
            anchors,
            gt_labels,
            gt_bboxes_xyxy,
        )

        loss_cls = ops.sigmoid_focal_loss(pred_scores, target_scores, alpha=0.25, gamma=2.0, reduction="sum")

        pos_pred_bboxes = pred_bboxes_xyxy[fg_mask]
        pos_target_bboxes = target_bboxes[fg_mask]
        if pos_pred_bboxes.shape[0] > 0:
            loss_box = custom_ciou_loss(pos_pred_bboxes, pos_target_bboxes, reduction="sum")
        else:
            loss_box = torch.tensor(0.0, device=pred_scores.device)

        num_positives = max(1.0, fg_mask.sum().item())
        loss_cls = loss_cls / num_positives
        loss_box = loss_box / num_positives
        total_loss = (loss_cls * self.cls_weight) + (loss_box * self.box_weight)

        return total_loss, {"loss_cls": loss_cls.item(), "loss_box": loss_box.item()}


def varify_dir_list(dir_list: list[str]) -> bool:
    for dir_path in dir_list:
        if not os.path.exists(dir_path):
            print(f"Directory {dir_path} does not exist.")
            return False
    return True


def mask_to_bbox(mask: np.ndarray) -> list[float]:
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x_min, x_max = np.min(coords[1]), np.max(coords[1])
    y_min, y_max = np.min(coords[0]), np.max(coords[0])
    x_center = float((x_min + x_max) / 2)
    y_center = float((y_min + y_max) / 2)
    width = float(x_max - x_min)
    height = float(y_max - y_min)
    return [x_center, y_center, width, height]


def gether_images_and_masks(dataset_path: str, output_dir: str) -> None:
    output_image_dir = os.path.join(output_dir, "images")
    output_mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    image_name_index = 0
    all_image_info = []

    for student_dir in tqdm(os.listdir(dataset_path), desc="Processing"):
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

            all_mask_dirs = (
                call_mask_dir
                + dislike_mask_dir
                + like_mask_dir
                + ok_mask_dir
                + one_mask_dir
                + palm_mask_dir
                + peace_mask_dir
                + rock_mask_dir
                + stop_mask_dir
                + three_mask_dir
            )

            try:
                for mask_dir in all_mask_dirs:
                    if not os.path.isdir(mask_dir):
                        continue

                    image_dir = mask_dir.replace("annotation", "rgb")
                    if not os.path.isdir(image_dir):
                        continue

                    for file in os.listdir(mask_dir):
                        if file.endswith(".png"):
                            mask_path = os.path.join(mask_dir, file)
                            image_path = os.path.join(image_dir, file)
                            if (not os.path.exists(image_path)) or (not os.path.exists(mask_path)):
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
                                "bbox": bbox,
                            }
                            all_image_info.append(image_info)
                            image_name_index += 1
            except Exception as error:
                print(f"Error processing student {student_dir}: {error}")
                continue

    with open(os.path.join(output_dir, "image_info.json"), "w", encoding="utf-8") as f:
        json.dump(all_image_info, f, indent=4)


def image_to_tensor(dataset_path: str) -> None:
    images_dir = os.path.join(dataset_path, "images")
    masks_dir = os.path.join(dataset_path, "masks")
    os.makedirs(os.path.join(dataset_path, "image_tensors"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "mask_tensors"), exist_ok=True)

    for file in tqdm(os.listdir(images_dir), desc=f"Processing images in {dataset_path}"):
        if file.endswith(".png"):
            image_path = os.path.join(images_dir, file)
            mask_path = os.path.join(masks_dir, file)
            image_tensor = tvf.to_tensor(Image.open(image_path).convert("RGB"))
            mask_tensor = tvf.to_tensor(Image.open(mask_path).convert("L"))
            torch.save(image_tensor, os.path.join(dataset_path, "image_tensors", file.replace(".png", ".pt")))
            torch.save(mask_tensor, os.path.join(dataset_path, "mask_tensors", file.replace(".png", ".pt")))


def balance_data_distribution(dataset_path: str) -> None:
    image_info_path = os.path.join(dataset_path, "image_info.json")
    if not os.path.exists(image_info_path):
        print(f"Error: image_info.json not found in {dataset_path}")
        return

    with open(image_info_path, "r", encoding="utf-8") as f:
        all_image_info = json.load(f)

    images_dir = os.path.join(dataset_path, "images")
    try:
        existing_images = set(os.listdir(images_dir))
    except FileNotFoundError:
        print(f"Error: images directory not found in {dataset_path}")
        return

    all_image_info = [info for info in all_image_info if info["new_image_name"] in existing_images]

    class_groups = {}
    for item in all_image_info:
        class_id = item["class_id"]
        class_groups.setdefault(class_id, []).append(item)

    if not class_groups:
        print("No classes found in dataset.")
        return

    min_samples = min(len(samples) for samples in class_groups.values())
    print(f"Balancing dataset to {min_samples} samples per class.")

    new_all_image_info = []
    files_to_remove = []
    for _, samples in class_groups.items():
        if len(samples) > min_samples:
            samples_to_keep = random.sample(samples, min_samples)
            samples_to_remove = [sample for sample in samples if sample not in samples_to_keep]

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
        if os.path.exists(file_path):
            os.remove(file_path)

    with open(image_info_path, "w", encoding="utf-8") as f:
        json.dump(new_all_image_info, f, indent=4)

    print(f"Balanced dataset saved: {dataset_path}")


def get_class_distribution(dataset_path: str) -> dict[str, int]:
    image_info_path = os.path.join(dataset_path, "image_info.json")
    if not os.path.exists(image_info_path):
        print(f"Warning: image_info.json not found in {dataset_path}")
        return {}

    with open(image_info_path, "r", encoding="utf-8") as f:
        image_info = json.load(f)

    distribution = {name: 0 for name in CLASS_ID_TO_NAME.values()}
    for item in image_info:
        class_id = int(item["class_id"])
        class_name = CLASS_ID_TO_NAME.get(class_id, f"class_{class_id}")
        distribution[class_name] = distribution.get(class_name, 0) + 1

    distribution = {name: count for name, count in distribution.items() if count > 0}
    return dict(sorted(distribution.items(), key=lambda pair: pair[0]))


def plot_distribution(distribution: dict[str, int], title: str, output_filename: str) -> None:
    if not distribution:
        print(f"Skipping plot for {title} due to empty distribution.")
        return

    labels = list(distribution.keys())
    values = list(distribution.values())

    plt.figure(figsize=(12, 7))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    ensure_dir(os.path.dirname(output_filename) or ".")
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved plot to {output_filename}")


def run_preprocess_pipeline(origin_dataset_path: str, output_dataset_path: str) -> None:
    print("Running gether_images_and_masks...")
    gether_images_and_masks(origin_dataset_path, output_dataset_path)

    print("Running image_to_tensor...")
    image_to_tensor(output_dataset_path)

    print("Running balance_data_distribution...")
    balance_data_distribution(output_dataset_path)

    print("Running get_class_distribution...")
    distribution = get_class_distribution(output_dataset_path)
    if distribution:
        print("Class distribution:")
        for class_name, count in distribution.items():
            print(f"  {class_name}: {count}")
    else:
        print("No class distribution found.")


def run_distribution_workflow(dataset_path: str, output_plot_path: str | None = None) -> None:
    distribution = get_class_distribution(dataset_path)
    if not distribution:
        print("No data found for distribution analysis.")
        return

    print("Class distribution:")
    for class_name, count in distribution.items():
        print(f"  {class_name}: {count}")

    if output_plot_path is not None:
        plot_distribution(distribution, "Dataset Class Distribution", output_plot_path)


def _prompt_non_empty(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Input cannot be empty. Please try again.")


def _run_utils_menu() -> None:
    print("\nSelect an option:")
    print("1) Preprocess data")
    print("2) Show data distribution")
    print("0) Exit")

    choice = input("Enter your choice (0/1/2): ").strip()

    if choice == "1":
        origin_dataset_path = _prompt_non_empty("Paste origin dataset folder path: ")
        output_dataset_path = _prompt_non_empty("Paste output dataset folder path: ")
        run_preprocess_pipeline(origin_dataset_path, output_dataset_path)
    elif choice == "2":
        dataset_path = _prompt_non_empty("Paste dataset folder path for distribution: ")
        output_plot_path = input("Optional output plot path (press Enter to skip): ").strip()
        run_distribution_workflow(dataset_path, output_plot_path or None)
    elif choice == "0":
        print("Exit.")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    _run_utils_menu()
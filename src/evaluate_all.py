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
from model import HandGestureMultiTask

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


def _clamp_xyxy(boxes, h, w):
    if boxes.numel() == 0:
        return boxes
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0].clamp(0.0, float(w - 1))
    boxes[:, 1] = boxes[:, 1].clamp(0.0, float(h - 1))
    boxes[:, 2] = boxes[:, 2].clamp(0.0, float(w - 1))
    boxes[:, 3] = boxes[:, 3].clamp(0.0, float(h - 1))
    return boxes


def _decode_detections(det_out, input_h, input_w, device):
    boxes_raw = det_out["boxes"]
    scores_raw = det_out["scores"]
    feats = det_out["feats"]

    centers, strides = _build_anchor_meta(feats, input_h, input_w, device)

    boxes_raw = boxes_raw.permute(0, 2, 1)[0]
    pred_xy = centers + torch.tanh(boxes_raw[:, :2]) * strides
    pred_wh = torch.exp(boxes_raw[:, 2:].clamp(-4.0, 4.0)) * strides
    decoded_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
    decoded_xyxy = _xywh_to_xyxy(decoded_xywh)

    scores = scores_raw.sigmoid().permute(0, 2, 1)[0]
    max_scores, class_preds = scores.max(dim=-1)
    return decoded_xyxy, max_scores, class_preds


def _postprocess_detections(
    boxes,
    scores,
    classes,
    conf_threshold,
    input_h,
    input_w,
    iou_threshold=0.45,
    min_box_size=4.0,
    max_det=100,
):
    if boxes.numel() == 0:
        return boxes, scores, classes

    boxes = _clamp_xyxy(boxes, input_h, input_w)

    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)
    keep = (scores >= conf_threshold) & (widths >= min_box_size) & (heights >= min_box_size)

    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]

    if boxes.numel() == 0:
        return boxes, scores, classes

    keep_idx = torchvision.ops.batched_nms(boxes, scores, classes, iou_threshold)
    if max_det > 0:
        keep_idx = keep_idx[:max_det]

    return boxes[keep_idx], scores[keep_idx], classes[keep_idx]


def _remove_small_components(mask_np, min_area=64):
    h, w = mask_np.shape
    visited = np.zeros((h, w), dtype=bool)
    output = np.zeros((h, w), dtype=bool)

    for y in range(h):
        for x in range(w):
            if not mask_np[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            component = []

            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))

                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if not visited[ny, nx] and mask_np[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            if len(component) >= min_area:
                ys, xs = zip(*component)
                output[np.array(ys), np.array(xs)] = True

    return output


def _postprocess_segmentation(seg_logits, threshold=0.5, min_area=64):
    seg_prob = seg_logits.sigmoid()[0, 0]
    mask = seg_prob > threshold

    if min_area > 1 and mask.any():
        mask_np = mask.detach().cpu().numpy().astype(bool)
        mask_np = _remove_small_components(mask_np, min_area=min_area)
        mask = torch.from_numpy(mask_np).to(device=seg_logits.device)

    return mask


def _extract_state_dict_and_cfg(ckpt_obj):
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"], ckpt_obj.get("cfg", {})
    if isinstance(ckpt_obj, dict):
        return ckpt_obj, {}
    raise TypeError(f"Unsupported checkpoint format: {type(ckpt_obj)}")


def _infer_model_hparams_from_state_dict(state_dict):
    num_classes = None
    reg_max = None

    cls_key = "detect.cv3.0.2.weight"
    if cls_key in state_dict:
        num_classes = int(state_dict[cls_key].shape[0])

    box_key = "detect.cv2.0.2.weight"
    if box_key in state_dict:
        reg_ch = int(state_dict[box_key].shape[0])
        if reg_ch % 4 == 0:
            reg_max = reg_ch // 4

    has_one2one = any(k.startswith("detect.one2one_cv2") for k in state_dict.keys())

    inferred_scales = []
    stem_key = "backbone.b0.conv.weight"
    if stem_key in state_dict:
        c1 = int(state_dict[stem_key].shape[0])
        if c1 == 16:
            inferred_scales = ["n"]
        elif c1 == 32:
            inferred_scales = ["s"]
        elif c1 == 96:
            inferred_scales = ["x"]
        elif c1 == 64:
            # m and l share channel widths; distinguish by depth keys if available.
            has_deeper_stage = any(k.startswith("backbone.b2.m.1.") for k in state_dict.keys())
            inferred_scales = ["l", "m"] if has_deeper_stage else ["m", "l"]

    return {
        "num_classes": num_classes,
        "reg_max": reg_max,
        "end2end": has_one2one,
        "scale_candidates": inferred_scales,
    }


def _unique_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if item is None:
            continue
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _build_and_load_model(state_dict, device, num_classes, end2end, reg_max, preferred_scale=None, inferred_scales=None):
    scale_candidates = _unique_keep_order([
        preferred_scale,
        *(inferred_scales or []),
        "n", "s", "m", "l", "x",
    ])

    last_err = None
    for scale in scale_candidates:
        try:
            model = HandGestureMultiTask(
                num_classes=num_classes,
                end2end=end2end,
                scale=scale,
                reg_max=reg_max,
            ).to(device)
            model.load_state_dict(state_dict, strict=True)
            return model, scale
        except RuntimeError as err:
            last_err = err

    raise RuntimeError(
        "Unable to match checkpoint weights to any known model scale. "
        f"Tried scales={scale_candidates}, num_classes={num_classes}, end2end={end2end}, reg_max={reg_max}.\n"
        f"Last load error:\n{last_err}"
    )

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
    det_iou_threshold=0.45,
    seg_threshold=0.5,
    seg_min_area=64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading MultiTask weights from {weights_path}...")

    # Load checkpoint first so we can mirror the training config.
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict, train_cfg = _extract_state_dict_and_cfg(ckpt)

    inferred = _infer_model_hparams_from_state_dict(state_dict)

    num_classes = int(train_cfg.get("num_classes", inferred["num_classes"] if inferred["num_classes"] is not None else 10))
    reg_max = int(train_cfg.get("reg_max", inferred["reg_max"] if inferred["reg_max"] is not None else 1))
    # Prefer cfg when provided, otherwise infer from state_dict keys.
    end2end = bool(train_cfg.get("end2end", inferred["end2end"]))
    scale_cfg = train_cfg.get("scale", None)
    input_size = train_cfg.get("resize_shape", [256, 256])  # (H, W)

    # 1. Initialize MultiTask Model (must match checkpoint architecture)
    model, loaded_scale = _build_and_load_model(
        state_dict=state_dict,
        device=device,
        num_classes=num_classes,
        end2end=end2end,
        reg_max=reg_max,
        preferred_scale=scale_cfg,
        inferred_scales=inferred["scale_candidates"],
    )
    model.eval()
    print(
        f"Loaded model settings -> scale={loaded_scale}, num_classes={num_classes}, "
        f"end2end={end2end}, reg_max={reg_max}, input_size={input_size}"
    )

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
        orig_w, orig_h = original_img.size
        # input_size is (H, W) -> PIL expects (W, H)
        in_h, in_w = int(input_size[0]), int(input_size[1])
        img_resized = original_img.resize((in_w, in_h), Image.Resampling.BILINEAR)
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
            mask_resized = _postprocess_segmentation(seg_out, threshold=seg_threshold, min_area=seg_min_area)

            # --- C. Process Detection ---
            input_h, input_w = img_tensor.shape[-2:]
            decoded_xyxy, max_scores, class_preds = _decode_detections(det_out, input_h, input_w, device)
            final_boxes, final_scores, final_classes = _postprocess_detections(
                decoded_xyxy,
                max_scores,
                class_preds,
                conf_threshold=conf_threshold,
                input_h=input_h,
                input_w=input_w,
                iou_threshold=det_iou_threshold,
            )

        # Project predictions back to original image coordinates.
        scale_x = float(orig_w) / float(in_w)
        scale_y = float(orig_h) / float(in_h)

        final_boxes_orig = final_boxes.clone()
        if final_boxes_orig.numel() > 0:
            final_boxes_orig[:, [0, 2]] *= scale_x
            final_boxes_orig[:, [1, 3]] *= scale_y
            final_boxes_orig = _clamp_xyxy(final_boxes_orig, orig_h, orig_w)

        mask_np_resized = mask_resized.detach().cpu().numpy().astype(np.uint8) * 255
        mask_orig = Image.fromarray(mask_np_resized, mode="L").resize((orig_w, orig_h), Image.Resampling.NEAREST)
        mask_np_orig = np.array(mask_orig) > 0
            
        # ---------------------------------------------------------------------------
        # Composite Drawing
        # ---------------------------------------------------------------------------
        
        # 1. Add Segmentation Overlay (Cyan, semi-transparent)
        img_rgba = original_img.convert("RGBA")
        mh, mw = mask_np_orig.shape
        mask_rgba = np.zeros((mh, mw, 4), dtype=np.uint8)
        mask_rgba[mask_np_orig] = [0, 255, 255, 120] # R, G, B, Alpha
        mask_pil = Image.fromarray(mask_rgba, mode="RGBA")
        
        img_composited = Image.alpha_composite(img_rgba, mask_pil)
        
        # 2. Draw Bounding Boxes
        draw = ImageDraw.Draw(img_composited)
        detections_text_list = []
        
        for box, score, cls_id in zip(final_boxes_orig, final_scores, final_classes):
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
        print(f"Processed {filename}: Found {len(final_boxes_orig)} objects.")

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
    parser.add_argument("--folder", type=str, help="Path to folder containing primary inference images", default="dataset/dataset_v1/val/images")
    parser.add_argument(
        "--test_folder",
        type=str,
        default="dataset/dataset_v1/val/images",
        help="Path to test image folder (JPEG set, e.g. 1600x1200)",
    )
    parser.add_argument("--weights", type=str, default="outputs/stage_2/train_2/best.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--det_iou", type=float, default=0.45, help="IoU threshold for class-aware NMS")
    parser.add_argument("--seg_thresh", type=float, default=0.5, help="Segmentation probability threshold")
    parser.add_argument("--seg_min_area", type=int, default=64, help="Minimum connected-component area kept in seg mask")
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
        det_iou_threshold=args.det_iou,
        seg_threshold=args.seg_thresh,
        seg_min_area=args.seg_min_area,
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
        det_iou_threshold=args.det_iou,
        seg_threshold=args.seg_thresh,
        seg_min_area=args.seg_min_area,
    )
import torch
import PIL.Image as Image
from model import HandGestureModel_v3, ModelConfig
from utils import decode_predictions


image_tensor = torch.load("dataset/dataset_v1/val/image_tensors/77.pt").to("cuda")
print(image_tensor.shape)


model = HandGestureModel_v3(ModelConfig()).to("cuda")
state = torch.load("/workspace/project_23047126_Liu/outputs/stage_1/train_3/stage_1_best.pt", map_location="cpu")
model.load_state_dict(state["state_dict"])
model.eval()
with torch.no_grad():
    cls_logits, bbox_preds, bbox_cls, seg_map = model(image_tensor.unsqueeze(0))
    print(f"cls_logits shape: {cls_logits.shape}, bbox_preds shape: {bbox_preds.shape}, bbox_cls shape: {bbox_cls.shape}, seg_map shape: {seg_map.shape}")
    pred_scores = bbox_cls.sigmoid().permute(0, 2, 1)
    print(f"pred_scores shape: {pred_scores.shape}")
    max_scores, cls_idx = pred_scores.max(dim=-1)
    print(f"max_scores shape: {max_scores.shape}, cls_idx shape: {cls_idx.shape}")

    max_scores = max_scores.squeeze(0).cpu().numpy()
    # plot score distribution
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterMathtext, LogLocator
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(max_scores, bins=40, range=(0.0, 1.0), alpha=0.8, edgecolor="black")
    axes[0].set_xlabel("Max class score")
    axes[0].set_ylabel("Anchor count")
    axes[0].set_title("Distribution of anchor max scores")
    axes[0].set_yscale("log", base=10)
    axes[0].yaxis.set_major_locator(LogLocator(base=10.0))
    axes[0].yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    sorted_scores = sorted(max_scores)
    cdf_y = [(i + 1) / len(sorted_scores) for i in range(len(sorted_scores))]
    axes[1].plot(sorted_scores, cdf_y)
    axes[1].set_xlabel("Max class score")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("CDF of anchor max scores")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("max_scores_distribution.png", dpi=150)

# decoded_preds = decode_predictions(bbox_preds, bbox_cls, image_tensor.shape[1], image_tensor.shape[2], conf_thresh=0.3)
# print(len(decoded_preds))
# print(decoded_preds[0].shape)




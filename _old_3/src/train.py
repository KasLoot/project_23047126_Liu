import math
import os
import argparse
import torch
import torch.nn as nn
from model import HandGestureModel_v3, ModelConfig
from torch.utils.data import DataLoader, Subset
from dataloader import HandGestureDataset, SegAugment
from utils import compute_detection_loss, decode_predictions, evaluate_batch, plot_confusion_matrix, plot_validation_samples, WarmupCosineScheduler
import random
import torch.nn.functional as F
import torchinfo
import matplotlib.pyplot as plt
import numpy as np




def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def empty_cache():
    torch.cuda.empty_cache()

class Stage_1_Config:
    epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    lr = 5e-4
    weight_decay = 1e-4

    output_dir = 'outputs/stage_1'


    model_desc = {
        'Model': 'HandGestureModel_v3',
        'epoch': None,
        'batch_size': batch_size,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'optimizer': 'AdamW',
        'scheduler': 'WarmupCosineScheduler',
        'augmentation': 'SegAugment',
        'state_dict': None
    }


def _plot_training_curves(train_losses, val_losses, iou50_scores, iou75_scores, iou95_scores, cls_acc, output_dir: str):
    epochs = list(range(1, len(train_losses) + 1))

    # Loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training / Validation Loss')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    # Metric curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, iou50_scores, label='IoU@0.50', linewidth=2)
    plt.plot(epochs, iou75_scores, label='IoU@0.75', linewidth=2)
    plt.plot(epochs, iou95_scores, label='IoU@0.95', linewidth=2)
    plt.plot(epochs, cls_acc, label='Cls Acc', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_curves.png'))
    plt.close()


def train_stage_1(model: HandGestureModel_v3, 
                  train_loader: DataLoader, 
                  val_loader: DataLoader, 
                  config: Stage_1_Config):

    os.makedirs(config.output_dir, exist_ok=True)

    for param in model.classifier_head.parameters(): param.requires_grad = False
    for param in model.segmentation_head.parameters(): param.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 1 Training (Backbone + Detection Head)\n Total Trainable Params: {trainable_params:,}")


    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    epochs = config.epochs
    total_steps = epochs * len(train_loader)
    warmup_steps = 3 * len(train_loader) # 3 epochs of warmup
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)


    best_val_loss = float('inf')
    train_losses, val_losses, iou50_scores, iou75_scores, iou95_scores, cls_acc = [], [], [], [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0

        for images, masks, class_ids, bboxes in train_loader:
            images = images.to(config.device)
            masks = masks.to(config.device)
            class_ids = class_ids.to(config.device)
            bboxes = bboxes.to(config.device)

            optimizer.zero_grad()
            cls_logits, bbox_preds, bbox_cls, seg_map = model(images)
            loss, c_loss, b_loss = compute_detection_loss(bbox_preds, bbox_cls, bboxes, class_ids, images.shape[2], images.shape[3])
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        iou50_correct, iou75_correct, iou95_correct = 0, 0, 0
        cls_correct, cls_total = 0, 0
        y_true_iou50, y_pred_iou50 = [], []
        vis_samples = []
        
        with torch.no_grad():
            for images, masks, class_ids, bboxes in val_loader:
                images = images.to(config.device)
                masks = masks.to(config.device)
                class_ids = class_ids.to(config.device)
                bboxes = bboxes.to(config.device)

                cls_logits, bbox_preds, bbox_cls, seg_map = model(images)
                loss, c_loss, b_loss = compute_detection_loss(bbox_preds, bbox_cls, bboxes, class_ids, images.shape[2], images.shape[3])
                epoch_val_loss += loss.item()

                decoded_preds = decode_predictions(bbox_preds, bbox_cls, images.shape[2], images.shape[3], conf_thresh=0.3)

                y_true_50, y_pred_50 = evaluate_batch(decoded_preds, bboxes, class_ids, images.shape[2], images.shape[3], iou_thresh=0.5)
                y_true_75, y_pred_75 = evaluate_batch(decoded_preds, bboxes, class_ids, images.shape[2], images.shape[3], iou_thresh=0.75)
                y_true_95, y_pred_95 = evaluate_batch(decoded_preds, bboxes, class_ids, images.shape[2], images.shape[3], iou_thresh=0.95)

                iou50_correct += sum(int(t == p) for t, p in zip(y_true_50, y_pred_50))
                iou75_correct += sum(int(t == p) for t, p in zip(y_true_75, y_pred_75))
                iou95_correct += sum(int(t == p) for t, p in zip(y_true_95, y_pred_95))

                cls_correct += sum(int(t == p) for t, p in zip(y_true_50, y_pred_50))
                cls_total += len(y_true_50)

                y_true_iou50.extend(y_true_50)
                y_pred_iou50.extend(y_pred_50)

                if len(vis_samples) < 10:
                    for bi in range(images.shape[0]):
                        if len(vis_samples) >= 10:
                            break
                        vis_samples.append((
                            images[bi].detach().cpu(),
                            bboxes[bi].detach().cpu(),
                            class_ids[bi].detach().cpu(),
                            decoded_preds[bi].detach().cpu() if len(decoded_preds[bi]) > 0 else decoded_preds[bi]
                        ))

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        total_samples = max(cls_total, 1)
        epoch_iou50 = iou50_correct / total_samples
        epoch_iou75 = iou75_correct / total_samples
        epoch_iou95 = iou95_correct / total_samples
        epoch_cls_acc = cls_correct / total_samples

        iou50_scores.append(epoch_iou50)
        iou75_scores.append(epoch_iou75)
        iou95_scores.append(epoch_iou95)
        cls_acc.append(epoch_cls_acc)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            config.model_desc['epoch'] = epoch
            config.model_desc['state_dict'] = model.state_dict()
            torch.save(config.model_desc, f'{config.output_dir}/best_model_stage_1.pt')

            if len(y_true_iou50) > 0:
                plot_confusion_matrix(y_true_iou50, y_pred_iou50, num_classes=10, output_dir=config.output_dir)
            if len(vis_samples) > 0:
                plot_validation_samples(vis_samples, config.output_dir)

        print(
            f"Epoch [{epoch}/{epochs}] - "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"IoU@0.50: {epoch_iou50:.4f} | "
            f"IoU@0.75: {epoch_iou75:.4f} | "
            f"IoU@0.95: {epoch_iou95:.4f} | "
            f"Cls Acc: {epoch_cls_acc:.4f}"
        )

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'iou50_scores': iou50_scores,
        'iou75_scores': iou75_scores,
        'iou95_scores': iou95_scores,
        'cls_acc': cls_acc,
    }
    torch.save(history, os.path.join(config.output_dir, 'stage1_history.pt'))

    with open(os.path.join(config.output_dir, 'stage1_metrics.txt'), 'w') as f:
        f.write('epoch,train_loss,val_loss,iou50,iou75,iou95,cls_acc\n')
        for i in range(len(train_losses)):
            f.write(
                f"{i+1},"
                f"{train_losses[i]:.6f},"
                f"{val_losses[i]:.6f},"
                f"{iou50_scores[i]:.6f},"
                f"{iou75_scores[i]:.6f},"
                f"{iou95_scores[i]:.6f},"
                f"{cls_acc[i]:.6f}\n"
            )

    _plot_training_curves(train_losses, val_losses, iou50_scores, iou75_scores, iou95_scores, cls_acc, config.output_dir)
        
    





def train():
    set_seed(42)
    config = Stage_1_Config()
    train_dataset = HandGestureDataset(root_dir='dataset/dataset_v1/train', transform=SegAugment())
    val_dataset = HandGestureDataset(root_dir='dataset/dataset_v1/val')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    model_config = ModelConfig()
    model = HandGestureModel_v3(model_config).to(config.device)
    print(model)
    torchinfo.summary(model, input_size=(1, 3, 480, 640), device=config.device)
    train_stage_1(model, train_loader, val_loader, config)


if __name__ == "__main__":
    train()
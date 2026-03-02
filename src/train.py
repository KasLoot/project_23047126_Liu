import math
import os
import argparse
import torch
import torch.nn as nn
from model import HandGestureModel_v3, ModelConfig
from torch.utils.data import DataLoader, Subset
from dataloader import HandGestureDataset, SegAugment
import random
import torch.nn.functional as F
import torchinfo
import matplotlib.pyplot as plt
import numpy as np

from utils import WarmupCosineScheduler




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



def train_stage_1(config: Stage_1_Config):
    os.makedirs(config.output_dir, exist_ok=True)

    # --- Dataset & Dataloader ---
    train_dataset = HandGestureDataset(root_dir='/workspace/project_23047126_Liu/dataset/dataset_v1/train', transform=SegAugment())
    val_dataset = HandGestureDataset(root_dir='/workspace/project_23047126_Liu/dataset/dataset_v1/val', transform=None)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    

    # --- Model ---
    model_config = ModelConfig()  # Assuming 10 gesture classes
    model = HandGestureModel_v3(model_config).to(config.device)
    for param in model.classifier_head.parameters(): param.requires_grad = False
    for param in model.segmentation_head.parameters(): param.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 1 Training (Backbone + Detection Head)\n Total Trainable Params: {trainable_params:,}")

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=int(0.1 * total_steps), total_steps=total_steps)

    



    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for image_tensor, mask_tensor, class_id, bbox in train_loader:
            image_tensor = image_tensor.to(config.device)
            # mask_tensor = mask_tensor.to(config.device)
            class_id = class_id.to(config.device)
            bbox = bbox.to(config.device)
            print(image_tensor.shape, class_id.shape, bbox.shape)

            optimizer.zero_grad()
            cls_logits, bbox_preds, bbox_cls, seg_map = model(image_tensor)

            # --- Loss Calculation ---
            # classification_loss = F.cross_entropy(cls_logits, class_id)
            # bbox_loss = F.mse_loss(bbox_preds, bbox)
            # loss = classification_loss + bbox_loss

            # loss.backward()
            # optimizer.step()
            # scheduler.step()

            # epoch_loss += loss.item()

            break

        print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader)}")
        break



if __name__ == "__main__":
    set_seed(42)
    empty_cache()
    config = Stage_1_Config()
    train_stage_1(config)
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
    full_dataset = HandGestureDataset(root_dir='data/processed', transform=SegAugment())
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    split_idx = int(0.8 * dataset_size)
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # --- Model ---
    model_config = ModelConfig(num_classes=10)  # Assuming 10 gesture classes
    model = HandGestureModel_v3(model_config).to(config.device)

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=int(0.1 * total_steps), total_steps=total_steps)

    



    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for image_tensor, mask_tensor, class_id, bbox in train_loader:
            image_tensor = image_tensor.to(config.device)
            mask_tensor = mask_tensor.to(config.device)
            class_id = class_id.to(config.device)
            bbox = bbox.to(config.device)

            optimizer.zero_grad()
            class_logits, bbox_preds = model(image_tensor)

            # --- Loss Calculation ---
            classification_loss = F.cross_entropy(class_logits, class_id)
            bbox_loss = F.mse_loss(bbox_preds, bbox)
            loss = classification_loss + bbox_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
        
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader)}")
        break
import argparse
import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import HandGestureDataset_v2, SegAugment_v2
from model import HandGestureMultiTask

class Train_S1_Config:
    epochs = 20
    batch_size = 32
    learning_rate = 5e-4
    weight_decay = 1e-2
    output_dir = "outputs/stage_1/train_1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_seed = 42
    train_data_dir = "dataset/dataset_v1/train"
    val_data_dir = "dataset/dataset_v1/val"
    resize_shape = (256, 256)  # Resize images and masks to this shape for training

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
def detection_collate_fn(batch):
    images, masks, class_ids, bboxes = zip(*batch)
    images = torch.stack(images, dim=0).float()
    masks = torch.stack(masks, dim=0)

    # Normalize per-sample targets to avoid shape mismatch between
    # augmented train set (bbox shape: [4]) and non-aug val set (bbox shape: [1, 4]).
    class_ids = torch.as_tensor(
        [int(torch.as_tensor(c).view(-1)[0].item()) for c in class_ids],
        dtype=torch.long,
    )
    bboxes = torch.stack(
        [torch.as_tensor(b, dtype=torch.float32).view(-1)[:4] for b in bboxes],
        dim=0,
    )
    return images, masks, class_ids, bboxes



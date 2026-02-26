import torch
import torch.nn as nn
from yolo26 import YOLO26MultiTask
from torch.utils.data import DataLoader, Subset
from dataloader import HandGestureDataset, SegAugment


def get_model(checkpoint_path: str | None = None) -> YOLO26MultiTask:
    model = YOLO26MultiTask(num_classes=10, end2end=True)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))
    return model


class Train_S1_Config:
    num_epochs = 20
    batch_size = 32
    train_split = 0.8
    learning_rate =5e-4
    weight_decay = 1e-4

    image_tensor_dir="/home/yuxin/Object_Detection/project_23047126_Liu/dataset/processed_dataset_v2/rgb_only_given/image_tensors"
    mask_tensor_dir="/home/yuxin/Object_Detection/project_23047126_Liu/dataset/processed_dataset_v2/rgb_only_given/mask_tensors"
    image_info_json="/home/yuxin/Object_Detection/project_23047126_Liu/dataset/processed_dataset_v2/rgb_only_given/image_info.json"

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def train_stage_1():

    train_stage_1_config = Train_S1_Config()
    
    print(f"Using device: {train_stage_1_config.device}")

    # --- 1. Initialize Model ---
    model = get_model()
    model.to(train_stage_1_config.device)

    

    

    


    
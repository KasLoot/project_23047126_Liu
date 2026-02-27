import math
import os
import torch
import torch.nn as nn
from yolo26 import YOLO26MultiTask
from model import HandGestureModel, ModelConfig
from torch.utils.data import DataLoader, Subset
from dataloader import HandGestureDataset, SegAugment
from utils import compute_detection_loss, decode_predictions, evaluate_batch, plot_confusion_matrix, plot_validation_samples
import random
import torch.nn.functional as F
import torchinfo

torch.manual_seed(42)
random.seed(42)




def get_model(checkpoint_path: str | None = None, model_config: ModelConfig | None = None) -> HandGestureModel:
    if model_config is None:
        model_config = ModelConfig()
    model = HandGestureModel(model_config=model_config)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))
    return model


class Train_S1_Config:
    num_epochs = 20
    batch_size = 16
    learning_rate = 1e-3
    weight_decay = 1e-4

    train_dataset_dir = "/home/yuxin/Object_Detection/project_23047126_Liu/dataset/dataset_v1/train"
    train_image_tensor_dir = os.path.join(train_dataset_dir, "image_tensors")
    train_mask_tensor_dir = os.path.join(train_dataset_dir, "mask_tensors")
    train_image_info_json = os.path.join(train_dataset_dir, "image_info.json")

    val_dataset_dir = "/home/yuxin/Object_Detection/project_23047126_Liu/dataset/dataset_v1/val"
    val_image_tensor_dir = os.path.join(val_dataset_dir, "image_tensors")
    val_mask_tensor_dir = os.path.join(val_dataset_dir, "mask_tensors")
    val_image_info_json = os.path.join(val_dataset_dir, "image_info.json")


    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def detection_collate_fn(batch):
    images, masks, class_ids, bboxes = zip(*batch)
    images = torch.stack(images, dim=0).float()
    masks = torch.stack(masks, dim=0)
    class_ids = torch.as_tensor(class_ids, dtype=torch.long)
    bboxes = torch.stack([b.float() for b in bboxes], dim=0)
    return images, masks, class_ids, bboxes


def train_stage_1():
    train_stage_1_config = Train_S1_Config()
    print(f"Using device: {train_stage_1_config.device}")

    output_dir = "outputs/stage_1_training"
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Initialize Model ---
    model = get_model(model_config=ModelConfig())
    model.to(train_stage_1_config.device)

    torchinfo.summary(model, input_size=(1, 3, 640, 480), device=train_stage_1_config.device)

    # --- 2. Freeze all heads EXCEPT the Detection Head ---
    for param in model.backbone.parameters(): param.requires_grad = False
    for param in model.classifier_head.parameters(): param.requires_grad = False
    for param in model.segmentation_head.parameters(): param.requires_grad = False
    
    # Verify what is trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 1 Training (Detection Head only)\n Total Trainable Params: {trainable_params:,}")

    # --- 3. Dataloaders ---
    train_dataset = HandGestureDataset(
        image_tensor_dir=train_stage_1_config.train_image_tensor_dir,
        mask_tensor_dir=train_stage_1_config.train_mask_tensor_dir,
        image_info_json=train_stage_1_config.train_image_info_json,
        transform=SegAugment()
    )
    val_dataset = HandGestureDataset(
        image_tensor_dir=train_stage_1_config.val_image_tensor_dir,
        mask_tensor_dir=train_stage_1_config.val_mask_tensor_dir,
        image_info_json=train_stage_1_config.val_image_info_json
    )
    
    # Using your existing custom collate function from train_1.py if applicable
    train_loader = DataLoader(train_dataset, batch_size=train_stage_1_config.batch_size, shuffle=True, collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_stage_1_config.batch_size, shuffle=False, collate_fn=detection_collate_fn)

    # --- 4. Optimizer ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=train_stage_1_config.learning_rate, 
        weight_decay=train_stage_1_config.weight_decay
    )

    # --- 5. Training Loop ---
    best_val_loss = float('inf')

    for epoch in range(train_stage_1_config.num_epochs):
        model.train()
        train_loss = 0.0
        
        # --- TRAINING LOOP ---
        for images, masks, class_ids, bboxes in train_loader:
            images, class_ids, bboxes = images.to(train_stage_1_config.device), class_ids.to(train_stage_1_config.device), bboxes.to(train_stage_1_config.device)

            optimizer.zero_grad()
            cls_logits, bbox_preds, bbox_cls, seg_map = model(images)
            
            input_h, input_w = images.shape[-2], images.shape[-1]
            loss, c_loss, b_loss = compute_detection_loss(bbox_preds, bbox_cls, bboxes, class_ids, input_h, input_w)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            train_loss += loss.item()

        # --- VALIDATION LOOP ---
        model.eval()
        val_loss = 0.0
        all_y_true, all_y_pred = [], []
        vis_samples = []
        
        with torch.no_grad():
            for images, masks, class_ids, bboxes in val_loader:
                images, class_ids, bboxes = images.to(train_stage_1_config.device), class_ids.to(train_stage_1_config.device), bboxes.to(train_stage_1_config.device)
                
                input_h, input_w = images.shape[-2], images.shape[-1]
                cls_logits, bbox_preds, bbox_cls, seg_map = model(images)
                
                # 1. Compute Validation Loss
                loss, c_loss, b_loss = compute_detection_loss(bbox_preds, bbox_cls, bboxes, class_ids, input_h, input_w)
                val_loss += loss.item()
                
                # 2. Decode Predictions for Metrics
                decoded_preds = decode_predictions(bbox_preds, bbox_cls, input_h, input_w, conf_thresh=0.05)
                
                # 3. Match Preds to GT for Confusion Matrix
                y_true, y_pred = evaluate_batch(decoded_preds, bboxes, class_ids, input_h, input_w)
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
                
                # 4. Save samples for visualization (collect up to 10)
                if len(vis_samples) < 10:
                    for bi in range(images.shape[0]):
                        if len(vis_samples) < 10:
                            vis_samples.append((images[bi].cpu(), bboxes[bi].cpu(), class_ids[bi].cpu(), decoded_preds[bi].detach().cpu()))

        # --- CALCULATE EPOCH METRICS ---
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate IoU@50 Accuracy (Ignoring BG classes for pure accuracy)
        correct = sum(1 for t, p in zip(all_y_true, all_y_pred) if t == p and p != 10)
        total_gt = len(all_y_true)
        iou50_acc = correct / total_gt if total_gt > 0 else 0.0
        
        print(f"Epoch {epoch+1} Results: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | IoU@50 Acc: {iou50_acc:.4f}")
        
        # --- SAVE PLOTS & METRICS ---
        with open(f"{output_dir}/val_metrics.txt", "a") as f:
            f.write(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, IoU@50: {iou50_acc:.4f}\n")
            
        plot_confusion_matrix(all_y_true, all_y_pred, num_classes=10, output_dir=output_dir)
        
        # Shuffle samples to get different visual results across epochs, then plot
        random.shuffle(vis_samples)
        plot_validation_samples(vis_samples, output_dir=output_dir)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{output_dir}/best_stage1_model.pt")
            print(f"--> Saved new best model to {output_dir}/best_stage1_model.pt")



if __name__ == "__main__":
    train_stage_1()
    

    

    


    
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your custom modules
from dataloader import HandGestureDataset_v2, SegAugment_v2, detection_collate_fn
from model import HandGestureMultiTask
from loss import YOLODetectionLoss

def train():
    # ==========================================
    # 1. Configuration & Hyperparameters
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dir = "dataset/dataset_v1/train"
    val_dir = "dataset/dataset_v1/val" 
    num_classes = 10
    batch_size = 16
    epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-4

    # Save directory for checkpoints
    os.makedirs("weights", exist_ok=True)

    # ==========================================
    # 2. Dataset & DataLoader Setup
    # ==========================================
    input_size = (480, 640)
    
    # Training dataset with augmentations
    train_transform = SegAugment_v2(out_size=input_size)
    train_dataset = HandGestureDataset_v2(
        root_dir=train_dir, 
        transform=train_transform,
        resize_shape=list(input_size)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, drop_last=True,
        collate_fn=detection_collate_fn,
    )

    # Validation dataset WITHOUT augmentations (transform=None)
    val_dataset = HandGestureDataset_v2(
        root_dir=val_dir, 
        transform=None,  # Only native resizing applied
        resize_shape=list(input_size)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, drop_last=False,
        collate_fn=detection_collate_fn,
    )

    # ==========================================
    # 3. Model & Loss & Optimizer Setup
    # ==========================================
    model = HandGestureMultiTask(num_classes=num_classes, reg_max=1).to(device)
    
    # Prior bias trick for stable initialization
    import math
    prior_prob = 0.01
    bias_value = -math.log((1 - prior_prob) / prior_prob)
    for branch in model.detect.cls_branches:
        nn_conv = branch[-1] # The final Conv2d layer
        torch.nn.init.constant_(nn_conv.bias, bias_value)

    criterion = YOLODetectionLoss(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    # ==========================================
    # 4. Main Training & Validation Loop
    # ==========================================
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, masks, class_ids, bboxes in pbar:
            images = images.to(device)
            bboxes = bboxes.to(device)       
            class_ids = class_ids.to(device) 

            preds = model(images, tasks=("det",))
            loss, loss_metrics = criterion(preds["det"], bboxes, class_ids)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "Cls": f"{loss_metrics['loss_cls']:.4f}", 
                "Box": f"{loss_metrics['loss_box']:.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.6f}"
            })

        avg_train_loss = epoch_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_cls = 0.0
        val_box = 0.0

        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ")
            for images, masks, class_ids, bboxes in vbar:
                images = images.to(device)
                bboxes = bboxes.to(device)
                class_ids = class_ids.to(device)

                preds = model(images, tasks=("det",))
                loss, loss_metrics = criterion(preds["det"], bboxes, class_ids)
                
                val_loss += loss.item()
                val_cls += loss_metrics["loss_cls"]
                val_box += loss_metrics["loss_box"]
                
                vbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        
        print(f"End of Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} (Cls: {val_cls/len(val_loader):.4f}, Box: {val_box/len(val_loader):.4f})")

        # --- Save Checkpoints ---
        # Save last model
        torch.save(model.state_dict(), "weights/last_model.pth")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "weights/best_model.pth")
            print(f"--> Saved new BEST model! (Val Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    train()
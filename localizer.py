import torch
import torch.nn as nn
from tqdm import tqdm
from models.localization import VGG11Localizer
from models.dataset import get_dataloaders  # Ensure this points to your dataset.py
from models.iou_loss import IoULoss

def train_localizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 1. Load Data
    # Point root to your oxford-iiit-pet folder
    loaders = get_dataloaders(root="data/oxford-iiit-pet", batch_size=32)
    train_loader = loaders["train"]
    val_loader   = loaders["val"]

    # 2. Build Model
    # Freeze encoder, train only the regression head
    model = VGG11Localizer(
        encoder_ckpt="checkpoints/classifier.pth", 
        freeze_encoder=True
    ).to(device)

    # 3. Loss & Optimizer
    criterion = IoULoss(reduction="mean")
    
    # Only pass parameters that require gradients (the head)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, 
        weight_decay=1e-4
    )
    
    # Train for 15 epochs (head-only training converges very fast)
    num_epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_iou = 0.0

    # 4. Training Loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            images = batch["image"].to(device)
            # Dataset returns [cx, cy, w, h] normalized in [0, 1]
            targets = batch["bbox"].to(device)
            
            optimizer.zero_grad()
            preds = model(images)
            
            # IoULoss expects both pred and target as [cx, cy, w, h] in [0,1]
            loss = criterion(preds, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # 5. Validation Loop
        model.eval()
        val_iou_sum = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = batch["image"].to(device)
                targets = batch["bbox"].to(device)
                
                preds = model(images)
                
                # Compute raw IoU scores using your IoULoss helper method
                # This returns an array of shape (N,)
                iou_scores = criterion.compute_iou_scores(preds, targets)
                val_iou_sum += iou_scores.mean().item()
                num_val_batches += 1
                
        avg_val_iou = val_iou_sum / num_val_batches
        print(f"Epoch {epoch+1} Summary: Train Loss={avg_train_loss:.4f} | Val IoU={avg_val_iou:.4f}")

        # 6. Save Checkpoint
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_iou": best_val_iou,
            }, "checkpoints/localizer.pth")
            print(f"  --> Saved new best checkpoint with IoU: {best_val_iou:.4f}")

    print(f"\nTraining complete. Best Validation IoU: {best_val_iou:.4f}")

if __name__ == "__main__":
    train_localizer()

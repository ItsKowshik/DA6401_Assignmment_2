"""
train_segmentation.py — Task 3: U-Net Semantic Segmentation
DA6401 Assignment 2

Runs three strategies for Section 2.3:
  python train_segmentation.py --freeze_mode frozen  --run_name seg_frozen
  python train_segmentation.py --freeze_mode partial --run_name seg_partial
  python train_segmentation.py --freeze_mode full    --run_name seg_finetune
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from data.pets_dataset import get_dataloaders
from models.segmentation import VGG11UNet
from losses.segmentation_loss import (
    SegmentationLoss, pixel_accuracy, dice_score
)

import wandb


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--root",           default="data/oxford-iiit-pet")
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--val_fraction",   type=float, default=0.1)

    # Model
    p.add_argument("--encoder_ckpt",   default="checkpoints/classifier.pth")
    p.add_argument("--freeze_mode",    default="full",
                   choices=["frozen", "partial", "full"])
    p.add_argument("--num_classes",    type=int,   default=3)

    # Training
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--mixed_prec",     action="store_true", default=True)
    p.add_argument("--scheduler",      default="cosine",
                   choices=["cosine", "step", "plateau"])

    # Loss
    p.add_argument("--ce_weight",      type=float, default=1.0)
    p.add_argument("--dice_weight",    type=float, default=1.0)

    # W&B
    p.add_argument("--project",        default="DL_A2")
    p.add_argument("--group",          default="Task3_Segmentation")
    p.add_argument("--run_name",       default=None)

    # Checkpoint
    p.add_argument("--checkpoint_dir", default="checkpoints")

    return p.parse_args()


# ── Grad norm helper ──────────────────────────────────────────────────────────

def compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


# ── Train epoch ───────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = total_ce = total_dice_l = 0.0
    total_pa   = total_ds = 0.0
    grad_norms = []
    n = 0

    for batch_idx, batch in enumerate(loader):
        imgs    = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device,  non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
            logits = model(imgs)
            loss, ce, dice_l = criterion(logits, targets, return_components=True)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = compute_grad_norm(model)
        grad_norms.append(grad_norm)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        pa = pixel_accuracy(logits.detach(), targets)
        ds = dice_score(logits.detach(), targets)

        total_loss   += loss.item()
        total_ce     += ce.item()
        total_dice_l += dice_l.item()
        total_pa     += pa
        total_ds     += ds
        n            += 1

        wandb.log({
            "train/batch_loss"     : loss.item(),
            "train/batch_ce"       : ce.item(),
            "train/batch_dice_loss": dice_l.item(),
            "train/batch_grad_norm": grad_norm,
            "step": epoch * len(loader) + batch_idx,
        })

    return {
        "loss"      : total_loss   / n,
        "ce"        : total_ce     / n,
        "dice_loss" : total_dice_l / n,
        "dice_score": total_ds     / n,
        "pixel_acc" : total_pa     / n,
        "grad_norm" : float(np.mean(grad_norms)),
    }


# ── Val epoch ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_ce = total_dice_l = 0.0
    total_pa   = total_ds = 0.0
    n = 0

    for batch in loader:
        imgs    = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device,  non_blocking=True).long()

        logits = model(imgs)
        loss, ce, dice_l = criterion(logits, targets, return_components=True)

        total_loss   += loss.item()
        total_ce     += ce.item()
        total_dice_l += dice_l.item()
        total_pa     += pixel_accuracy(logits, targets)
        total_ds     += dice_score(logits, targets)
        n            += 1

    return {
        "loss"      : total_loss   / n,
        "ce"        : total_ce     / n,
        "dice_loss" : total_dice_l / n,
        "dice_score": total_ds     / n,
        "pixel_acc" : total_pa     / n,
    }


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, metrics, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict"  : model.state_dict(),
        "optimizer"   : optimizer.state_dict(),
        "epoch"       : epoch,
        "best_dice"   : metrics["val_dice"],
        "val_loss"    : metrics["val_loss"],
        "freeze_mode" : model.freeze_mode,
    }, path)
    print(f"  ✓ Checkpoint saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    run_name = args.run_name or f"seg_{args.freeze_mode}_lr{args.lr}"
    ckpt_path = os.path.join(
        args.checkpoint_dir, f"segmenter_{args.freeze_mode}.pth"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ── W&B ──────────────────────────────────────────────────────────────────
    run = wandb.init(
        project = args.project,
        group   = args.group,
        name    = run_name,
        tags    = ["task3", "segmentation", args.freeze_mode],
        config  = {
            "encoder_ckpt"  : args.encoder_ckpt,
            "freeze_mode"   : args.freeze_mode,
            "epochs"        : args.epochs,
            "batch_size"    : args.batch_size,
            "lr"            : args.lr,
            "weight_decay"  : args.weight_decay,
            "scheduler"     : args.scheduler,
            "ce_weight"     : args.ce_weight,
            "dice_weight"   : args.dice_weight,
            "mixed_prec"    : args.mixed_prec,
            "num_classes"   : args.num_classes,
            "optimizer"     : "adam",
        }
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading data...")
    loaders = get_dataloaders(
        root         = args.root,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        val_fraction = args.val_fraction,
    )
    print(f"  Train: {len(loaders['train'].dataset)} | "
          f"Val: {len(loaders['val'].dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = VGG11UNet(
        encoder_ckpt = args.encoder_ckpt,
        num_classes  = args.num_classes,
        freeze_mode  = args.freeze_mode,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,}")
    wandb.config.update({
        "n_trainable_params": n_trainable,
        "n_total_params"    : n_total,
    })

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
    criterion = SegmentationLoss(
        num_classes = args.num_classes,
        ce_weight   = args.ce_weight,
        dice_weight = args.dice_weight,
    )

    optimizer = torch.optim.Adam(
        model.trainable_parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )

    scaler = torch.amp.GradScaler(
        'cuda', enabled=(args.mixed_prec and device.type == "cuda")
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_dice = 0.0

    print(f"\n{'='*60}")
    print(f"  Starting training — {args.epochs} epochs")
    print(f"  Run: {run_name}  |  Freeze: {args.freeze_mode}")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):

        t0      = time.time()
        train_m = train_epoch(model, loaders["train"], optimizer,
                              criterion, scaler, device, epoch)
        t_train = time.time() - t0

        t0    = time.time()
        val_m = val_epoch(model, loaders["val"], criterion, device)
        t_val = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        if args.scheduler == "plateau":
            scheduler.step(val_m["dice_score"])
        else:
            scheduler.step()

        wandb.log({
            # Train metrics
            "train/loss"       : train_m["loss"],
            "train/ce_loss"    : train_m["ce"],
            "train/dice_loss"  : train_m["dice_loss"],
            "train/dice_score" : train_m["dice_score"],
            "train/pixel_acc"  : train_m["pixel_acc"],
            "train/grad_norm"  : train_m["grad_norm"],
            # Val metrics
            "val/loss"         : val_m["loss"],
            "val/ce_loss"      : val_m["ce"],
            "val/dice_loss"    : val_m["dice_loss"],
            "val/dice_score"   : val_m["dice_score"],
            "val/pixel_acc"    : val_m["pixel_acc"],
            # Gaps (overfitting monitors)
            "gap/loss"         : train_m["loss"]       - val_m["loss"],
            "gap/dice_score"   : train_m["dice_score"] - val_m["dice_score"],
            "gap/pixel_acc"    : train_m["pixel_acc"]  - val_m["pixel_acc"],
            # Training dynamics
            "lr"               : current_lr,
            "best_val_dice"    : best_dice,
            # Timing
            "time/train_epoch_sec" : t_train,
            "time/val_epoch_sec"   : t_val,
            "time/total_epoch_sec" : t_train + t_val,
            "epoch"            : epoch,
        })

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train loss={train_m['loss']:.4f} "
            f"dice={train_m['dice_score']:.4f} "
            f"px_acc={train_m['pixel_acc']:.4f} | "
            f"Val loss={val_m['loss']:.4f} "
            f"dice={val_m['dice_score']:.4f} "
            f"px_acc={val_m['pixel_acc']:.4f} | "
            f"LR={current_lr:.2e} | "
            f"Time={t_train+t_val:.1f}s"
        )

        if val_m["dice_score"] > best_dice:
            best_dice = val_m["dice_score"]
            save_checkpoint(
                model, optimizer, epoch,
                {"val_dice": val_m["dice_score"], "val_loss": val_m["loss"]},
                ckpt_path,
            )
            wandb.log({"best_val_dice": best_dice, "best_epoch": epoch})

    wandb.summary["best_val_dice"]  = best_dice
    wandb.summary["freeze_mode"]    = args.freeze_mode
    wandb.summary["checkpoint"]     = ckpt_path

    print(f"\n{'='*60}")
    print(f"  Training complete! Best val Dice: {best_dice:.4f}")
    print(f"  W&B run: {run.url}")
    print(f"{'='*60}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
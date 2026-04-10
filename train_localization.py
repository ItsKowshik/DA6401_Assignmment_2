"""
train_localization.py — Task 2: VGG11 Bounding Box Regression
DA6401 Assignment 2
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

import wandb

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

sys.path.insert(0, str(Path(__file__).parent))
from data.pets_dataset import get_dataloaders
from models.localization import VGG11Localizer
from losses.iou_loss import IoULoss


def parse_args():
    p = argparse.ArgumentParser(description="VGG11 Localizer — Task 2")

    p.add_argument("--root", default="data/oxford-iiit-pet")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_fraction", type=float, default=0.1)

    p.add_argument("--encoder_ckpt", default="checkpoints/classifier.pth")
    p.add_argument("--freeze_encoder", action="store_true", default=False)
    p.add_argument("--dropout_p", type=float, default=0.3)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--mixed_prec", action="store_true", default=True)

    p.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "plateau"])

    p.add_argument("--project", default="DL_A2")
    p.add_argument("--run_name", default=None)

    p.add_argument("--checkpoint_dir", default="checkpoints")

    return p.parse_args()


def box_scale_like(imgs: torch.Tensor) -> torch.Tensor:
    h, w = imgs.shape[-2], imgs.shape[-1]
    return imgs.new_tensor([w, h, w, h])


def ensure_pixel_boxes(boxes: torch.Tensor, imgs: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    if boxes.detach().amax() <= 2.0:
        return boxes * box_scale_like(imgs)
    return boxes


@torch.no_grad()
def mean_iou(pred_px: torch.Tensor, gt_px: torch.Tensor, eps: float = 1e-7) -> float:
    def to_xyxy(b):
        x1 = b[:, 0] - b[:, 2] / 2
        y1 = b[:, 1] - b[:, 3] / 2
        x2 = b[:, 0] + b[:, 2] / 2
        y2 = b[:, 1] + b[:, 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    p = to_xyxy(pred_px.float())
    g = to_xyxy(gt_px.float())

    inter_x1 = torch.max(p[:, 0], g[:, 0])
    inter_y1 = torch.max(p[:, 1], g[:, 1])
    inter_x2 = torch.min(p[:, 2], g[:, 2])
    inter_y2 = torch.min(p[:, 3], g[:, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    pred_area = (p[:, 2] - p[:, 0]).clamp(0) * (p[:, 3] - p[:, 1]).clamp(0)
    gt_area = (g[:, 2] - g[:, 0]).clamp(0) * (g[:, 3] - g[:, 1]).clamp(0)
    union = pred_area + gt_area - inter + eps
    return (inter / union).mean().item()


def train_epoch(model, loader, optimizer, criterion_iou, criterion_mse, scaler, device, epoch, grad_clip):
    model.train()
    total_loss, total_iou, n = 0.0, 0.0, 0

    for batch_idx, batch in enumerate(loader):
        imgs = batch["image"].to(device, non_blocking=True)
        gt = batch["bbox"].to(device, non_blocking=True).float()

        has_bbox = batch.get("has_bbox", None)
        if has_bbox is not None:
            has_bbox = has_bbox.to(device, non_blocking=True).bool()
            if has_bbox.sum() == 0:
                continue
            imgs = imgs[has_bbox]
            gt = gt[has_bbox]

        gt_px = ensure_pixel_boxes(gt, imgs)
        scale = box_scale_like(imgs)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=(scaler.is_enabled() and device.type == "cuda")):
            pred_px = model(imgs)
            pred_norm = pred_px / scale
            gt_norm = gt_px / scale

            iou_loss = criterion_iou(pred_px, gt_px)
            mse_loss = criterion_mse(pred_norm, gt_norm)
            loss = iou_loss + mse_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_iou = mean_iou(pred_px.detach(), gt_px)
        bs = imgs.size(0)

        total_loss += loss.item() * bs
        total_iou += batch_iou * bs
        n += bs

        wandb.log({
            "train/batch_loss": loss.item(),
            "train/batch_iou": batch_iou,
            "step": (epoch - 1) * len(loader) + batch_idx,
        })

    return {
        "loss": total_loss / max(n, 1),
        "iou": total_iou / max(n, 1),
    }


@torch.no_grad()
def val_epoch(model, loader, criterion_iou, criterion_mse, device):
    model.eval()
    total_loss, total_iou, acc05_sum, acc075_sum, n = 0.0, 0.0, 0.0, 0.0, 0

    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        gt = batch["bbox"].to(device, non_blocking=True).float()

        has_bbox = batch.get("has_bbox", None)
        if has_bbox is not None:
            has_bbox = has_bbox.to(device, non_blocking=True).bool()
            if has_bbox.sum() == 0:
                continue
            imgs = imgs[has_bbox]
            gt = gt[has_bbox]

        gt_px = ensure_pixel_boxes(gt, imgs)
        scale = box_scale_like(imgs)

        pred_px = model(imgs)
        pred_norm = pred_px / scale
        gt_norm = gt_px / scale

        iou_loss = criterion_iou(pred_px, gt_px)
        mse_loss = criterion_mse(pred_norm, gt_norm)
        loss = iou_loss + mse_loss

        batch_iou = mean_iou(pred_px, gt_px)

        def to_xyxy(b):
            x1 = b[:, 0] - b[:, 2] / 2
            y1 = b[:, 1] - b[:, 3] / 2
            x2 = b[:, 0] + b[:, 2] / 2
            y2 = b[:, 1] + b[:, 3] / 2
            return torch.stack([x1, y1, x2, y2], dim=1)

        p = to_xyxy(pred_px.float())
        g = to_xyxy(gt_px.float())

        inter_x1 = torch.max(p[:, 0], g[:, 0])
        inter_y1 = torch.max(p[:, 1], g[:, 1])
        inter_x2 = torch.min(p[:, 2], g[:, 2])
        inter_y2 = torch.min(p[:, 3], g[:, 3])
        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        pred_area = (p[:, 2] - p[:, 0]).clamp(0) * (p[:, 3] - p[:, 1]).clamp(0)
        gt_area = (g[:, 2] - g[:, 0]).clamp(0) * (g[:, 3] - g[:, 1]).clamp(0)
        ious = inter / (pred_area + gt_area - inter + 1e-7)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_iou += batch_iou * bs
        acc05_sum += (ious >= 0.50).float().sum().item()
        acc075_sum += (ious >= 0.75).float().sum().item()
        n += bs

    return {
        "loss": total_loss / max(n, 1),
        "iou": total_iou / max(n, 1),
        "acc05": acc05_sum / max(n, 1),
        "acc075": acc075_sum / max(n, 1),
    }


def save_checkpoint(model, optimizer, epoch, metrics, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_iou": metrics["val_iou"],
        "val_loss": metrics["val_loss"],
        "freeze_encoder": model.freeze_encoder,
    }, path)
    print(f"  Checkpoint saved -> {path}")


def main():
    args = parse_args()
    enc_mode = "frozen" if args.freeze_encoder else "finetuned"
    run_name = args.run_name or f"localizer_{enc_mode}_lr{args.lr}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    run = wandb.init(
        project=args.project,
        group="Task2_Localization",
        name=run_name,
        tags=["task2", "localization", enc_mode],
        config={
            "encoder_ckpt": args.encoder_ckpt,
            "freeze_encoder": args.freeze_encoder,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "dropout_p": args.dropout_p,
            "mixed_prec": args.mixed_prec,
            "optimizer": "adam",
            "loss": "IoU + MSE(normalized)",
        }
    )

    print("\nLoading data...")
    loaders = get_dataloaders(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
    )
    print(f"  Train: {len(loaders['train'].dataset)} | Val: {len(loaders['val'].dataset)}")

    print("\nBuilding model...")
    model = VGG11Localizer(
        encoder_ckpt=args.encoder_ckpt,
        freeze_encoder=args.freeze_encoder,
        dropout_p=args.dropout_p,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,}")
    wandb.config.update({
        "n_trainable_params": n_trainable,
        "n_total_params": n_total,
    })

    criterion_iou = IoULoss(reduction="mean")
    criterion_mse = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
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

    scaler = torch.amp.GradScaler("cuda", enabled=(args.mixed_prec and device.type == "cuda"))
    ckpt_path = os.path.join(args.checkpoint_dir, "localizer.pth")

    best_val_iou = 0.0

    print(f"\n{'=' * 60}")
    print(f"  Starting training — {args.epochs} epochs")
    print(f"  Run: {run_name} | Encoder: {enc_mode}")
    print(f"{'=' * 60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_epoch(
            model, loaders["train"], optimizer,
            criterion_iou, criterion_mse, scaler, device, epoch, args.grad_clip
        )
        train_time = time.time() - t0

        t0 = time.time()
        val_m = val_epoch(model, loaders["val"], criterion_iou, criterion_mse, device)
        val_time = time.time() - t0
        epoch_time = train_time + val_time

        if val_m["iou"] > best_val_iou:
            best_val_iou = val_m["iou"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {"val_iou": val_m["iou"], "val_loss": val_m["loss"]},
                ckpt_path,
            )

        current_lr = optimizer.param_groups[0]["lr"]
        if args.scheduler == "plateau":
            scheduler.step(val_m["iou"])
        else:
            scheduler.step()

        wandb.log({
            "epoch": epoch,
            "train/loss": train_m["loss"],
            "train/iou": train_m["iou"],
            "val/loss": val_m["loss"],
            "val/iou": val_m["iou"],
            "val/acc@iou0.5": val_m["acc05"],
            "val/acc@iou0.75": val_m["acc075"],
            "gap/loss": train_m["loss"] - val_m["loss"],
            "gap/iou": train_m["iou"] - val_m["iou"],
            "lr": current_lr,
            "best_val_iou": best_val_iou,
            "time/epoch_sec": epoch_time,
            "time/train_epoch_sec": train_time,
            "time/val_epoch_sec": val_time,
        })

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train loss={train_m['loss']:.4f} iou={train_m['iou']:.4f} | "
            f"Val loss={val_m['loss']:.4f} iou={val_m['iou']:.4f} | "
            f"Acc@0.5={val_m['acc05']:.4f} Acc@0.75={val_m['acc075']:.4f} | "
            f"LR={current_lr:.2e} | Time={epoch_time:.1f}s"
            + ("  *** best" if val_m["iou"] >= best_val_iou else "")
        )

    wandb.summary["best_val_iou"] = best_val_iou
    wandb.summary["freeze_encoder"] = args.freeze_encoder
    wandb.summary["checkpoint"] = ckpt_path

    print(f"\n{'=' * 60}")
    print(f"  Training complete! Best val IoU: {best_val_iou:.4f}")
    print(f"  W&B run: {run.url}")
    print(f"{'=' * 60}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
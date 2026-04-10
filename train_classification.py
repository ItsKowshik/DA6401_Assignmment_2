# train_classification.py  Task 1 – VGG11 Classification Training  DA6401 Assignment 2
#
# Usage examples:
#   python train_classification.py --root data/oxford-iiit-pet                        # Standard run BN + Dropout p=0.5
#   python train_classification.py --root data/oxford-iiit-pet --no_bn --run_name classifier_nobn          # Section 2.1 without BN
#   python train_classification.py --root data/oxford-iiit-pet --dropout_p 0.0 --run_name classifier_nodropout  # Section 2.2 No dropout
#   python train_classification.py --root data/oxford-iiit-pet --dropout_p 0.2 --run_name classifier_dropout0.2  # Section 2.2 Dropout p=0.2
#   python train_classification.py --root data/oxford-iiit-pet --dropout_p 0.5 --run_name classifier_dropout0.5  # Section 2.2 Dropout p=0.5


import argparse
import os
import sys
import time
from pathlib import Path


from scipy.stats import gaussian_kde
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
import wandb


sys.path.insert(0, str(Path(__file__).parent))
from data.pets_dataset import get_dataloaders
from models.classification import VGG11Classifier



# ─── Argument parser ──────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="VGG11 Classifier Task 1")


    # Data
    p.add_argument("--root",         default="data/oxford-iiit-pet")
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--val_fraction", type=float, default=0.1)


    # Architecture
    p.add_argument("--num_classes", type=int,   default=37)
    p.add_argument("--dropout_p",   type=float, default=0.5,
                   help="CustomDropout probability (0.0 = no dropout)")
    p.add_argument("--no_bn", action="store_true",
                   help="Disable BatchNorm for Section 2.1 experiment")


    # Training
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--optimizer",    default="sgd", choices=["sgd", "adam"])
    p.add_argument("--lr",           type=float, default=0.01)
    p.add_argument("--momentum",     type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--label_smooth", type=float, default=0.1)
    p.add_argument("--grad_clip",    type=float, default=5.0)
    p.add_argument("--mixed_prec",   action="store_true", default=True,
                   help="Use torch.cuda.amp mixed precision")


    # Scheduler
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "plateau"])
    p.add_argument("--lr_step",   type=int,   default=10)
    p.add_argument("--lr_gamma",  type=float, default=0.1)


    # W&B
    p.add_argument("--project",  default="DL_A2")
    p.add_argument("--run_name", default=None,
                   help="W&B run name (auto-generated if None)")


    # Checkpoints
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--save_best_only", action="store_true", default=True)


    # Misc
    p.add_argument("--log_activations_every", type=int, default=5,
                   help="Log 3rd conv activation distribution every N epochs")
    p.add_argument("--experiment", default=None,
                   choices=["standard", "bn_effect", "dropout_effect"],
                   help="Experiment type – controls W&B group assignment")


    return p.parse_args()



# ─── Helpers ──────────────────────────────────────────────────────────────────


def resolve_group(args) -> str:
    """Each experiment type gets its own W&B group."""
    if args.experiment == "bn_effect":      return "Task1_BN_Effect"
    if args.experiment == "dropout_effect": return "Task1_Dropout_Effect"
    if args.experiment == "standard":       return "Task1_Classification"
    # Auto-detect from flags if --experiment not given
    if args.no_bn:              return "Task1_BN_Effect"
    if args.dropout_p != 0.5:  return "Task1_Dropout_Effect"
    return "Task1_Classification"



def auto_run_name(args) -> str:
    bn_tag  = "bn"  if not args.no_bn else "nobn"
    do_tag  = f"do{args.dropout_p}"
    opt_tag = args.optimizer
    return f"classifier_{bn_tag}_{do_tag}_{opt_tag}_lr{args.lr}"



def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Compute top-k accuracy for a batch."""
    with torch.no_grad():
        _, topk = logits.topk(k, dim=1)
        correct = topk.eq(labels.view(-1, 1).expand_as(topk))
        return correct.any(dim=1).float().mean().item()



def compute_grad_norm(model: nn.Module) -> float:
    """L2 norm of all gradients."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5



def log_activation_distribution(model, loader, device, epoch, step, n_batches=5):
    """
    Pass n_batches through the model, collect 3rd conv activations,
    log native W&B interactive histogram + KDE line chart.
    Required for Section 2.1.
    """
    activations = []


    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().flatten().numpy())


    target_layer = None
    for layer in model.encoder.block3:
        if isinstance(layer, nn.Conv2d):
            target_layer = layer
            break
    if target_layer is None:
        return


    hook = target_layer.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            model(batch["image"].to(device))
    hook.remove()
    model.train()


    if not activations:
        return


    all_acts = np.concatenate(activations)
    mu  = float(all_acts.mean())
    std = float(all_acts.std())


    # Sample for performance (10k points is plenty for visualisation)
    sample = np.random.choice(all_acts, size=min(10000, len(all_acts)), replace=False)
    table = wandb.Table(data=[[float(v)] for v in sample], columns=["activation"])
    
    wandb.log({
        "activations/3rd_conv_histogram": wandb.plot.histogram(
            table, "activation",
            title=f"3rd Conv Activation Distribution  μ={mu:.3f}, σ={std:.3f}"
        ),
        "activations/3rd_conv_mean": mu,
        "activations/3rd_conv_std":  std,
        "epoch": epoch,
    }, step=step)



# ─── Train / Val loops ────────────────────────────────────────────────────────


def train_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    top1_sum = top5_sum = 0.0
    n_batches = 0
    grad_norms = []


    for batch_idx, batch in enumerate(loader):
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)


        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            logits = model(imgs)
            loss   = criterion(logits, labels)


        scaler.scale(loss).backward()
        # Gradient clipping – unscale first for accurate clip
        scaler.unscale_(optimizer)
        grad_norm = compute_grad_norm(model)
        grad_norms.append(grad_norm)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()


        total_loss += loss.item()
        top1_sum   += topk_accuracy(logits, labels, k=1)
        top5_sum   += topk_accuracy(logits, labels, k=5)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())               
        n_batches  += 1

        wandb.log({
            "train/batch_loss":      loss.item(),
            "train/batch_grad_norm": grad_norm,
        }, step=(epoch - 1) * len(loader) + batch_idx)


    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return dict(
        loss      = total_loss / n_batches,
        acc_top1  = top1_sum   / n_batches,
        acc_top5  = top5_sum   / n_batches,
        f1_macro  = f1,
        grad_norm = float(np.mean(grad_norms)),
    )



@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    top1_sum = top5_sum = 0.0
    n_batches = 0


    for batch in loader:
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(imgs)
        loss   = criterion(logits, labels)


        total_loss += loss.item()
        top1_sum   += topk_accuracy(logits, labels, k=1)
        top5_sum   += topk_accuracy(logits, labels, k=5)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())  
        all_labels.extend(labels.cpu().tolist())               
        n_batches  += 1


    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return dict(
        loss     = total_loss / n_batches,
        acc_top1 = top1_sum   / n_batches,
        acc_top5 = top5_sum   / n_batches,
        f1_macro = f1,
    )



# ─── Checkpoint ───────────────────────────────────────────────────────────────


def save_checkpoint(model, optimizer, epoch, metrics, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "epoch":      epoch,
        "best_metric": metrics["val_acc_top1"],
        "val_f1_macro": metrics["val_f1_macro"],
        "val_loss":    metrics["val_loss"],
    }, path)
    print(f"  ✓ Checkpoint saved → {path}")



# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    args    = parse_args()
    use_bn  = not args.no_bn
    run_name = args.run_name or auto_run_name(args)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")


    group = resolve_group(args)


    run = wandb.init(
        project = args.project,
        group   = group,
        name    = run_name,
        tags    = ["task1", "classification", "bn" if use_bn else "no_bn",
                   f"dropout{args.dropout_p}"],
        config  = dict(
            # Architecture
            num_classes = args.num_classes,
            dropout_p   = args.dropout_p,
            use_bn      = use_bn,
            image_size  = 224,
            # Training
            epochs      = args.epochs,
            batch_size  = args.batch_size,
            optimizer   = args.optimizer,
            lr          = args.lr,
            momentum    = args.momentum,
            weight_decay= args.weight_decay,
            label_smoothing = args.label_smooth,
            grad_clip   = args.grad_clip,
            scheduler   = args.scheduler,
            mixed_precision = args.mixed_prec,
            # Dataset
            dataset     = "Oxford-IIIT-Pet",
            val_fraction= args.val_fraction,
            # Device
            device      = str(device),
            gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        ),
    )


    # ── Data ──
    print("Loading data...")
    loaders = get_dataloaders(
        root        = args.root,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        val_fraction= args.val_fraction,
    )
    print(f"  Train: {len(loaders['train'].dataset)} | "
          f"Val: {len(loaders['val'].dataset)} | "
          f"Test: {len(loaders['test'].dataset)}")


    # ── Model ──
    print("Building model...")
    model = VGG11Classifier(
        num_classes = args.num_classes,
        in_channels = 3,
        dropout_p   = args.dropout_p,
        use_bn      = use_bn,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params})


    # ── Loss ──
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)


    # ── Optimizer ──
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay, nesterov=True,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr, weight_decay=args.weight_decay,
        )


    # ── Scheduler ──
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5, verbose=True)


    # ── Mixed precision ──
    scaler = torch.amp.GradScaler("cuda",
        enabled=args.mixed_prec and device.type == "cuda")


    best_val_acc = 0.0
    ckpt_path    = os.path.join(args.checkpoint_dir, "classifier.pth")


    print("=" * 60)
    print(f"  Starting training — {args.epochs} epochs")
    print(f"  Run: {run_name}  |  W&B group: {group}")
    print("=" * 60)


    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()


        # Train
        t0 = time.time()
        train_metrics = train_epoch(
            model, loaders["train"], optimizer, criterion, scaler, device, epoch)
        train_time = time.time() - t0


        # Validate
        t0 = time.time()
        val_metrics = val_epoch(model, loaders["val"], criterion, device)
        val_time = time.time() - t0


        total_epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]


        # LR scheduler step
        if args.scheduler == "plateau":
            scheduler.step(val_metrics["acc_top1"])
        else:
            scheduler.step()


        # Generalisation gaps
        gap_loss    = train_metrics["loss"]     - val_metrics["loss"]
        gap_acc_top1= train_metrics["acc_top1"] - val_metrics["acc_top1"]
        gap_f1_macro= train_metrics["f1_macro"] - val_metrics["f1_macro"]


        current_step = epoch * len(loaders["train"]) - 1

        wandb.log({
            "epoch": epoch,
            "lr":    current_lr,
            # Train metrics
            "train/loss":      train_metrics["loss"],
            "train/acc_top1":  train_metrics["acc_top1"],
            "train/acc_top5":  train_metrics["acc_top5"],
            "train/f1_macro":  train_metrics["f1_macro"],
            "train/grad_norm": train_metrics["grad_norm"],
            # Val metrics
            "val/loss":      val_metrics["loss"],
            "val/acc_top1":  val_metrics["acc_top1"],
            "val/acc_top5":  val_metrics["acc_top5"],
            "val/f1_macro":  val_metrics["f1_macro"],
            # Generalisation gaps
            "gap/loss":     gap_loss,
            "gap/acc_top1": gap_acc_top1,
            "gap/f1_macro": gap_f1_macro,
            # Best so far
            "best_val_acc": best_val_acc,
            # Computation time
            "time/train_epoch_sec": train_time,
            "time/val_epoch_sec":   val_time,
            "time/total_epoch_sec": total_epoch_time,
        }, step=current_step)


        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train loss={train_metrics['loss']:.4f} acc={train_metrics['acc_top1']:.4f} "
            f"f1={train_metrics['f1_macro']:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} acc={val_metrics['acc_top1']:.4f} "
            f"f1={val_metrics['f1_macro']:.4f} | "
            f"LR={current_lr:.2e} Time={total_epoch_time:.1f}s"
        )


        # Save best checkpoint
        if val_metrics["acc_top1"] > best_val_acc:
            best_val_acc = val_metrics["acc_top1"]
            save_checkpoint(model, optimizer, epoch, {
                "val_acc_top1": val_metrics["acc_top1"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_loss":     val_metrics["loss"],
            }, ckpt_path)
            wandb.log({"best_val_acc": best_val_acc, "best_epoch": epoch}, step=current_step)


        # Activation distribution logging (Sections 2.1 & 2.4)
        if epoch % args.log_activations_every == 0 or epoch == args.epochs:
            log_activation_distribution(
                model, loaders["val"], device, epoch, step=current_step, n_batches=5)


    # Final W&B summary
    wandb.summary["best_val_acc"]   = best_val_acc
    wandb.summary["checkpoint"]     = ckpt_path
    wandb.summary["total_params"]   = n_params
    wandb.summary["use_bn"]         = use_bn
    wandb.summary["dropout_p"]      = args.dropout_p


    print("=" * 60)
    print("  Training complete!")
    print(f"  Best val acc: {best_val_acc:.4f}")
    print(f"  Checkpoint:   {ckpt_path}")
    print(f"  W&B run:      {run.url}")
    print("=" * 60)
    wandb.finish()



if __name__ == "__main__":
    main()
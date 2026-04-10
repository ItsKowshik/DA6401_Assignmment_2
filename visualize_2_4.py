import argparse
import torch
import torch.nn as nn
import numpy as np
import wandb
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from data.pets_dataset import get_dataloaders
from models.classification import VGG11Classifier
from models.utils import _remap_vgg_state


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root",        default="data/oxford-iiit-pet")
    p.add_argument("--checkpoint",  default="checkpoints/classifier.pth")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--project",     default="DL_A2")
    p.add_argument("--max_channels",type=int, default=16,
                   help="Max feature map channels to visualize per layer")
    return p.parse_args()


def normalize_map(fm: np.ndarray) -> np.ndarray:
    """Normalize a single feature map to [0, 1] for visualization."""
    lo, hi = fm.min(), fm.max()
    return (fm - lo) / (hi - lo + 1e-8)


def log_feature_grid(name: str, feat_tensor: torch.Tensor, max_ch: int):
    """
    Log up to max_ch feature maps as individual W&B images.
    feat_tensor shape: (1, C, H, W)
    """
    maps = feat_tensor[0]          # (C, H, W)
    n    = min(max_ch, maps.shape[0])
    images = []
    for i in range(n):
        fm = normalize_map(maps[i].numpy())
        images.append(wandb.Image(
            fm,
            caption=f"Channel {i} | shape={list(maps.shape[1:])}"
        ))
    wandb.log({f"feature_maps/{name}": images})


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = VGG11Classifier(
        num_classes=37, in_channels=3, dropout_p=0.5, use_bn=True
    ).to(device)
    sd = _remap_vgg_state(ckpt["state_dict"])
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ── Pick one image from val set ───────────────────────────────────────────
    loaders = get_dataloaders(
        root=args.root, batch_size=1,
        num_workers=args.num_workers, val_fraction=0.1
    )
    
    # Iterate through the validation set until we find a dog.
    # In the Oxford-IIIT Pet dataset, dog filenames always start with a lowercase letter.
    for b in loaders["val"]:
        stem = b["stem"][0]
        if stem[0].islower(): # Found a dog!
            batch = b
            break

    img    = batch["image"].to(device)    # (1, 3, 224, 224)
    label  = batch["label"].item()
    print(f"Selected image: {batch['stem'][0]} (Class ID: {label})")
    # ── Register hooks on first and last conv ─────────────────────────────────
    features = {}

    def make_hook(key):
        def hook(module, inp, out):
            features[key] = out.detach().cpu()
        return hook

    # First conv in block1
    first_conv = None
    for layer in model.encoder.block1:
        if isinstance(layer, nn.Conv2d):
            first_conv = layer
            break

    # Last conv in block5
    last_conv = None
    for layer in model.encoder.block5:
        if isinstance(layer, nn.Conv2d):
            last_conv = layer

    first_conv.register_forward_hook(make_hook("first_conv"))
    last_conv.register_forward_hook(make_hook("last_conv"))

    with torch.no_grad():
        logits = model(img)
        pred   = logits.argmax(dim=1).item()

    print(f"  True label: {label} | Predicted: {pred}")
    print(f"  first_conv output shape : {features['first_conv'].shape}")
    print(f"  last_conv  output shape : {features['last_conv'].shape}")

    # ── W&B init ──────────────────────────────────────────────────────────────
    run = wandb.init(
        project = args.project,
        group   = "Task1_FeatureMaps",
        name    = "feature_maps_2_4",
        tags    = ["task1", "visualization", "feature_maps", "section2.4"],
        config  = {
            "checkpoint"    : args.checkpoint,
            "true_label"    : label,
            "predicted_label": pred,
            "first_conv_shape": list(features["first_conv"].shape),
            "last_conv_shape" : list(features["last_conv"].shape),
            "max_channels"  : args.max_channels,
        }
    )

    # ── Log input image ───────────────────────────────────────────────────────
    # Denormalize ImageNet stats for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    display_img = (img.cpu().squeeze(0) * std + mean).clamp(0, 1)
    display_np  = display_img.permute(1, 2, 0).numpy()  # (H, W, 3)

    wandb.log({
        "input/image": wandb.Image(
            display_np,
            caption=f"True: {label} | Pred: {pred}"
        )
    })

    # ── Log feature maps ──────────────────────────────────────────────────────
    log_feature_grid(
        "01_first_conv_block1",
        features["first_conv"],
        args.max_channels
    )
    log_feature_grid(
        "02_last_conv_block5",
        features["last_conv"],
        args.max_channels
    )

    # ── Log activation statistics as a summary table ──────────────────────────
    for key in ["first_conv", "last_conv"]:
        fm   = features[key][0].numpy()   # (C, H, W)
        flat = fm.reshape(fm.shape[0], -1)
        table = wandb.Table(columns=["channel", "mean", "std", "min", "max"])
        for i in range(min(args.max_channels, fm.shape[0])):
            table.add_data(
                i,
                round(float(flat[i].mean()), 4),
                round(float(flat[i].std()),  4),
                round(float(flat[i].min()),  4),
                round(float(flat[i].max()),  4),
            )
        wandb.log({f"feature_stats/{key}": table})

    print(f"\nFeature maps logged → {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
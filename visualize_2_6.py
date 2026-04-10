import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import wandb

sys.path.insert(0, str(Path(__file__).parent))
from data.pets_dataset import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from models.segmentation import VGG11UNet
from losses.segmentation_loss import pixel_accuracy, dice_score
from models.utils import _remap_vgg_state


TRIMAP_PALETTE = np.array([
    [0, 200, 0],
    [20, 20, 20],
    [255, 200, 0],
], dtype=np.uint8)

MASK_CLASS_LABELS = {
    0: "pet",
    1: "background",
    2: "border",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/oxford-iiit-pet")
    p.add_argument("--segmenter_ckpt", default="checkpoints/unet.pth")
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--project", default="DL_A2")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--group", default="Task3_2_6")
    p.add_argument("--run_name", default="section_2.6_segmentation")
    return p.parse_args()


def mask_to_rgb(mask_np: np.ndarray) -> np.ndarray:
    return TRIMAP_PALETTE[mask_np.clip(0, 2)]


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=args.project,
        group=args.group,
        name=args.run_name,
        tags=["task3", "visualization", "section2.6", "overlay"],
    )

    loaders = get_dataloaders(
        root=args.root,
        batch_size=1,
        num_workers=args.num_workers,
    )

    ckpt = torch.load(args.segmenter_ckpt, map_location="cpu")
    model = VGG11UNet(
        encoder_ckpt="checkpoints/classifier.pth",
        freeze_mode=ckpt.get("freeze_mode", "full"),
    ).to(device)
    model.load_state_dict(_remap_vgg_state(ckpt["state_dict"]), strict=False)
    model.eval()    
    print(f"Loaded segmenter — best dice: {ckpt.get('best_dice', '?')}")

    table = wandb.Table(columns=[
        "sample_id",
        "original_image",
        "gt_overlay",
        "pred_overlay",
        "ground_truth_trimap",
        "predicted_trimap",
        "pixel_accuracy",
        "dice_score",
        "pa_minus_dice",
    ])

    all_pa = []
    all_ds = []
    collected = 0

    with torch.no_grad():
        for batch in loaders["test"]:
            if collected >= args.n_samples:
                break

            imgs = batch["image"].to(device)
            targets = batch["mask"].to(device).long()

            logits = model(imgs)
            preds = logits.argmax(dim=1)

            pa = float(pixel_accuracy(logits, targets))
            ds = float(dice_score(logits, targets))
            all_pa.append(pa)
            all_ds.append(ds)

            orig_np = denormalize(imgs[0])
            gt_mask_np = targets[0].detach().cpu().numpy().astype(np.uint8)
            pred_mask_np = preds[0].detach().cpu().numpy().astype(np.uint8)
            gt_np = mask_to_rgb(gt_mask_np)
            pred_np = mask_to_rgb(pred_mask_np)

            gt_overlay = wandb.Image(
                orig_np,
                masks={
                    "ground_truth": {
                        "mask_data": gt_mask_np,
                        "class_labels": MASK_CLASS_LABELS,
                    }
                },
                caption="GT overlay on original image",
            )

            pred_overlay = wandb.Image(
                orig_np,
                masks={
                    "prediction": {
                        "mask_data": pred_mask_np,
                        "class_labels": MASK_CLASS_LABELS,
                    }
                },
                caption=f"Prediction overlay | PA={pa:.3f} Dice={ds:.3f}",
            )

            table.add_data(
                collected + 1,
                wandb.Image(orig_np, caption="Original Image"),
                gt_overlay,
                pred_overlay,
                wandb.Image(gt_np, caption="Ground Truth Trimap\n0=pet(green) 1=bg(black) 2=border(yellow)"),
                wandb.Image(pred_np, caption=f"Predicted Trimap | PA={pa:.3f} Dice={ds:.3f}"),
                round(pa, 4),
                round(ds, 4),
                round(pa - ds, 4),
            )

            print(
                f"  [{collected+1:02d}/{args.n_samples}] "
                f"pixel_acc={pa:.4f} dice={ds:.4f} gap={pa-ds:.4f}"
            )
            collected += 1

    mean_pa = float(np.mean(all_pa)) if all_pa else 0.0
    mean_ds = float(np.mean(all_ds)) if all_ds else 0.0
    mean_gap = mean_pa - mean_ds

    pa_vs_dice_table = wandb.Table(
        columns=["sample_id", "pixel_accuracy", "dice_score"],
        data=[[i + 1, round(all_pa[i], 4), round(all_ds[i], 4)] for i in range(collected)],
    )

    wandb.log({
        "segmentation/sample_table": table,
        "segmentation/mean_pixel_accuracy": mean_pa,
        "segmentation/mean_dice_score": mean_ds,
        "segmentation/mean_pa_minus_dice": mean_gap,
        "segmentation/pa_bar": wandb.plot.bar(
            pa_vs_dice_table,
            label="sample_id",
            value="pixel_accuracy",
            title="Pixel Accuracy per Sample",
        ),
        "segmentation/pa_vs_dice_line": wandb.plot.line_series(
            xs=list(range(1, collected + 1)),
            ys=[all_pa, all_ds],
            keys=["Pixel Accuracy", "Dice Score"],
            title="PA vs Dice — Inflation Gap (Section 2.6)",
            xname="Sample Index",
        ),
        "segmentation/pixel_accuracy_dist": wandb.plot.histogram(
            wandb.Table(columns=["pixel_accuracy"], data=[[v] for v in all_pa]),
            value="pixel_accuracy",
            title="Pixel Accuracy Distribution",
        ),
        "segmentation/dice_score_dist": wandb.plot.histogram(
            wandb.Table(columns=["dice_score"], data=[[v] for v in all_ds]),
            value="dice_score",
            title="Dice Score Distribution",
        ),
        "segmentation/pa_vs_dice_scatter": wandb.plot.scatter(
            wandb.Table(
                columns=["pixel_accuracy", "dice_score"],
                data=[[all_pa[i], all_ds[i]] for i in range(collected)],
            ),
            x="pixel_accuracy",
            y="dice_score",
            title="Pixel Accuracy vs Dice Score",
        ),
    })

    wandb.summary["test_mean_pixel_accuracy"] = mean_pa
    wandb.summary["test_mean_dice_score"] = mean_ds
    wandb.summary["test_pa_dice_gap"] = mean_gap

    print("\nSection 2.6 complete:")
    print(f"  Samples        : {collected}")
    print(f"  Mean Pixel Acc : {mean_pa:.4f}")
    print(f"  Mean Dice Score: {mean_ds:.4f}")
    print(f"  Gap (PA - Dice): {mean_gap:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
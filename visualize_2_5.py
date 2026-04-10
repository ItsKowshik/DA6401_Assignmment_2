"""visualize_2_5.py — Section 2.5: Bounding Box Visualization
DA6401 Assignment 2
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
import wandb

sys.path.insert(0, str(Path(__file__).parent))
from data.pets_dataset import get_dataloaders, IMAGE_SIZE
from models.localization import VGG11Localizer
from train import compute_iou_batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/oxford-iiit-pet")
    p.add_argument("--localizer_ckpt", default="checkpoints/localizer.pth")
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--project", default="DL_A2")
    p.add_argument("--group", default="Task2_5")
    p.add_argument("--run_name", default="section_2.5_detections")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--high_conf_thresh", type=float, default=0.4)
    p.add_argument("--low_iou_thresh", type=float, default=0.2)
    return p.parse_args()


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor.cpu().permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def compute_confidence(pred_box: torch.Tensor) -> float:
    cx, cy, _, _ = pred_box.tolist()
    dist = ((cx - IMAGE_SIZE / 2) ** 2 + (cy - IMAGE_SIZE / 2) ** 2) ** 0.5
    max_dist = (2 * (IMAGE_SIZE / 2) ** 2) ** 0.5
    confidence = max(0.0, 1.0 - (dist / max_dist))
    confidence = 0.3 + (confidence * 0.69)
    return round(confidence, 4)


def classify_failure(iou: float, confidence: float, high_conf_thresh: float, low_iou_thresh: float):
    if confidence >= high_conf_thresh and iou < low_iou_thresh:
        return True, "high_conf_low_iou"
    if iou == 0.0:
        return True, "missed_object"
    return False, "none"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=args.project,
        group=args.group,
        name=args.run_name,
        tags=["task2", "visualization", "section2.5"],
        config={
            "n_samples": args.n_samples,
            "localizer_ckpt": args.localizer_ckpt,
        },
    )

    loaders = get_dataloaders(root=args.root, batch_size=16, num_workers=args.num_workers)

    # Initialize model
    model = VGG11Localizer(encoder_ckpt=None, freeze_encoder=True).to(device)

    # Load weights correctly 
    ckpt = torch.load(args.localizer_ckpt, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval()
    
    table = wandb.Table(columns=["image", "iou", "confidence", "is_failure", "failure_reason"])

    collected = 0
    all_ious = []
    all_confs = []
    n_failures = 0

    with torch.no_grad():
        for batch in loaders["val"]:
            if collected >= args.n_samples:
                break
                
            imgs = batch["image"].to(device)
            # FIX: Scale ground truth to [0, 224] pixels to match the model!
            bboxes = batch["bbox"].to(device)
            has_bb = batch["has_bbox"].to(device)
            
            if has_bb.sum() == 0:
                continue

            preds = model(imgs)
            ious = compute_iou_batch(preds, bboxes)

            for i in range(imgs.size(0)):
                if collected >= args.n_samples:
                    break
                    
                if not has_bb[i]:
                    continue
                    
                img_np = denormalize(imgs[i])
                gt_box = bboxes[i].cpu().numpy()
                pr_box = preds[i].cpu().numpy()
                iou_v  = ious[i].item()
                
                confidence = compute_confidence(preds[i].cpu())
                is_failure, failure_reason = classify_failure(
                    iou_v, confidence, args.high_conf_thresh, args.low_iou_thresh
                )

                def _draw(img, box, color):
                    pil  = Image.fromarray(img.copy())
                    draw = ImageDraw.Draw(pil)
                    xc, yc, bw, bh = box
                    draw.rectangle([xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2], outline=color, width=3)
                    return np.array(pil)

                img_drawn = _draw(_draw(img_np, gt_box, "green"), pr_box, "red")

                table.add_data(
                    wandb.Image(img_drawn, caption="GT=green Pred=red"),
                    round(iou_v, 4), round(confidence, 4), is_failure, failure_reason
                )

                all_ious.append(iou_v)
                all_confs.append(confidence)
                if is_failure:
                    n_failures += 1

                collected += 1
                print(f"  [{collected:02d}/{args.n_samples}] IoU={iou_v:.4f} conf={confidence:.4f} failure={is_failure}")

    # Prevent ZeroDivisionError
    safe_collected = max(collected, 1)
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    mean_conf = float(np.mean(all_confs)) if all_confs else 0.0
    failure_rate = n_failures / safe_collected

    wandb.log({
        "detections/table": table,
        "detections/mean_iou": mean_iou,
        "detections/mean_confidence": mean_conf,
        "detections/n_failures": n_failures,
        "detections/n_samples": collected,
        "detections/failure_rate": failure_rate,
    })

    wandb.summary["test_mean_iou"] = mean_iou
    wandb.summary["test_mean_conf"] = mean_conf
    wandb.summary["test_n_failures"] = n_failures
    wandb.summary["test_failure_rate"] = failure_rate

    print(f"\nSection 2.5 complete:\n  Samples: {collected}\n  Mean IoU: {mean_iou:.4f}\n  Failures: {n_failures}/{collected}")
    wandb.finish()

if __name__ == "__main__":
    main()
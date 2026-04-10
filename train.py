import argparse
import time
import torch
import torch.nn as nn
import wandb

from models.multitask import MultiTaskPerceptionModel
from data.pets_dataset import get_dataloaders, IMAGE_SIZE
from losses.iou_loss import IoULoss
from losses.segmentation_loss import SegmentationLoss


def compute_iou_batch(pred_boxes, tgt_boxes, eps=1e-6):
    px1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    py1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    px2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    py2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    tx1 = tgt_boxes[:, 0] - tgt_boxes[:, 2] / 2
    ty1 = tgt_boxes[:, 1] - tgt_boxes[:, 3] / 2
    tx2 = tgt_boxes[:, 0] + tgt_boxes[:, 2] / 2
    ty2 = tgt_boxes[:, 1] + tgt_boxes[:, 3] / 2

    iw = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    ih = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = iw * ih

    pa = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    ta = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = pa + ta - inter

    return inter / (union + eps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--run_name", type=str, default="multitask")
    parser.add_argument("--group", type=str, default="task4-multitask")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskPerceptionModel(
        num_breeds=37,
        seg_classes=3,
        classifier_path="checkpoints_perfect/classifier.pth",
        localizer_path="checkpoints_perfect/localizer.pth",
        unet_path="checkpoints_perfect/unet.pth",
    ).to(device)

    run = wandb.init(
        project="DL_A2",
        group=args.group,
        name=args.run_name,
        config=vars(args),
    )

    cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    mse_fn = nn.MSELoss()
    iou_fn = IoULoss(reduction="mean")
    seg_crit = SegmentationLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    loaders = get_dataloaders(
        root="data/oxford-iiit-pet",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    best_combined = -1.0

    for epoch in range(1, args.epochs + 1):
        train_start = time.time()
        model.train()

        tr_cls_correct = 0
        tr_seg_dice_sum = 0.0
        tr_loc_iou_sum = 0.0
        tr_loss_sum = 0.0
        n = 0
        n_loc = 0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bbox"].to(device)
            masks = batch["mask"].to(device)
            has_bbox = batch["has_bbox"].to(device)

            optimizer.zero_grad()
            out = model(imgs)

            cls_out = out["classification"]
            loc_out = out["localization"]
            seg_out = out["segmentation"]

            cls_loss = cls_criterion(cls_out, labels)
            seg_loss, _, l_dice = seg_crit(seg_out, masks, return_components=True)

            if has_bbox.sum() > 0:
                pred_loc = loc_out[has_bbox] / float(IMAGE_SIZE)  # normalize
                tgt_loc = bboxes[has_bbox] / float(IMAGE_SIZE)    # normalize

                loc_loss = mse_fn(pred_loc, tgt_loc)  # MSE only, normalized scale

                train_ious = compute_iou_batch(loc_out[has_bbox], bboxes[has_bbox])  # IoU on pixel for metric
                tr_loc_iou_sum += train_ious.sum().item()
                n_loc += has_bbox.sum().item()
            else:
                loc_loss = torch.tensor(0.0, device=device)

            loss = cls_loss + 0.5*loc_loss + seg_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = imgs.size(0)
            tr_loss_sum += loss.item() * bs
            tr_cls_correct += (cls_out.argmax(1) == labels).sum().item()
            tr_seg_dice_sum += (1.0 - l_dice.item()) * bs
            n += bs

        train_time = time.time() - train_start
        scheduler.step()

        val_start = time.time()
        model.eval()

        v_cls_correct = 0
        v_seg_dice_sum = 0.0
        v_loc_iou_sum = 0.0
        v_loss_sum = 0.0
        vn = 0
        vn_loc = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device)
                bboxes = batch["bbox"].to(device)
                masks = batch["mask"].to(device)
                has_bbox = batch["has_bbox"].to(device)

                out = model(imgs)

                cls_out = out["classification"]
                loc_out = out["localization"]
                seg_out = out["segmentation"]

                cls_loss = cls_criterion(cls_out, labels)
                seg_loss, _, l_dice = seg_crit(seg_out, masks, return_components=True)

                if has_bbox.sum() > 0:
                    pred_loc = loc_out[has_bbox]
                    tgt_loc = bboxes[has_bbox]

                    pred_loc_norm = pred_loc / float(IMAGE_SIZE)
                    tgt_loc_norm = tgt_loc / float(IMAGE_SIZE)

                    loc_loss = mse_fn(pred_loc_norm, tgt_loc_norm) + iou_fn(pred_loc, tgt_loc)

                    val_ious = compute_iou_batch(pred_loc, tgt_loc)
                    v_loc_iou_sum += val_ious.sum().item()
                    vn_loc += has_bbox.sum().item()
                else:
                    loc_loss = torch.tensor(0.0, device=device)

                loss = cls_loss + loc_loss + seg_loss

                bs = imgs.size(0)
                v_loss_sum += loss.item() * bs
                v_cls_correct += (cls_out.argmax(1) == labels).sum().item()
                v_seg_dice_sum += (1.0 - l_dice.item()) * bs
                vn += bs

        val_time = time.time() - val_start

        train_loss = tr_loss_sum / max(n, 1)
        train_cls_acc = tr_cls_correct / max(n, 1)
        train_seg_dice = tr_seg_dice_sum / max(n, 1)
        train_loc_iou = tr_loc_iou_sum / max(n_loc, 1)

        val_loss = v_loss_sum / max(vn, 1)
        val_cls_acc = v_cls_correct / max(vn, 1)
        val_seg_dice = v_seg_dice_sum / max(vn, 1)
        val_loc_iou = v_loc_iou_sum / max(vn_loc, 1)
        combined = (val_cls_acc + val_loc_iou + val_seg_dice) / 3.0

        wandb.log({
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train/comp_time": train_time,
            "val/comp_time": val_time,
            "train/loss_total": train_loss,
            "train/cls_acc": train_cls_acc,
            "train/loc_iou": train_loc_iou,
            "train/seg_dice": train_seg_dice,
            "val/loss_total": val_loss,
            "val/cls_acc": val_cls_acc,
            "val/loc_iou": val_loc_iou,
            "val/seg_dice": val_seg_dice,
            "val/combined": combined,
        }, step=epoch)

        print(
            f"[{epoch:02d}/{args.epochs}]  "
            f"t_train={train_time:.1f}s  "
            f"t_val={val_time:.1f}s  "
            f"val_loss={val_loss:.4f}  "
            f"cls_acc={val_cls_acc:.4f}  "
            f"loc_iou={val_loc_iou:.4f}  "
            f"seg_dice={val_seg_dice:.4f}  "
            f"combined={combined:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if combined > best_combined:
            best_combined = combined
            run.summary.update({
                "best_epoch": epoch,
                "best_val_loss": val_loss,
                "best_val_acc": val_cls_acc,
                "best_val_iou": val_loc_iou,
                "best_val_dice": val_seg_dice,
                "best_val_combined": best_combined,
            })

    run.finish()
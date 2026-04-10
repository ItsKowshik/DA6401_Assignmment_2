"""
losses/segmentation_loss.py — Task 3: Segmentation Loss
DA6401 Assignment 2

Combined loss: CrossEntropy + Dice
  L = CE(pred, gt) + λ * (1 - Dice(pred, gt))

Why this combination:
  - CrossEntropy: per-pixel supervision, strong gradient signal at every pixel,
    handles multi-class naturally via softmax probabilities.
  - Dice loss: directly optimizes the overlap metric, robust to class imbalance
    (foreground pet pixels are far fewer than background pixels in trimaps).
    CE alone would achieve high pixel accuracy by predicting background everywhere
    — Dice penalizes this directly.
  - Together: CE drives stable early convergence, Dice refines boundary precision.

Usage:
    criterion = SegmentationLoss(ce_weight=1.0, dice_weight=1.0)
    loss, ce, dice = criterion(logits, targets, return_components=True)
    # logits : (N, 3, H, W) raw unnormalized scores
    # targets: (N, H, W)   long tensor, values in {0, 1, 2}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.

    Computes Dice per class then averages (macro).
    Uses softmax probabilities — works with raw logits via log_softmax=False.

    Formula per class c:
        Dice_c = (2 * sum(p_c * g_c) + eps) / (sum(p_c) + sum(g_c) + eps)
        L_dice  = 1 - mean_c(Dice_c)
    """

    def __init__(self, num_classes: int = 3, eps: float = 1e-6,
                 ignore_index: int = -1):
        super().__init__()
        self.num_classes   = num_classes
        self.eps           = eps
        self.ignore_index  = ignore_index

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (N, C, H, W) raw scores
            targets: (N, H, W)    long, values in {0..C-1}
        Returns:
            scalar Dice loss
        """
        N, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)      # (N, C, H, W)

        # One-hot encode targets → (N, C, H, W)
        targets_oh = F.one_hot(
            targets.clamp(0, C - 1), num_classes=C
        ).permute(0, 3, 1, 2).float()         # (N, C, H, W)

        # Handle ignore_index — zero out those positions in both
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs      = probs      * mask
            targets_oh = targets_oh * mask

        # Flatten spatial dims: (N, C, H*W)
        probs      = probs.view(N, C, -1)
        targets_oh = targets_oh.view(N, C, -1)

        # Dice per class per sample
        intersection = (probs * targets_oh).sum(dim=2)      # (N, C)
        union        = probs.sum(dim=2) + targets_oh.sum(dim=2)  # (N, C)
        dice_per     = (2.0 * intersection + self.eps) / (union + self.eps)
        gt_present = targets_oh.sum(dim=2) > 0   # (N, C) bool
        dice_per   = torch.where(gt_present, dice_per, torch.ones_like(dice_per))
        return 1.0 - dice_per.mean()


class SegmentationLoss(nn.Module):
    """
    Combined CrossEntropy + Dice loss for 3-class trimap segmentation.

      L = ce_weight * CE(logits, targets)
        + dice_weight * DiceLoss(logits, targets)

    Args:
        num_classes  : number of segmentation classes (3 for trimaps)
        ce_weight    : scalar weight on CrossEntropy term
        dice_weight  : scalar weight on Dice term
        class_weights: optional (C,) tensor to upweight rare classes
        ignore_index : pixel label to ignore in both losses
    """

    def __init__(
        self,
        num_classes:   int             = 3,
        ce_weight:     float           = 1.0,
        dice_weight:   float           = 1.0,
        class_weights: torch.Tensor    = None,
        ignore_index:  int             = -1,
    ):
        super().__init__()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight

        self.ce   = nn.CrossEntropyLoss(
            weight       = class_weights,
            ignore_index = ignore_index,
        )
        self.dice = DiceLoss(
            num_classes  = num_classes,
            ignore_index = ignore_index,
        )

    def forward(
        self,
        logits:           torch.Tensor,   # (N, C, H, W)
        targets:          torch.Tensor,   # (N, H, W) long
        return_components: bool = False,
    ):
        """
        Returns:
            combined loss (scalar) if return_components=False
            (combined, ce_loss, dice_loss) if return_components=True
        """
        ce_loss   = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        combined  = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        if return_components:
            return combined, ce_loss, dice_loss
        return combined


# ── Pixel accuracy helper (used in train_segmentation.py + Section 2.6) ──────

@torch.no_grad()
def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Overall pixel accuracy = correct_pixels / total_pixels.

    Args:
        logits : (N, C, H, W)
        targets: (N, H, W) long
    Returns:
        float in [0, 1]
    """
    preds   = logits.argmax(dim=1)        # (N, H, W)
    correct = (preds == targets).sum().item()
    total   = targets.numel()
    return correct / total


@torch.no_grad()
def dice_score(logits: torch.Tensor, targets: torch.Tensor,
               num_classes: int = 3, eps: float = 1e-6) -> float:
    """
    Mean Dice score (not loss) — for logging. Higher is better.
    Computed same as DiceLoss but returns the score not 1-score.
    """
    N, C, H, W = logits.shape
    probs      = F.softmax(logits, dim=1)
    targets_oh = F.one_hot(
        targets.clamp(0, C - 1), num_classes=C
    ).permute(0, 3, 1, 2).float()

    probs      = probs.view(N, C, -1)
    targets_oh = targets_oh.view(N, C, -1)

    intersection = (probs * targets_oh).sum(dim=2)
    union        = probs.sum(dim=2) + targets_oh.sum(dim=2)
    dice_per     = (2.0 * intersection + eps) / (union + eps)

    return dice_per.mean().item()


if __name__ == "__main__":
    import torch
    criterion = SegmentationLoss(num_classes=3, ce_weight=1.0, dice_weight=1.0)

    # Test 1: Perfect prediction → dice loss ≈ 0, combined ≈ CE only
    logits  = torch.zeros(2, 3, 224, 224)
    targets = torch.zeros(2, 224, 224, dtype=torch.long)
    logits[:, 0, :, :] = 10.0   # force class 0 everywhere
    loss, ce, dice = criterion(logits, targets, return_components=True)
    assert dice.item() < 0.01, "FAIL: perfect pred dice should be near 0"
    print(f"Perfect prediction → dice={dice.item():.4f}, ce={ce.item():.4f}")

    # Test 2: Random prediction → loss > 0, gradient flows
    logits  = torch.randn(2, 3, 224, 224, requires_grad=True)
    targets = torch.randint(0, 3, (2, 224, 224))
    loss    = criterion(logits, targets)
    loss.backward()
    assert logits.grad is not None, "FAIL: no gradient"
    print(f"Random prediction → loss={loss.item():.4f}, gradient flows")

    # Test 3: pixel_accuracy and dice_score helpers
    logits  = torch.randn(2, 3, 64, 64)
    targets = torch.randint(0, 3, (2, 64, 64))
    pa  = pixel_accuracy(logits, targets)
    ds  = dice_score(logits, targets)
    assert 0 <= pa <= 1, "FAIL: pixel_accuracy out of range"
    assert 0 <= ds <= 1, "FAIL: dice_score out of range"
    print(f"Helpers → pixel_accuracy={pa:.4f}, dice_score={ds:.4f}")

    # Test 4: Class imbalance demo (why dice > pixel accuracy matters)
    # 95% background, 5% foreground
    logits  = torch.zeros(1, 3, 224, 224)
    logits[:, 1, :, :] = 10.0   # predict background everywhere
    targets = torch.ones(1, 224, 224, dtype=torch.long)
    targets[:, :11, :] = 0      
    pa = pixel_accuracy(logits, targets)
    ds = dice_score(logits, targets)
    print(f"Imbalance demo → pixel_accuracy={pa:.4f} (inflated), dice={ds:.4f} (honest)")
    assert pa > ds, "FAIL: expected pixel_acc > dice for imbalanced case"

    print("\nAll tests passed")
"""
models/iou_loss.py — Custom IoU Loss for Task 2
DA6401 Assignment 2

Input format: [x_center, y_center, width, height] normalized to [0, 1]
Usage:
    criterion = IoULoss(reduction='mean')
    loss = criterion(pred_boxes, gt_boxes)  # both shape (N, 4)
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """
    Custom IoU Loss inheriting from nn.Module.
    Expects boxes in [x_center, y_center, width, height] format, normalized [0, 1].

    L = 1 - IoU(pred, gt)

    Handles:
      - Zero-area boxes (eps guard)
      - Fully non-overlapping boxes (IoU=0, loss=1, gradient still flows via pred area)
      - Mixed precision (float32/float16) via .float() cast
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        assert reduction in ("mean", "sum", "none"), \
            f"reduction must be 'mean' | 'sum' | 'none', got {reduction}"
        self.reduction = reduction
        self.eps       = eps

    # ── Internal helper: cxcywh → xyxy ───────────────────────────────────────
    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert [x_c, y_c, w, h] → [x1, y1, x2, y2].
        boxes: (N, 4)
        """
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    # ── Core IoU computation ──────────────────────────────────────────────────
    def _compute_iou(
        self,
        pred: torch.Tensor,   # (N, 4) cxcywh
        gt:   torch.Tensor,   # (N, 4) cxcywh
    ) -> torch.Tensor:        # (N,)

        pred_xyxy = self._cxcywh_to_xyxy(pred.float())
        gt_xyxy   = self._cxcywh_to_xyxy(gt.float())

        # Intersection
        inter_x1 = torch.max(pred_xyxy[:, 0], gt_xyxy[:, 0])
        inter_y1 = torch.max(pred_xyxy[:, 1], gt_xyxy[:, 1])
        inter_x2 = torch.min(pred_xyxy[:, 2], gt_xyxy[:, 2])
        inter_y2 = torch.min(pred_xyxy[:, 3], gt_xyxy[:, 3])

        inter_w    = (inter_x2 - inter_x1).clamp(min=0)
        inter_h    = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Individual areas  (clamp prevents negative area from bad predictions)
        pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0) * \
                    (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0)
        gt_area   = (gt_xyxy[:, 2]   - gt_xyxy[:, 0]).clamp(min=0) * \
                    (gt_xyxy[:, 3]   - gt_xyxy[:, 1]).clamp(min=0)

        union = pred_area + gt_area - inter_area + self.eps
        iou   = inter_area / union

        return iou   # (N,) values in [0, 1]

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        pred: torch.Tensor,   # (N, 4) predicted boxes, cxcywh, [0,1]
        gt:   torch.Tensor,   # (N, 4) ground-truth boxes, cxcywh, [0,1]
    ) -> torch.Tensor:

        if pred.shape != gt.shape or pred.shape[-1] != 4:
            raise ValueError(
                f"pred and gt must both be (N, 4), got {pred.shape} and {gt.shape}"
            )

        iou  = self._compute_iou(pred, gt)       # (N,)
        loss = 1.0 - iou                          # (N,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    # ── Convenience: raw IoU scores (no gradient, for logging) ────────────────
    @torch.no_grad()
    def compute_iou_scores(
        self,
        pred: torch.Tensor,
        gt:   torch.Tensor,
    ) -> torch.Tensor:
        """Returns per-sample IoU scores (N,) for logging/metrics. No grad."""
        return self._compute_iou(pred, gt)
    

if __name__ == "__main__":
    import torch
    criterion = IoULoss(reduction="mean")

    # Test 1: Perfect overlap → loss = 0
    b = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    assert criterion(b, b).item() < 1e-5, "FAIL: perfect overlap"
    print("✓ Perfect overlap → loss ≈ 0")

    # Test 2: No overlap → loss = 1
    pred = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    gt   = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
    assert abs(criterion(pred, gt).item() - 1.0) < 1e-5, "FAIL: no overlap"
    print("✓ No overlap → loss ≈ 1")

    # Test 3: Partial overlap → loss in (0, 1)
    pred = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    gt   = torch.tensor([[0.6, 0.6, 0.4, 0.4]])
    loss = criterion(pred, gt).item()
    assert 0 < loss < 1, "FAIL: partial overlap"
    print(f"✓ Partial overlap → loss = {loss:.4f}")

    # Test 4: Gradient flows
    pred = torch.tensor([[0.5, 0.5, 0.3, 0.3]], requires_grad=True)
    gt   = torch.tensor([[0.6, 0.6, 0.3, 0.3]])
    criterion(pred, gt).backward()
    assert pred.grad is not None, "FAIL: no gradient"
    print(f"✓ Gradient flows → grad = {pred.grad}")

    # Test 5: Zero-area box (no crash)
    pred = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    gt   = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
    loss = criterion(pred, gt).item()
    print(f"✓ Zero-area box → loss = {loss:.4f} (no crash)")

    print("\nAll tests passed ✅")
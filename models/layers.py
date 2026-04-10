"""Reusable custom layers 
"""
import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """
    Inverted dropout — scales surviving activations by 1/(1-p) at train time
    so eval mode is a pure identity (no rescaling needed).
    nn.Dropout and F.dropout are NOT used anywhere.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x                                       # identity at eval
        mask = torch.bernoulli(torch.full_like(x, 1.0 - self.p))
        return x * mask / (1.0 - self.p)                  # inverted scaling

    def extra_repr(self) -> str:
        return f"p={self.p}"

"""Classification model — DA6401 Assignment 2"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder (conv backbone) + FC classification head.

    ── Head Architecture ────────────────────────────────────────────────────
    Bottleneck [B,512,7,7]
        → Flatten → [B, 25088]
        → Linear(25088→4096) → BN1d → ReLU → CustomDropout(p)
        → Linear(4096→4096)  → BN1d → ReLU → CustomDropout(p)
        → Linear(4096→num_classes)
        → [B, 37] logits

    ── Design Choices (required justification) ──────────────────────────────
    1. BatchNorm1d in FC layers:
       BN after each FC (before ReLU) prevents the dense activations from
       exploding or collapsing, enabling stable training at higher LR.
       It also acts as a mild regulariser in the classifier head.

    2. CustomDropout placement (after BN, before next Linear):
       Dropping AFTER BN means the mask acts on already-normalised values,
       giving the downstream layer a consistent zero-mean signal even with
       dropped neurons — more stable gradient estimates than dropping raw
       pre-BN activations.

    3. No Dropout after the final Linear:
       The final layer outputs class logits directly fed to CrossEntropyLoss.
       Adding dropout here would corrupt the probability distribution at
       training time without any regularisation benefit.

    4. Bias=False in the two hidden FC layers:
       BN's β parameter subsumes the bias term, removing a redundant
       degree of freedom.
    """

    def __init__(self, num_classes: int = 37, in_channels: int = 3,
                 dropout_p: float = 0.5, use_bn: bool = True):
        super().__init__()
        self.encoder    = VGG11Encoder(in_channels=in_channels, use_bn=use_bn)
        self.classifier = self._build_head(num_classes, dropout_p, use_bn)
        self._init_weights()

    def _build_head(self, num_classes, dropout_p, use_bn) -> nn.Sequential:
        layers = [nn.Flatten()]

        # FC1 block (Force bias=True to match checkpoint)
        layers.append(nn.Linear(512 * 7 * 7, 4096, bias=True))
        if use_bn:
            layers.append(nn.BatchNorm1d(4096))
        layers += [nn.ReLU(inplace=True), CustomDropout(p=dropout_p)]

        # FC2 block (Force bias=True to match checkpoint)
        layers.append(nn.Linear(4096, 4096, bias=True))
        if use_bn:
            layers.append(nn.BatchNorm1d(4096))
        layers += [nn.ReLU(inplace=True), CustomDropout(p=dropout_p)]

        # Output — raw logits
        layers.append(nn.Linear(4096, num_classes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))   # [B, num_classes]
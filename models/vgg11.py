"""VGG11 Encoder — DA6401 Assignment 2

Architecture strictly follows Configuration A from the official VGG paper:
    Simonyan & Zisserman, "Very Deep Convolutional Networks for
    Large-Scale Image Recognition", ICLR 2015.
    https://arxiv.org/abs/1409.1556

Modifications:
    • BatchNorm2d inserted after EVERY Conv2d, BEFORE ReLU.
      Justification: BN normalises each layer's input distribution,
      eliminating internal covariate shift, stabilising gradient flow,
      allowing higher learning rates, and reducing dependence on careful
      weight initialisation — all without adding inference cost.
    • bias=False on all Conv2d layers (BN's learnable β subsumes the bias).

Feature map dimensions (fixed input 224×224 per VGG paper):
    block1 pre-pool : [B,  64, 224, 224]
    after pool1     : [B,  64, 112, 112]
    block2 pre-pool : [B, 128, 112, 112]
    after pool2     : [B, 128,  56,  56]
    block3 pre-pool : [B, 256,  56,  56]
    after pool3     : [B, 256,  28,  28]
    block4 pre-pool : [B, 512,  28,  28]
    after pool4     : [B, 512,  14,  14]
    block5 pre-pool : [B, 512,  14,  14]
    bottleneck      : [B, 512,   7,   7]  ← returned by forward()
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class VGG11Encoder(nn.Module):
    """
    VGG11 convolutional backbone — Simonyan & Zisserman (2014), Table 1 column A.

    VGG11 block structure (8 conv layers total):
        Block 1 : Conv(64)               → MaxPool/2
        Block 2 : Conv(128)              → MaxPool/2
        Block 3 : Conv(256) → Conv(256)  → MaxPool/2
        Block 4 : Conv(512) → Conv(512)  → MaxPool/2
        Block 5 : Conv(512) → Conv(512)  → MaxPool/2

    All Conv: kernel=3×3, padding=1.
    BN placement (our design): Conv → BN → ReLU  (standard VGG-BN).
    When use_bn=False: Conv → ReLU  (original VGG11 without BN).
    Output bottleneck: [B, 512, 7, 7]
    """

    def __init__(self, in_channels: int = 3, use_bn: bool = True):
        super().__init__()
        self.use_bn = use_bn

        self.block1 = self._make_block(in_channels, [64])
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2 = self._make_block(64, [128])
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3 = self._make_block(128, [256, 256])
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block4 = self._make_block(256, [512, 512])
        self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block5 = self._make_block(512, [512, 512])
        self.pool5  = nn.MaxPool2d(kernel_size=2, stride=2)

        self._init_weights()

    def _make_block(self, in_ch: int, out_channels: list) -> nn.Sequential:
        """Build one VGG conv block: [Conv → (BN) → ReLU] × len(out_channels)."""
        layers = []
        ch = in_ch
        for out_ch in out_channels:
            layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=1,
                                    bias=True))
            if self.use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            ch = out_ch
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Args:
            x               : [B, C, 224, 224]
            return_features : if True, also return pre-pool skip connections
                              for U-Net decoder.
        Returns:
            bottleneck [B, 512, 7, 7]          when return_features=False
            (bottleneck, feature_dict)          when return_features=True
                feature_dict keys: 'block1'..'block5' (pre-pool outputs)
        """
        f1 = self.block1(x)          # [B,  64, 224, 224]
        x  = self.pool1(f1)          # [B,  64, 112, 112]

        f2 = self.block2(x)          # [B, 128, 112, 112]
        x  = self.pool2(f2)          # [B, 128,  56,  56]

        f3 = self.block3(x)          # [B, 256,  56,  56]
        x  = self.pool3(f3)          # [B, 256,  28,  28]

        f4 = self.block4(x)          # [B, 512,  28,  28]
        x  = self.pool4(f4)          # [B, 512,  14,  14]

        f5 = self.block5(x)          # [B, 512,  14,  14]
        bn = self.pool5(f5)          # [B, 512,   7,   7]  ← bottleneck

        if return_features:
            return bn, {'block1': f1, 'block2': f2, 'block3': f3,
                        'block4': f4, 'block5': f5}
        return bn


# ── Autograder alias: `from models.vgg11 import VGG11` ───────────────────────
VGG11 = VGG11Encoder

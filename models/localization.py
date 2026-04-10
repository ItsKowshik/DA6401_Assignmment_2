"""
models/localization.py — Task 2: VGG11 Encoder + Regression Head
DA6401 Assignment 2

Output:
    Bounding boxes in pixel-space format:
    [x_center, y_center, width, height]
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.classification import VGG11Classifier
from models.layers import CustomDropout
from models.utils import _remap_vgg_state


class RegressionHead(nn.Module):
    def __init__(self, in_features: int, dropout_p: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class VGG11Localizer(nn.Module):
    def __init__(
        self,
        encoder_ckpt: str = "checkpoints/classifier.pth",
        freeze_encoder: bool = True,
        dropout_p: float = 0.3,
        num_classes: int = 37,
    ):
        super().__init__()

        base = VGG11Classifier(
            num_classes=num_classes,
            in_channels=3,
            dropout_p=0.5,
            use_bn=True,
        )
        self.encoder = base.encoder

        if encoder_ckpt is not None:
            ckpt = torch.load(encoder_ckpt, map_location="cpu")
            sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            sd = _remap_vgg_state(sd)

            enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
            if not enc_sd:
                enc_sd = sd

            self.encoder.load_state_dict(enc_sd, strict=False)
            print(
                f"  Loaded Task 1 encoder from epoch {ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'} "
                f"(val_acc={ckpt.get('best_metric', 0.0) if isinstance(ckpt, dict) else 0.0:.4f})"
            )
        else:
            print("  Encoder: weights to be loaded externally")

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("  Encoder: FROZEN (feature extractor mode)")
        else:
            print("  Encoder: FINE-TUNING (end-to-end mode)")

        self.freeze_encoder = freeze_encoder

        self.reg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=False),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=False),
            CustomDropout(p=dropout_p),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        features = self.encoder(x)
        scale = x.new_tensor([w, h, w, h])
        return self.reg_head(features) * scale

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def unfreeze_encoder_blocks(self, blocks: list):
        for name, param in self.encoder.named_parameters():
            for block in blocks:
                if name.startswith(block):
                    param.requires_grad = True
                    break
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Unfroze blocks {blocks} -> {n_trainable:,} trainable params")


if __name__ == "__main__":
    model = VGG11Localizer(
        encoder_ckpt="checkpoints/classifier.pth",
        freeze_encoder=True,
    )
    x = torch.randn(4, 3, 224, 224)
    boxes = model(x)
    print(f"Output shape : {boxes.shape}")
    print(f"Output range : [{boxes.min():.3f}, {boxes.max():.3f}]")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable    : {n_trainable:,} / {n_total:,}")
    print("Sanity check passed")
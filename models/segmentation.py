"""
models/segmentation.py — Task 3: U-Net Style Semantic Segmentation
DA6401 Assignment 2

Architecture:
  VGG11 Encoder (Task 1 backbone) → Bottleneck → Symmetric Decoder
  Skip connections: encoder feature maps concatenated at each decoder stage
  Upsampling: ConvTranspose2d ONLY (bilinear/interpolation strictly prohibited)
  Output: (N, 3, 224, 224) — 3-class logits (0=pet, 1=bg, 2=border)

Usage:
    model = VGG11UNet(encoder_ckpt="checkpoints/classifier.pth",
                      freeze_mode="frozen")  # "frozen" | "partial" | "full"
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.classification import VGG11Classifier
from models.layers import CustomDropout
from models.utils import _remap_vgg_state
# ── Decoder block: ConvTranspose2d upsample + conv refinement ─────────────────

class DecoderBlock(nn.Module):
    """
    One decoder stage:
      1. ConvTranspose2d to 2× upsample (learnable, not bilinear)
      2. Concatenate skip connection from encoder
      3. Two Conv2d layers to refine fused features
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()

        # Learnable upsample — doubles spatial resolution
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2
        )

        # After concat with skip: (in_channels//2 + skip_channels) → out_channels
        fused = in_channels // 2 + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(fused, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Handle spatial size mismatch (odd input dimensions)
        if x.shape != skip.shape:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="nearest"
            )

        x = torch.cat([x, skip], dim=1)   # channel concat along dim=1
        return self.conv(x)


# ── VGG11 Encoder with skip connection extraction ────────────────────────────

class VGG11Encoder(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        # Mirror VGG11Classifier.encoder's named attributes exactly
        # → state_dict keys stay encoder.block1.X, encoder.block2.X, etc.
        # → matches checkpoint keys directly after _remap_vgg_state
        self.block1 = backbone.block1
        self.block2 = backbone.block2
        self.block3 = backbone.block3
        self.block4 = backbone.block4
        self.block5 = backbone.block5

    def forward(self, x: torch.Tensor):
        skips = []
        for block in [self.block1, self.block2, self.block3,
                    self.block4, self.block5]:
            found_pool = False
            for layer in block:
                if isinstance(layer, nn.MaxPool2d):
                    skips.append(x)   # pre-pool snapshot
                    found_pool = True
                x = layer(x)

            # MaxPool not inside block — snapshot then pool manually
            if not found_pool:
                skips.append(x)
                x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        # skips[0]=64@224, [1]=128@112, [2]=256@56, [3]=512@28, [4]=512@14
        # x = bottleneck 512@7
        return x, skips

# ── Full U-Net model ──────────────────────────────────────────────────────────

class VGG11UNet(nn.Module):
    """
    U-Net style segmentation using VGG11 encoder from Task 1.

    Args:
        encoder_ckpt : path to classifier.pth from Task 1
        num_classes  : segmentation classes (3 for trimaps)
        freeze_mode  : "frozen"  — encoder fully frozen
                       "partial" — freeze blocks 1-3, unfreeze blocks 4-5
                       "full"    — entire network trainable
        num_classes_cls: number of Task 1 classifier output classes (for loading)
    """

    def __init__(
        self,
        encoder_ckpt:    str = "checkpoints_perfect/classifier.pth",
        num_classes:     int = 3,
        freeze_mode:     str = "full",
        num_classes_cls: int = 37,
    ):
        super().__init__()
        assert freeze_mode in ("frozen", "partial", "full"), \
            f"freeze_mode must be 'frozen'|'partial'|'full', got {freeze_mode}"

        self.freeze_mode = freeze_mode
        self.num_classes = num_classes

        # ── Load Task 1 encoder ───────────────────────────────────────────
        base = VGG11Classifier(
            num_classes=num_classes_cls,
            in_channels=3,
            dropout_p=0.5,
            use_bn=True,
        )
        if encoder_ckpt is not None:
            ckpt = torch.load(encoder_ckpt, map_location="cpu")
            sd   = _remap_vgg_state(ckpt["state_dict"])
            base.encoder.load_state_dict(sd, strict=False)   # <--- FIXED!
            print(f"  Loaded Task 1 encoder from epoch {ckpt.get('epoch', '?')} "
                f"(val_acc={ckpt.get('best_metric', 0):.4f})")
        else:
            print("  Encoder: weights to be loaded externally via multitask model")

        # ── Wrap encoder for skip connection extraction ────────────────────
        self.encoder = VGG11Encoder(base.encoder)

        # ── Apply freeze mode ─────────────────────────────────────────────
        self._apply_freeze(freeze_mode)

        # ── Symmetric decoder (mirrors VGG11 channel progression) ─────────
        #  Bottleneck: 512ch, 7×7
        #  DecoderBlock(in, skip, out):
        #    skip channels come from corresponding encoder block output
        def _dec_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = _dec_block(1024, 512)   # 512+512

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = _dec_block(768,  256)   # 256+512

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _dec_block(384,  128)   # 128+256

        self.up2 = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)
        self.dec2 = _dec_block(192,   64)   # 64+128

        self.up1 = nn.ConvTranspose2d(64,  64,  kernel_size=2, stride=2)
        self.dec1 = _dec_block(128,   64)   # 64+64

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _apply_freeze(self, freeze_mode: str):
        """Apply encoder freezing strategy."""
        if freeze_mode == "frozen":
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("  Encoder: FULLY FROZEN")

        elif freeze_mode == "partial":
            # Freeze blocks 0,1,2 (low-level features: edges, gradients)
            # Unfreeze blocks 3,4 (high-level semantic features)
            blocks_list = [self.encoder.block1, self.encoder.block2, self.encoder.block3, self.encoder.block4, self.encoder.block5]
            for i, block in enumerate(blocks_list):
                for p in block.parameters():
                    p.requires_grad = (i >= 3)
            n_frozen = sum(
                p.numel() for i, b in enumerate(blocks_list)
                for p in b.parameters() if i < 3
            )
            print(f"  Encoder: PARTIAL FINE-TUNE "
                  f"(blocks 0-2 frozen, blocks 3-4 trainable) "
                  f"| frozen params: {n_frozen:,}")


        else:  # full
            for p in self.encoder.parameters():
                p.requires_grad = True
            print("  Encoder: FULL FINE-TUNE")

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        # skips[4]=512@14, [3]=512@28, [2]=256@56, [1]=128@112, [0]=64@224

        d = torch.cat([self.up5(bottleneck), skips[4]], dim=1)
        d = self.dec5(d)

        d = torch.cat([self.up4(d), skips[3]], dim=1)
        d = self.dec4(d)

        d = torch.cat([self.up3(d), skips[2]], dim=1)
        d = self.dec3(d)

        d = torch.cat([self.up2(d), skips[1]], dim=1)
        d = self.dec2(d)

        d = torch.cat([self.up1(d), skips[0]], dim=1)
        d = self.dec1(d)

        return self.final_conv(d)    # (N, 3, 224, 224)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
    

if __name__ == "__main__":
    import torch
    for mode in ["frozen", "partial", "full"]:
        print(f"\n── freeze_mode={mode} ──")
        model = VGG11UNet(
            encoder_ckpt="checkpoints/classifier.pth",
            freeze_mode=mode,
        )
        x      = torch.randn(2, 3, 224, 224)
        out    = model(x)
        assert out.shape == (2, 3, 224, 224), \
            f"FAIL: expected (2,3,224,224), got {out.shape}"

        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"  Output : {out.shape} ✓")
        print(f"  Params : {n_train:,} trainable / {n_total:,} total")

    print("\nAll sanity checks passed ✅")

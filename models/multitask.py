"""Unified multi-task model"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.classification import VGG11Classifier
from models.localization   import VGG11Localizer
from models.segmentation   import VGG11UNet, VGG11Encoder

from models.utils import _remap_vgg_state

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path:  str = "checkpoints/localizer.pth",
        unet_path:       str = "checkpoints/unet.pth",
        num_breeds:      int = 37,
        seg_classes:     int = 3,
        in_channels:     int = 3,
        freeze_mode:     str = "frozen",   # "frozen" | "partial" | "full"
    ):

        super().__init__()
        import gdown
        gdown.download(id="1D99fZYGGKiYf3UA-cZbzlNEu0VwLPZKQ", output=classifier_path, quiet=False)
        gdown.download(id="1193Rw7OdbA67Ao78JzK2FoIjCZvPFnq2",  output=localizer_path,  quiet=False)
        gdown.download(id="17OLC6FKd6S9OHhCWOa9QKLHt7YLxe3qK",       output=unet_path,       quiet=False)
        
        # type-safety guards — argparse can pass strings
        num_breeds  = int(num_breeds)
        seg_classes = int(seg_classes)
        in_channels = int(in_channels)

        self.num_breeds  = num_breeds
        self.seg_classes = seg_classes

        # ── Step 1: Load Task 1 classifier → shared encoder ──────────────────
        classifier = VGG11Classifier(
            num_classes = num_breeds,
            in_channels = in_channels,
            use_bn      = True,
        )
        cls_ckpt = torch.load(classifier_path, map_location="cpu")
        cls_sd   = _remap_vgg_state(cls_ckpt["state_dict"])
        classifier.load_state_dict(cls_sd, strict=False)

        # ── Shared VGG11 encoder backbone (from Task 1) ───────────────────────
        self.encoder = VGG11Encoder(classifier.encoder)
        # Freeze shared encoder — Task 4 trains only the heads
        if freeze_mode == "frozen":
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("  Encoder: FROZEN")

        elif freeze_mode == "partial":
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.encoder.block4.parameters():
                p.requires_grad = True
            for p in self.encoder.block5.parameters():
                p.requires_grad = True

        elif freeze_mode == "full":
            for p in self.encoder.parameters():
                p.requires_grad = True
            print("  Encoder: FULL FINE-TUNE")

        # ── Head 1: Classification (Task 1 FC head) ───────────────────────────
        # Strip any Flatten already inside classifier.classifier to avoid double
        cls_layers = [m for m in classifier.classifier.children()
                      if not isinstance(m, nn.Flatten)]
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            *cls_layers,
        )

        # ── Head 2: Localization (Task 2 regression head) ─────────────────────
        localizer = VGG11Localizer(
            encoder_ckpt   = None,              
            freeze_encoder = True,
        )
        loc_ckpt = torch.load(localizer_path, map_location="cpu")
        loc_sd   = _remap_vgg_state(loc_ckpt["state_dict"])
        localizer.load_state_dict(loc_sd, strict=False)
        
        ckpt_head_keys = [k for k in loc_ckpt["state_dict"].keys() if "head" in k or "reg" in k]
        weight_keys = sorted([k for k in ckpt_head_keys if k.endswith("weight")])
        bias_keys   = sorted([k for k in ckpt_head_keys if k.endswith("bias")])
        
        # FIXED: Extract linear layers from the correct reg_head module
        model_linears = [m for m in localizer.reg_head if isinstance(m, nn.Linear)]
        
        if len(weight_keys) == 3 and len(bias_keys) == 3 and len(model_linears) == 3:
            with torch.no_grad():
                for i in range(3):
                    model_linears[i].weight.copy_(loc_ckpt["state_dict"][weight_keys[i]])
                    model_linears[i].bias.copy_(loc_ckpt["state_dict"][bias_keys[i]])
            print(" Successfully force-loaded trained regression head weights!")
        else:
            print(" Could not force-load head weights, sizes mismatch.")
            print(f"      Found {len(weight_keys)} weights, {len(bias_keys)} biases, {len(model_linears)} model layers.")

        self.loc_encoder = localizer.encoder 
        
        self.loc_head = localizer.reg_head

        # ── Head 3: Segmentation decoder (Task 3 U-Net decoder) ───────────────
        unet = VGG11UNet(
            encoder_ckpt = None,               
            num_classes  = seg_classes,
            freeze_mode  = "full",
        )
        unet_ckpt = torch.load(unet_path, map_location="cpu")
        unet_sd   = _remap_vgg_state(unet_ckpt["state_dict"])
        unet.load_state_dict(unet_sd, strict=False)
        self.up5        = unet.up5
        self.dec5       = unet.dec5
        self.up4        = unet.up4
        self.dec4       = unet.dec4
        self.up3        = unet.up3
        self.dec3       = unet.dec3
        self.up2        = unet.up2
        self.dec2       = unet.dec2
        self.up1        = unet.up1
        self.dec1       = unet.dec1
        self.final_conv = unet.final_conv
        print(f"  Loaded U-Net       (epoch={unet_ckpt.get('epoch','?')} "
              f"val_dice={unet_ckpt.get('best_dice', 0):.4f})")

    def forward(self, x: torch.Tensor):
        """Single forward pass → all three task outputs simultaneously.
        Args:
            x: (B, C, H, W)
        Returns:
            cls_logits:  (B, num_breeds)
            loc_out:     (B, 4)
            seg_logits:  (B, seg_classes, H, W)
        """
        # ── Shared encoder — run once, reused by all heads ────────────────────
        bottleneck, skips = self.encoder(x)
        # skips: [s1(64,112), s2(128,56), s3(256,28), s4(512,14)]

        # ── Head 1: Classification ────────────────────────────────────────────
        cls_logits = self.cls_head(bottleneck)           # (B, 37)

        # ── Head 2: Localization ──────────────────────────────────────────────
        loc_features = self.loc_encoder(x)

        normalized_coords = self.loc_head(loc_features)  # (B, 4) in [0, 1]
        
        # Scale to absolute image pixels [0, 224]
        loc_out = normalized_coords * 224.0

        # ── Head 3: Segmentation ──────────────────────────────────────────────
        d = torch.cat([self.up5(bottleneck), skips[4]], dim=1); d = self.dec5(d)
        d = torch.cat([self.up4(d),          skips[3]], dim=1); d = self.dec4(d)
        d = torch.cat([self.up3(d),          skips[2]], dim=1); d = self.dec3(d)
        d = torch.cat([self.up2(d),          skips[1]], dim=1); d = self.dec2(d)
        d = torch.cat([self.up1(d),          skips[0]], dim=1); d = self.dec1(d)
        seg_logits = self.final_conv(d)
        
        return {
            "classification": cls_logits,   # (B, 37)
            "localization":   loc_out,      # (B, 4)
            "segmentation":   seg_logits,   # (B, 3, 224, 224)
        }

if __name__ == "__main__":
    model = MultiTaskPerceptionModel(
        classifier_path = "checkpoints/classifier.pth",
        localizer_path  = "checkpoints/localizer.pth",
        unet_path       = "checkpoints/unet.pth",
    )
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    cls_out = out["classification"]
    loc_out = out["localization"]
    seg_out = out["segmentation"]

    assert cls_out.shape == (2, 37),           f"FAIL cls: {cls_out.shape}"
    assert loc_out.shape == (2, 4),            f"FAIL loc: {loc_out.shape}"
    assert seg_out.shape == (2, 3, 224, 224),  f"FAIL seg: {seg_out.shape}"

    n_total  = sum(p.numel() for p in model.parameters())
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"✓ classification : {cls_out.shape}")
    print(f"✓ localization   : {loc_out.shape}")
    print(f"✓ segmentation   : {seg_out.shape}")
    print(f"✓ Total params   : {n_total:,}  |  Frozen: {n_frozen:,}")
    print("\nAll sanity checks passed ")
# report_2_7.py
import os
import torch
import wandb
import requests
import numpy as np

from PIL import Image, ImageDraw
from io import BytesIO
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.multitask import MultiTaskPerceptionModel  # ← back to multitask
from data.pets_dataset import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


IMAGE_GROUPS = [
    [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg",
        "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?auto=format&fit=crop&w=1200&q=80",
    ],
    [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg",
        "https://images.unsplash.com/photo-1511044568932-338cba0ad803?auto=format&fit=crop&w=1200&q=80",
    ],
    [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Dog_Breeds.jpg/1200px-Dog_Breeds.jpg",
        "https://images.unsplash.com/photo-1548199973-03cce0bbc87b?auto=format&fit=crop&w=1200&q=80",
    ],
]

BREED_NAMES = [
    "Abyssinian","Bengal","Birman","Bombay","British_Shorthair",
    "Egyptian_Mau","Maine_Coon","Persian","Ragdoll","Russian_Blue",
    "Siamese","Sphynx","american_bulldog","american_pit_bull_terrier",
    "basset_hound","beagle","boxer","chihuahua","english_cocker_spaniel",
    "english_setter","german_shorthaired","great_pyrenees","havanese",
    "japanese_chin","keeshond","leonberger","miniature_pinscher",
    "newfoundland","pomeranian","pug","saint_bernard","samoyed",
    "scottish_terrier","shiba_inu","staffordshire_bull_terrier",
    "wheaten_terrier","yorkshire_terrier",
]

MASK_CLASS_LABELS = {0: "pet", 1: "background", 2: "border"}
MASK_COLORS       = {0: (0, 200, 0), 1: (0, 0, 0), 2: (255, 165, 0)}

REQUEST_HEADERS = {
    "User-Agent": "DA6401-A2-Report/1.0 (student project; contact: local-script)",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model loading ──────────────────────────────────────────────────────────────
model = MultiTaskPerceptionModel(
    num_breeds=37,
    seg_classes=3,
    classifier_path="checkpoints_perfect/classifier.pth",
    localizer_path="checkpoints_perfect/localizer.pth",
    unet_path="checkpoints_perfect/unet.pth",
).to(device)
model.eval()

# ── Transform (albumentations — matches training pipeline exactly) ─────────────
tfm = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def cxcywh_px_to_xyxy(box):
    cx, cy, w, h = box
    x1 = max(0, int(cx - w / 2))
    y1 = max(0, int(cy - h / 2))
    x2 = min(IMAGE_SIZE, int(cx + w / 2))
    y2 = min(IMAGE_SIZE, int(cy + h / 2))
    return x1, y1, x2, y2


def colorize_trimap(mask_idx):
    rgb = np.zeros((mask_idx.shape[0], mask_idx.shape[1], 3), dtype=np.uint8)
    for cls_id, color in MASK_COLORS.items():
        rgb[mask_idx == cls_id] = color
    return Image.fromarray(rgb)


def fetch_image_with_fallback(url_group):
    session = requests.Session()
    for url in url_group:
        try:
            resp = session.get(url, headers=REQUEST_HEADERS, timeout=15)
            resp.raise_for_status()
            if "image" not in resp.headers.get("Content-Type", ""):
                continue
            return Image.open(BytesIO(resp.content)).convert("RGB"), url
        except Exception:
            continue
    raise RuntimeError(f"Failed to download any image from: {url_group}")


# ── W&B ────────────────────────────────────────────────────────────────────────
run = wandb.init(
    project="DL_A2",
    group="Task_2_7",
    name="report_2_7",
    config={"image_size": IMAGE_SIZE, "n_images": len(IMAGE_GROUPS)},
)

table = wandb.Table(columns=[
    "original", "bbox_overlay", "segmentation_overlay", "seg_mask",
    "predicted_breed", "confidence", "source_url", "analysis",
])

for i, url_group in enumerate(IMAGE_GROUPS):
    print(f"Processing image {i+1}/{len(IMAGE_GROUPS)} ...")

    pil_raw, used_url = fetch_image_with_fallback(url_group)
    img_np = np.array(pil_raw.resize((IMAGE_SIZE, IMAGE_SIZE)))

    inp = tfm(image=img_np)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)

    cls_o = out["classification"]
    loc_o = out["localization"]
    seg_o = out["segmentation"]

    # Direct output — no conditional scaling hack needed now that weights load correctly
    box        = loc_o[0].cpu().numpy()
    probs      = cls_o.softmax(dim=1)[0].cpu()
    pred_cls   = probs.argmax().item()
    confidence = probs.max().item()
    breed_name = BREED_NAMES[pred_cls] if pred_cls < len(BREED_NAMES) else str(pred_cls)

    x1, y1, x2, y2 = cxcywh_px_to_xyxy(box)
    print(f"  bbox: {box.tolist()} → ({x1},{y1},{x2},{y2})")

    pil_orig = Image.fromarray(img_np)
    pil_bbox = pil_orig.copy()
    draw = ImageDraw.Draw(pil_bbox)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.text((x1 + 4, y1 + 2), f"{breed_name} ({confidence:.2f})", fill="red")

    mask_idx     = seg_o[0].detach().cpu().argmax(dim=0).numpy().astype(np.uint8)
    pil_seg_mask = colorize_trimap(mask_idx)

    seg_overlay = wandb.Image(
        img_np,
        masks={"prediction": {"mask_data": mask_idx, "class_labels": MASK_CLASS_LABELS}},
        caption=f"{breed_name} | conf={confidence:.2f}",
    )

    box_area_pct = ((max(0, x2-x1) * max(0, y2-y1)) / float(IMAGE_SIZE * IMAGE_SIZE)) * 100.0
    fg_pct       = float((mask_idx == 0).mean() * 100.0)
    border_pct   = float((mask_idx == 2).mean() * 100.0)

    analysis = (
        f"Predicted: {breed_name} ({confidence * 100:.1f}% confidence). "
        f"Bounding box covers ~{box_area_pct:.1f}% of the image. "
        f"Predicted pet pixels: {fg_pct:.1f}%. "
        f"Predicted border pixels: {border_pct:.1f}%."
    )

    table.add_data(
        wandb.Image(pil_orig), wandb.Image(pil_bbox),
        seg_overlay, wandb.Image(pil_seg_mask),
        breed_name, round(confidence, 4), used_url, analysis,
    )
    print(f"  -> {breed_name} ({confidence * 100:.1f}%)")

run.log({"task_2_7_pipeline_showcase": table})
run.finish()
print("\nSection 2.7 logged to W&B")

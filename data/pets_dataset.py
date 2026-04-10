"""Oxford-IIIT Pet multi-task dataset — DA6401 Assignment 2"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import torch


IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(split: str) -> A.Compose:
    """
    Get data augmentation transforms for the given split.
    For training, we apply a variety of augmentations to improve generalization.
    For validation and testing, we only resize and normalize the images.

    Args:
        split (str): One of "train", "val", or "test".
    Returns:
        A.Compose: An albumentations Compose object with the appropriate transforms.
    """
    if split == "train":
        return A.Compose(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1,
                    p=0.5,
                ),
                A.Rotate(limit=15, p=0.3),
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(8, 32),
                    hole_width_range=(8, 32),
                    fill=0,
                    p=0.3,
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["bbox_labels"],
                min_visibility=0.1,
            ),
        )
    else:
        return A.Compose(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["bbox_labels"],
                min_visibility=0.1,
            ),
        )


def _load_bbox_xml(xml_path: str) -> Optional[List[float]]:
    """
    Load bounding box from the given XML file in COCO format.
    If the file does not exist, is malformed, or does not contain a valid bounding box, return None.
    Args:
        xml_path (str): Path to the XML file containing bounding box annotations.
    Returns:
        Optional[List[float]]: A list [xc, yc, w, h] representing the bounding box in COCO format,
        or None if the bounding box cannot be loaded.

    """
    if not os.path.exists(xml_path):
        return None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj = root.find("object")
        if obj is None:
            return None
        bndbox = obj.find("bndbox")
        if bndbox is None:
            return None

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        if xmax <= xmin or ymax <= ymin:
            return None

        xc = (xmin + xmax) / 2.0
        yc = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin
        return [xc, yc, w, h]
    except Exception:
        return None


class OxfordIIITPetDataset(Dataset):
    """
    PyTorch Dataset for the Oxford-IIIT Pet multi-task dataset.
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        assert split in ("train", "val", "test")

        self.root = Path(root)
        self.split = split
        self.transform = transform or get_transforms("train" if split == "train" else "val")

        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "annotations" / "trimaps"
        self.xmls_dir = self.root / "annotations" / "xmls"

        name_to_class: Dict[str, int] = {}
        with open(self.root / "annotations" / "list.txt") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                name_to_class[parts[0]] = int(parts[1]) - 1

        split_file = self.root / "annotations" / ("trainval.txt" if split in ("train", "val") else "test.txt")
        all_names: List[str] = []
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                name = line.split()[0]
                if name in name_to_class:
                    all_names.append(name)

        if split in ("train", "val"):
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(all_names)).tolist()
            n_val = max(1, int(len(all_names) * val_fraction))
            val_set = set(indices[:n_val])
            if split == "val":
                all_names = [all_names[i] for i in range(len(all_names)) if i in val_set]
            else:
                all_names = [all_names[i] for i in range(len(all_names)) if i not in val_set]

        self.samples: List[Dict] = []
        for name in all_names:
            img_path = self.images_dir / f"{name}.jpg"
            if not img_path.exists():
                continue
            self.samples.append(
                {
                    "image_path": str(img_path),
                    "mask_path": str(self.masks_dir / f"{name}.png"),
                    "xml_path": str(self.xmls_dir / f"{name}.xml"),
                    "label": name_to_class[name],
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]

        image = np.array(Image.open(s["image_path"]).convert("RGB"))
        orig_h, orig_w = image.shape[:2]

        mask_path = s["mask_path"]
        if os.path.exists(mask_path):
            raw_mask = np.array(Image.open(mask_path))
            mask = np.clip(raw_mask.astype(np.int32) - 1, 0, 2).astype(np.uint8)
        else:
            mask = np.ones((orig_h, orig_w), dtype=np.uint8)

        raw_bbox = _load_bbox_xml(s["xml_path"])
        has_bbox = raw_bbox is not None

        if has_bbox:
            xc, yc, bw, bh = raw_bbox
            x1 = max(0.0, xc - bw / 2.0)
            y1 = max(0.0, yc - bh / 2.0)
            coco_box = [x1, y1, bw, bh]
        else:
            coco_box = [0.0, 0.0, float(orig_w), float(orig_h)]

        out = self.transform(
            image=image,
            mask=mask,
            bboxes=[coco_box],
            bbox_labels=[0],
        )

        t_image = out["image"]
        t_mask = out["mask"].long()
        t_bboxes = out["bboxes"]

        if has_bbox and len(t_bboxes) == 0:
            has_bbox = False

        if has_bbox and len(t_bboxes) > 0:
            x1t, y1t, bwt, bht = t_bboxes[0]
            xct = x1t + bwt / 2.0
            yct = y1t + bht / 2.0
            bbox_tensor = torch.tensor([xct, yct, bwt, bht], dtype=torch.float32)
        else:
            bbox_tensor = torch.tensor(
                [IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0, float(IMAGE_SIZE), float(IMAGE_SIZE)],
                dtype=torch.float32,
            )

        return {
            "image": t_image,
            "label": torch.tensor(s["label"], dtype=torch.long),
            "bbox": bbox_tensor,
            "mask": t_mask,
            "has_bbox": torch.tensor(has_bbox, dtype=torch.bool),
            "stem": Path(s["image_path"]).stem,
        }

    def get_class_names(self) -> List[str]:
        idx_to_breed: Dict[int, str] = {}
        with open(self.root / "annotations" / "list.txt") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                img_name = parts[0]
                class_id = int(parts[1])
                breed = " ".join(img_name.split("_")[:-1])
                idx_to_breed[class_id - 1] = breed
        return [idx_to_breed[i] for i in range(37)]

    def __repr__(self) -> str:
        return f"OxfordIIITPetDataset(split='{self.split}', n_samples={len(self)}, image_size={IMAGE_SIZE})"


def get_dataloaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for the Oxford-IIIT Pet multi-task dataset.
    This function initializes the OxfordIIITPetDataset for each split (train, val, test
) and creates corresponding DataLoaders.
    Args:
        root (str): Root directory of the dataset.
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of worker processes for data loading.
        val_fraction (float): Fraction of training data to use for validation.
        seed (int): Random seed for reproducibility when splitting train/val.

    Returns:
        Dict[str, DataLoader]: A dictionary containing DataLoaders for "train", "val", and "test" splits.
    """
    datasets = {
        split: OxfordIIITPetDataset(
            root=root,
            split=split,
            val_fraction=val_fraction,
            seed=seed,
        )
        for split in ("train", "val", "test")
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return loaders
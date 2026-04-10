# DA6401 Assignment 2 — Multi-Task Pet Perception with VGG11

## WandB Report
**https://api.wandb.ai/links/k-indian-institute-of-technology-madras/8n0x53jy**

---

A multi-task deep learning pipeline built on VGG11 for the Oxford-IIIT Pet dataset, covering:
- **Task 1** — 37-class breed classification
- **Task 2** — Single-object bounding box localization
- **Task 3** — Trimap segmentation (foreground / background / boundary)
- **Task 4** — Unified multi-task model combining all three heads

All experiments are tracked with [Weights & Biases](https://wandb.ai).

---

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset Setup](#dataset-setup)
- [Environment Setup](#environment-setup)
- [Training](#training)
  - [Task 1 — Classification](#task-1--classification)
  - [Task 2 — Localization](#task-2--localization)
  - [Task 3 — Segmentation](#task-3--segmentation)
  - [Task 4 — Multi-Task](#task-4--multi-task)
- [Experiments (W&B)](#experiments-wb)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [File Reference](#file-reference)
- [Common Errors & Fixes](#common-errors--fixes)

---

## Project Structure
```text
A2/
├── data/
│ └── oxford-iiit-pet/
│ ├── images/ # .jpg pet images
│ ├── annotations/
│ │ ├── list.txt # image stems + class labels
│ │ ├── trimaps/ # .png trimap masks (values 1/2/3)
│ │ └── xmls/ # .xml bounding box annotations
├── models/
│ ├── vgg11.py # VGG11Encoder backbone
│ ├── classification.py # VGG11Classifier head
│ ├── localization.py # VGG11Localizer regression head
│ ├── segmentation.py # VGG11UNet decoder
│ ├── multitask.py # MultiTaskPerceptionModel (all heads)
│ └── layers.py # CustomDropout
├── data/
│ └── pets_dataset.py # OxfordIIITPetDataset + dataloaders
├── losses/
│ ├── iou_loss.py # IoULoss
│ └── segmentation_loss.py # CE + Dice combined loss
├── checkpoints/ # Saved model weights (auto-created)
├── train_classification.py # Task 1 training script
├── train_localization.py # Task 2 training script
├── train_segmentation.py # Task 3 training script
├── train.py # Unified script for all tasks
├── visualize_2_1.sh #All WandB experiments
├── visualize_2_2.sh
├── visualize_2_3.sh
├── visualize_2_4.py
├── visualize_2_5.py
├── visualize_2_6.py
├── visualize_2_7.py
└── README.md
```
---

## Dataset Setup

Download the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/):

```bash
mkdir -p data/oxford-iiit-pet
cd data/oxford-iiit-pet

# Images
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
tar -xf images.tar.gz

# Annotations (masks + xmls + list.txt)
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xf annotations.tar.gz

cd ../..
```

Expected layour afer extraction

```text
data/oxford-iiit-pet/
├── images/          # ~7000 .jpg files
└── annotations/
    ├── list.txt
    ├── trimaps/     # ~7000 .png files
    └── xmls/        # ~3600 .xml files
```

---
## Environment Setup

```bash
# Create and activate conda environment
conda create -n da6401-a2 python=3.11
conda activate da6401-a2

# Install PyTorch (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install wandb albumentations scikit-learn scipy numpy Pillow

# Login to W&B
wandb login
```
---

## Training
#### Task 1 Classification

```python
python train_classification.py \
  --root data/oxford-iiit-pet \
  --epochs 30 \
  --batch_size 32 \
  --lr 0.01 \
  --optimizer sgd \
  --dropout_p 0.5 \
  --run_name Standard_Run
```
#### Section 2.1 BatchNorm Effect

```python
# With BN (default)
python train_classification.py --root data/oxford-iiit-pet --experiment bn_effect --run_name cls_bn

# Without BN
python train_classification.py --root data/oxford-iiit-pet --experiment bn_effect --no_bn --run_name cls_nobn

```
#### Section 2.2 Dropout Effect

```python
python train_classification.py --root data/oxford-iiit-pet --experiment dropout_effect --dropout_p 0.0 --run_name cls_nodropout
python train_classification.py --root data/oxford-iiit-pet --experiment dropout_effect --dropout_p 0.2 --run_name cls_dropout0.2
python train_classification.py --root data/oxford-iiit-pet --experiment dropout_effect --dropout_p 0.5 --run_name cls_dropout0.5

```

### Task 2 Localization
#### Training
Fine-tune full network (recommended):

```python
python train_localization.py \
  --root data/oxford-iiit-pet \
  --encoder_ckpt checkpoints/classifier.pth \
  --epochs 30 \
  --lr 0.001 \
  --batch_size 32
```
Frozen encoder (feature extractor only):

```python
python train_localization.py \
  --root data/oxford-iiit-pet \
  --encoder_ckpt checkpoints/classifier.pth \
  --epochs 30 \
  --lr 0.001 \
  --freeze_encoder \
  --batch_size 32

```
### Task 3 — Segmentation
#### Training
```python
python train_segmentation.py \
  --root data/oxford-iiit-pet \
  --encoder_ckpt checkpoints/classifier.pth \
  --epochs 30 \
  --lr 0.001 \
  --batch_size 32
```
#### Multiple Strategies Training
```python
python train_segmentation.py --root data/oxford-iiit-pet --epochs 30 --batch_size 32 --lr 0.01 --encoder_ckpt checkpoints_perfect/classifier.pth --freeze_mode frozen --run_name "Seg_Strict_Frozen" --val_frac 0.05
python train_segmentation.py --root data/oxford-iiit-pet --epochs 30 --batch_size 32 --lr 0.01 --encoder_ckpt checkpoints_perfect/classifier.pth --freeze_mode partial --run_name "Seg_Partial_Finetune" --val_frac 0.05
python train_segmentation.py --root data/oxford-iiit-pet --epochs 30 --batch_size 32 --lr 0.01 --encoder_ckpt checkpoints_perfect/classifier.pth --freeze_mode full --run_name "Full_Finetune" --val_frac 0.05
```
### Task 4 - Multitask
```bash
python train.py
```

---

## WandB Experiments
```bash
./viusalize_2_1.sh
./visualize_2_2.sh
./visualize_2_3.sh
```
```python
python visualize_2_4.py
python visualize_2_5.py
python visualize_2_6.py
python visualize_2_7.py
```

---
## Model Architecture

### VGG11 Encoder (Shared Backbone)
- 5 convolutional blocks with MaxPool  
- BatchNorm after each `Conv2d`  
- CustomDropout for regularization  
- **Output:** `[B, 512, 7, 7]` feature map  

### Classification Head (Task 1)
```
Flatten  
→ Linear(25088, 4096) → BatchNorm → ReLU → Dropout  
→ Linear(4096, 4096) → BatchNorm → ReLU → Dropout  
→ Linear(4096, 37)
```
- **Output:** 37 breed classes  

###  Localization Head (Task 2)
```
AdaptiveAvgPool2d(4,4)  
→ Flatten  
→ Linear(8192, 1024) → ReLU → Dropout  
→ Linear(1024, 256) → ReLU → Dropout  
→ Linear(256, 4) → Sigmoid
```
- **Output:** `[B, 4]` → normalized `(cx, cy, w, h)` in `[0, 1]`  

### Segmentation Head — U-Net Decoder (Task 3)
- 5 transposed convolution upsampling blocks  
- Skip connections from encoder  
- **Output:** `[B, 3, H, W]` logits for 3-class trimap  

###  Multi-Task Model (Task 4)
- Shared encoder for **classification + segmentation**  
- Separate localization encoder (independent copy)  
- All three heads run in a **single forward pass**

---

## Common Errors & Fixes

### CUBLAS_STATUS_ALLOC_FAILED
GPU out of memory. Reduce batch size:

```bash
# Add to your command
--batch_size 16
```

Or add this at the top of your script:

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

### W&B step warning (step is less than current step)
Caused by mixing explicit `step=` in per-batch logs with auto-step in per-epoch logs.

Fix per-batch logging:

```python
wandb.log({"train/batch_loss": loss.item()}, commit=False)
```

### IoU = 0.00 during localization training
The model outputs normalized `[0,1]` boxes but the dataset returned pixel-space `[0,224]` boxes.

Ensure `pets_dataset.py` normalizes bounding boxes:

```python
bbox_tensor = bbox_tensor / float(IMAGE_SIZE)  # normalize to [0, 1]
```
### train_localization.py --freeze_encoder gives loss ~48, IoU ~0.00
The frozen classification encoder has no spatial sensitivity for localization.

Use full fine-tuning (remove `--freeze_encoder`). This is expected behavior.


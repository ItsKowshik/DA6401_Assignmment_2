#!/bin/bash

set -e

echo "Starting Experiment 2.3"

python train_segmentation.py --root data/oxford-iiit-pet --epochs 30 --batch_size 32 --lr 0.01 --encoder_ckpt checkpoints_perfect/classifier.pth --freeze_mode frozen --run_name "Seg_Strict_Frozen" --val_frac 0.05
python train_segmentation.py --root data/oxford-iiit-pet --epochs 30 --batch_size 32 --lr 0.01 --encoder_ckpt checkpoints_perfect/classifier.pth --freeze_mode partial --run_name "Seg_Partial_Finetune" --val_frac 0.05
python train_segmentation.py --root data/oxford-iiit-pet --epochs 30 --batch_size 32 --lr 0.01 --encoder_ckpt checkpoints_perfect/classifier.pth --freeze_mode full --run_name "Full_Finetune" --val_frac 0.05

echo "Experiment 2.3 finished successfully!"
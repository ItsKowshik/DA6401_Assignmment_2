#!/bin/bash

set -e

echo "Training localization model"
python train_localization.py --root data/oxford-iiit-pet --run_name localizer_finetune --val_frac 0.05 --lr 0.0001 --encoder_ckpt checkpoints_perfect/classifier.pth
python train_localization.py --root data/oxford-iiit-pet --run_name localizer_finetune --val_frac 0.05 --lr 0.0005 --encoder_ckpt checkpoints_perfect/classifier.pth
python train_localization.py --root data/oxford-iiit-pet --run_name localizer_finetune --val_frac 0.05 --lr 0.001 --encoder_ckpt checkpoints_perfect/classifier.pth
echo "Localization model trained successfully!"
#!/bin/bash

set -e

echo "Starting Experiment 2.1"
python train_classification.py --root data/oxford-iiit-pet --epochs 30 --batch_size 32 --lr 0.1  --project DL_A2 --run_name "Cls_No_BatchNorm" --experiment bn_effect --val_frac 0.05 --no_bn
python train_classification.py --root data/oxford-iiit-pet --epochs 30 --batch_size 32 --lr 0.1  --project DL_A2 --run_name "Cls_BatchNorm" --experiment bn_effect --val_frac 0.05
echo "Experiment 2.1 finished successfully!"
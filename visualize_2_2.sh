#!/bin/bash

set -e

echo "Starting Experiment 2.2"
python train_classification.py --root data/oxford-iiit-pet --epochs 30 --batch_size 32 --lr 0.01  --project DL_A2 --run_name "Dropout p_0.0" --experiment dropout_effect --val_frac 0.05 --dropout_p 0.0
python train_classification.py --root data/oxford-iiit-pet --epochs 30 --batch_size 32 --lr 0.01  --project DL_A2 --run_name "Dropout p_0.2" --experiment dropout_effect --val_frac 0.05 --dropout_p 0.2
python train_classification.py --root data/oxford-iiit-pet --epochs 30 --batch_size 32 --lr 0.01  --project DL_A2 --run_name "Dropout p_0.5" --experiment dropout_effect --val_frac 0.05 --dropout_p 0.5
echo "Experiment 2.2 finished successfully!"
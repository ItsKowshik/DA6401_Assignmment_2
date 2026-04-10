#!/bin/bash

set -e

echo "Training classification model"
python train_classification.py --root data/oxford-iiit-pet --epochs 30 --batch_size 32 --lr 0.01  --project DL_A2 --run_name "Standard Run" --experiment standard --val_fraction 0.05
python train_classification.py --root data/oxford-iiit-pet --epochs 30 --batch_size 32 --lr 0.001  --project DL_A2 --run_name "Standard Run" --experiment standard --val_fraction 0.05
python train_classification.py --root data/oxford-iiit-pet --epochs 30 --batch_size 32 --lr 0.0001  --project DL_A2 --run_name "Standard Run" --experiment standard --val_fraction 0.05

echo "Classification model trained successfully!"
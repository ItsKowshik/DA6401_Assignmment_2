#!/bin/bash

set -e

echo "Starting experiment 2.8"
python train.py
python train.py --lr 0.0005
python train.py --batch_size 64
echo "Experiment 2.8 completed"
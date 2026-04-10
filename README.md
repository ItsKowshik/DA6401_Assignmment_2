# DA6401 Assignment-2 Skeleton Guide

This repository is an instructional skeleton for building the complete visual perception pipeline on Oxford-IIIT Pet.


### ADDITIONAL INSTRUCTIONS FOR ASSIGNMENT2:
- Ensure VGG11 is implemented according to the official paper(https://arxiv.org/abs/1409.1556). The only difference being injecting BatchNorm and CustomDropout layers is your design choice.
- Train all the networks on normalized images as input (as the test set given by autograder will be normalized images).
- The output of Localization model = [x_center, y_center, width, height] all these numbers are with respect to image coordinates, in pixel space (not normalized)
- Train the object localization network with the following loss function: MSE + custom_IOU_loss.
- Make sure the custom_IOU loss is in range: [0,1]
- In the custom IOU loss, you have to implement all the two reduction types: ["mean", "sum"] and the default reduction type should be "mean". You may include any other reduction type as well, which will help your network learn better.
- multitask.py shd load the saved checkpoints (classifier.pth, localizer.pth, unet.pth), initialize the shared backbone and heads with these trained weights and do prediction.
- Keep paths as relative paths for loading in multitask.py
- Assume input image size is fixed according to vgg11 paper(can be hardcoded need not pass as args)
- Stick to the arguments of the functions and classes given in the github repo, if you include any additional arguments make sure they always have some default value.
- Do not import any other python packages apart from the ones mentioned in assignment pdf, if you do so the autograder will instantly crash and your submission will not be evaluated.
- The following classes will be used by autograder: 
    ```
        from models.vgg11 import VGG11
        from models.layers import CustomDropout
        from losses.iou_loss import IoULoss
        from multitask import MultiTaskPerceptionModel
    ```
- The submission link for this assignment will be available by Saturday(04/04/2026) on gradescope





### GENERAL INSTRUCTIONS:
- From this assignment onwards, if we find any wandb report which is private/inaccessible while grading, there wont be any second chance, that submission will be marked 0 for wandb marks.
- The entireity of plots presented in the wandb report should be interactive and logged in the wandb project. Any screenshot or images of plots will straightly be marked 0 for that question.
- Gradescope offers an option to activate whichever submission you want to, and that submission will be used for evaluation. Under any circumstances, no requests to be raised to TAs to activate any of your prior submissions. It is the student's responsibility to do so(if required) before submission deadline.
- Assignment2 discussion forum has been opened on moodle for any doubt clarification/discussion.   




## Contact

For questions or issues, please contact the teaching staff or post on the course forum.

---

Good luck with your implementation!

Task 1
python train.py   --root data/oxford-iiit-pet   --experiment standard   --run_name classifier_with_bn   --epochs 80   --lr 0.01   --weight_decay 1e-3   --val_fraction 0.05  --batch_size 32



python visualize_2_4.py   --root data/oxford-iiit-pet   --checkpoint checkpoints/classifier.pth   --max_channels 16

Task 2
python train_localization.py   --root data/oxford-iiit-pet   --run_name localizer_finetune   --epochs 30   --batch_size 32   --lr 1e-4 --val_fraction 0.05

python train_localization.py   --root data/oxford-iiit-pet   --freeze_encoder   --run_name localizer_frozen   --epochs 30   --batch_size 32   --lr 1e-3 --val_fraction 0.05

python visualize_2_5.py \
    --root data/oxford-iiit-pet \
    --localizer_ckpt checkpoints/localizer.pth \
    --n_samples 20

Task 3
# Run 1 — Frozen encoder
python train_segmentation.py \
    --root data/oxford-iiit-pet \
    --freeze_mode frozen \
    --run_name seg_frozen \
    --lr 1e-3 \
    --epochs 30

# Run 2 — Partial fine-tune (after run 1 finishes)
python train_segmentation.py \
    --root data/oxford-iiit-pet \
    --freeze_mode partial \
    --run_name seg_partial \
    --lr 1e-4 \
    --epochs 30

# Run 3 — Full fine-tune (after run 2 finishes)
python train_segmentation.py \
    --root data/oxford-iiit-pet \
    --freeze_mode full \
    --run_name seg_finetune \
    --lr 1e-4 \
    --epochs 30



Task 1 — VGG11 Classification
bash
# Train (3 dropout experiments for section 2.2)
python classifier.py --group "task1-classification" --run_name "cls-no-dropout"    --dropout 0.0
python classifier.py --group "task1-classification" --run_name "cls-dropout-0.2"  --dropout 0.2
python classifier.py --group "task1-classification" --run_name "cls-dropout-0.5"  --dropout 0.5

# Sanity check
python models/classification.py
Task 2 — Object Localization
bash
# Train (frozen vs fine-tune for section 2.3 reference)
python localizer.py --group "task2-localization" --run_name "loc-frozen"   --freeze_encoder
python localizer.py --group "task2-localization" --run_name "loc-finetune" --no-freeze_encoder

# Sanity check
python models/localization.py
Task 3 — U-Net Segmentation
bash
# Train 3 freeze modes (section 2.3)
python segmenter.py --group "task3-segmentation" --run_name "seg-frozen"  --freeze_mode frozen
python segmenter.py --group "task3-segmentation" --run_name "seg-partial" --freeze_mode partial
python segmenter.py --group "task3-segmentation" --run_name "seg-full"    --freeze_mode full

# Copy best checkpoint for Task 4
cp checkpoints/segmenter_full.pth checkpoints/unet.pth

# Sanity check
python models/segmentation.py
Task 4 — Multi-Task Pipeline
bash
# Sanity check model only
python models/multitask.py

# Train run 1 (balanced lambdas)
python multitask.py \
    --group "task4-multitask" --run_name "multitask-lam1-1-1" \
    --lambda_cls 1.0 --lambda_loc 1.0 --lambda_seg 1.0

# Train run 2 (loc-heavy)
python multitask.py \
    --group "task4-multitask" --run_name "multitask-lam1-2-1" \
    --lambda_cls 1.0 --lambda_loc 2.0 --lambda_seg 1.0
W&B Report Scripts
bash
# Section 2.5 — BBox prediction table
python report_2_5.py

# Section 2.6 — Segmentation sample images
python report_2_6.py

# Section 2.7 — Novel images pipeline showcase
python report_2_7.py
Checkpoint Verification
bash
# Verify all checkpoints exist and are non-empty
ls -lh checkpoints/classifier.pth \
        checkpoints/localizer.pth  \
        checkpoints/unet.pth       \
        checkpoints/multitask_best.pth

# Inspect any checkpoint metadata
python -c "
import torch
for f in ['classifier','localizer','unet','multitask_best']:
    ckpt = torch.load(f'checkpoints/{f}.pth', map_location='cpu')
    print(f, {k:v for k,v in ckpt.items() if k!='state_dict' and k!='optimizer'})
"
Quick Sanity — All Models
bash
python models/classification.py
python models/localization.py
python models/segmentation.py
python models/multitask.py
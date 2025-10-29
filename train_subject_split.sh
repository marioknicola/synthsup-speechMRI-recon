#!/bin/bash
# Training script for U-Net with subject-based split
# 
# Train/Val/Test Split:
# - Train: Subject0023, Subject0024, Subject0025, Subject0026, Subject0027
# - Val: Subject0022
# - Test: Subject0021

cd "/Users/marioknicola/MSc Project/synthsup-speechMRI-recon"

python3 train_unet_subject_split.py \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --output-dir ../outputs_subject_split \
    --epochs 200 \
    --batch-size 4 \
    --lr 1e-4 \
    --base-filters 32 \
    --loss-alpha 0.7 \
    --num-workers 4 \
    --save-freq 10

# Notes:
# - Training on Subjects 0023, 0024, 0025, 0026, 0027
# - Validation on Subject 0022
# - Test set (Subject 0021) reserved for final evaluation
# - Using base_filters=32 for same model capacity as before
# - Loss: 70% L2 + 30% SSIM (alpha=0.7)
# - Checkpoints saved every 10 epochs
# - Best model saved based on validation loss

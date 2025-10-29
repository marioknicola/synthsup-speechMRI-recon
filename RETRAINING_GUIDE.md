# Retraining Setup - Subject-Based Split

## Data Summary

Your processed data is ready for training:

### Dataset Split
- **Train**: Subject0025, Subject0026, Subject0027 (33 files)
- **Validation**: Subject0023 (11 files)  
- **Test**: Subject0024 (12 files)

### Data Locations
- Input (undersampled): `Synth_LR_nii/` - 56 files
- Target (fully-sampled): `HR_nii/` - 56 files

## Files Created

### 1. Training Script: `train_unet_subject_split.py`
Main training script with subject-based data splitting. Key features:
- Automatically splits data by subject ID extracted from filenames
- Same U-Net architecture as before (configurable base_filters)
- Combined L2 + SSIM loss (70% L2, 30% SSIM by default)
- TensorBoard logging
- Automatic checkpoint saving
- Learning rate scheduling with ReduceLROnPlateau

### 2. Data Verification: `check_data_ready.py`
Verifies your data is properly organized and counts files per subject.
Run this anytime to check data status:
```bash
python check_data_ready.py
```

### 3. Training Command: `train_subject_split.sh`
Ready-to-run bash script with recommended hyperparameters.

## How to Start Training

### Option 1: Using the shell script (recommended)
```bash
cd "/Users/marioknicola/MSc Project/synthsup-speechMRI-recon"
./train_subject_split.sh
```

### Option 2: Direct Python command
```bash
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
```

### Option 3: Resume from checkpoint
```bash
python3 train_unet_subject_split.py \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --output-dir ../outputs_subject_split \
    --epochs 200 \
    --resume ../outputs_subject_split/checkpoints/best_model.pth
```

## Training Configuration

### Recommended Hyperparameters
- **Epochs**: 200 (or more if needed)
- **Batch size**: 4 (adjust based on GPU memory)
- **Learning rate**: 1e-4
- **Base filters**: 32 (same as previous model)
- **Loss weights**: 70% L2 + 30% SSIM (alpha=0.7)
- **Optimizer**: Adam with weight decay 1e-5
- **LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=10)

### Model Architecture
- U-Net with bilinear upsampling
- Same architecture as your previous training
- Configurable depth via `--base-filters` parameter

## Output Structure

Training will create:
```
outputs_subject_split/
├── checkpoints/
│   ├── best_model.pth              # Best model based on val loss
│   ├── checkpoint_epoch_10.pth     # Regular checkpoints
│   ├── checkpoint_epoch_20.pth
│   └── ...
├── logs/                           # TensorBoard logs
└── dataset_split.txt               # Subject split info
```

## Monitoring Training

### TensorBoard (recommended)
```bash
tensorboard --logdir ../outputs_subject_split/logs
```
Then open http://localhost:6006 in your browser

### Console Output
Training progress will show:
- Epoch number and progress bars
- Train/Val loss per epoch
- Current learning rate
- Time per epoch
- Best model saves

## Key Differences from Previous Training

1. **Subject-based split** instead of random 80/10/10 split
   - Ensures no data leakage between subjects
   - More realistic evaluation (tests generalization to new subjects)

2. **Explicit train/val/test designation**
   - Train: Subjects 0025, 0026, 0027
   - Val: Subject 0023
   - Test: Subject 0024 (for final evaluation after training)

3. **Split information saved**
   - `dataset_split.txt` contains exact indices used
   - Reproducible and traceable

## After Training

### Evaluate on Test Set (Subject 0024)
After training completes, you can evaluate on the held-out test set:
```bash
python evaluate_test_set.py \
    --model-path ../outputs_subject_split/checkpoints/best_model.pth \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --test-subject 0024
```

### Compare with Previous Model
You can compare this new model (trained on subjects 0025-0027) with your previous model to see if the subject-based split provides better generalization.

## Troubleshooting

### If training is too slow
- Reduce `--num-workers` (try 2 or 0)
- Increase `--batch-size` if you have more GPU memory
- Train on GPU/Colab instead of local machine

### If GPU out of memory
- Reduce `--batch-size` (try 2 or 1)
- Reduce `--base-filters` (try 16 instead of 32)

### If loss doesn't decrease
- Check data loading with `check_data_ready.py`
- Verify normalization is working (check console output)
- Try adjusting `--loss-alpha` (more L2 weight)
- Reduce learning rate to 5e-5

## Next Steps

1. ✅ Data is verified and ready
2. ⏳ Run training with `./train_subject_split.sh`
3. ⏳ Monitor training with TensorBoard
4. ⏳ Evaluate best model on test set (Subject 0024)
5. ⏳ Compare with previous model performance
6. ⏳ Run inference on Dynamic_SENSE data if needed

---
**Status**: Ready to train! 🚀

# Cross-Validation Training for U-Net MRI Super-Resolution

## Overview

This directory contains scripts for **subject-level cross-validation** training, designed for proper model evaluation across subjects.

## Files

- **`train_cross_validation.py`** - Main training script with explicit train/test/val split definition
- **`colab_cross_validation.ipynb`** - Google Colab notebook with easy configuration
- **`utils/cv_config_generator.py`** - Helper to generate common CV split strategies

## Quick Start (Colab)

1. Open `colab_cross_validation.ipynb` in Google Colab
2. Configure your CV strategy (Option A, B, or C)
3. Set training parameters
4. Run all cells

## Quick Start (Local/Terminal)

### Example 1: Leave-One-Out CV (No Validation Set)

**Recommended for final model evaluation**

```bash
# Fold 1: Test on Subject 0026
python train_cross_validation.py \
    --train-subjects 0023 0024 0025 \
    --test-subjects 0026 \
    --fold-name fold_test0026

# Fold 2: Test on Subject 0025
python train_cross_validation.py \
    --train-subjects 0023 0024 0026 \
    --test-subjects 0025 \
    --fold-name fold_test0025

# Continue for all subjects...
```

### Example 2: 3-Fold CV

```bash
# Fold 1
python train_cross_validation.py \
    --train-subjects 0023 0024 \
    --test-subjects 0025 0026 \
    --fold-name fold1

# Fold 2
python train_cross_validation.py \
    --train-subjects 0025 0026 \
    --test-subjects 0021 0022 \
    --fold-name fold2

# Fold 3
python train_cross_validation.py \
    --train-subjects 0021 0022 \
    --test-subjects 0023 0024 \
    --fold-name fold3
```

### Example 3: With Validation Set (for hyperparameter tuning)

```bash
python train_cross_validation.py \
    --train-subjects 0023 0024 0025 \
    --val-subjects 0022 \
    --test-subjects 0021 \
    --fold-name fold_val0022_test0021
```

## When to Use Validation Set?

### ❌ Don't use validation set for:
- **Final model evaluation** - Use all data for training/testing
- **Limited subjects** - Every subject should contribute to training or testing
- **Reporting final performance** - Avoid information leakage

### ✅ Use validation set for:
- **Hyperparameter tuning** - Before final evaluation
- **Early stopping** - Prevent overfitting during development
- **Architecture search** - Comparing different models

## Arguments

### Required:
- `--train-subjects`: Space-separated list of training subject IDs
- `--test-subjects`: Space-separated list of test subject IDs
- `--fold-name`: Name for this fold (e.g., 'fold1', 'fold_test0026')

### Optional:
- `--val-subjects`: Validation subject IDs (optional)
- `--input-dir`: Input (LR) images directory (default: `../Synth_LR_unpadded_nii`)
- `--target-dir`: Target (HR) images directory (default: `../Dynamic_SENSE_padded`)
- `--output-dir`: Output directory (default: `cross_validation_results`)
- `--epochs`: Number of epochs (default: 200)
- `--batch-size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 1e-5)
- `--base-filters`: Base filters in U-Net (default: 32)
- `--loss-alpha`: MSE weight in combined loss (default: 0.7)

## Output Structure

```
cross_validation_results/
├── fold1/
│   ├── config.json              # Configuration for this fold
│   ├── training_history.json    # Loss/metrics per epoch
│   ├── fold1_best.pth          # Best model checkpoint
│   └── fold1_latest.pth        # Latest checkpoint
├── fold2/
│   └── ...
└── fold3/
    └── ...
```

## Generate CV Configurations

Use the helper script to generate common CV strategies:

```bash
python utils/cv_config_generator.py
```

This generates:
- `cv_leave_one_out.json` - Leave-one-out configuration
- `cv_3fold.json` - 3-fold configuration
- Training commands for each fold

## Cross-Validation Strategies

### 1. Leave-One-Out CV (LOOCV)
- **When:** Limited subjects (4-6)
- **Folds:** N folds (N = number of subjects)
- **Each fold:** Train on N-1, test on 1
- **Pros:** Maximum training data per fold
- **Cons:** Computationally expensive

### 2. K-Fold CV
- **When:** More subjects available (>6)
- **Folds:** K folds (typically 3-5)
- **Each fold:** Train on (K-1)/K data, test on 1/K
- **Pros:** Faster than LOOCV, still robust
- **Cons:** Less training data per fold

### 3. Stratified CV with Validation
- **When:** Hyperparameter tuning needed
- **Splits:** Train / Val / Test per fold
- **Use:** Development phase only
- **Final evaluation:** Remove validation set

## Best Practices

### For Final Model Evaluation:
1. ✅ Use all subjects in either training or testing
2. ✅ Use leave-one-out or k-fold WITHOUT validation set
3. ✅ Train all folds to completion
4. ✅ Report average ± std across folds
5. ❌ Don't use validation set (wastes data)

### For Hyperparameter Tuning:
1. ✅ Include validation set in each fold
2. ✅ Use validation loss for early stopping
3. ✅ Select hyperparameters based on validation performance
4. ✅ After tuning, retrain with full CV (no validation) for final eval

### For Reporting:
```
Example: Leave-one-out CV with 4 subjects
- Fold 1: Train on [0023,0024,0025], Test on [0026]
- Fold 2: Train on [0023,0024,0026], Test on [0025]
- Fold 3: Train on [0023,0025,0026], Test on [0024]
- Fold 4: Train on [0024,0025,0026], Test on [0023]

Results: PSNR = 28.5 ± 1.2 dB, SSIM = 0.73 ± 0.02
```

## Comparison with train_unet_subject_split.py

| Feature | train_unet_subject_split.py | train_cross_validation.py |
|---------|---------------------------|--------------------------|
| Purpose | Single train/val/test split | Multiple CV folds |
| Use case | Quick training, development | Final evaluation |
| Subject definition | Hardcoded in script | Command-line arguments |
| Validation set | Always present | Optional |
| Output | Single model | One model per fold |
| Flexibility | Low | High |

## Common Issues

### "No samples found for subject X"
- Check subject ID format (e.g., '0023' not '23')
- Verify files exist with pattern `Subject00XX_*.nii`

### Out of Memory
- Reduce `--batch-size` (try 2 or 1)
- Reduce `--base-filters` (try 16)

### Training too slow
- Use GPU (Colab: Runtime → Change runtime type → GPU)
- Reduce `--epochs` for testing
- Use fewer subjects initially

## Example Workflow

1. **Development phase:**
   ```bash
   # Quick test with validation set
   python train_cross_validation.py \
       --train-subjects 0023 0024 \
       --val-subjects 0025 \
       --test-subjects 0026 \
       --fold-name test_run \
       --epochs 10
   ```

2. **Full evaluation:**
   ```bash
   # Run all leave-one-out folds
   for test_subj in 0023 0024 0025 0026; do
       train_subjs=$(echo "0023 0024 0025 0026" | sed "s/$test_subj//")
       python train_cross_validation.py \
           --train-subjects $train_subjs \
           --test-subjects $test_subj \
           --fold-name fold_test$test_subj \
           --epochs 200
   done
   ```

3. **Evaluate results:**
   ```bash
   # Run inference on each fold's test set
   # Compute metrics
   # Average across folds
   ```

## Support

For issues or questions:
1. Check this README
2. Review example notebook: `colab_cross_validation.ipynb`
3. Check original training script: `train_unet_subject_split.py`

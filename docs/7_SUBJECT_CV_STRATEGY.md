# 7-Subject Cross-Validation Strategy

## Overview

For your MSc abstract with 7 subjects, we've implemented a **6-fold CV + 1 held-out subject** strategy.

## Workflow

### 1. Hold Out 1 Subject (e.g., Subject 0021)
- This subject is **completely excluded** from all training and model selection
- It will only be used for the **final test** after selecting the best model
- Ensures unbiased final performance estimate

### 2. 6-Fold Cross-Validation on Remaining 6 Subjects
- Each fold:
  - **Train on 5 subjects**
  - **Test on 1 subject** (the held-out fold)
- Produces 6 independent models with 6 test performance estimates
- Example for subjects 0022-0027:
  ```
  Fold 1: Train=[0023,0024,0025,0026,0027], Test=[0022]
  Fold 2: Train=[0022,0024,0025,0026,0027], Test=[0023]
  Fold 3: Train=[0022,0023,0025,0026,0027], Test=[0024]
  Fold 4: Train=[0022,0023,0024,0026,0027], Test=[0025]
  Fold 5: Train=[0022,0023,0024,0025,0027], Test=[0026]
  Fold 6: Train=[0022,0023,0024,0025,0026], Test=[0027]
  ```

### 3. Select Best Model
After all 6 folds complete:
1. Run inference on each fold's test subject
2. Collect test metrics (PSNR, SSIM) for all 6 folds
3. Select best fold using one of these criteria:
   - **Option A:** Highest average test PSNR
   - **Option B:** Highest average test SSIM
   - **Option C:** Best average rank across both metrics
   - **Option D:** Lowest test loss (less recommended)

### 4. Final Test on Held-Out Subject
- Take the best model from Step 3
- Run inference on Subject 0021 (never seen before)
- This gives an unbiased estimate of generalization performance

## Implementation

### Files Created
1. **`train_cross_validation.py`** - Main training script
   - Takes `--train-subjects`, `--test-subjects`, `--fold-name` as arguments
   - No hardcoded subject lists
   - Flexible for any CV configuration

2. **`colab_cross_validation.ipynb`** - Colab notebook
   - Automated 6-fold generation
   - ZIP file upload for data
   - Inference and evaluation cells
   - Reporting guidance

3. **`utils/cv_config_generator.py`** - Helper scripts
   - Functions to generate CV configurations
   - Can create bash scripts for local training

## Reporting for Abstract

### Report These Metrics:

1. **Cross-Validation Performance (6 subjects)**
   ```
   PSNR: 28.5 ± 1.2 dB (mean ± SD across 6 folds)
   SSIM: 0.89 ± 0.03 (mean ± SD across 6 folds)
   ```

2. **Held-Out Performance (1 subject)**
   ```
   PSNR: 27.8 dB on held-out subject
   SSIM: 0.87 on held-out subject
   ```

### Methods Section Template:

> "We trained U-Net models using 6-fold cross-validation on 6 subjects (n=5 for training, n=1 for testing per fold). The best-performing model was selected based on average cross-validation test PSNR/SSIM and evaluated on a held-out subject that was never used during training or model selection. This approach ensures unbiased performance estimation while maximizing use of limited data."

## Why This Approach?

### ✅ Advantages
1. **Unbiased evaluation** - Held-out subject never seen during CV
2. **Robust model selection** - 6 folds provide stable performance estimate
3. **Maximizes training data** - Each fold uses 5/6 subjects
4. **Standard practice** - Well-accepted for limited subject counts
5. **Good for abstracts** - Clear methodology, defensible choices

### ❌ Alternative (Not Recommended)
**Simple train/test split (e.g., 5 train, 2 test):**
- Only 1 model trained
- Less robust performance estimate
- Arbitrary split affects results
- No model selection process

## Usage

### In Colab:
1. Upload `Synth_LR_nii.zip` and `HR_nii.zip`
2. Configure subjects in cell 3:
   ```python
   ALL_SUBJECTS = ['0021', '0022', '0023', '0024', '0025', '0026', '0027']
   HELD_OUT_SUBJECT = '0021'
   ```
3. Run all cells to train 6 folds
4. Check test inference metrics in cell 7
5. Update `BEST_FOLD_NAME` in cell 8 based on results
6. Run final test on held-out subject

### Locally:
```bash
# Generate configuration
python utils/cv_config_generator.py \
    --subjects 0022 0023 0024 0025 0026 0027 \
    --held-out 0021 \
    --strategy 6fold \
    --output configs/

# Train all folds (can run in parallel if multiple GPUs)
bash configs/train_all_folds.sh

# Evaluate and select best model
python utils/evaluate_cv_folds.py --results-dir results/cross_validation
```

## Timeline Estimate (Colab with T4 GPU)

- **Per fold:** ~2-3 hours (200 epochs)
- **All 6 folds:** ~12-18 hours (if run sequentially)
- **Inference:** ~5-10 minutes per fold
- **Total:** ~13-19 hours

**Tip:** Use Colab Pro for longer runtimes and better GPUs (A100 = ~2x faster)

## Troubleshooting

### Colab Disconnects
Run in browser console (F12):
```javascript
setInterval(() => {
    document.querySelector("colab-connect-button").click()
}, 60000)
```

### Out of Memory
- Reduce `batch_size` from 4 to 2
- Clear cache between folds: `torch.cuda.empty_cache()`
- Monitor: `!nvidia-smi`

### Which Fold to Select?
1. Run inference on all 6 test sets first
2. Check `metrics_summary.json` in each fold's `test_inference/` directory
3. Select fold with highest average PSNR or SSIM
4. If metrics are close, check training curves for stability

## References

This strategy follows best practices from:
- Varoquaux et al. (2017) "Assessing and tuning brain decoders"
- Kohavi (1995) "A study of cross-validation and bootstrap"
- Standard ML practice for small datasets (~10 subjects or fewer)

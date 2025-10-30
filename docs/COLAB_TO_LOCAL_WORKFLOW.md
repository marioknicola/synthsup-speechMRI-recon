# Train in Colab → Evaluate Locally Workflow

Complete guide for training models in Google Colab and running inference/evaluation on your local machine.

## Overview

**Why this workflow?**
- ✅ Train on Colab's free GPU (faster than local)
- ✅ Evaluate locally with your full dataset
- ✅ Keep large data files on your machine
- ✅ Only download trained models (~30-50 MB each)

---

## Part 1: Training in Colab

### 1.1 Prepare Your Data

Package your data as ZIP files for faster upload:

```bash
# On your local machine
cd "/Users/marioknicola/MSc Project"

# Create ZIP files
zip -r Synth_LR_nii.zip Synth_LR_nii/
zip -r HR_nii.zip HR_nii/

# Check sizes
ls -lh *.zip
```

### 1.2 Open Colab Notebook

1. Upload `colab_cross_validation.ipynb` to Google Colab
2. Enable GPU: **Runtime → Change runtime type → GPU (T4)**
3. Run all cells in order

### 1.3 Upload Data

When prompted in **Section 2**, upload your ZIP files:
- `Synth_LR_nii.zip` (~500 MB)
- `HR_nii.zip` (~200 MB)

They will be extracted to `/content/data/`

### 1.4 Configure CV Strategy

In **Section 3**, set your subjects:

```python
ALL_SUBJECTS = ['0021', '0022', '0023', '0024', '0025', '0026', '0027']
HELD_OUT_SUBJECT = '0021'  # For final testing
```

This creates 6-fold CV on subjects 0022-0027.

### 1.5 Start Training

**Section 5** trains all 6 folds sequentially (~12-18 hours total).

**⚠️ Prevent disconnection:**
Open browser console (F12) and run:
```javascript
setInterval(() => {
    document.querySelector("colab-connect-button").click()
}, 60000)
```

### 1.6 Monitor Progress

Each fold saves:
- `fold_name_best.pth` - Best model checkpoint (~30-50 MB)
- `config.json` - Training configuration
- `training_history.json` - Loss/metrics per epoch

Files are saved to Google Drive:
```
/content/drive/MyDrive/MSc_Project/cross_validation_results/
├── fold1/
│   ├── fold1_best.pth
│   ├── config.json
│   └── training_history.json
├── fold2/
│   └── ...
└── ...
```

**Note:** Only best models are saved to conserve storage!

---

## Part 2: Download Models to Local

### 2.1 Get Instructions

```bash
cd "/Users/marioknicola/MSc Project/synthsup-speechMRI-recon"

python utils/download_colab_models.py --output-dir ./cv_models
```

This prints detailed download instructions.

### 2.2 Download via Google Drive

**Option A: Web Interface**
1. Go to https://drive.google.com/
2. Navigate to: `MyDrive → MSc_Project → cross_validation_results`
3. Download all fold directories
4. Extract to `./cv_models/`

**Option B: Google Drive Desktop** (if installed)
```bash
cp -r ~/Google\ Drive/My\ Drive/MSc_Project/cross_validation_results/* ./cv_models/
```

**Option C: Using the script (if Drive Desktop available)**
```bash
python utils/download_colab_models.py --output-dir ./cv_models
# Follow prompts to auto-copy
```

### 2.3 Verify Downloads

```bash
ls -lh cv_models/*/fold*_best.pth
```

Expected output:
```
cv_models/fold1/fold1_best.pth
cv_models/fold2/fold2_best.pth
...
```

---

## Part 3: Run Inference Locally

### 3.1 Single Fold Inference

Test one fold on its test subject:

```bash
python inference_unet.py \
    --checkpoint ./cv_models/fold1/fold1_best.pth \
    --input-dir "../Synth_LR_nii" \
    --target-dir "../HR_nii" \
    --output-dir ./inference_results/fold1 \
    --file-pattern "*Subject0022*.nii" \
    --compute-metrics \
    --visualize
```

### 3.2 Batch Evaluation (All Folds)

Evaluate all folds automatically:

```bash
python utils/evaluate_all_folds.py \
    --models-dir ./cv_models \
    --input-dir "../Synth_LR_nii" \
    --target-dir "../HR_nii" \
    --output-dir ./evaluation_results
```

This script:
- ✅ Reads each fold's config to get test subjects
- ✅ Runs inference automatically
- ✅ Computes metrics (PSNR, SSIM, MSE)
- ✅ Prints summary table
- ✅ Recommends best fold

**Output:**
```
================================================================================
CROSS-VALIDATION RESULTS SUMMARY
================================================================================

Fold       Test Subject(s)     PSNR (dB)       SSIM            MSE            
--------------------------------------------------------------------------------
fold1      0022                28.45 ± 0.00    0.8912 ± 0.0000 0.001432
fold2      0023                29.12 ± 0.00    0.9034 ± 0.0000 0.001225
fold3      0024                27.89 ± 0.00    0.8745 ± 0.0000 0.001628
...
--------------------------------------------------------------------------------
OVERALL    (across folds)      28.52 ± 0.51    0.8897 ± 0.0144 0.001435
================================================================================

BEST MODELS
================================================================================
Highest PSNR: fold2 (29.12 dB)
Highest SSIM: fold2 (0.9034)
Lowest MSE:   fold2 (0.001225)

🎯 RECOMMENDED: fold2 (best PSNR)
================================================================================
```

### 3.3 Evaluate on Held-Out Subject

After selecting best fold, test on held-out subject (0021):

```bash
BEST_FOLD=fold2  # Based on evaluation results

python inference_unet.py \
    --checkpoint ./cv_models/${BEST_FOLD}/${BEST_FOLD}_best.pth \
    --input-dir "../Synth_LR_nii" \
    --target-dir "../HR_nii" \
    --output-dir ./inference_results/heldout_test \
    --file-pattern "*Subject0021*.nii" \
    --compute-metrics \
    --visualize
```

---

## Part 4: Analyze Results

### 4.1 Check Metrics

```bash
# CV fold results
cat evaluation_results/cv_summary.json

# Held-out test results
cat inference_results/heldout_test/metrics_summary.json
```

### 4.2 View Visualizations

```bash
open inference_results/heldout_test/comparisons/
```

Each comparison shows:
- Input (LR)
- U-Net output (SR)
- Target (HR)
- Metrics

### 4.3 Training History

Check training curves:

```python
import json
import matplotlib.pyplot as plt

# Load history
with open('cv_models/fold2/training_history.json', 'r') as f:
    history = json.load(f)

# Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(history['train_ssim'], label='Train')
plt.plot(history['val_ssim'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.legend()
plt.title('SSIM')

plt.tight_layout()
plt.savefig('training_curves.png')
```

---

## Part 5: Reporting (For Abstract)

### 5.1 Cross-Validation Performance

Report mean ± std across all folds:

```
PSNR: 28.52 ± 0.51 dB (6-fold CV)
SSIM: 0.8897 ± 0.0144 (6-fold CV)
```

### 5.2 Held-Out Performance

Report best model on held-out subject:

```
PSNR: 27.89 dB (held-out subject 0021)
SSIM: 0.8756 (held-out subject 0021)
```

### 5.3 Methods Template

> "We trained U-Net models using 6-fold cross-validation on 6 subjects (0022-0027), with each fold using 5 subjects for training and 1 for testing. Models were trained for 200 epochs with Adam optimizer (learning rate: 1×10⁻⁵) using combined MSE and SSIM loss (α=0.7). The best-performing fold was selected based on test set PSNR and evaluated on a held-out subject (0021) that was never used during training or model selection. All training was performed on Google Colab with NVIDIA T4 GPU."

---

## Troubleshooting

### Colab Issues

**Problem:** Session disconnects during training
**Solution:** Run console script (see Section 1.5) or use Colab Pro

**Problem:** Out of memory
**Solution:** Reduce batch size from 4 to 2 in Section 4:
```python
TRAINING_CONFIG = {
    'batch_size': 2,  # Reduced from 4
    ...
}
```

**Problem:** Training too slow
**Solution:** 
- Upgrade to Colab Pro for A100 GPU (~2x faster)
- Reduce epochs from 200 to 100 for initial testing

### Download Issues

**Problem:** Google Drive storage full
**Solution:** 
- Each model is only ~30-50 MB (not GB)
- Delete old Colab outputs: `/content/drive/MyDrive/Colab Notebooks/`

**Problem:** Can't find models in Drive
**Solution:**
- Check: `MyDrive/MSc_Project/cross_validation_results/`
- Verify Colab mounted Drive correctly in Section 1

### Local Inference Issues

**Problem:** `ModuleNotFoundError`
**Solution:**
```bash
pip install -r requirements.txt
```

**Problem:** `FileNotFoundError` for data
**Solution:**
- Verify paths: `ls ../Synth_LR_nii`
- Update paths in commands

**Problem:** `RuntimeError: CUDA out of memory`
**Solution:**
- You're running on CPU locally (normal)
- Will be slower but works fine
- Or reduce batch size in inference script

---

## File Organization

Recommended directory structure:

```
MSc Project/
├── synthsup-speechMRI-recon/          # Code repository
│   ├── train_cross_validation.py      # Training script
│   ├── inference_unet.py              # Inference script
│   ├── colab_cross_validation.ipynb   # Colab notebook
│   ├── utils/
│   │   ├── download_colab_models.py   # Download helper
│   │   └── evaluate_all_folds.py      # Batch evaluation
│   ├── cv_models/                     # Downloaded models
│   │   ├── fold1/
│   │   ├── fold2/
│   │   └── ...
│   ├── evaluation_results/            # Evaluation outputs
│   └── inference_results/             # Inference outputs
│       ├── fold1/
│       └── heldout_test/
├── Synth_LR_nii/                      # Input data
├── HR_nii/                            # Target data
├── Synth_LR_nii.zip                   # For Colab upload
└── HR_nii.zip                         # For Colab upload
```

---

## Quick Reference

### Train in Colab
1. Upload ZIP files
2. Configure subjects
3. Run all cells
4. Wait 12-18 hours

### Download Models
```bash
python utils/download_colab_models.py --output-dir ./cv_models
# Follow instructions
```

### Evaluate All Folds
```bash
python utils/evaluate_all_folds.py \
    --models-dir ./cv_models \
    --input-dir "../Synth_LR_nii" \
    --target-dir "../HR_nii" \
    --output-dir ./evaluation_results
```

### Test Best Model on Held-Out
```bash
python inference_unet.py \
    --checkpoint ./cv_models/fold2/fold2_best.pth \
    --input-dir "../Synth_LR_nii" \
    --target-dir "../HR_nii" \
    --output-dir ./inference_results/heldout_test \
    --file-pattern "*Subject0021*.nii" \
    --compute-metrics --visualize
```

---

## Need Help?

Check these files:
- Training setup: `docs/7_SUBJECT_CV_STRATEGY.md`
- CV details: `docs/CROSS_VALIDATION.md`
- Model architecture: `unet_model.py`
- Dataset format: `dataset.py`

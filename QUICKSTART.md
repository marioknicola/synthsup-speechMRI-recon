# Quick Start: Train in Colab → Evaluate Locally

## 🚀 In 5 Minutes

### Step 1: Prepare Data (2 min)
```bash
cd "/Users/marioknicola/MSc Project"
zip -r Synth_LR_nii.zip Synth_LR_nii/
zip -r HR_nii.zip HR_nii/
```

### Step 2: Train in Colab (12-18 hours)
1. Open `colab_cross_validation.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Upload ZIP files when prompted
4. Set held-out subject (default: 0021)
5. Run all cells
6. Wait for training to complete

### Step 3: Download Models (5 min)
```bash
cd synthsup-speechMRI-recon
python utils/download_colab_models.py --output-dir ./cv_models
# Follow instructions to download from Google Drive
```

### Step 4: Evaluate All Folds (30 min)
```bash
python utils/evaluate_all_folds.py \
    --models-dir ./cv_models \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --output-dir ./evaluation_results
```

### Step 5: Test Best Model on Held-Out (5 min)
```bash
# Use fold recommended by evaluate_all_folds.py
python inference_unet.py \
    --checkpoint ./cv_models/fold2/fold2_best.pth \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --output-dir ./inference_results/heldout \
    --file-pattern "*Subject0021*.nii" \
    --compute-metrics --visualize
```

## 📊 Results

**Cross-validation (6 folds):**
```bash
cat evaluation_results/cv_summary.json
```

**Held-out test:**
```bash
cat inference_results/heldout/metrics_summary.json
```

**Visualizations:**
```bash
open inference_results/heldout/comparisons/
```

## 📝 For Your Abstract

Report:
- **CV performance:** Mean ± SD from `cv_summary.json`
- **Held-out performance:** Metrics from held-out test
- **Methods:** See template in `docs/COLAB_TO_LOCAL_WORKFLOW.md`

## 🆘 Need Help?

- **Full guide:** `docs/COLAB_TO_LOCAL_WORKFLOW.md`
- **CV strategy:** `docs/7_SUBJECT_CV_STRATEGY.md`
- **Troubleshooting:** See workflow guide Part 6

## ✅ Checklist

- [ ] Created ZIP files for upload
- [ ] Trained all 6 folds in Colab
- [ ] Downloaded models to `./cv_models/`
- [ ] Ran batch evaluation
- [ ] Tested best model on held-out subject
- [ ] Collected metrics for abstract

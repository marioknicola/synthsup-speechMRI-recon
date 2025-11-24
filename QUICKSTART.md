# Quick Start: Train in Colab → Evaluate Locally

## 🚀 In 5 Steps

### Step 1: Prepare Data (2 min)
```bash
cd "/Users/marioknicola/MSc Project"
zip -r Synth_LR_nii.zip Synth_LR_nii/
zip -r HR_nii.zip HR_nii/
```

### Step 2: Train in Colab (12-18 hours)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marioknicola/synthsup-speechMRI-recon/blob/main/colab_cross_validation.ipynb)

1. Click badge above to open notebook
2. Enable GPU: Runtime → Change runtime type → GPU (T4)
3. Upload ZIP files when prompted
4. Run all cells
5. Wait for training (~12-18 hours)

### Step 3: Download Results (5 min)
```bash
# After training completes in Colab, download and extract:
cd "/Users/marioknicola/MSc Project/synthsup-speechMRI-recon"
unzip ~/Downloads/cross_validation_results.zip -d ./cv_models
```

### Step 4: Evaluate All Folds (30 min)
```bash
python evaluate_cv_sliding_folds.py \
    --models-dir ./cv_models \
    --input-dir ../Synth_LR_unpadded_nii \
    --target-dir ../Dynamic_SENSE_padded \
    --output-dir ./evaluation_results
```

### Step 5: Test Best Model (5 min)
```bash
# Use fold recommended by evaluation
python inference_test_subject.py \
    --checkpoint ./cv_models/fold_2/checkpoints/best_model.pth \
    --input-file ../Synth_LR_unpadded_nii/Subject0021_aa.nii \
    --output-dir ./inference_results/best_fold
```

---

## 📊 View Results

```bash
# CV summary
cat evaluation_results/cv_summary.json

# Best model metrics
cat inference_results/best_fold/metrics_summary.json

# Visualizations
open inference_results/best_fold/comparisons/
```

---

## 📖 Need More Details?

- **Complete workflow:** [`docs/COLAB_TO_LOCAL_WORKFLOW.md`](docs/COLAB_TO_LOCAL_WORKFLOW.md)
- **CV strategy:** [`docs/7_SUBJECT_CV_STRATEGY.md`](docs/7_SUBJECT_CV_STRATEGY.md)
- **Inference guide:** [`docs/INFERENCE.md`](docs/INFERENCE.md)
- **Main README:** [`README.md`](README.md)

---

**Last Updated:** November 24, 2025

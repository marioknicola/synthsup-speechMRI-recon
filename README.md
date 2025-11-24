# Synthetically Supervised Deep Learning for Dynamic Speech MRI Reconstruction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marioknicola/synthsup-speechMRI-recon/blob/main/colab_cross_validation.ipynb)

A comprehensive toolkit for accelerated MRI reconstruction combining classical SENSE algorithms with deep learning (U-Net) for dynamic speech MRI imaging.

---

## 🎯 Overview

This repository provides:

- **Classical SENSE Reconstruction:** Generalized reconstruction for 22-coil undersampled k-space data
- **Deep Learning Pipeline:** U-Net training with synthetic supervision for image quality enhancement
- **Cross-Validation Framework:** 5-fold and 7-subject CV strategies with automatic evaluation
- **Data Utilities:** Synthetic undersampling, normalization, metrics (PSNR/SSIM), format conversion

**Key Features:**
- 📊 Automatic train/val/test splitting with reproducible seeds
- 🔬 Lightweight U-Net baseline (~7.8M parameters)
- 📈 Built-in TensorBoard logging and metric tracking
- 🌐 Google Colab integration for free GPU training
- ⚡ Normalized data pipeline (all outputs in [0,1] range)

---

## 🚀 Quick Start

### Option 1: Train in Google Colab (Recommended)

**Best for:** Fast training with free GPU, no local setup required

1. **Click the badge:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marioknicola/synthsup-speechMRI-recon/blob/main/colab_cross_validation.ipynb)

2. **Prepare your data:**
   ```bash
   # On your local machine
   cd "/Users/marioknicola/MSc Project"
   zip -r Synth_LR_nii.zip Synth_LR_nii/
   zip -r HR_nii.zip HR_nii/
   ```

3. **In Colab:**
   - Enable GPU: Runtime → Change runtime type → GPU
   - Upload ZIP files when prompted
   - Run all cells
   - Download trained models after completion

4. **Evaluate locally:**
   ```bash
   # Extract downloaded results
   unzip ~/Downloads/cross_validation_results.zip -d ./cv_models
   
   # Batch evaluate all folds
   python evaluate_cv_sliding_folds.py \
       --models-dir ./cv_models \
       --input-dir ../Synth_LR_unpadded_nii \
       --target-dir ../Dynamic_SENSE_padded \
       --output-dir ./evaluation_results
   ```

📖 **Full workflow guide:** See [`QUICKSTART.md`](QUICKSTART.md)

---

### Option 2: Local Training & Inference

```bash
# Clone the repository
cd "MSc Project"
git clone https://github.com/marioknicola/synthsup-speechMRI-recon
cd synthsup-speechMRI-recon

# Install dependencies
pip install -r requirements.txt

# Train with 5-fold CV
python train_cross_validation_sliding.py \
    --fold 1 \
    --input-dir ../Synth_LR_unpadded_nii \
    --target-dir ../Dynamic_SENSE_padded \
    --output-dir ../cv_results \
    --epochs 100 \
    --batch-size 4

# Evaluate all folds
python evaluate_cv_sliding_folds.py \
    --models-dir ../cv_results \
    --input-dir ../Synth_LR_unpadded_nii \
    --target-dir ../Dynamic_SENSE_padded \
    --output-dir ../evaluation_results

# Monitor training
tensorboard --logdir ../cv_results/fold_1/logs
```

---

## 📂 Project Structure

Expected directory layout (data folders live **outside** repository):

```
MSc Project/
├── synthsup-speechMRI-recon/       # This repository (code only)
│   ├── train_cross_validation_sliding.py  # 5-fold CV training
│   ├── train_cross_validation.py         # Flexible N-fold CV
│   ├── evaluate_cv_sliding_folds.py      # Batch evaluation
│   ├── inference_test_subject.py         # Single subject inference
│   ├── inference_dynamic_speech.py       # Dynamic volume inference
│   ├── sense_reconstruction.py           # Classical SENSE
│   ├── unet_model.py                     # U-Net architecture
│   ├── dataset.py                        # PyTorch dataloaders
│   ├── utils/                            # Utilities
│   │   ├── dataloading_currentlyUNUSED.py  # Legacy HR/LR generation
│   │   ├── synthetic_undersampling.py      # K-space utilities
│   │   ├── crop_dynamic_volumes.py
│   │   ├── transform_nifti_files.py
│   │   ├── PSNR_and_SSIM.py
│   │   └── ...
│   └── docs/                             # Documentation
│       ├── QUICKSTART.md
│       ├── COLAB_TO_LOCAL_WORKFLOW.md
│       ├── 7_SUBJECT_CV_STRATEGY.md
│       └── INFERENCE.md
│
├── kspace_mat_FS/                  # Fully-sampled k-space (MAT files)
├── kspace_mat_US/                  # Undersampled k-space (MAT files)
├── sensitivity_maps/               # Coil sensitivity maps
├── HR_nii/                         # High-res ground truth (normalized)
├── Synth_LR_nii/                   # Synthetic LR inputs (normalized)
├── Synth_LR_cropped/               # Cropped LR variants
├── Dynamic_SENSE/                  # Classical reconstructions
├── Dynamic_SENSE_padded/           # Padded dynamic data
└── cv_results/                     # Training outputs
```

---

## 🔬 Pipeline Components

### 1. Data Generation (Preprocessing)

Generate normalized HR/LR training pairs from fully-sampled k-space:

```bash
cd synthsup-speechMRI-recon
python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from utils.dataloading_currentlyUNUSED import process_folder

process_folder(
    folder_path='../kspace_mat_FS',
    crop_size=40,
    output_folder_fullres='../HR_nii',
    output_folder_lowres='../Synth_LR_nii',
    output_folder_cropped='../Synth_LR_cropped'
)
PY
```

**Output:** All images normalized to [0,1] range automatically.

---

### 2. Classical SENSE Reconstruction

Reconstruct undersampled dynamic data:

```bash
python sense_reconstruction.py \
    --kspace ../kspace_mat_US/ \
    --coilmap ../sensitivity_maps/ \
    --output-dir ../ \
    --plot
```

---

### 3. Deep Learning Training

**5-Fold Sliding Window CV:**
```bash
# Train all folds
python train_cross_validation_sliding.py --all-folds \
    --input-dir ../Synth_LR_unpadded_nii \
    --target-dir ../Dynamic_SENSE_padded \
    --output-dir ../cv_results \
    --epochs 100

# Or train single fold
python train_cross_validation_sliding.py --fold 1 \
    --input-dir ../Synth_LR_unpadded_nii \
    --target-dir ../Dynamic_SENSE_padded \
    --output-dir ../cv_results \
    --epochs 100
```

**7-Subject CV (6-fold + 1 held-out):**
```bash
# Use colab_cross_validation.ipynb for automated setup
# See docs/7_SUBJECT_CV_STRATEGY.md for details
```

---

### 4. Evaluation & Inference

**Batch evaluation (all folds):**
```bash
python evaluate_cv_sliding_folds.py \
    --models-dir ../cv_results \
    --input-dir ../Synth_LR_unpadded_nii \
    --target-dir ../Dynamic_SENSE_padded \
    --output-dir ../evaluation_results
```

**Single subject inference:**
```bash
python inference_test_subject.py \
    --checkpoint ../cv_results/fold_1/checkpoints/best_model.pth \
    --input-file ../Synth_LR_nii/LR_kspace_Subject0026_aa.nii \
    --output-dir ../inference_results
```

**Dynamic volume inference:**
```bash
python inference_dynamic_speech.py \
    --checkpoint ../cv_results/best_model.pth \
    --input-file ../Dynamic_SENSE_padded/Subject0021_speech.nii \
    --output-file ../Dynamic_SENSE_padded/unet_speaking.nii
```

---

## 📊 Data Format & Normalization

**Critical:** All image outputs are normalized to [0,1] range:
- ✅ `HR_nii/` - min=0.0, max=1.0
- ✅ `Synth_LR_nii/` - min=0.0, max=1.0
- ✅ `Synth_LR_cropped/` - min=0.0, max=1.0
- ✅ `Dynamic_SENSE/` - min=0.0, max=1.0 (if regenerated)

**K-space data:**
- Shape: `(Ny=312, Nx=410, Nc=22 coils)` for fully-sampled
- Shape: `(Ny=80, Nx=82, Nc=22 coils)` for undersampled
- Orientation: `rot90(k=-1) + flip(axis=1)` applied during preprocessing

---

## 🔧 Model Configuration

**Default U-Net (Lightweight Baseline):**
- Base filters: 32
- Parameters: ~7.8M
- Training memory: ~5-8 GB (batch_size=4)
- Inference: ~30-60 ms/frame (GPU)

**Heavier Variant (Optional):**
```bash
python train_cross_validation_sliding.py --base-filters 64  # ~31M parameters
```

---

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup guide
- **[docs/COLAB_TO_LOCAL_WORKFLOW.md](docs/COLAB_TO_LOCAL_WORKFLOW.md)** - Complete Colab training workflow
- **[docs/7_SUBJECT_CV_STRATEGY.md](docs/7_SUBJECT_CV_STRATEGY.md)** - Cross-validation strategy
- **[docs/INFERENCE.md](docs/INFERENCE.md)** - Testing and visualization guide

---

## 🐛 Troubleshooting

**CUDA Out of Memory:**
```bash
python train_cross_validation_sliding.py --batch-size 2 --base-filters 16
```

**Import Errors:**
```bash
pip install -r requirements.txt --upgrade
```

**Data Loading Issues:**
- Verify NIfTI files exist in specified directories
- Check normalization: `python -c "import nibabel as nib; print(nib.load('HR_nii/kspace_Subject0026_sh.nii').get_fdata().min(), nib.load('HR_nii/kspace_Subject0026_sh.nii').get_fdata().max())"`
- Should output: `(0.0, 1.0)`

---

## 📝 Key Scripts Reference

| Script | Purpose |
|--------|---------|
| `train_cross_validation_sliding.py` | 5-fold sliding window CV training |
| `train_cross_validation.py` | Flexible N-fold CV (for custom splits) |
| `evaluate_cv_sliding_folds.py` | Batch evaluation of all CV folds |
| `inference_test_subject.py` | Single subject/file inference |
| `inference_dynamic_speech.py` | Dynamic volume (multi-frame) inference |
| `sense_reconstruction.py` | Classical SENSE reconstruction |
| `utils/dataloading_currentlyUNUSED.py` | Generate normalized HR/LR pairs |
| `utils/synthetic_undersampling.py` | K-space manipulation utilities |

---

## 🎓 Citation

If you use this code for your research, please cite:

```bibtex
@mastersthesis{knicola2025synthsup,
  title={Synthetically Supervised Deep Learning for Dynamic Speech MRI Reconstruction},
  author={Knicola, Mario},
  year={2025},
  school={University}
}
```

---

**Last Updated:** November 24, 2025

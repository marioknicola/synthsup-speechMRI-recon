# Synthetically Supervised Deep Learning for Dynamic Speech MRI Reconstruction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marioknicola/synthsup-speechMRI-recon/blob/main/colab_cross_validation.ipynb)

A comprehensive toolkit for MRI reconstruction combining classical SENSE algorithms with deep learning (U-Net) approaches for accelerated dynamic speech MRI imaging.

> 🚀 **Quick Start:** Click the "Open in Colab" badge above to train models with 6-fold cross-validation on free GPU!

## 🎯 Overview

This repository provides tools for:
- **Classical SENSE Reconstruction:** For 22 coil data, the filled lines of kspace can be modified to fit the trajectory.
- **Deep Learning Reconstruction:** Lightweight U-Net baseline for synthetically supervised learning
- **Data Utilities:** Synthetic undersampling via kspace truncation and noise injection, image quality metrics, format conversion
- **Training Pipeline:** Complete PyTorch training infrastructure with TensorBoard logging (in progress)

**Key Features:**
- 📊 Automatic 80/10/10 train/val/test splitting with reproducible seeds, default is 42 (of course)
- 📈 Built-in metrics (PSNR, SSIM) and visualization
- 🔬 Lightweight baseline model (~7.8M parameters)

---

## 📦 Quick Start

### 🌐 Option 1: Train in Google Colab (Recommended)

**Best for:** Fast training with free GPU, no local setup required

1. **Click the badge:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marioknicola/synthsup-speechMRI-recon/blob/main/colab_cross_validation.ipynb)

2. **Prepare your data:**
   ```bash
   # On your local machine
   cd "/path/to/MSc Project"
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
   python utils/evaluate_all_folds.py \
       --models-dir ./cv_models \
       --input-dir ../Synth_LR_nii \
       --target-dir ../HR_nii \
       --output-dir ./evaluation_results
   ```

📖 **Full workflow guide:** See [`docs/COLAB_TO_LOCAL_WORKFLOW.md`](docs/COLAB_TO_LOCAL_WORKFLOW.md)

---

### 💻 Option 2: Local Installation

```bash
# Clone the repository
cd "MSc Project"
git clone  https://github.com/marioknicola/synthsup-speechMRI-recon
cd synthsup-speechMRI-recon

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

**1. Classical SENSE Reconstruction:**
```bash
python3 sense_reconstruction.py \
    --kspace ../kspace_mat_US/ \
    --coilmap ../sensitivity_maps/ \
    --output-dir ../ \
    --plot
```

**2. Train U-Net (80/10/10 automatic split):**
```bash
# Standard training (pre-computed NIfTI pairs)
python3 train_unet.py \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --output-dir ../outputs \
    --epochs 100 \
    --batch-size 4

# Enhanced training (with k-space undersampling)
python3 train_unet_kspace.py \
    --synth-lr-dir ../Synth_LR_nii \
    --hr-dir ../HR_nii \
    --kspace-fs-dir ../kspace_mat_FS \
    --output-dir ../outputs_kspace \
    --epochs 100 \
    --batch-size 4
```

**3. Run Inference:**
```bash
python3 inference_unet.py \
    --checkpoint ../outputs/checkpoints/best_model.pth \
    --input-dir ../Synth_LR_nii \
    --output-dir ../reconstructions \
    --compute-metrics \
    --visualize
```

**4. Monitor Training:**
```bash
tensorboard --logdir ../outputs/logs
```

---

## 📂 Repository Structure

```
synthsup-speechMRI-recon/
├── sense_reconstruction.py      # Classical SENSE reconstruction
├── unet_model.py                # U-Net architecture (32 base filters as default)
├── unet_with_dc.py              # U-Net with data consistency layer
├── dataset.py                   # PyTorch data loaders (NIfTI pairs)
├── dataset_kspace.py            # Enhanced dataset with k-space undersampling
├── train_unet.py                # Training script (NIfTI pairs, 80/10/10 split)
├── train_unet_kspace.py         # Enhanced training (NIfTI + k-space, 80/10/10 split)
├── inference_unet.py            # Inference with metrics
├── utils/                       # Utility scripts
│   ├── synthetic_undersampling.py
│   ├── PSNR_and_SSIM.py
│   ├── niftNormaliser.py
│   ├── nifti2png.py
│   ├── gaussian_noise_injection.py
│   ├── rician_noise_injection.py
│   └── resample.py
├── docs/                        # Documentation
│   ├── GETTING_STARTED.md       # Step-by-step tutorial
│   ├── UNET_README.md           # Quick reference
│   ├── UNET_ARCHITECTURE.md     # Architecture details
│   ├── KSPACE_TRAINING_GUIDE.md # K-space training guide (NEW!)
│   ├── DATA_CONSISTENCY_GUIDE.md # Data consistency implementation
│   ├── QUICK_REFERENCE.md       # Command cheat sheet
│   └── CHANGELOG.md             # Version history
├── .github/
│   └── copilot-instructions.md  # guide for copilot
├── requirements.txt             # dependencies
└── README.md                    # This file
```

---

## 📚 Documentation

- **[Getting Started Guide](docs/GETTING_STARTED.md)** - Complete tutorial from setup to inference
- **[Inference Guide](docs/INFERENCE_GUIDE.md)** - Testing trained models and visualization
- **[Data Consistency Guide](docs/DATA_CONSISTENCY_GUIDE.md)** - Physics-guided reconstruction with k-space enforcement
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Common commands and workflows
- **[U-Net Documentation](docs/UNET_README.md)** - Daily usage reference
- **[Architecture Details](docs/UNET_ARCHITECTURE.md)** - Technical deep dive
- **[Changelog](docs/CHANGELOG.md)** - Recent updates and migration guide

---

## 🔬 Training Data Strategy

The pipeline uses a **80/10/10 automatic split** from synthetic training pairs:

```
../Synth_LR_nii/  (input)  ──┐
                              ├──> 80% train / 10% val / 10% test
../HR_nii/        (target) ──┘

../Dynamic_SENSE/ (reserved for final testing after training)
```

- Fixed seed (42) ensures reproducible splits
- Test indices saved to `../outputs/test_indices.txt`
- Dynamic_SENSE used for final independent validation

---

## 🎓 Google Colab Training

**train on Colab**:

### Option 1: Mount Google Drive

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Upload data to Drive (do this once via Drive web interface)
#    - Upload Synth_LR_nii/ and HR_nii/ folders to Drive
#    - Create "outputs" folder in Drive

# 3. Clone repository
!cd /content && git clone <your-repo-url> synthsup-speechMRI-recon
%cd /content/synthsup-speechMRI-recon

# 4. Install dependencies
!pip install -r requirements.txt

# 5. Train with Drive paths
!python3 train_unet.py \
    --input-dir "/content/drive/MyDrive/MRI_Data/Synth_LR_nii" \
    --target-dir "/content/drive/MyDrive/MRI_Data/HR_nii" \
    --output-dir "/content/drive/MyDrive/MRI_Data/outputs" \
    --epochs 100 \
    --batch-size 4
```

### Option 2: Direct Upload to Colab VM (easiest i think)

```python
# 1. Upload data directly (smaller datasets)
from google.colab import files
import zipfile

# Upload zipped data
uploaded = files.upload()  # Upload Synth_LR_nii.zip and HR_nii.zip

# Extract
!unzip Synth_LR_nii.zip -d /content/
!unzip HR_nii.zip -d /content/

# 2. Clone and train
!git clone https://github.com/marioknicola/synthsup-speechMRI-recon
%cd synthsup-speechMRI-recon
!pip install -r requirements.txt

!python3 train_unet.py \
    --input-dir /content/Synth_LR_nii \
    --target-dir /content/HR_nii \
    --output-dir /content/outputs \
    --epochs 100 \ 
    --batch-size 4
```

### Tips for Colab Training:

1. **Save Checkpoints to Drive:** Always use `--output-dir` pointing to Google Drive to avoid losing progress
2. **Monitor with TensorBoard:**
   ```python
   %load_ext tensorboard
   %tensorboard --logdir /content/drive/MyDrive/MRI_Data/outputs/logs
   ```
3. **Prevent Disconnection:** Keep browser tab active or use:
   ```javascript
   // Run in browser console
   function ClickConnect(){
     console.log("Clicking connect"); 
     document.querySelector("colab-connect-button").click()
   }
   setInterval(ClickConnect, 60000)
   ```
^^^ not tried this
4. **Use TPU (optional):** Colab also offers TPUs, but requires code modifications for PyTorch XLA
5. **Expected Runtime:** 100 epochs on T4 GPU ≈ 4-6 hours (depends on dataset size)

---

## 🔧 Model Configuration

**Default U-Net (Lightweight Baseline):**
- Base filters: 32
- Parameters: ~7.8M
- Training memory: ~5-8 GB (batch_size=4)
- Inference: ~30-60 ms/frame (GPU)

**Heavier Variant (Optional):**
```bash
python3 train_unet.py --base-filters 64  # ~31M parameters
```

---

## 📊 Data Format

**Input Requirements:**
- NIfTI format (.nii) for images
- MATLAB .mat files for k-space data
- Shape convention: (Ny=80, Nx=82, Nc=22 coils, Nf=100 frames) (frames for dynamic data)
- Orientation: `rot90(k=-1, axes=(0,1))` + `flip(axis=1)` (to deal with strange nifti convention)

**Output Locations:**
- All outputs save to parent directory (`../`)
- Never writes inside repository folder
- Organises into: `outputs/`, `reconstructions/`, `Dynamic_SENSE/`

---

## 🐛 Troubleshooting

**CUDA Out of Memory:**
```bash
python3 train_unet.py --batch-size 2 --base-filters 16 ...
```

**Import Errors:**
```bash
pip install -r requirements.txt --upgrade
```

**Data Loading Issues:**
- Verify NIfTI files exist in specified directories
- Check file permissions
- Ensure consistent naming convention

See [GETTING_STARTED.md](docs/GETTING_STARTED.md) for detailed troubleshooting.

---

**Last Updated:** October 27, 2025

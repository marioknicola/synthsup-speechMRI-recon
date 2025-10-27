# Synthetically Supervised Deep Learning for Dynamic Speech MRI Reconstruction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive toolkit for MRI reconstruction combining classical SENSE algorithms with deep learning (U-Net) approaches for accelerated dynamic speech MRI imaging.

## 🎯 Overview

This repository provides tools for:
- **Classical SENSE Reconstruction:** Generalized SENSE implementation for multi-coil MRI data
- **Deep Learning Reconstruction:** Lightweight U-Net baseline for synthetically supervised learning
- **Data Utilities:** Synthetic undersampling, image quality metrics, format conversion
- **Training Pipeline:** Complete PyTorch training infrastructure with TensorBoard logging

**Key Features:**
- 🚀 Configurable CLI tools for all major operations
- 📊 Automatic 80/10/10 train/val/test splitting with reproducible seeds
- 💾 Safe output management (never writes to repo directory)
- 📈 Built-in metrics (PSNR, SSIM) and visualization
- 🔬 Lightweight baseline model (~7.8M parameters) suitable for comparison studies

---

## 📦 Quick Start

### Installation

```bash
# Clone the repository
cd "MSc Project"
git clone <repository-url> synthsup-speechMRI-recon
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
python3 train_unet.py \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --output-dir ../outputs \
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
├── unet_model.py                # U-Net architecture (32 base filters)
├── dataset.py                   # PyTorch data loaders
├── train_unet.py                # Training script with auto-split
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
│   ├── QUICK_REFERENCE.md       # Command cheat sheet
│   └── CHANGELOG.md             # Version history
├── .github/
│   └── copilot-instructions.md  # AI assistant guide
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 📚 Documentation

- **[Getting Started Guide](docs/GETTING_STARTED.md)** - Complete tutorial from setup to inference
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

**Yes, you can train on Colab!** Here's how:

### Option 1: Mount Google Drive (Recommended)

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

### Option 2: Direct Upload to Colab VM

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
!git clone <your-repo-url> synthsup-speechMRI-recon
%cd synthsup-speechMRI-recon
!pip install -r requirements.txt

!python3 train_unet.py \
    --input-dir /content/Synth_LR_nii \
    --target-dir /content/HR_nii \
    --output-dir /content/outputs \
    --epochs 100 \
    --batch-size 4
```

### Option 3: Use Colab's GPU Efficiently

```python
# Check GPU availability
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Monitor GPU during training
!nvidia-smi

# For Colab's T4 GPU (16GB), use:
!python3 train_unet.py \
    --input-dir <path> \
    --target-dir <path> \
    --output-dir <path> \
    --batch-size 8 \
    --base-filters 32 \
    --epochs 100
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
- Shape convention: (Ny=80, Nx=82, Nc=22 coils, Nf=100 frames)
- Orientation: `rot90(k=-1, axes=(0,1))` + `flip(axis=1)`

**Output Locations:**
- All outputs save to parent directory (`../`)
- Never writes inside repository folder
- Organizes into: `outputs/`, `reconstructions/`, `Dynamic_SENSE/`

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

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{synthsup-speechmri,
  author = {Mario Knicola},
  title = {Synthetically Supervised Deep Learning for Dynamic Speech MRI Reconstruction},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/marioknicola/synthsup-speechMRI-recon}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Contact

- **Author:** Mario Klitos Nicola
- **Project:** MSc Thesis - Speech MRI Reconstruction
- **Institution:** King's College London

---

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- NiBabel developers for NIfTI support
- scikit-image for metrics implementation

---

**Last Updated:** October 27, 2025

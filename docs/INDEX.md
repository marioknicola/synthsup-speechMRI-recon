# Documentation Index

Quick reference for all project documentation.

---

## Quick Start

**New to the project?** → [`GETTING_STARTED.md`](GETTING_STARTED.md)

**Training in Colab?** → [`COLAB_TRAINING_GUIDE.md`](COLAB_TRAINING_GUIDE.md)

**K-space training details?** → [`KSPACE_TRAINING.md`](KSPACE_TRAINING.md)

---

## Core Documentation

### Training & Data

| Document | Description |
|----------|-------------|
| [`KSPACE_TRAINING.md`](KSPACE_TRAINING.md) | Complete guide for training with k-space data |
| [`COLAB_TRAINING_GUIDE.md`](COLAB_TRAINING_GUIDE.md) | Step-by-step Colab training instructions |
| [`COLAB_QUICKSTART.md`](COLAB_QUICKSTART.md) | Fast-track Colab training (copy-paste ready) |
| [`DATA_CONSISTENCY_GUIDE.md`](DATA_CONSISTENCY_GUIDE.md) | Data consistency overview & theory |
| [`DATA_CONSISTENCY_COLAB.md`](DATA_CONSISTENCY_COLAB.md) | **Data consistency in Colab (step-by-step)** |

### Model & Architecture

| Document | Description |
|----------|-------------|
| [`UNET_ARCHITECTURE.md`](UNET_ARCHITECTURE.md) | U-Net architecture details |
| [`UNET_README.md`](UNET_README.md) | U-Net training reference |

### Usage & Inference

| Document | Description |
|----------|-------------|
| [`INFERENCE_GUIDE.md`](INFERENCE_GUIDE.md) | Running inference on test data |
| [`GETTING_STARTED.md`](GETTING_STARTED.md) | Initial setup and usage |

### Reference

| Document | Description |
|----------|-------------|
| [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) | Command reference |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history |

---

## Training Decision Tree

```
Do you want to train the model?
│
├─ YES → Do you have k-space data?
│        │
│        ├─ YES → Use train_unet_kspace.py
│        │        → See: KSPACE_TRAINING.md
│        │        → Colab: COLAB_TRAINING_GUIDE.md
│        │
│        └─ NO  → Use train_unet.py
│                 → See: UNET_README.md
│
└─ NO  → Run inference only
         → Use inference_unet.py
         → See: INFERENCE_GUIDE.md
```

---

## Common Tasks

### Train Model with K-Space (Recommended)

```bash
python3 train_unet_kspace.py \
    --synth-lr-dir ../Synth_LR_nii \
    --hr-dir ../HR_nii \
    --kspace-fs-dir ../kspace_mat_FS \
    --batch-size 4 \
    --epochs 100
```

**Guide**: [`KSPACE_TRAINING.md`](KSPACE_TRAINING.md)  
**Colab**: [`COLAB_TRAINING_GUIDE.md`](COLAB_TRAINING_GUIDE.md)

### Train Model (Standard)

```bash
python3 train_unet.py \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --epochs 100
```

**Guide**: [`UNET_README.md`](UNET_README.md)

### Run Inference

```bash
python3 inference_unet.py \
    --checkpoint ../outputs/checkpoints/best_model.pth \
    --input-dir ../Dynamic_SENSE_padded \
    --output-dir ../reconstructions \
    --compute-metrics \
    --visualize
```

**Guide**: [`INFERENCE_GUIDE.md`](INFERENCE_GUIDE.md)

---

## Key Concepts

### Data Sources

1. **NIfTI Pairs** (`Synth_LR_nii` + `HR_nii`)
   - Pre-computed synthetic undersampling
   - 21 static image pairs
   
2. **K-Space** (`kspace_mat_FS`)
   - Fully-sampled k-space
   - On-the-fly SENSE undersampling
   - 21 k-space files
   
3. **Dynamic Test Data** (`Dynamic_SENSE_padded`)
   - Real undersampled sequences
   - 100 frames × 4 subjects
   - For testing only (not training)

### Training Approaches

| Approach | Data | Samples | Best For |
|----------|------|---------|----------|
| **Standard** | NIfTI only | 21 | Quick baseline |
| **Enhanced** | NIfTI + k-space | 42 | Better performance |

**Recommendation**: Use enhanced training (k-space) for better results.

### SENSE Undersampling

- **Pattern**: 40 out of 82 k-space columns (~2.05× acceleration)
- **Center**: 18 columns fully sampled (100% acquisition)
- **Periphery**: R=3 undersampling
- **Applied at**: 82-column resolution (matches Dynamic acquisition)

---

## Implementation Notes

### Critical FFT Convention

**IMPORTANT**: Use `axes=(0,)` for fftshift:

```python
# Correct
img = scipy.fft.ifftshift(
    scipy.fft.ifft2(scipy.fft.ifftshift(kspace, axes=(0,)), axes=(0, 1)),
    axes=(0,)
)
```

This matches MRI physics and prevents artifacts.

### K-Space Processing Pipeline

```
1. Truncate to 80×82 (centered on k-space peak)
2. Apply SENSE pattern (40/82, center 18 fully sampled)
3. IFFT2 + RSS → 80×82 image
4. Zero-pad to 312×410 via k-space
```

**Result**: Input (undersampled) + Target (fully-sampled) pair at 312×410

---

## File Organization

```
synthsup-speechMRI-recon/
├── docs/                           # Documentation
│   ├── INDEX.md                    # This file
│   ├── KSPACE_TRAINING.md          # K-space training guide
│   ├── COLAB_TRAINING_GUIDE.md     # Colab instructions
│   ├── INFERENCE_GUIDE.md          # Inference guide
│   ├── GETTING_STARTED.md          # Getting started
│   ├── UNET_ARCHITECTURE.md        # Architecture details
│   ├── UNET_README.md              # U-Net training
│   ├── DATA_CONSISTENCY_GUIDE.md   # Data consistency
│   ├── QUICK_REFERENCE.md          # Command reference
│   └── CHANGELOG.md                # Version history
├── dataset_kspace.py               # K-space dataset loader
├── train_unet_kspace.py            # K-space training script
├── train_unet.py                   # Standard training script
├── inference_unet.py               # Inference script
└── unet_model.py                   # U-Net model
```

---

## Getting Help

1. **Check documentation**: See relevant guide above
2. **Check FAQ**: Most guides have FAQ sections
3. **Check troubleshooting**: Common issues and solutions
4. **Test dataset**: Run `python3 dataset_kspace.py` to verify

---

## Status

- ✅ K-space training implemented and verified
- ✅ FFT convention corrected (`axes=(0,)`)
- ✅ K-space centering on peak (not geometric center)
- ✅ Documentation consolidated and updated
- ✅ Colab training guide created
- ✅ Ready for production use

**Last Updated**: 27 October 2025

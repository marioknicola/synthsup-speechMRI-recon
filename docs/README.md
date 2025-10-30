# U-Net for MRI Reconstruction

Complete U-Net deep learning pipeline for synthetically supervised MRI reconstruction with detailed architecture information.

---

## 🎯 Overview

This pipeline learns to reconstruct fully-sampled MRI images from undersampled/aliased inputs using synthetic supervision.

**Training Strategy:**
- **Input:** Synthetically undersampled images (`Synth_LR_nii/`)
- **Target:** Fully-sampled high-resolution images (`HR_nii/`)
- **Test:** Real dynamic SENSE data (`Dynamic_SENSE_padded/`)

**Key Components:**
- `unet_model.py` - U-Net architecture - `dataset.py` - PyTorch data loaders with [0,1] normalization
- `train_unet.py` - Training script with L1+SSIM loss
- `inference_unet.py` - Inference with metrics and visualization

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python3 train_unet.py \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --epochs 200 \
    --batch-size 4 \
    --base-filters 32
```

**Outputs:**
- `../outputs/checkpoints/best_model.pth` - Best model
- `../outputs/training.log` - Training history
- `../outputs/test_indices.txt` - Test set split

**Monitor training:**
```bash
tensorboard --logdir ../outputs/logs
```

### 3. Run Inference

```bash
python3 inference_unet.py \
    --checkpoint ../outputs/checkpoints/best_model.pth \
    --input-dir ../Dynamic_SENSE_padded_95f \
    --output-dir ../reconstructions \
    --compute-metrics \
    --visualize
```

---

### Parameter Configuration

```python
# Lightweight baseline (default)
model = UNet(in_channels=1, out_channels=1, base_filters=32)


# Heavier variant
model = UNet(in_channels=1, out_channels=1, base_filters=64)

```

### Layer Dimensions

For input shape `(1, 312, 410)` with `base_filters=32`:

| Layer | Shape | Channels | Parameters | Notes |
|-------|-------|----------|------------|-------|
| **Input** | (1, 312, 410) | 1 | - | Normalized [0,1] |
| **Inc** | (32, 312, 410) | 32 | 1,856 | Initial features |
| **Down1** | (64, 156, 205) | 64 | 37,376 | ×1/2 resolution |
| **Down2** | (128, 78, 102) | 128 | 148,224 | ×1/4 resolution |
| **Down3** | (256, 39, 51) | 256 | 591,360 | ×1/8 resolution |
| **Down4** | (256, 19, 25) | 256 | 591,360 | ×1/16 (bottleneck) |
| **Up1** | (128, 39, 51) | 128 | 591,360 | ×1/8 resolution |
| **Up2** | (64, 78, 102) | 64 | 148,224 | ×1/4 resolution |
| **Up3** | (32, 156, 205) | 32 | 37,376 | ×1/2 resolution |
| **Up4** | (32, 312, 410) | 32 | 18,752 | Full resolution |
| **Output** | (1, 312, 410) | 1 | 33 | Final prediction |
---
^^^ this doesn't include the batch size


## 🎓 Training Details

### Data Normalization

**Per-file [0, 1] normalization** (automatic):
- Computes min/max statistics for entire file at initialization
- All frames from same file use same normalization
- Prevents frame-to-frame intensity variations
- Automatic denormalization during inference

**Benefits:**
- Solves 4.8× intensity mismatch between datasets
- Prevents negative values in predictions
- Stable gradients during training
- Better generalization to test data

### Loss Function

Combined L1 + SSIM loss:

```python
Loss = 0.70 × L1(pred, target) + 0.16 × SSIM_loss(pred, target)
```

- **L1 Loss:** Pixel-wise accuracy
- **SSIM Loss:** Perceptual quality and structure preservation
- **Balance:** Prevents blurry outputs while maintaining fidelity

### Training Configuration

Default settings (80/10/10 split):
```bash
python3 train_unet.py \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --epochs 100 \
    --batch-size 4 \
    --lr 1e-4 \
    --base-filters 32 \
    --output-dir ../outputs
```

Advanced options:
```bash
# Longer training
--epochs 200

# Larger model
--base-filters 64 --batch-size 2

# Different loss balance
--alpha 0.5  # 50% L1, 50% SSIM

# Lower learning rate
--lr 5e-5
```

### Resume Training

```bash
python3 train_unet.py \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --resume ../outputs/checkpoints/latest_model.pth \
    --epochs 200
```

---

## 📊 Expected Results

### Training Metrics (After 100 Epochs)

| Metric | Initial | After Training |
|--------|---------|----------------|
| **Val PSNR** | ~16 dB | **27-32 dB** |
| **Val SSIM** | ~0.22 | **0.75-0.85** |
| **Val Loss** | ~0.20 | **0.05-0.10** |
| **Training Time** | - | 2-4 hours (GPU) |

### Improvements Over Zero-Filled

- ✅ Reduced aliasing artifacts
- ✅ Better edge preservation
- ✅ Improved SNR
- ✅ More natural appearance

---

## 🔧 Advanced Features

## 🗂️ Directory Structure

```
MSc Project/
├── synthsup-speechMRI-recon/        # Repository (code only)
│   ├── unet_model.py
│   ├── dataset.py
│   ├── train_unet.py
│   ├── train_cross_validation.py
│   ├── inference_unet.py
│   ├── remove_first_5_frames.py
│   ├── utils/
│   │   ├── evaluate_all_folds.py
│   │   └── download_colab_models.py
│   └── docs/
│
├── Synth_LR_nii/                    # Training input
├── HR_nii/                          # Training target
├── Dynamic_SENSE_padded_95f/        # Test data (95 frames)
│
├── cv_models/                       # Cross-validation models
│   ├── fold1/
│   ├── fold2/
│   ├── ...
│   └── fold6/
│
├── evaluation_results/              # CV evaluation results
│   ├── fold1/
│   ├── ...
│   └── cv_summary.json
│
├── inference_results/               # Inference outputs
│   └── heldout_0021/
│
├── outputs/                         # Training outputs
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   └── latest_model.pth
│   ├── logs/
│   └── training.log
│
└── reconstructions/                 # Inference outputs
```

**⚠️ Important:** Never save outputs inside the repository folder!

---

## 🧪 Testing

### Quick Model Test
```bash
python3 unet_model.py
```

### Quick Dataset Test
```bash
python3 dataset.py
```

### Full Inference Test
```bash
python3 inference_unet.py \
    --checkpoint ../outputs/checkpoints/best_model.pth \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --output-dir ../reconstructions_test \
    --compute-metrics \
    --visualize
```

---

## 🔍 Troubleshooting

### CUDA Out of Memory

Reduce batch size or model size:
```bash
python3 train_unet.py --batch-size 2 --base-filters 16 ...
```

### Dataset Not Found

Verify data structure:
```bash
ls ../Synth_LR_nii/*.nii | wc -l 
ls ../HR_nii/*.nii | wc -l         
```

## 📚 See Also

- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Step-by-step tutorial
- **[USE_IN_COLAB.md](USE_IN_COLAB.md)** - Google Colab training guide
- **[INFERENCE.md](INFERENCE.md)** - Inference and evaluation guide
- **[DATA_CONSISTENCY.md](DATA_CONSISTENCY.md)** - Physics-based constraints
- **[CHANGELOG.md](CHANGELOG.md)** - Version history


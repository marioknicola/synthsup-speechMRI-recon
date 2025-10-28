# U-Net Deep Learning Pipeline for MRI Reconstruction

This directory includes a complete U-Net-based deep learning pipeline for synthetically supervised MRI reconstruction.

## 🎯 Overview

The pipeline learns to reconstruct fully-sampled MRI images from undersampled/aliased inputs using synthetic supervision.

**Training Strategy:**
- **Training data:** `../Synth_LR_nii/` (input) → `../HR_nii/` (target)

**Key Components:**
- `unet_model.py` - U-Net architecture with optional data consistency
- `dataset.py` - PyTorch data loaders for NIfTI and k-space MAT files
- `train_unet.py` - Training script with L1+SSIM loss
- `inference_unet.py` - Inference with metrics and visualization

## ⚠️ Output Directory Convention

**NEVER save outputs inside the `synthsup-speechMRI-recon/` repository folder!**

All generated files must go to parent `../` (MSc Project) directory:
- Checkpoints: `../outputs/checkpoints/`
- Logs: `../outputs/logs/`
- Reconstructions: `../reconstructions/`

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Training Data

**Your data should already exist in the parent directory:**
image pairs
- `../Synth_LR_nii/` - Training inputs (synthetic low-resolution)
- `../HR_nii/` - Training targets (high-resolution ground truth)


**For SENSE recon:**
```bash
python3 sense_reconstruction.py --kspace ../kspace_mat_US/kspace_Subject0026_vv.mat \
                                --coilmap ../sensitivity_maps/sens_Subject0026_Exam17853_80x82x100_nC22.mat \
                                --output-dir .. \
                                --subject-id Subject0026_vv \
                                --save-nifti
```

This creates:
- `../Dynamic_SENSE/` - SENSE reconstructions unpadded
- `../Synth_LR_nii/` - Synthetically undersampled inputs
- `../HR_nii/` - High-resolution ground truth targets
- `../Dynamic_SENSE_padded/` - SENSE reconstructions padded to HR size

### 3. Train the U-Net (80/10/10 split from synthetic pairs)

```bash
python3 train_unet.py --input-dir ../Synth_LR_nii \
                      --target-dir ../HR_nii \
                      --output-dir ../outputs \
                      --epochs 100 \
                      --batch-size 4 \
                      --lr 1e-4 \
                      --base-filters 32
```

**Training outputs:**
- `../outputs/checkpoints/` - Model checkpoints
- `../outputs/logs/` - TensorBoard logs
- `../outputs/test_indices.txt` - Saved test set indices for reproducibility

**Data split:**
- 80% training, 10% validation, 10% test (seed=42)
- Dynamic_SENSE reserved for final testing after training

**Monitor training:**
```bash
tensorboard --logdir ../outputs/logs
```

### 4. Run Inference

```bash
python3 inference_unet.py --checkpoint ../outputs/checkpoints/best_model.pth \
                          --input-dir ../Synth_LR_nii \
                          --output-dir ../reconstructions \
                          --compute-metrics \
                          --target-dir ../HR_nii \
                          --visualize
```

**Inference outputs:**
- `../reconstructions/*.nii` - Reconstructed NIfTI files
- `../reconstructions/visualizations/` - Comparison plots
- Console output with PSNR/SSIM metrics

## 📊 Model Architecture

**U-Net Configuration:**
- Input: Single-channel magnitude MRI images (1, H, W)
- Output: Reconstructed magnitude images (1, H, W)
- Default: 32 base filters (~7.8M parameters) - lightweight baseline
- Loss: Combined L1 (84%) + SSIM (16%)
- Optimiser: Adam with ReduceLROnPlateau scheduling

**Key Features:**
- Skip connections for preserving fine details
- Batch normalization for stable training
- Optional data consistency layer for k-space enforcement
- Handles arbitrary image sizes (tested with 312×410)

## 🔧 Advanced Usage

### Custom Training Configuration

```bash
python3 train_unet.py \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --output-dir ../outputs \
    --epochs 200 \
    --batch-size 8 \
    --lr 5e-4 \
    --base-filters 32 \
    --loss-alpha 0.7 \
    --save-freq 5
```

### Resume Training

```bash
python3 train_unet.py \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --output-dir ../outputs \
    --resume ../outputs/checkpoints/checkpoint_epoch_50.pth
```

### K-Space Dataset (On-the-fly Undersampling)

The `KSpaceDataset` class in `dataset.py` supports loading fully-sampled k-space and applying synthetic undersampling during training:

```python
from dataset import KSpaceDataset

dataset = KSpaceDataset(
    kspace_dir='../kspace_mat_FS/',
    sensitivity_dir='../sensitivity_maps/',
    acquired_indices=[0, 3, 6, 9, ...],  # R=3 undersampling pattern
    normalize=True
)
```

## 📈 Expected Results

**Typical metrics after 100 epochs:**
- PSNR: 30-35 dB (depending on undersampling factor)
- SSIM: 0.85-0.95
- Training time: ~2-4 hours on GPU for 100 epochs with ~1000 frames

**Improvements over zero-filled reconstruction:**
- Reduced aliasing artifacts
- Better edge preservation
- Improved SNR

## 🧪 Testing

Quick model test:
```bash
python3 unet_model.py
```

Quick dataset test:
```bash
python3 dataset.py
```

## 📝 Notes

- **Orientation:** Both SENSE and U-Net outputs use the same orientation convention (rot90 + flip)
- **Normalization:** Images are normalized to [0, 1] during training for stable convergence
- **GPU:** Training will automatically use CUDA if available
- **Memory:** Reduce batch size if encountering OOM errors
- **Output Location:** All outputs saved to `../` (parent directory), never inside repository

## 📊 Training Data Structure

```
Training:
  Input:  ../Synth_LR_nii/      (synthetic low-resolution)
  Target: ../HR_nii/            (high-resolution ground truth)

Validation:
  Input:  ../Dynamic_SENSE/     (SENSE reconstructions)
  Target: ../Dynamic_SENSE_padded/  (padded SENSE)
```

## 🔍 Troubleshooting

**Issue: CUDA out of memory**
```bash
python3 train_unet.py --batch-size 2 --base-filters 16 ...
```

**Issue: Dataset not found**
- Ensure `../Synth_LR_nii/` and `../HR_nii/` exist
- Check files with `ls ../Synth_LR_nii/ | head`
- Verify you're running from inside `synthsup-speechMRI-recon/` directory

**Issue: Poor convergence**
- Try different learning rates (1e-3, 5e-4, 1e-4)
- Adjust loss weighting with `--loss-alpha`
- Increase training epochs

## 📚 References

- U-Net: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- SENSE: Pruessmann et al., "SENSE: Sensitivity Encoding for Fast MRI" (1999)
- SSIM Loss: Wang et al., "Image Quality Assessment" (2004)

## 🤝 Contributing

When modifying the pipeline:
1. Maintain compatibility with existing data formats
2. Update this README with new features
3. Test on a small subset before full training
4. Document new hyperparameters and their effects
5. **Never commit generated data (checkpoints, NIfTIs) to the repository**

## 📚 See Also

- **`GETTING_STARTED.md`** - Complete step-by-step tutorial for beginners
- **`UNET_ARCHITECTURE.md`** - Detailed U-Net architecture explanation with diagrams
- **`.github/copilot-instructions.md`** - AI coding assistant guidelines
- **`README.md`** - Project overview

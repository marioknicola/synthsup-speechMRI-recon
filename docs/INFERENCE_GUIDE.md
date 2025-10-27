# Inference Guide: Testing Your Trained Model

This guide explains how to test your trained U-Net model and visualize performance.

---

## 📥 Step 1: Get Your Trained Model

### From Google Colab:

**Option A: Download from Drive (Recommended)**
1. Open Google Drive
2. Navigate to your outputs folder (e.g., `MRI_Data/outputs/checkpoints/`)
3. Download `best_model.pth`
4. Place it in your local `MSc Project/outputs/checkpoints/` folder

**Option B: Download from Colab Notebook**
```python
from google.colab import files
files.download('/content/drive/MyDrive/MRI_Data/outputs/checkpoints/best_model.pth')
```

### From Local Training:
If you trained locally, your model is already in `../outputs/checkpoints/`

---

## 🧪 Step 2: Choose Your Test Set

### Option 1: Held-Out Test Set (Recommended First)

Test on the 10% of Synth_LR_nii/HR_nii that the model never saw during training:

```bash
python3 inference_unet.py \
    --checkpoint ../outputs/checkpoints/best_model.pth \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --test-indices ../outputs/test_indices.txt \
    --output-dir ../reconstructions_test \
    --compute-metrics \
    --visualize
```

**What this does:**
- Loads only test set samples using saved indices
- Computes PSNR and SSIM for each frame
- Generates comparison visualizations
- Saves reconstructed NIfTI files

### Option 2: All Synthetic Data

Test on all available synthetic pairs (includes training data):

```bash
python3 inference_unet.py \
    --checkpoint ../outputs/checkpoints/best_model.pth \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --output-dir ../reconstructions_all \
    --compute-metrics \
    --visualize
```

### Option 3: Dynamic_SENSE (Final Independent Validation)

Test on real SENSE-reconstructed data (completely independent):

```bash
python3 inference_unet.py \
    --checkpoint ../outputs/checkpoints/best_model.pth \
    --input-dir ../Dynamic_SENSE_padded \
    --output-dir ../reconstructions_dynamic \
    --compute-metrics \
    --visualize
```

**Note:** Dynamic_SENSE won't have PSNR/SSIM unless you have ground truth targets.

---

## 📊 Step 3: Analyze Results

### Check Metrics Summary

```bash
cat ../reconstructions_test/metrics_summary.txt
```

**Example output:**
```
=== Metrics Summary ===
Total samples: 42
Average PSNR: 32.45 dB
Average SSIM: 0.8234

Per-file metrics:
Subject0023_aa.nii: PSNR=33.12 dB, SSIM=0.8456
Subject0024_aa.nii: PSNR=31.87 dB, SSIM=0.8123
...
```

### View Visualizations

```bash
# macOS
open ../reconstructions_test/visualizations/

# Linux
xdg-open ../reconstructions_test/visualizations/

# Windows
explorer ..\reconstructions_test\visualizations\
```

**Visualization shows 4 panels:**
1. **Input** - Undersampled image (what model receives)
2. **Output** - Model reconstruction
3. **Target** - Ground truth
4. **Difference** - |Output - Target| (error map)

### Load Reconstructed Images in Python

```python
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load reconstruction
img = nib.load('../reconstructions_test/recon_Subject0023_aa.nii')
data = img.get_fdata()

# Visualize a frame
frame_idx = 50
plt.figure(figsize=(10, 8))
plt.imshow(data[:, :, frame_idx], cmap='gray')
plt.title(f'Reconstructed Frame {frame_idx}')
plt.colorbar()
plt.show()

print(f"Shape: {data.shape}")  # (H, W, num_frames)
```

---

## 🔍 Step 4: Compare Multiple Models (Optional)

If you trained multiple models with different configurations:

```bash
# Baseline model (32 base filters)
python3 inference_unet.py \
    --checkpoint ../outputs/model_32filters.pth \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --test-indices ../outputs/test_indices.txt \
    --output-dir ../reconstructions_baseline \
    --compute-metrics

# Heavier model (64 base filters)
python3 inference_unet.py \
    --checkpoint ../outputs/model_64filters.pth \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --test-indices ../outputs/test_indices.txt \
    --output-dir ../reconstructions_heavy \
    --compute-metrics

# Compare results
echo "Baseline (32 filters):"
cat ../reconstructions_baseline/metrics_summary.txt

echo "\nHeavier (64 filters):"
cat ../reconstructions_heavy/metrics_summary.txt
```

---

## 🎨 Step 5: Advanced Visualization

### Create Custom Comparison Plots

```python
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load data
input_img = nib.load('../Synth_LR_nii/Subject0023_aa.nii').get_fdata()
recon_img = nib.load('../reconstructions_test/recon_Subject0023_aa.nii').get_fdata()
target_img = nib.load('../HR_nii/kspace_Subject0023_aa.nii').get_fdata()

# Plot multiple frames
frames = [0, 25, 50, 75, 99]
fig, axes = plt.subplots(3, len(frames), figsize=(20, 10))

for i, frame in enumerate(frames):
    axes[0, i].imshow(input_img[:, :, frame], cmap='gray', vmin=0, vmax=1)
    axes[0, i].set_title(f'Input - Frame {frame}')
    axes[0, i].axis('off')
    
    axes[1, i].imshow(recon_img[:, :, frame], cmap='gray', vmin=0, vmax=1)
    axes[1, i].set_title(f'Reconstruction - Frame {frame}')
    axes[1, i].axis('off')
    
    axes[2, i].imshow(target_img[:, :, frame], cmap='gray', vmin=0, vmax=1)
    axes[2, i].set_title(f'Target - Frame {frame}')
    axes[2, i].axis('off')

plt.tight_layout()
plt.savefig('comparison_multi_frame.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Create Error Heatmaps

```python
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

# Compute frame-by-frame SSIM
num_frames = recon_img.shape[2]
ssim_scores = []
psnr_scores = []

for frame in range(num_frames):
    ssim = structural_similarity(
        target_img[:, :, frame],
        recon_img[:, :, frame],
        data_range=1.0
    )
    ssim_scores.append(ssim)
    
    # PSNR
    mse = np.mean((target_img[:, :, frame] - recon_img[:, :, frame])**2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    psnr_scores.append(psnr)

# Plot temporal evolution
fig, axes = plt.subplots(2, 1, figsize=(15, 8))

axes[0].plot(ssim_scores, 'b-', linewidth=2)
axes[0].set_ylabel('SSIM', fontsize=12)
axes[0].set_title('Reconstruction Quality Over Time', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1])

axes[1].plot(psnr_scores, 'r-', linewidth=2)
axes[1].set_xlabel('Frame', fontsize=12)
axes[1].set_ylabel('PSNR (dB)', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quality_over_time.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")
```

---

## 📋 Common Inference Commands

### Quick Test (Single Subject)
```bash
python3 inference_unet.py \
    --checkpoint ../outputs/checkpoints/best_model.pth \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --output-dir ../quick_test \
    --max-files 1 \
    --compute-metrics \
    --visualize
```

### Batch Processing (All Subjects)
```bash
python3 inference_unet.py \
    --checkpoint ../outputs/checkpoints/best_model.pth \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --output-dir ../reconstructions_all \
    --compute-metrics
```

### Test Different Checkpoint
```bash
# Test epoch 50 checkpoint
python3 inference_unet.py \
    --checkpoint ../outputs/checkpoints/checkpoint_epoch_50.pth \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --output-dir ../reconstructions_epoch50 \
    --compute-metrics

# Compare with best model
python3 inference_unet.py \
    --checkpoint ../outputs/checkpoints/best_model.pth \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --output-dir ../reconstructions_best \
    --compute-metrics
```

---

## 🔧 Troubleshooting

### Model File Not Found
```bash
# Check if checkpoint exists
ls -lh ../outputs/checkpoints/

# Verify path
python3 -c "import os; print(os.path.abspath('../outputs/checkpoints/best_model.pth'))"
```

### CUDA/GPU Errors
If you get GPU memory errors during inference:
```bash
# Force CPU inference (slower but works)
CUDA_VISIBLE_DEVICES="" python3 inference_unet.py ...

# Or reduce batch size (if using batched inference)
python3 inference_unet.py --batch-size 1 ...
```

### Dimension Mismatch
Make sure your input images match the training data format:
```python
import nibabel as nib
img = nib.load('../Synth_LR_nii/Subject0023_aa.nii')
print(f"Shape: {img.shape}")  # Should be (H, W, num_frames)
print(f"Data type: {img.get_data_dtype()}")
```

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```

---

## 📈 Interpreting Results

### Good Performance Indicators:
- **PSNR > 30 dB** - Generally good reconstruction quality
- **PSNR > 35 dB** - Very good quality
- **SSIM > 0.80** - Good structural similarity
- **SSIM > 0.90** - Excellent structural preservation

### What to Look For:
1. **Quantitative metrics:** PSNR and SSIM scores
2. **Visual quality:** Sharp edges, preserved details
3. **Artifacts:** Check for:
   - Blurring
   - Checkerboard patterns
   - Missing details
   - Hallucinated features

### Red Flags:
- ⚠️ PSNR < 25 dB - Poor reconstruction
- ⚠️ SSIM < 0.70 - Structural distortion
- ⚠️ High variance across frames - Unstable model
- ⚠️ Visual artifacts in error maps - Systematic errors

---

## 🎯 Next Steps

After inference:

1. **Analyze metrics** - Compare to baseline (SENSE reconstruction)
2. **Identify failure cases** - Which subjects/frames perform poorly?
3. **Iterate on training** - Adjust hyperparameters, add data augmentation
4. **Compare architectures** - Try different base_filters values
5. **Final validation** - Test on Dynamic_SENSE data

For detailed analysis, see:
- [GETTING_STARTED.md](GETTING_STARTED.md) - Training workflow
- [UNET_ARCHITECTURE.md](UNET_ARCHITECTURE.md) - Model details
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command cheatsheet

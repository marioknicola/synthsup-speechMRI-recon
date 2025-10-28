# Google Colab Training Guide

Step-by-step instructions for training the U-Net with k-space data in Google Colab.

---

## Prerequisites

### 1. Prepare Data Locally

Create zip files of your data directories:

```bash
# On your local machine:
cd /path/to/your/MSc\ Project

zip -r kspace_mat_FS.zip kspace_mat_FS/
zip -r Synth_LR_nii.zip Synth_LR_nii/
zip -r HR_nii.zip HR_nii/
zip -r Dynamic_SENSE_padded.zip Dynamic_SENSE_padded/  # Optional, for testing
```

### 2. File Sizes (Approximate)

- `kspace_mat_FS.zip`: ~500 MB (21 .mat files)
- `Synth_LR_nii.zip`: ~50 MB (21 .nii files)
- `HR_nii.zip`: ~50 MB (21 .nii files)
- `Dynamic_SENSE_padded.zip`: ~100 MB (42 .nii files, optional)

**Total**: ~700 MB (reasonable for Colab upload)

---

## Step-by-Step Instructions

### Step 1: Create New Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create new notebook: **File → New notebook**
3. Enable GPU: **Runtime → Change runtime type → Hardware accelerator → GPU (T4)**

### Step 2: Clone Repository

```python
# Cell 1: Clone the repository
!git clone https://github.com/YOUR_USERNAME/synthsup-speechMRI-recon.git
%cd synthsup-speechMRI-recon
!pwd
```

Expected output: `/content/synthsup-speechMRI-recon`

### Step 3: Upload Data Files

Upload your prepared zip files using Colab's file upload:

```python
# Cell 2: Upload data files
from google.colab import files
import os

# Create data directory
!mkdir -p /content/data

print("Upload the following zip files:")
print("  1. kspace_mat_FS.zip")
print("  2. Synth_LR_nii.zip")
print("  3. HR_nii.zip")
print("  4. Dynamic_SENSE_padded.zip (optional)")

uploaded = files.upload()

# Move uploaded files to data directory
for filename in uploaded.keys():
    !mv {filename} /content/data/
```

**Note**: Click "Choose Files" button that appears and upload your zip files. This may take a few minutes depending on file size.

### Step 4: Extract Data

```python
# Cell 3: Extract data files
%cd /content/data

!echo "=== Extracting zip files ==="
!unzip -q kspace_mat_FS.zip
!unzip -q Synth_LR_nii.zip
!unzip -q HR_nii.zip

# Optional: Extract test data
!if [ -f Dynamic_SENSE_padded.zip ]; then unzip -q Dynamic_SENSE_padded.zip; fi

!echo ""
!echo "=== Verification ==="
!ls -d kspace_mat_FS Synth_LR_nii HR_nii

%cd /content/synthsup-speechMRI-recon
```

### Step 5: Verify Data

```python
# Cell 4: Check data files
!echo "=== K-Space Files ==="
!ls -1 /content/data/kspace_mat_FS/*.mat | wc -l
!echo ""
!echo "=== NIfTI Files ==="
!ls -1 /content/data/Synth_LR_nii/*.nii | wc -l
!ls -1 /content/data/HR_nii/*.nii | wc -l
```

Expected output:
```
=== K-Space Files ===
21

=== NIfTI Files ===
21
21
```

### Step 6: Install Dependencies (if needed)

```python
# Cell 5: Install packages (usually pre-installed)
!pip install torch torchvision nibabel scipy scikit-image matplotlib
```

### Step 7: Test Dataset

```python
# Cell 6: Test dataset loading
!python3 -c "
from dataset_kspace import create_combined_dataset
ds = create_combined_dataset(
    '/content/data/Synth_LR_nii',
    '/content/data/HR_nii',
    '/content/data/kspace_mat_FS',
    normalize=True
)
print(f'✓ Dataset loaded: {len(ds)} samples')
sample = ds[0]
print(f'✓ Input shape: {sample[\"input\"].shape}')
print(f'✓ Target shape: {sample[\"target\"].shape}')
"
```

Expected output:
```
✓ Dataset loaded: 42 samples
✓ Input shape: torch.Size([1, 312, 410])
✓ Target shape: torch.Size([1, 312, 410])
```

### Step 8: Start Training

```python
# Cell 7: Train model
!python3 train_unet_kspace.py \
    --synth-lr-dir /content/data/Synth_LR_nii \
    --hr-dir /content/data/HR_nii \
    --kspace-fs-dir /content/data/kspace_mat_FS \
    --batch-size 8 \
    --epochs 100 \
    --output-dir /content/outputs_kspace \
    --num-workers 2
```

**Training Configuration:**
- Batch size: 8 (Colab GPU can handle this)
- Epochs: 100 (about 30-45 minutes on T4 GPU)
- Workers: 2 (for data loading)

### Step 9: Monitor Training

Training will display:
```
Epoch [1/100]
Training: 100%|████████| 5/5 [00:08<00:00,  1.65s/it]
Validation: 100%|████████| 1/1 [00:01<00:00,  1.20s/it]

  Train Loss: 0.1234 (L1: 0.1100, SSIM: 0.0134)
  Val Loss:   0.1156 | PSNR: 28.45 dB | SSIM: 0.8234
  Time: 9.2s
  ✓ Saved best model
```

**Expected training time:** ~30-45 minutes (100 epochs on T4 GPU)

### Step 10: Check Results

```python
# Cell 8: View training log
!tail -n 50 /content/outputs_kspace/training.log
```

### Step 11: Test on Dynamic Data (Optional)

```python
# Cell 9: Run inference on Dynamic data
!python3 inference_unet.py \
    --checkpoint /content/outputs_kspace/checkpoints/best_model.pth \
    --input-dir /content/data/Dynamic_SENSE_padded \
    --output-dir /content/reconstructions_dynamic \
    --compute-metrics \
    --visualize
```

---

## Quick Reference: All Commands in One Cell

```python
# === COMPLETE COLAB TRAINING SCRIPT ===

# Clone repository
!git clone https://github.com/YOUR_USERNAME/synthsup-speechMRI-recon.git
%cd synthsup-speechMRI-recon

# Upload data (will prompt for file selection)
from google.colab import files
!mkdir -p /content/data
%cd /content/data
print("Upload: kspace_mat_FS.zip, Synth_LR_nii.zip, HR_nii.zip")
uploaded = files.upload()

# Extract data
!unzip -q kspace_mat_FS.zip
!unzip -q Synth_LR_nii.zip
!unzip -q HR_nii.zip

# Verify data
!ls -1 /content/data/kspace_mat_FS/*.mat | wc -l
!ls -1 /content/data/Synth_LR_nii/*.nii | wc -l
!ls -1 /content/data/HR_nii/*.nii | wc -l

# Train
%cd /content/synthsup-speechMRI-recon
!python3 train_unet_kspace.py \
    --synth-lr-dir /content/data/Synth_LR_nii \
    --hr-dir /content/data/HR_nii \
    --kspace-fs-dir /content/data/kspace_mat_FS \
    --batch-size 8 \
    --epochs 100 \
    --output-dir /content/outputs_kspace \
    --num-workers 2

# View results
!tail -n 20 /content/outputs_kspace/training.log
```

---

## Important Notes

### GPU Settings

- **Use GPU**: Runtime → Change runtime type → T4 GPU
- **Don't use CPU**: Training will be 10-20× slower (3-5 hours)

### Batch Size

- **GPU (T4)**: Use `--batch-size 8`
- **GPU (V100/A100)**: Use `--batch-size 16` or higher
- **If OOM error**: Reduce to `--batch-size 4` or `--batch-size 2`

### Session Timeout

Colab disconnects after:
- **Free tier**: 12 hours max session
- **Idle timeout**: 90 minutes

**Solution**: Training saves checkpoints automatically. Resume with:
```python
!python3 train_unet_kspace.py \
    --resume /content/outputs_kspace/checkpoints/latest_model.pth \
    --epochs 200 \
    --synth-lr-dir /content/data/Synth_LR_nii \
    --hr-dir /content/data/HR_nii \
    --kspace-fs-dir /content/data/kspace_mat_FS \
    --batch-size 8
```

**Note**: If session disconnects, you'll need to re-upload your data zips and extract them again.

### Saving Results

Download trained model:
```python
from google.colab import files
files.download('/content/outputs_kspace/checkpoints/best_model.pth')
```

---

## Troubleshooting

### "No module named 'dataset_kspace'"

**Solution**: Check working directory
```python
import os
print(os.getcwd())  # Should be: /content/synthsup-speechMRI-recon
%cd /content/synthsup-speechMRI-recon
```

### "FileNotFoundError: kspace_mat_FS"

**Solution**: Verify data was extracted
```python
!ls /content/data/kspace_mat_FS/
# If empty, re-upload and extract the zip file
```

### "CUDA out of memory"

**Solution**: Reduce batch size
```python
--batch-size 4  # or 2
```

### Upload interrupted

**Solution**: Colab may timeout on large uploads
- Try uploading one zip at a time
- Compress files further if needed
- Or host files online and use `!wget https://your-url.com/data.zip`

### Training very slow

**Check**: GPU is enabled
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

Expected: `GPU available: True`, `Device: Tesla T4`

---

## Expected Results

### Training Metrics (After 100 Epochs)

| Metric | Expected Value |
|--------|----------------|
| **Val PSNR** | 27-32 dB |
| **Val SSIM** | 0.75-0.85 |
| **Train Time** | 30-45 min (GPU) |
| **Final Loss** | 0.05-0.10 |

### Checkpoint Files

After training, you'll have:
```
outputs_kspace/
├── checkpoints/
│   ├── best_model.pth        # Best validation loss
│   ├── latest_model.pth       # Most recent epoch
│   └── epoch_*.pth            # Per-epoch checkpoints
├── training.log               # Training history
└── test_indices.txt           # Test set indices
```

---

## Next Steps After Training

1. **Download model**: Get `best_model.pth` from Drive
2. **Test on Dynamic data**: Run inference (see Step 10)
3. **Compare results**: vs model trained on NIfTI only
4. **Analyze metrics**: Check PSNR/SSIM improvements

---

## Advanced: Custom Training

### Longer Training

```python
!python3 train_unet_kspace.py \
    --synth-lr-dir /content/data/Synth_LR_nii \
    --hr-dir /content/data/HR_nii \
    --kspace-fs-dir /content/data/kspace_mat_FS \
    --batch-size 8 \
    --epochs 200 \
    --output-dir /content/outputs_kspace_long
```

### Different Loss Weight

```python
# More SSIM emphasis
!python3 train_unet_kspace.py \
    --synth-lr-dir /content/data/Synth_LR_nii \
    --hr-dir /content/data/HR_nii \
    --kspace-fs-dir /content/data/kspace_mat_FS \
    --alpha 0.5  # 50% L1, 50% SSIM \
    --batch-size 8 \
    --epochs 100 \
    --output-dir /content/outputs_kspace
```

### Larger Model

```python
# 31M parameters (vs 7.8M default)
!python3 train_unet_kspace.py \
    --synth-lr-dir /content/data/Synth_LR_nii \
    --hr-dir /content/data/HR_nii \
    --kspace-fs-dir /content/data/kspace_mat_FS \
    --base-filters 64 \
    --batch-size 8 \
    --epochs 100 \
    --output-dir /content/outputs_kspace
```

---

## FAQ

**Q: How much does Colab cost?**  
A: Free tier is sufficient. GPU access included.

**Q: Can I use Colab Pro?**  
A: Yes! Faster GPUs (V100/A100) → 2-3× faster training.

**Q: How long does training take?**  
A: ~30-45 minutes (100 epochs, T4 GPU, batch size 8).

**Q: What if Colab disconnects?**  
A: You'll need to re-upload data, then resume with `--resume /content/outputs_kspace/checkpoints/latest_model.pth`

**Q: Can I train without GPU?**  
A: Yes, but 10-20× slower (~3-5 hours).

**Q: How do I know if training is working?**  
A: Val PSNR should improve from ~16 dB to ~28-32 dB.

---

**Status**: ✅ Ready for Colab training  
**Last Updated**: 27 October 2025

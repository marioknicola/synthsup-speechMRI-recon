# 🚀 Quick Start: Training in Google Colab

**Complete instructions for training the U-Net with k-space data in Google Colab.**

---

## ⚡ Fast Track (Copy & Paste)

### 1. Clone Repo & Upload Data

```python
# Clone repository
!git clone https://github.com/marioknicola/synthsup-speechMRI-recon.git
%cd synthsup-speechMRI-recon

# Upload your data as zip files (use Colab's upload button)
from google.colab import files
import zipfile
import os

# Upload kspace_mat_FS.zip
print("Upload kspace_mat_FS.zip...")
uploaded = files.upload()

# Extract
with zipfile.ZipFile('kspace_mat_FS.zip', 'r') as zip_ref:
    zip_ref.extractall('../')
print(f"✓ Extracted kspace_mat_FS/ ({len(os.listdir('../kspace_mat_FS'))} files)")

# Upload Synth_LR_nii.zip
print("\nUpload Synth_LR_nii.zip...")
uploaded = files.upload()
with zipfile.ZipFile('Synth_LR_nii.zip', 'r') as zip_ref:
    zip_ref.extractall('../')
print(f"✓ Extracted Synth_LR_nii/ ({len(os.listdir('../Synth_LR_nii'))} files)")

# Upload HR_nii.zip
print("\nUpload HR_nii.zip...")
uploaded = files.upload()
with zipfile.ZipFile('HR_nii.zip', 'r') as zip_ref:
    zip_ref.extractall('../')
print(f"✓ Extracted HR_nii/ ({len(os.listdir('../HR_nii'))} files)")

print("\n✓ All data uploaded and extracted!")
```

### 2. Verify Data

```python
!ls ../kspace_mat_FS/*.mat | wc -l  # Should be: 21
!ls ../Synth_LR_nii/*.nii | wc -l   # Should be: 21
!ls ../HR_nii/*.nii | wc -l         # Should be: 21
```

### 3. Train Model

```python
!python3 train_unet_kspace.py \
    --synth-lr-dir ../Synth_LR_nii \
    --hr-dir ../HR_nii \
    --kspace-fs-dir ../kspace_mat_FS \
    --batch-size 8 \
    --epochs 100 \
    --output-dir ../outputs_kspace \
    --num-workers 2
```

**Training time**: ~30-45 minutes on T4 GPU ⏱️

---

## 📋 Prerequisites

### Required Files (Prepare as ZIP archives)

Before starting Colab:

1. **kspace_mat_FS.zip** (~500 MB) - 21 .mat files
2. **Synth_LR_nii.zip** (~50 MB) - 21 .nii files  
3. **HR_nii.zip** (~50 MB) - 21 .nii files
4. *Optional:* **Dynamic_SENSE_padded.zip** (~100 MB) - Test data

**How to create zips** (on your local machine):
```bash
cd "MSc Project"
zip -r kspace_mat_FS.zip kspace_mat_FS/
zip -r Synth_LR_nii.zip Synth_LR_nii/
zip -r HR_nii.zip HR_nii/
zip -r Dynamic_SENSE_padded.zip Dynamic_SENSE_padded/
```

### GPU Setup

1. Create new Colab notebook
2. **Runtime → Change runtime type**
3. Select **GPU** (T4 recommended)
4. Save

---

## 📊 Expected Results

### Training Metrics (After 100 Epochs)

| Metric | Before Training | After Training |
|--------|----------------|----------------|
| **Val PSNR** | ~16 dB | **27-32 dB** |
| **Val SSIM** | ~0.22 | **0.75-0.85** |

### Output Files

```
outputs_kspace/
├── checkpoints/
│   ├── best_model.pth         # Best model (use this!)
│   └── latest_model.pth        # Most recent
├── training.log                # Training history
└── test_indices.txt            # Test set split
```

---

## 🔧 Troubleshooting

### Problem: "CUDA out of memory"

**Solution**: Reduce batch size

```python
--batch-size 4  # or even 2
```

### Problem: "No module named 'dataset_kspace'"

**Solution**: Check you're in the repo directory

```python
import os
print(os.getcwd())  # Should be: /content/synthsup-speechMRI-recon
%cd /content/synthsup-speechMRI-recon
```

### Problem: "FileNotFoundError: kspace_mat_FS"

**Solution**: Verify data was extracted

```python
!ls ../kspace_mat_FS/
# If empty, re-upload and extract the zip file
```

### Problem: Colab disconnected

**Solution**: Resume training

```python
!python3 train_unet_kspace.py \
    --resume ../outputs_kspace/checkpoints/latest_model.pth \
    --epochs 200 \
    ...
```

### Problem: Upload interrupted

**Solution**: Colab may timeout on large uploads

- Try uploading one zip at a time
- Compress files further if needed
- Or host files online and use `!wget`

### Problem: Training is slow

**Check GPU is enabled:**

```python
import torch
print(f"GPU: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

Should show: `GPU: True`, `Device: Tesla T4`

---

## 📈 Monitor Training Progress

### View Real-Time Logs

```python
!tail -f ../outputs_kspace/training.log
```

Press **Ctrl+C** to stop.

### Check Last 20 Lines

```python
!tail -n 20 ../outputs_kspace/training.log
```

### What to Look For

Good training shows:
- ✅ Val PSNR increasing (16 → 28+ dB)
- ✅ Val loss decreasing
- ✅ Train loss decreasing smoothly
- ✅ "Saved best model" messages

---

## 🎯 After Training: Test Model

### On Dynamic Data

```python
!python3 inference_unet.py \
    --checkpoint ../outputs_kspace/checkpoints/best_model.pth \
    --input-dir ../Dynamic_SENSE_padded \
    --output-dir ../reconstructions_dynamic \
    --compute-metrics \
    --visualize
```

### Download Trained Model

```python
from google.colab import files
files.download('../outputs_kspace/checkpoints/best_model.pth')
```

Or access directly from Google Drive on your computer.

---

## 🎨 Advanced Options

### Longer Training

```python
--epochs 200  # More training (better results, takes longer)
```

### Different Loss Balance

```python
--alpha 0.5  # 50% L1, 50% SSIM (default: 84% L1, 16% SSIM)
```

### Larger Model

```python
--base-filters 64  # 31M params (vs 7.8M default)
```

### Lower Learning Rate

```python
--lr 5e-5  # Slower but more stable (default: 1e-4)
```

---

## 📚 Full Documentation

- **Complete Guide**: [`docs/KSPACE_TRAINING.md`](../docs/KSPACE_TRAINING.md)
- **Detailed Colab Guide**: [`docs/COLAB_TRAINING_GUIDE.md`](../docs/COLAB_TRAINING_GUIDE.md)
- **All Documentation**: [`docs/INDEX.md`](../docs/INDEX.md)

---

## ✅ Checklist

Before training:
- [ ] Data uploaded to Google Drive
- [ ] GPU enabled in Colab
- [ ] Verified 21 files in each directory
- [ ] Working directory is correct

After training:
- [ ] Val PSNR reached 27+ dB
- [ ] Downloaded `best_model.pth`
- [ ] Tested on Dynamic data
- [ ] Compared with baseline model

---

## 🆘 Need Help?

1. Check **[Full Colab Guide](../docs/COLAB_TRAINING_GUIDE.md)** for detailed instructions
2. Check **[K-Space Training Guide](../docs/KSPACE_TRAINING.md)** for technical details
3. Check **[Troubleshooting](#-troubleshooting)** section above

---

**Status**: ✅ Ready for Colab training  
**Last Updated**: 27 October 2025  
**Training Time**: ~30-45 minutes (100 epochs, T4 GPU)

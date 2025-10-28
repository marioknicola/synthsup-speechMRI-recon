# Google Colab Training Guide

Complete guide for training the U-Net model in Google Colab with GPU acceleration.

---

## 🚀 Quick Start (5 Minutes)

### 1. Prepare Data on Your Computer

Create zip files of your data:

```bash
cd /path/to/MSc\ Project
zip -r Synth_LR_nii.zip Synth_LR_nii/
zip -r HR_nii.zip HR_nii/
zip -r Dynamic_SENSE_padded.zip Dynamic_SENSE_padded/  # Optional, for testing
```

**File sizes**: ~200 MB total

### 2. Set Up Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create new notebook
3. **Runtime → Change runtime type → GPU (T4)**

### 3. Run Training

Copy-paste this complete script:

```python
# Clone repository
!git clone https://github.com/YOUR_USERNAME/synthsup-speechMRI-recon.git
%cd synthsup-speechMRI-recon

# Upload data
from google.colab import files
import zipfile
import os

!mkdir -p ../data
%cd ../data

print("📤 Upload Synth_LR_nii.zip...")
uploaded = files.upload()
!unzip -q Synth_LR_nii.zip

print("📤 Upload HR_nii.zip...")
uploaded = files.upload()
!unzip -q HR_nii.zip

# Verify
!echo "Files found:"
!ls -1 Synth_LR_nii/*.nii | wc -l
!ls -1 HR_nii/*.nii | wc -l

# Train
%cd /content/synthsup-speechMRI-recon
!python3 train_unet.py \
    --input-dir ../data/Synth_LR_nii \
    --target-dir ../data/HR_nii \
    --batch-size 8 \
    --epochs 100 \
    --output-dir ../outputs \
    --num-workers 2

# View results
!tail -n 20 ../outputs/training.log
```

**Training time**: 30-45 minutes on T4 GPU

---

## 📋 Detailed Instructions

### Step 1: Enable GPU

1. Open new Colab notebook
2. **Runtime → Change runtime type**
3. Select **Hardware accelerator: GPU**
4. Choose **T4** (free tier) or **V100/A100** (Colab Pro)
5. Click **Save**

Verify GPU is active:

```python
import torch
print(f"GPU: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

Expected: `GPU: True`, `Device: Tesla T4`

### Step 2: Clone Repository

```python
!git clone https://github.com/YOUR_USERNAME/synthsup-speechMRI-recon.git
%cd synthsup-speechMRI-recon
!pwd  # Should show: /content/synthsup-speechMRI-recon
```

### Step 3: Upload and Extract Data

```python
from google.colab import files
import zipfile

# Create data directory
!mkdir -p ../data
%cd ../data

# Upload Synth_LR_nii.zip
print("Upload Synth_LR_nii.zip (click Choose Files button)")
uploaded = files.upload()
!unzip -q Synth_LR_nii.zip
print(f"✓ Extracted {len(!ls -1 Synth_LR_nii/*.nii)} LR files")

# Upload HR_nii.zip
print("\nUpload HR_nii.zip")
uploaded = files.upload()
!unzip -q HR_nii.zip
print(f"✓ Extracted {len(!ls -1 HR_nii/*.nii)} HR files")

%cd /content/synthsup-speechMRI-recon
```

### Step 4: Verify Data

```python
!echo "=== Data Verification ==="
!ls -1 ../data/Synth_LR_nii/*.nii | wc -l  # Should be: 21
!ls -1 ../data/HR_nii/*.nii | wc -l         # Should be: 21
```

### Step 5: Train Model

```python
!python3 train_unet.py \
    --input-dir ../data/Synth_LR_nii \
    --target-dir ../data/HR_nii \
    --batch-size 8 \
    --epochs 100 \
    --output-dir ../outputs \
    --num-workers 2
```

**Training configuration:**
- Batch size: 8 (T4 GPU can handle this)
- Epochs: 100 (~30-45 minutes)
- Base filters: 32 (default, 7.8M parameters)

### Step 6: Monitor Training

View training progress:

```python
# View last 20 lines
!tail -n 20 ../outputs/training.log

# Or follow in real-time (Ctrl+C to stop)
!tail -f ../outputs/training.log
```

Expected output:
```
Epoch [50/100]
Training: 100%|████████| 5/5 [00:08<00:00]
Validation: 100%|████████| 1/1 [00:01<00:00]

  Train Loss: 0.0852 (L1: 0.0756, SSIM: 0.0096)
  Val Loss:   0.0794 | PSNR: 28.45 dB | SSIM: 0.8234
  ✓ Saved best model
```

### Step 7: Download Trained Model

```python
from google.colab import files
files.download('../outputs/checkpoints/best_model.pth')
```

---

## 🎯 Testing the Model

### On Validation Data

```python
!python3 inference_unet.py \
    --checkpoint ../outputs/checkpoints/best_model.pth \
    --input-dir ../data/Synth_LR_nii \
    --output-dir ../reconstructions_val \
    --compute-metrics
```

### On Dynamic Data (Optional)

First, upload Dynamic_SENSE_padded.zip:

```python
%cd ../data
print("Upload Dynamic_SENSE_padded.zip")
uploaded = files.upload()
!unzip -q Dynamic_SENSE_padded.zip

%cd /content/synthsup-speechMRI-recon

!python3 inference_unet.py \
    --checkpoint ../outputs/checkpoints/best_model.pth \
    --input-dir ../data/Dynamic_SENSE_padded \
    --output-dir ../reconstructions_dynamic \
    --compute-metrics \
    --visualize
```

---

## 📊 Expected Results

### Training Metrics (After 100 Epochs)

| Metric | Initial | After Training |
|--------|---------|----------------|
| **Val PSNR** | ~16 dB | **27-32 dB** |
| **Val SSIM** | ~0.22 | **0.75-0.85** |
| **Val Loss** | ~0.20 | **0.05-0.10** |

### Output Files

```
outputs/
├── checkpoints/
│   ├── best_model.pth         # Use this for inference
│   └── latest_model.pth        # Most recent checkpoint
├── training.log                # Complete training history
└── test_indices.txt            # Test set split (reproducible)
```

---

## 🔧 Troubleshooting

### Problem: CUDA Out of Memory

**Solution**: Reduce batch size

```python
--batch-size 4  # or even 2
```

### Problem: Session Disconnected

Colab disconnects after ~12 hours (or 90 min idle).

**Solution**: Resume training from checkpoint

```python
!python3 train_unet.py \
    --resume ../outputs/checkpoints/latest_model.pth \
    --input-dir ../data/Synth_LR_nii \
    --target-dir ../data/HR_nii \
    --epochs 200 \
    --batch-size 8 \
    --output-dir ../outputs
```

**Note**: You'll need to re-upload your data after reconnecting.

### Problem: Upload Interrupted

**Solution 1**: Upload one file at a time

**Solution 2**: Host files online and download with wget
```python
!wget https://your-url.com/Synth_LR_nii.zip
!unzip -q Synth_LR_nii.zip
```

### Problem: "No module named 'dataset'"

**Solution**: Verify working directory

```python
import os
print(os.getcwd())  # Should be: /content/synthsup-speechMRI-recon

# If not, navigate to it:
%cd /content/synthsup-speechMRI-recon
```

### Problem: FileNotFoundError

**Solution**: Check data was extracted

```python
!ls ../data/Synth_LR_nii/
# If empty, re-upload and extract zip file
```

### Problem: Training Very Slow

**Check GPU is enabled:**

```python
import torch
print(f"GPU: {torch.cuda.is_available()}")
# Should show: GPU: True
```

If False, go to **Runtime → Change runtime type → GPU**

---

## ⚙️ Advanced Options

### Longer Training (Better Results)

```python
--epochs 200  # Takes ~60-90 minutes
```

### Larger Model (More Parameters)

```python
--base-filters 64  # 31M params (vs 7.8M default)
--batch-size 4     # Reduce batch size for larger model
```

### Different Loss Balance

```python
--alpha 0.5  # 50% L1, 50% SSIM (default: 84% L1, 16% SSIM)
```

### Lower Learning Rate (More Stable)

```python
--lr 5e-5  # Default: 1e-4
```

### Save Checkpoints More Frequently

```python
--save-interval 5  # Save every 5 epochs (default: 10)
```

---

## 💡 Tips for Colab

### Keep Session Alive

Colab disconnects after 90 min of inactivity. To prevent this:

1. Keep the browser tab open
2. Occasionally run a cell (e.g., `!date`)
3. Or use browser extensions to keep tab active

### Use Google Drive for Persistence

Mount Drive to save outputs:

```python
from google.colab import drive
drive.mount('/content/drive')

# Train with outputs in Drive
!python3 train_unet.py \
    --input-dir ../data/Synth_LR_nii \
    --target-dir ../data/HR_nii \
    --output-dir /content/drive/MyDrive/msc_outputs \
    --batch-size 8 \
    --epochs 100
```

Benefits:
- Outputs persist after disconnect
- Can resume training later
- Access files from any device

### Monitor GPU Usage

```python
# Check GPU memory
!nvidia-smi

# More detailed monitoring
!watch -n 1 nvidia-smi  # Updates every second
```

---

## 📚 Additional Resources

- **Getting Started**: [`GETTING_STARTED.md`](GETTING_STARTED.md)
- **Inference Guide**: [`INFERENCE.md`](INFERENCE.md)
- **U-Net Architecture**: [`README.md`](README.md)
- **Full Documentation**: [`INDEX.md`](INDEX.md)

---

## ✅ Pre-Flight Checklist

Before training:
- [ ] GPU enabled in Colab
- [ ] Data zipped on local machine
- [ ] Repository cloned
- [ ] Data uploaded and extracted
- [ ] Verified 21 files in each directory

After training:
- [ ] Val PSNR reached 27+ dB
- [ ] Downloaded `best_model.pth`
- [ ] Saved training.log for reference

---

**Status**: ✅ Ready for production use  
**Last Updated**: 28 October 2025  
**Training Time**: 30-45 minutes (100 epochs, T4 GPU)

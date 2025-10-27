# Getting Started with Synthetically Supervised MRI Reconstruction

This guide will walk you through setting up and using the complete pipeline for MRI reconstruction using both classical SENSE and deep learning (U-Net) approaches.

## 📁 Project Structure

Your workspace should look like this:

```
MSc Project/
├── synthsup-speechMRI-recon/         # THIS REPOSITORY (code only)
│   ├── .github/
│   │   └── copilot-instructions.md
│   ├── sense_reconstruction.py        # Classical SENSE reconstruction
│   ├── unet_model.py                  # U-Net architecture (lightweight baseline)
│   ├── dataset.py                     # PyTorch data loaders
│   ├── train_unet.py                  # Training script (80/10/10 split)
│   ├── inference_unet.py              # Inference script
│   ├── utils/                         # Utility scripts
│   │   ├── __init__.py
│   │   ├── synthetic_undersampling.py
│   │   ├── PSNR_and_SSIM.py
│   │   ├── niftNormaliser.py
│   │   ├── nifti2png.py
│   │   ├── gaussian_noise_injection.py
│   │   ├── rician_noise_injection.py
│   │   └── resample.py
│   ├── dataloading_currentlyUNUSED.py # Deprecated script
│   ├── requirements.txt
│   ├── README.md
│   ├── UNET_README.md
│   ├── UNET_ARCHITECTURE.md           # Detailed architecture docs
│   ├── QUICK_REFERENCE.md             # Command cheat sheet
│   └── GETTING_STARTED.md             # This file
│
├── kspace_mat_US/                     # Undersampled k-space data (MAT files)
│   ├── kspace_Subject0023_aa.mat
│   ├── kspace_Subject0024_aa.mat
│   └── ...
│
├── kspace_mat_FS/                     # Fully-sampled k-space (if available)
│   └── ...
│
├── sensitivity_maps/                  # Coil sensitivity maps (MAT files)
│   ├── sens_Subject0023_*.mat
│   └── ...
│
├── Synth_LR_nii/                      # 🎯 TRAINING INPUTS (synthetic undersampled)
│   └── *.nii files                    #    Used for 80/10/10 train/val/test split
│
├── HR_nii/                            # 🎯 TRAINING TARGETS (high-res ground truth)
│   └── *.nii files                    #    Paired with Synth_LR_nii
│
├── Dynamic_SENSE/                     # 🧪 FINAL TEST DATA (real accelerated scans)
│   └── *.nii files                    #    Reserved for post-training evaluation
│
├── Dynamic_SENSE_padded/              # Padded version of Dynamic_SENSE
│   └── *.nii files
│
├── outputs/                           # Training outputs (created automatically)
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   └── checkpoint_epoch_*.pth
│   └── logs/                          # TensorBoard logs
│
└── reconstructions/                   # Inference outputs (created automatically)
    ├── recon_*.nii
    └── visualizations/
```

## ⚠️ CRITICAL RULE

**NEVER save generated data inside `synthsup-speechMRI-recon/` folder!**

All outputs (NIfTI files, checkpoints, logs, MAT files) must go to parent `MSc Project/` directory.

---

## 🚀 Step 1: Initial Setup

### 1.1 Navigate to Repository

```bash
cd "MSc Project/synthsup-speechMRI-recon"
```

### 1.2 Install Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- `numpy`, `scipy`, `matplotlib` - Scientific computing
- `nibabel` - NIfTI file handling
- `pydicom` - DICOM support
- `scikit-image` - Image metrics (PSNR, SSIM)
- `torch`, `torchvision` - PyTorch for U-Net
- `tqdm` - Progress bars
- `h5py` - HDF5 file support

### 1.3 Verify Installation

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Expected output:**
```
PyTorch: 2.x.x
CUDA available: True  # or False if no GPU
```

---

## 📊 Step 2: Verify Data Structure

### 2.1 Check Training Data

```bash
cd ..  # Go to MSc Project/
ls Synth_LR_nii/ | head -5
ls HR_nii/ | head -5
```

**Expected:** You should see matching `.nii` files in both directories.

### 2.2 Check Validation Data

```bash
ls Dynamic_SENSE/ | head -5
ls Dynamic_SENSE_padded/ | head -5
```

### 2.3 Check K-Space Data

```bash
ls kspace_mat_US/*.mat | head -3
ls sensitivity_maps/*.mat | head -3
```

**If data is missing:** Contact your supervisor or check the data preparation pipeline.

---

## 🧪 Step 3: Test Classical SENSE Reconstruction (Optional)

This step is optional but helps verify your setup and generate validation data if needed.

### 3.1 Quick Smoke Test

```bash
cd synthsup-speechMRI-recon/
python3 sense_reconstruction.py --smoke-test --plot
```

**What happens:**
- Loads default k-space and coil maps
- Processes only 5 frames, 20 rows (quick test)
- Shows visualization plots
- Saves to `../Dynamic_SENSE/` and `../Dynamic_SENSE_padded/`

**Expected time:** ~1-2 minutes

### 3.2 Full SENSE Reconstruction (Optional)

```bash
python3 sense_reconstruction.py \
    --kspace ../kspace_mat_US/kspace_Subject0026_vv.mat \
    --coilmap ../sensitivity_maps/sens_Subject0026_Exam17853_80x82x100_nC22.mat \
    --output-dir .. \
    --subject-id Subject0026_vv
```

**Expected time:** ~30-60 minutes (depends on data size)

---

## 🧠 Step 4: Test U-Net Model

### 4.1 Verify Model Loads

```bash
python3 unet_model.py
```

**Expected output:**
```
Using device: cuda  # or cpu
Model architecture:
Input shape: torch.Size([2, 1, 312, 410])
Output shape: torch.Size([2, 1, 312, 410])

Total trainable parameters: 31,037,633

✓ Model test passed!
```

### 4.2 Test Dataset Loader

```bash
python3 dataset.py
```

**Expected output:**
```
Found X input files
Found X target files
Total frames in dataset: XXX

Sample keys: dict_keys(['input', 'target', 'file_idx', 'frame_idx', 'filename'])
Input shape: torch.Size([1, H, W])
Target shape: torch.Size([1, H, W])
...
✓ Dataset test passed!
```

**If this fails:** Check that `../Synth_LR_nii/` and `../HR_nii/` directories exist and contain `.nii` files.

---

## 🏋️ Step 5: Train the U-Net

### 5.1 Start Training

```bash
python3 train_unet.py \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --output-dir ../outputs \
    --epochs 100 \
    --batch-size 4 \
    --lr 1e-4 \
    --base-filters 32
```

**Command breakdown:**
- `--input-dir`: Directory with synthetic undersampled inputs (Synth_LR_nii)
- `--target-dir`: Directory with high-resolution ground truth (HR_nii)
- `--output-dir`: Where to save checkpoints, logs, and test indices
- `--epochs`: Number of training epochs
- `--batch-size`: Images per batch (reduce if GPU OOM)
- `--lr`: Learning rate
- `--base-filters`: U-Net width (32 = lightweight baseline, ~8M parameters)

**Data Split:**
- Automatically splits data 80/10/10 (train/val/test) with seed=42
- Test indices saved to `../outputs/test_indices.txt` for reproducibility
- Dynamic_SENSE reserved for final testing after training complete

### 5.2 Monitor Training

**Option 1: Terminal output**

You'll see progress bars showing:
```
Epoch 0 [Train]: 100%|████████| 250/250 [02:15<00:00, 1.85it/s, loss=0.0234]
Epoch 0 [Val]: 100%|██████████| 63/63 [00:30<00:00, 2.07it/s, loss=0.0198]

Epoch 0/100
  Train Loss: 0.0234
  Val Loss:   0.0198
  LR: 0.000100
  Time: 165.3s
```

**Option 2: TensorBoard (recommended)**

Open a new terminal:
```bash
cd "MSc Project"
tensorboard --logdir outputs/logs
```

Then open browser to: http://localhost:6006

You'll see real-time plots of:
- Training loss
- Validation loss
- Learning rate schedule

### 5.3 Training Time Estimates

| Hardware | Batch Size | Time per Epoch | 100 Epochs |
|----------|------------|----------------|------------|
| RTX 3090 | 8 | ~2 min | ~3.5 hours |
| RTX 3080 | 4 | ~3 min | ~5 hours |
| RTX 2080 Ti | 4 | ~4 min | ~7 hours |
| CPU only | 2 | ~30 min | ~50 hours ⚠️ |

### 5.4 What If Training Crashes?

**CUDA Out of Memory:**
```bash
python3 train_unet.py \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --output-dir ../outputs \
    --batch-size 2 \          # ← Reduced
    --base-filters 16 \       # ← Further reduced
    --epochs 100
```

**Resume from Checkpoint:**
```bash
python3 train_unet.py \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --val-input-dir ../Dynamic_SENSE \
    --val-target-dir ../Dynamic_SENSE_padded \
    --output-dir ../outputs \
    --resume ../outputs/checkpoints/checkpoint_epoch_50.pth \  # ← Resume
    --epochs 100
```

### 5.5 When is Training Done?

Monitor validation loss in TensorBoard:
- **Good:** Val loss decreases and plateaus
- **Overfitting:** Train loss ↓ but val loss ↑ (early stop!)
- **Underfitting:** Both losses still decreasing (train longer)

**Best model is saved automatically** to `../outputs/checkpoints/best_model.pth`

---

## 🔮 Step 6: Run Inference

### 6.1 Basic Inference

```bash
python3 inference_unet.py \
    --checkpoint ../outputs/checkpoints/best_model.pth \
    --input-dir ../Synth_LR_nii \
    --output-dir ../reconstructions
```

**What happens:**
- Loads trained model
- Processes all `.nii` files in input directory
- Saves reconstructed files to output directory

### 6.2 Inference with Metrics

```bash
python3 inference_unet.py \
    --checkpoint ../outputs/checkpoints/best_model.pth \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --output-dir ../reconstructions \
    --compute-metrics \
    --visualize \
    --num-vis 10
```

**Additional features:**
- `--compute-metrics`: Calculate PSNR and SSIM
- `--visualize`: Generate comparison plots
- `--num-vis`: Number of frames to visualize per file

**Expected output:**
```
Metrics for Subject0023_aa.nii:
  Avg PSNR: 32.45 dB
  Avg SSIM: 0.8923

Metrics for Subject0024_aa.nii:
  Avg PSNR: 31.87 dB
  Avg SSIM: 0.8856
...

OVERALL METRICS
Average PSNR: 32.16 dB
Average SSIM: 0.8890
```

### 6.3 Inference on Validation Set

```bash
python3 inference_unet.py \
    --checkpoint ../outputs/checkpoints/best_model.pth \
    --input-dir ../Dynamic_SENSE \
    --target-dir ../Dynamic_SENSE_padded \
    --output-dir ../reconstructions_val \
    --compute-metrics \
    --visualize
```

---

## 📈 Step 7: Analyze Results

### 7.1 Check Reconstructed Images

```bash
cd ../reconstructions
ls *.nii
```

Open with your favorite NIfTI viewer (e.g., ITK-SNAP, FSLeyes, 3D Slicer).

### 7.2 Review Visualizations

```bash
cd visualizations/
open *.png  # macOS
# or: eog *.png  # Linux
# or: explorer .  # Windows
```

Each image shows: Input | Reconstruction | Target

### 7.3 Compare Metrics

Create a simple comparison script or use Excel/Python:

```python
import pandas as pd

results = {
    'Method': ['Zero-filled', 'SENSE', 'U-Net'],
    'PSNR (dB)': [28.5, 30.2, 32.1],
    'SSIM': [0.78, 0.85, 0.89]
}

df = pd.DataFrame(results)
print(df)
```

---

## 🛠️ Common Issues and Solutions

### Issue 1: "No files found in directory"

**Solution:** Check your directory structure matches Step 2.

```bash
cd "MSc Project"
ls Synth_LR_nii/ | wc -l  # Should show file count
ls HR_nii/ | wc -l
```

### Issue 2: "CUDA out of memory"

**Solution:** Reduce batch size or model size:

```bash
--batch-size 2 --base-filters 32
```

### Issue 3: "Module not found: torch"

**Solution:** Reinstall requirements:

```bash
pip install --upgrade -r requirements.txt
```

### Issue 4: Training loss not decreasing

**Possible causes:**
1. **Learning rate too high/low:** Try `--lr 5e-4` or `--lr 5e-5`
2. **Bad data normalization:** Check dataset.py is normalizing correctly
3. **Data mismatch:** Verify input/target pairs align correctly

### Issue 5: Validation loss higher than training loss

**This is normal!** If gap is large:
- Add data augmentation (future work)
- Reduce model complexity (`--base-filters 32`)
- Get more training data

### Issue 6: "Permission denied" when saving

**Solution:** Check you're not trying to save inside the repository:

```bash
# ❌ WRONG
--output-dir ./outputs

# ✅ CORRECT
--output-dir ../outputs
```

---

## 📚 Next Steps

### Advanced Training

**Experiment with hyperparameters:**

```bash
# More aggressive training
python3 train_unet.py \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --val-input-dir ../Dynamic_SENSE \
    --val-target-dir ../Dynamic_SENSE_padded \
    --epochs 200 \
    --lr 5e-4 \
    --loss-alpha 0.7  # More SSIM weight

# Smaller, faster model
python3 train_unet.py \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --val-input-dir ../Dynamic_SENSE \
    --val-target-dir ../Dynamic_SENSE_padded \
    --base-filters 32 \  # Fewer parameters
    --batch-size 8      # Larger batches
```

### Custom Data

**Add your own subjects:**

1. Place k-space MAT files in `../kspace_mat_US/`
2. Generate coil maps in `../sensitivity_maps/`
3. Run SENSE reconstruction
4. Retrain U-Net with new data

### Production Deployment

**Export trained model:**

```python
import torch
model = torch.load('../outputs/checkpoints/best_model.pth')
torch.save(model['model_state_dict'], '../model_weights_only.pth')
```

---

## 📖 Additional Documentation

- **`UNET_ARCHITECTURE.md`** - Detailed U-Net explanation with diagrams
- **`UNET_README.md`** - Quick reference for U-Net pipeline
- **`.github/copilot-instructions.md`** - AI agent guidelines
- **`README.md`** - Project overview

---

## 🎓 Learning Resources

### Understanding U-Net
- Original paper: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- [Stanford CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)

### MRI Reconstruction
- [fastMRI Dataset](https://fastmri.org/)
- SENSE paper: [Pruessmann et al., 1999](https://doi.org/10.1002/mrm.1910420517)

### PyTorch
- [Official PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)

---

## ✅ Checklist: Am I Ready to Start?

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] PyTorch with CUDA working (if GPU available)
- [ ] Data directories exist and contain files:
  - [ ] `../Synth_LR_nii/` (training inputs)
  - [ ] `../HR_nii/` (training targets)
  - [ ] `../Dynamic_SENSE/` (validation inputs)
  - [ ] `../Dynamic_SENSE_padded/` (validation targets)
- [ ] Model test passes (`python3 unet_model.py`)
- [ ] Dataset test passes (`python3 dataset.py`)
- [ ] Have 10-50 GB free disk space for outputs
- [ ] Have 10+ GB GPU memory (or patience for CPU training)

**If all checked:** You're ready! Start with Step 5 (Training).

---

## 🆘 Getting Help

If stuck:
1. Read error messages carefully
2. Check this guide's "Common Issues" section
3. Review `.github/copilot-instructions.md`
4. Ask your supervisor
5. Check GitHub Issues (if public repo)

**Include in your help request:**
- Full error message
- Command you ran
- Output of `python3 --version` and `pip list | grep torch`
- GPU info (if applicable): `nvidia-smi`

---

## 🎉 Success Looks Like

After completing this guide, you should have:

✅ Trained U-Net model in `../outputs/checkpoints/best_model.pth`  
✅ Training curves showing decreasing loss  
✅ PSNR > 30 dB, SSIM > 0.85 on validation  
✅ Reconstructed NIfTI files in `../reconstructions/`  
✅ Visualization images showing input/output/target comparisons  
✅ Understanding of the complete pipeline  

**Congratulations!** You've successfully set up and trained a deep learning model for MRI reconstruction! 🎊

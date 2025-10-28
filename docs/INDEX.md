# Documentation Index

Complete reference for synthsup-speechMRI-recon project documentation.

---

## 🚀 Quick Start

**New to the project?** → [`GETTING_STARTED.md`](GETTING_STARTED.md)  
**Training in Google Colab?** → [`USE_IN_COLAB.md`](USE_IN_COLAB.md)  
**Want to understand U-Net?** → [`README.md`](README.md)

---

## 📚 Core Documentation

| Document | Description |
|----------|-------------|
| **[GETTING_STARTED.md](GETTING_STARTED.md)** | Step-by-step local setup and usage guide |
| **[USE_IN_COLAB.md](USE_IN_COLAB.md)** | Complete Google Colab training guide |
| **[README.md](README.md)** | U-Net architecture and training details |
| **[INFERENCE.md](INFERENCE.md)** | Running inference and evaluation |
| **[DATA_CONSISTENCY.md](DATA_CONSISTENCY.md)** | Physics-based reconstruction constraints (optional) |
| **[CHANGELOG.md](CHANGELOG.md)** | Version history and updates |

---

## 🎯 Common Tasks

### Train Model Locally

```bash
python3 train_unet.py \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --epochs 100 \
    --batch-size 4
```

**See:** [README.md](README.md) for full training details

### Train in Google Colab

```python
# Complete script in USE_IN_COLAB.md
!git clone https://github.com/YOUR_USERNAME/synthsup-speechMRI-recon.git
# ... upload data ...
!python3 train_unet.py --input-dir ../data/Synth_LR_nii --target-dir ../data/HR_nii
```

**See:** [USE_IN_COLAB.md](USE_IN_COLAB.md) for step-by-step guide

### Run Inference

```bash
python3 inference_unet.py \
    --checkpoint ../outputs/checkpoints/best_model.pth \
    --input-dir ../Dynamic_SENSE_padded_95f \
    --output-dir ../reconstructions \
    --compute-metrics \
    --visualize
```

**See:** [INFERENCE.md](INFERENCE.md) for full inference options

---

## 🗂️ Key Concepts

### Data Structure

```
MSc Project/
├── synthsup-speechMRI-recon/       # Repository (code only)
│   ├── unet_model.py
│   ├── dataset.py
│   ├── train_unet.py
│   ├── inference_unet.py
│   └── docs/
│
├── Synth_LR_nii/                   # Training input (21 files)
├── HR_nii/                         # Training target (21 files)
├── Dynamic_SENSE_padded_95f/       # Test data (43 files × 95 frames)
│
├── outputs/                        # Training outputs
│   ├── checkpoints/
│   ├── logs/
│   └── training.log
│
└── reconstructions/                # Inference outputs
```

### Data Normalization

**Per-file [0, 1] normalization** (automatic):
- Computes min/max for entire file at initialization
- All frames use same normalization statistics
- Prevents frame-to-frame intensity variations
- Automatic denormalization during inference

**Benefits:**
- Solves intensity mismatch between datasets
- Prevents negative values in predictions
- Stable training with consistent gradients

### Frame Preprocessing

**First 5 frames removal:**
```bash
python remove_first_5_frames.py \
    --dynamic-dir ../Dynamic_SENSE_padded \
    --dynamic-output ../Dynamic_SENSE_padded_95f
```

**Rationale:** First 5 frames are 2.3× brighter due to sequence dynamics

---

## 📊 Expected Results

After 100 epochs of training:

| Metric | Initial | After Training |
|--------|---------|----------------|
| **Val PSNR** | ~16 dB | **27-32 dB** |
| **Val SSIM** | ~0.22 | **0.75-0.85** |
| **Training Time** | - | 30-45 min (Colab T4 GPU) |

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution | Guide |
|-------|----------|-------|
| CUDA out of memory | Reduce `--batch-size 2` | README.md |
| Dataset not found | Check data paths with `ls` | GETTING_STARTED.md |
| Poor convergence | Adjust `--lr` or `--alpha` | README.md |
| Colab disconnected | Use `--resume` to continue | USE_IN_COLAB.md |

---

## 📖 Project Structure

```
docs/
├── INDEX.md                    # This file
├── GETTING_STARTED.md          # Local setup guide
├── USE_IN_COLAB.md             # Google Colab guide
├── README.md                   # U-Net architecture & training
├── INFERENCE.md                # Inference & evaluation
├── DATA_CONSISTENCY.md         # Physics constraints (optional)
└── CHANGELOG.md                # Version history
```

---

## ✅ Status

- ✅ Per-file [0,1] normalization implemented
- ✅ First 5 frames preprocessing available
- ✅ Documentation consolidated (15 → 7 files)
- ✅ Google Colab training guide complete
- ✅ Ready for production use

**Last Updated:** 28 October 2025

````

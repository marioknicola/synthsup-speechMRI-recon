# Changelog

## Documentation Reorganization & Colab Support (October 27, 2025)

### Changes

#### 1. Documentation Organization ✅
- **Moved all .md files to `docs/` folder** (except README.md)
  - `docs/GETTING_STARTED.md` - Step-by-step tutorial
  - `docs/UNET_README.md` - Quick reference
  - `docs/UNET_ARCHITECTURE.md` - Architecture details
  - `docs/QUICK_REFERENCE.md` - Command cheat sheet
  - `docs/CHANGELOG.md` - This file
- **Updated README.md** - Comprehensive project overview with badges, quick start, and documentation links

#### 2. Google Colab Support ✅
- **Added `colab_training_template.ipynb`** - Ready-to-use Colab notebook
- **Includes:**
  - GPU detection and verification
  - Google Drive mounting
  - Automatic dependency installation
  - Training with Drive-backed persistence
  - TensorBoard integration
  - Checkpoint management
  - Resume training after disconnection
  - Tips for preventing Colab timeouts

#### 3. Updated References
- All internal documentation links updated to reflect `docs/` structure
- `.github/copilot-instructions.md` updated with new paths

### New Files
- `colab_training_template.ipynb` - Complete Colab training workflow
- Enhanced `README.md` with Colab instructions

---

## Repository Reorganization & Baseline Configuration (October 27, 2025)

### Major Changes

#### 1. Training Data Strategy ✅
- **Changed from:** Separate validation directories (Dynamic_SENSE/Dynamic_SENSE_padded)
- **Changed to:** 80/10/10 split from Synth_LR_nii/HR_nii synthetic pairs
- **Rationale:** 
  - More consistent evaluation as more training data is added
  - Dynamic_SENSE reserved for final independent testing after training
  - Reproducible splits with fixed seed=42
  - Test indices saved to `outputs/test_indices.txt`

#### 2. U-Net Model Configuration ✅
- **Changed from:** `base_filters=64` (~31M parameters)
- **Changed to:** `base_filters=32` (~7.8M parameters, default)
- **Rationale:**
  - Lightweight baseline for fair comparison
  - Project goal is establishing baseline, not optimized model
  - Reduces training time and memory requirements
  - Still achieves good reconstruction quality

#### 3. Repository Structure ✅
- **New structure:**
  ```
  synthsup-speechMRI-recon/
  ├── sense_reconstruction.py
  ├── unet_model.py
  ├── dataset.py
  ├── train_unet.py
  ├── inference_unet.py
  ├── utils/                          # NEW
  │   ├── __init__.py
  │   ├── synthetic_undersampling.py
  │   ├── PSNR_and_SSIM.py
  │   ├── niftNormaliser.py
  │   ├── nifti2png.py
  │   ├── gaussian_noise_injection.py
  │   ├── rician_noise_injection.py
  │   └── resample.py
  ├── dataloading_currentlyUNUSED.py  # RENAMED
  ├── requirements.txt
  └── [documentation files]
  ```

#### 4. CLI Changes ✅

**train_unet.py:**
- **Removed arguments:**
  - `--val-input-dir` (no longer needed)
  - `--val-target-dir` (no longer needed)
  - `--train-split` (fixed at 0.8)
- **Changed defaults:**
  - `--base-filters`: 64 → 32
- **New behavior:**
  - Automatically splits data 80/10/10 with seed=42
  - Saves test indices to `{output_dir}/test_indices.txt`
  - Prints clear dataset split information

**Example old command:**
```bash
python3 train_unet.py \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --val-input-dir ../Dynamic_SENSE \
    --val-target-dir ../Dynamic_SENSE_padded \
    --base-filters 64
```

**Example new command:**
```bash
python3 train_unet.py \
    --input-dir ../Synth_LR_nii \
    --target-dir ../HR_nii \
    --base-filters 32  # Default, can be omitted
```

### Documentation Updates ✅

All documentation updated to reflect new structure:
- ✅ `.github/copilot-instructions.md` - Repository conventions and structure
- ✅ `GETTING_STARTED.md` - Complete tutorial with new commands
- ✅ `UNET_README.md` - Quick reference updated
- ✅ `QUICK_REFERENCE.md` - Command cheat sheet updated
- ✅ `UNET_ARCHITECTURE.md` - Parameter counts for base_filters=32 and 64

### Migration Guide

If you have existing training runs or scripts:

1. **Update training commands:**
   - Remove `--val-input-dir` and `--val-target-dir` arguments
   - Optionally change `--base-filters 64` to `--base-filters 32`
   - Data will be automatically split 80/10/10

2. **Update imports for utility functions:**
   ```python
   # Old
   from PSNR_and_SSIM import calculate_psnr
   
   # New
   from utils.PSNR_and_SSIM import calculate_psnr
   ```

3. **Test set evaluation:**
   - After training, use saved test indices from `outputs/test_indices.txt`
   - Use `inference_unet.py` with the held-out test set
   - Then evaluate on Dynamic_SENSE for final independent validation

### Performance Expectations

With `base_filters=32`:
- Training memory: ~5-8 GB (batch_size=4)
- Inference time: ~30-60 ms per frame (GPU)
- Parameters: ~7.8M (vs ~31M for base_filters=64)

### Backward Compatibility

- ⚠️ **Breaking:** Old training commands with `--val-input-dir` will error
- ✅ **Compatible:** Existing trained models work with `inference_unet.py`
- ✅ **Compatible:** All data formats and file structures unchanged

---

## Previous Changes

### Initial U-Net Pipeline Setup
- Created complete U-Net training pipeline
- Added `unet_model.py`, `dataset.py`, `train_unet.py`, `inference_unet.py`
- Added comprehensive documentation

### SENSE Reconstruction Enhancement
- Made `sense_reconstruction.py` configurable with argparse
- Changed from hardcoded absolute paths to configurable defaults
- All outputs default to parent directory (`../`)

### Dependency Management
- Created `requirements.txt` with pinned versions
- Ensures reproducible environment setup

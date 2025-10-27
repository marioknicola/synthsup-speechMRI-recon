## Goal
Make AI coding assistants productive quickly in this repository: explain the big-picture architecture, key data shapes and file conventions, common entry points, and gotchas that appear repeatedly in the code.

## High-level architecture (big picture)
- Purpose: synth-supervised deep learning + classical recon utilities for dynamic speech MRI. Primary flows are data preparation (NIfTI / resampling / synthetic undersampling), classical reconstruction (SENSE), U-Net training for DL reconstruction, and metrics/visualisation.
- **Repository structure:**
  - **Core scripts (root):**
    - `sense_reconstruction.py` — Classical SENSE reconstruction (single-file entry point)
    - `unet_model.py` — Lightweight U-Net baseline (default: 32 base filters, ~7.8M params)
    - `dataset.py` — PyTorch data loaders for paired undersampled→fully-sampled data
    - `train_unet.py` — Training script with 80/10/10 split, L1+SSIM loss, checkpointing
    - `inference_unet.py` — Inference with metrics computation and visualization
    - `dataloading_currentlyUNUSED.py` — Deprecated script marked as unused
  - **Utilities (utils/):**
    - `synthetic_undersampling.py` — Creates undersampled k-space from NIfTI
    - `PSNR_and_SSIM.py` — Image quality metrics
    - `niftNormaliser.py` — Normalize NIfTI images
    - `nifti2png.py` — Convert NIfTI to PNG for visualization
    - `gaussian_noise_injection.py` / `rician_noise_injection.py` — Add noise
    - `resample.py` — Image resampling utilities
  - **Documentation (docs/):**
    - `.github/copilot-instructions.md` — This file (AI assistant guide)
    - `docs/GETTING_STARTED.md` — Complete step-by-step tutorial
    - `docs/UNET_ARCHITECTURE.md` — Detailed architecture documentation
    - `docs/UNET_README.md` — Quick reference for daily use
    - `docs/QUICK_REFERENCE.md` — Command cheat sheet
    - `docs/CHANGELOG.md` — Version history and migration guide
- Data directories (referenced across code): `../kspace_mat_US/`, `../sensitivity_maps/`, `../Dynamic_SENSE/`, `../Synth_LR_nii/`, `../HR_nii/`, `../outputs/`, `../reconstructions/`.
- **Training pipeline:** 80/10/10 split from Synth_LR_nii→HR_nii pairs, Dynamic_SENSE reserved for final testing.

## Important repository conventions & patterns (project-specific)

### Data Organization & Output Paths ⚠️ CRITICAL
**NEVER save generated data (images, .mat files, checkpoints) inside the `synthsup-speechMRI-recon/` repository folder.**

All outputs must go to parent `MSc Project/` folder in organized subdirectories:
- **Training data:**
  - `../Synth_LR_nii/` - Synthetic low-resolution inputs (undersampled)
  - `../HR_nii/` - High-resolution ground truth targets
  - `../Dynamic_SENSE_padded/` - Validation data (from SENSE reconstruction)
- **Generated outputs:**
  - `../outputs/` - Model checkpoints, logs, TensorBoard data
  - `../reconstructions/` - Inference results
  - `../Dynamic_SENSE/` - SENSE reconstruction outputs (unpadded)
  - `../kspace_mat_US/`, `../kspace_mat_FS/` - K-space data
  - `../sensitivity_maps/` - Coil sensitivity maps

**Default CLI behavior:** All scripts default to saving outside the repo. Example:
```python
parser.add_argument('--output-dir', default='../outputs')  # NOT './outputs'
```

**U-Net Model Configuration:**
- Default: `base_filters=32` (~7.8M parameters) - lightweight baseline for fair comparison
- Heavier variant: `base_filters=64` (~31M parameters) - not recommended for initial baseline
- More training data will be added over time, so model should remain simple initially

### MAT File Conventions
- k-space MATs: loaded with `loadmat(...)["kspace"]` and expected shape (Ny, Nx, Nc, Nf). Example default in `sense_reconstruction.py`: (80, 82, 22, 100).
- coil sensitivity maps: loaded with `loadmat(...)["coilmap"]`. In `sense_reconstruction.py` the author applies `np.transpose(coilmap, (0,1,3,2))` — be careful: axis order in the MAT may differ and the code actively reorders channels/frames.

### Other Conventions
- Undersampling axis / indexing: undersampling is along Nx (axis=1) in the SENSE code and in `synthetic_undersampling.py` rows/columns selection is explicit. When modifying masks or synthetic pipelines, keep the axis semantics consistent.
- Orientation corrections: multiple files apply rotation/flip when saving or visualising NIfTI. In `sense_reconstruction.py` NIfTI saving uses rotation then flip:
  - rot90 with `k=-1` on (0,1) axes followed by `np.flip(..., axis=1)`. Follow the same convention when producing or reading saved NIfTIs to avoid mismatched orientations.
- Heavy linear algebra in loops: `sense_reconstruction.py` constructs an encoding matrix per y-row and solves dense linear systems per frame/row using `scipy.linalg.solve` (with a small regulariser). Expect long runtime and high memory use; profiling / vectorisation / batching are typical improvement targets.

## Integration points & external dependencies
- Code reads MATLAB `.mat` files (scipy.io.loadmat), NIfTI (nibabel), DICOM (pydicom in metric script), and uses standard numerical libs: `numpy`, `scipy`, `matplotlib`, `scikit-image` (metrics), `nibabel`.
- Deep learning stack: PyTorch (`torch`, `torchvision`) for U-Net model, training, and inference.
- Complete dependency list in `requirements.txt` — install with: `pip install -r requirements.txt`
- MAT files must contain the expected variable names (`kspace`, `coilmap`). If you change MAT generation, keep those names or update the consumers.

## Quick start (how to run the main workflows)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Classical SENSE Reconstruction
```bash
# Run with default settings (uses hardcoded MAT files)
python3 sense_reconstruction.py

# Run with custom paths and smoke test mode
python3 sense_reconstruction.py --kspace ../kspace_mat_US/kspace_Subject0026_vv.mat \
                                --coilmap ../sensitivity_maps/sens_Subject0026_Exam17853_80x82x100_nC22.mat \
                                --output-dir .. \
                                --smoke-test --plot
```

### U-Net Training (Deep Learning Pipeline)
```bash
# Train U-Net on synthetic pairs: Synth_LR_nii → HR_nii
# Validation uses Dynamic_SENSE_padded
python3 train_unet.py --input-dir ../Synth_LR_nii \
                      --target-dir ../HR_nii \
                      --val-input-dir ../Dynamic_SENSE \
                      --val-target-dir ../Dynamic_SENSE_padded \
                      --output-dir ../outputs \
                      --epochs 100 \
                      --batch-size 4 \
                      --lr 1e-4

# Monitor training with TensorBoard
tensorboard --logdir ../outputs/logs
```

### U-Net Inference
```bash
# Run inference with trained model
python3 inference_unet.py --checkpoint ../outputs/checkpoints/best_model.pth \
                          --input-dir ../Synth_LR_nii \
                          --output-dir ../reconstructions \
                          --compute-metrics \
                          --target-dir ../HR_nii \
                          --visualize
```

## Examples of project-specific edits an AI assistant might make
- Convert absolute save paths to configurable arguments (use `argparse`) — ✅ now done in `sense_reconstruction.py`.
- Extract the per-row solver into a vectorised routine or use a sparse representation for the encoding matrix to speed computation in `sense_reconstruction.py`.
- Add data augmentation transforms to `dataset.py` for improved U-Net generalization.
- Implement learning rate warmup or cosine annealing in `train_unet.py`.
- Add k-space data consistency layer enforcement during U-Net inference for physics-guided reconstruction (see `DataConsistencyLayer` in `unet_model.py`).
- Create cross-validation splits for more robust model evaluation.

## Quick checklist for code edits / PRs
- Preserve variable names when altering MAT loading (`kspace`, `coilmap`) or update all call sites.
- Maintain orientation corrections (rot90 + flip) unless you update all consumers/visualisers — this applies to both SENSE and U-Net outputs.
- When adding dependencies, update `requirements.txt` with pinned versions.
- For U-Net changes: test with a small dataset first using `--smoke-test` or limited epochs to validate before full training.
- Ensure NIfTI outputs from both SENSE and U-Net use consistent orientation conventions for fair comparison.

## Where to look for examples
- **Classical reconstruction:** `sense_reconstruction.py` — canonical SENSE flow and file I/O conventions (NIfTI saving and orientation). Use this file as the primary reference for k-space shapes and expected MAT variables.
- **U-Net architecture:** `unet_model.py` — standard U-Net with skip connections, includes optional `DataConsistencyLayer` for k-space enforcement.
- **Training loop:** `train_unet.py` — PyTorch training with combined L1+SSIM loss, learning rate scheduling, checkpointing, and TensorBoard logging.
- **Dataset loading:** `dataset.py` — shows how to load paired NIfTI data and on-the-fly k-space undersampling from MAT files.
- **Synthetic undersampling:** `synthetic_undersampling.py` — example of undersampling and the repo's NIfTI flip/rotate convention.
- **Metrics:** `PSNR_and_SSIM.py` — example of metric calculation with `skimage.metrics` and how the repo applies noise injection routines.

## Training data preparation workflow
1. Generate fully-sampled reconstructions using SENSE: `sense_reconstruction.py` → outputs to `../Dynamic_SENSE/` and `../Dynamic_SENSE_padded/`
2. Main training uses 80/10/10 split from: `../Synth_LR_nii/` (input) and `../HR_nii/` (target) — these are pre-generated synthetic pairs
3. Training script automatically splits data with seed=42 for reproducibility, saves test indices to `../outputs/test_indices.txt`
4. Dynamic_SENSE data reserved for final testing after training complete (not used during training/validation)
5. Train U-Net: `train_unet.py --input-dir ../Synth_LR_nii --target-dir ../HR_nii --output-dir ../outputs`
6. Inference: `inference_unet.py` applies trained model to held-out test set or Dynamic_SENSE data

**Note:** More training data will be added to Synth_LR_nii/HR_nii over time. The 80/10/10 split ensures consistent evaluation.

If any part is unclear or you want these instructions adapted (e.g., add more detailed training tips, or add smoke tests), tell me which change you prefer and I'll update this file accordingly.

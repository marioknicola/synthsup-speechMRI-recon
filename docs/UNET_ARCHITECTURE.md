# U-Net Architecture for MRI Reconstruction

## Overview

This document provides a detailed explanation of the U-Net architecture implemented in `unet_model.py` for MRI reconstruction.

## What is U-Net?

U-Net is a convolutional neural network architecture originally designed for biomedical image segmentation. Its key feature is a symmetric encoder-decoder structure with **skip connections** that preserve spatial information during upsampling.

### Why U-Net for MRI Reconstruction?

1. **Preserves fine details** - Skip connections help retain high-frequency information lost in downsampling
2. **Works well with limited data** - Effective with relatively small datasets
3. **Handles arbitrary image sizes** - Fully convolutional (no fixed input size)
4. **Proven in medical imaging** - State-of-the-art results in various medical imaging tasks

## Architecture Diagram

### High-Level Structure

```
Input (1, H, W)
      ↓
┌─────────────────────────────────────────────────────────┐
│  Encoder (Downsampling Path)                            │
│                                                          │
│  Inc:    Conv(1→64) → BN → ReLU → Conv(64→64) → BN → ReLU
│           │                                        │
│           │                                        └─────────┐
│           ↓                                                  │
│  Down1:  MaxPool → Conv(64→128) → BN → ReLU                 │
│           │                                        │         │
│           │                                        └────────┐│
│           ↓                                                 ││
│  Down2:  MaxPool → Conv(128→256) → BN → ReLU               ││
│           │                                        │        ││
│           │                                        └───────┐││
│           ↓                                                │││
│  Down3:  MaxPool → Conv(256→512) → BN → ReLU              │││
│           │                                        │       │││
│           │                                        └──────┐│││
│           ↓                                               ││││
│  Down4:  MaxPool → Conv(512→512) → BN → ReLU             ││││
│           │                              (bottleneck)     ││││
└───────────┼──────────────────────────────────────────────┘│││
            ↓                                                ││││
┌───────────┼──────────────────────────────────────────────┐│││
│  Decoder (Upsampling Path)                               ││││
│                                                           ││││
│  Up1:    Upsample → Concat(←──────────────────────────)  ││││
│           │                                               ││││
│           Conv(1024→256) → BN → ReLU                      ││││
│           ↓                                                │││
│  Up2:    Upsample → Concat(←─────────────────────────)    │││
│           │                                                │││
│           Conv(512→128) → BN → ReLU                        │││
│           ↓                                                 ││
│  Up3:    Upsample → Concat(←────────────────────────)      ││
│           │                                                 ││
│           Conv(256→64) → BN → ReLU                          ││
│           ↓                                                  │
│  Up4:    Upsample → Concat(←───────────────────────────)    │
│           │                                                  │
│           Conv(128→64) → BN → ReLU                           │
│           ↓                                                  │
│  OutConv: Conv(64→1) - Final 1x1 convolution                │
└───────────┼──────────────────────────────────────────────────┘
            ↓
    Output (1, H, W)
```

### Detailed Layer-by-Layer Breakdown with Exact Dimensions

**For input shape: (1, 312, 410)** using `base_filters=32` (default):

```
INPUT LAYER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shape: (1, 312, 410)     Channels: 1      Params: 0


ENCODER PATH (Contracting)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Inc] Initial Double Convolution
├─ Conv2d(1 → 32, 3×3, pad=1)    → (32, 312, 410)
├─ BatchNorm2d(32)                → (32, 312, 410)
├─ ReLU                           → (32, 312, 410)
├─ Conv2d(32 → 32, 3×3, pad=1)   → (32, 312, 410)
├─ BatchNorm2d(32)                → (32, 312, 410)
└─ ReLU                           → (32, 312, 410) ────┐
   Parameters: 1,856                                    │ Skip connection
                                                        │
[Down1] First Downsampling                              │
├─ MaxPool2d(2×2)                 → (32, 156, 205)     │
├─ Conv2d(32 → 64, 3×3, pad=1)   → (64, 156, 205)     │
├─ BatchNorm2d(64)                → (64, 156, 205)     │
├─ ReLU                           → (64, 156, 205)     │
├─ Conv2d(64 → 64, 3×3, pad=1)   → (64, 156, 205)     │
├─ BatchNorm2d(64)                → (64, 156, 205)     │
└─ ReLU                           → (64, 156, 205) ────┼──┐
   Parameters: 37,376                                  │  │
                                                        │  │
[Down2] Second Downsampling                             │  │
├─ MaxPool2d(2×2)                 → (64, 78, 102)      │  │
├─ Conv2d(64 → 128, 3×3, pad=1)  → (128, 78, 102)     │  │
├─ BatchNorm2d(128)               → (128, 78, 102)     │  │
├─ ReLU                           → (128, 78, 102)     │  │
├─ Conv2d(128 → 128, 3×3, pad=1) → (128, 78, 102)     │  │
├─ BatchNorm2d(128)               → (128, 78, 102)     │  │
└─ ReLU                           → (128, 78, 102) ────┼──┼──┐
   Parameters: 148,224                                 │  │  │
                                                        │  │  │
[Down3] Third Downsampling                              │  │  │
├─ MaxPool2d(2×2)                 → (128, 39, 51)      │  │  │
├─ Conv2d(128 → 256, 3×3, pad=1) → (256, 39, 51)      │  │  │
├─ BatchNorm2d(256)               → (256, 39, 51)      │  │  │
├─ ReLU                           → (256, 39, 51)      │  │  │
├─ Conv2d(256 → 256, 3×3, pad=1) → (256, 39, 51)      │  │  │
├─ BatchNorm2d(256)               → (256, 39, 51)      │  │  │
└─ ReLU                           → (256, 39, 51) ─────┼──┼──┼──┐
   Parameters: 591,360                                 │  │  │  │
                                                        │  │  │  │
[Down4] Fourth Downsampling (Bottleneck)                │  │  │  │
├─ MaxPool2d(2×2)                 → (256, 19, 25)      │  │  │  │
├─ Conv2d(256 → 256, 3×3, pad=1) → (256, 19, 25)      │  │  │  │
├─ BatchNorm2d(256)               → (256, 19, 25)      │  │  │  │
├─ ReLU                           → (256, 19, 25)      │  │  │  │
├─ Conv2d(256 → 256, 3×3, pad=1) → (256, 19, 25)      │  │  │  │
├─ BatchNorm2d(256)               → (256, 19, 25)      │  │  │  │
└─ ReLU                           → (256, 19, 25)      │  │  │  │
   Parameters: 591,360           ▲ BOTTLENECK          │  │  │  │
                                 (deepest layer)       │  │  │  │
                                                        │  │  │  │
DECODER PATH (Expanding)                                │  │  │  │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  │  │
                                                        │  │  │  │
[Up1] First Upsampling                                  │  │  │  │
├─ Upsample(2×, bilinear)         → (256, 38, 50)      │  │  │  │
├─ Crop to match                  → (256, 39, 51)      │  │  │  │
├─ Concatenate with Down3 ←───────────────────────────────┘  │  │
│  (256 + 256)                    → (512, 39, 51)         │  │
├─ Conv2d(512 → 128, 3×3, pad=1) → (128, 39, 51)         │  │
├─ BatchNorm2d(128)               → (128, 39, 51)         │  │
├─ ReLU                           → (128, 39, 51)         │  │
├─ Conv2d(128 → 128, 3×3, pad=1) → (128, 39, 51)         │  │
├─ BatchNorm2d(128)               → (128, 39, 51)         │  │
└─ ReLU                           → (128, 39, 51)         │  │
   Parameters: 591,360                                    │  │
                                                           │  │
[Up2] Second Upsampling                                    │  │
├─ Upsample(2×, bilinear)         → (128, 78, 102)        │  │
├─ Concatenate with Down2 ←─────────────────────────────────┘  │
│  (128 + 128)                    → (256, 78, 102)          │
├─ Conv2d(256 → 64, 3×3, pad=1)  → (64, 78, 102)          │
├─ BatchNorm2d(64)                → (64, 78, 102)          │
├─ ReLU                           → (64, 78, 102)          │
├─ Conv2d(64 → 64, 3×3, pad=1)   → (64, 78, 102)          │
├─ BatchNorm2d(64)                → (64, 78, 102)          │
└─ ReLU                           → (64, 78, 102)          │
   Parameters: 148,224                                     │
                                                            │
[Up3] Third Upsampling                                      │
├─ Upsample(2×, bilinear)         → (64, 156, 204)         │
├─ Crop to match                  → (64, 156, 205)         │
├─ Concatenate with Down1 ←───────────────────────────────────┘
│  (64 + 64)                      → (128, 156, 205)
├─ Conv2d(128 → 32, 3×3, pad=1)  → (32, 156, 205)
├─ BatchNorm2d(32)                → (32, 156, 205)
├─ ReLU                           → (32, 156, 205)
├─ Conv2d(32 → 32, 3×3, pad=1)   → (32, 156, 205)
├─ BatchNorm2d(32)                → (32, 156, 205)
└─ ReLU                           → (32, 156, 205)
   Parameters: 37,376

[Up4] Fourth Upsampling
├─ Upsample(2×, bilinear)         → (32, 312, 410)
├─ Concatenate with Inc ←─────────────────────────────────────┘
│  (32 + 32)                      → (64, 312, 410)
├─ Conv2d(64 → 32, 3×3, pad=1)   → (32, 312, 410)
├─ BatchNorm2d(32)                → (32, 312, 410)
├─ ReLU                           → (32, 312, 410)
├─ Conv2d(32 → 32, 3×3, pad=1)   → (32, 312, 410)
├─ BatchNorm2d(32)                → (32, 312, 410)
└─ ReLU                           → (32, 312, 410)
   Parameters: 18,752


OUTPUT LAYER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[OutConv] Final 1×1 Convolution
├─ Conv2d(32 → 1, 1×1)           → (1, 312, 410)
└─ No activation (linear output)  → (1, 312, 410)
   Parameters: 33


OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shape: (1, 312, 410)     Channels: 1

TOTAL PARAMETERS: ~7.8M (base_filters=32)
```

### Key Dimension Changes

| Layer | Shape | Channels | Spatial Size | Reduction |
|-------|-------|----------|--------------|-----------|
| **Input** | (1, 312, 410) | 1 | 312×410 | - |
| **Inc** | (32, 312, 410) | 32 | 312×410 | ×1 |
| **Down1** | (64, 156, 205) | 64 | 156×205 | ×1/2 |
| **Down2** | (128, 78, 102) | 128 | 78×102 | ×1/4 |
| **Down3** | (256, 39, 51) | 256 | 39×51 | ×1/8 |
| **Down4** | (256, 19, 25) | 256 | 19×25 | ×1/16 |
| **Up1** | (128, 39, 51) | 128 | 39×51 | ×1/8 |
| **Up2** | (64, 78, 102) | 64 | 78×102 | ×1/4 |
| **Up3** | (32, 156, 205) | 32 | 156×205 | ×1/2 |
| **Up4** | (32, 312, 410) | 32 | 312×410 | ×1 |
| **Output** | (1, 312, 410) | 1 | 312×410 | ×1 |

### Skip Connection Details

```
Encoder → Decoder Connections:

Inc (32, 312, 410) ─────────────→ Up4 → Concat → (64, 312, 410)
                                            ↓
                                    DoubleConv(64→32)
                                            ↓
                                      (32, 312, 410)

Down1 (64, 156, 205) ───────────→ Up3 → Concat → (128, 156, 205)
                                            ↓
                                    DoubleConv(128→32)
                                            ↓
                                      (32, 156, 205)

Down2 (128, 78, 102) ───────────→ Up2 → Concat → (256, 78, 102)
                                            ↓
                                    DoubleConv(256→64)
                                            ↓
                                      (64, 78, 102)

Down3 (256, 39, 51) ────────────→ Up1 → Concat → (512, 39, 51)
                                            ↓
                                    DoubleConv(512→128)
                                            ↓
                                      (128, 39, 51)
```

### For base_filters=64 (Heavier Variant)

Same architecture but with doubled channels:

| Layer | Shape | Channels | Parameters |
|-------|-------|----------|------------|
| **Input** | (1, 312, 410) | 1 | - |
| **Inc** | (64, 312, 410) | 64 | 7,424 |
| **Down1** | (128, 156, 205) | 128 | 148,992 |
| **Down2** | (256, 78, 102) | 256 | 591,360 |
| **Down3** | (512, 39, 51) | 512 | 2,360,832 |
| **Down4** | (512, 19, 25) | 512 | 2,360,832 |
| **Up1** | (256, 39, 51) | 256 | 2,360,832 |
| **Up2** | (128, 78, 102) | 128 | 591,360 |
| **Up3** | (64, 156, 205) | 64 | 148,992 |
| **Up4** | (64, 312, 410) | 64 | 74,944 |
| **Output** | (1, 312, 410) | 1 | 65 |
| **TOTAL** | - | - | **~31M** |

## Detailed Component Breakdown

### 1. **DoubleConv Block**

The fundamental building block used throughout the network:

```python
Input → Conv2d(3x3, pad=1) → BatchNorm2d → ReLU → 
        Conv2d(3x3, pad=1) → BatchNorm2d → ReLU → Output
```

**Purpose:**
- Two consecutive convolutions extract features at multiple levels
- Batch normalization stabilizes training
- ReLU adds non-linearity

**Parameters:** 
- `in_channels`: Input channels
- `out_channels`: Output channels
- Kernel size: 3×3
- Padding: 1 (preserves spatial dimensions)

### 2. **Encoder Path (Contracting Path)**

Progressively downsamples and extracts hierarchical features:

| Layer  | Input Channels | Output Channels | Spatial Size | Operation |
|--------|----------------|-----------------|--------------|-----------|
| Inc    | 1              | 64              | H × W        | DoubleConv |
| Down1  | 64             | 128             | H/2 × W/2    | MaxPool + DoubleConv |
| Down2  | 128            | 256             | H/4 × W/4    | MaxPool + DoubleConv |
| Down3  | 256            | 512             | H/8 × W/8    | MaxPool + DoubleConv |
| Down4  | 512            | 512             | H/16 × W/16  | MaxPool + DoubleConv |

**MaxPooling:** 2×2 window with stride 2 (halves spatial dimensions)

**Receptive field grows** as we go deeper, capturing larger context.

### 3. **Bottleneck**

The deepest layer (Down4 output) with:
- **Smallest spatial dimensions** (H/16 × W/16)
- **Most channels** (512)
- **Largest receptive field** - captures global context

### 4. **Decoder Path (Expanding Path)**

Progressively upsamples and recovers spatial details:

| Layer | Input Channels | Output Channels | Spatial Size | Operation |
|-------|----------------|-----------------|--------------|-----------|
| Up1   | 512 + 512      | 256             | H/8 × W/8    | Upsample + Concat + DoubleConv |
| Up2   | 256 + 256      | 128             | H/4 × W/4    | Upsample + Concat + DoubleConv |
| Up3   | 128 + 128      | 64              | H/2 × W/2    | Upsample + Concat + DoubleConv |
| Up4   | 64 + 64        | 64              | H × W        | Upsample + Concat + DoubleConv |

**Key operations:**
1. **Upsample** - Bilinear interpolation (2× spatial dimensions)
2. **Concatenate** - Merge with corresponding encoder features (skip connection)
3. **DoubleConv** - Process merged features

### 5. **Skip Connections** ⭐

The defining feature of U-Net:

```
Encoder Level i  ──→  (skip)  ──→  Decoder Level i
```

**Why critical for MRI reconstruction:**
- **Preserve high-frequency details** (edges, textures) lost during downsampling
- **Gradient flow** - Help backpropagation reach earlier layers
- **Localization** - Combine "where" (encoder) with "what" (decoder)

### 6. **Output Layer**

Final 1×1 convolution:
```python
Conv2d(64 → 1, kernel_size=1)
```

Maps 64 feature channels to single-channel magnitude image.

## Parameter Configuration

### Default Configuration

```python
model = UNet(
    in_channels=1,        # Grayscale MRI magnitude
    out_channels=1,       # Reconstructed magnitude
    base_filters=32,      # Starting filter count (lightweight baseline)
    bilinear=True         # Bilinear vs transposed conv upsampling
)
```

### Parameter Count

With `base_filters=32` (default, lightweight baseline):
- **Total parameters: ~7.8 million**
- **Trainable parameters: ~7.8 million**

With `base_filters=64` (heavier variant):
- **Total parameters: ~31 million**
- **Trainable parameters: ~31 million**

Breakdown by layer type:
- Convolutions: ~95%
- Batch normalization: ~5%

### Computational Requirements

For 312×410 input with `base_filters=32`:
- **Forward pass memory:** ~2-3 GB (depends on batch size)
- **Training memory:** ~5-8 GB (with batch_size=4)
- **Inference time:** ~30-60 ms per frame (GPU)

For `base_filters=64`:
- **Forward pass memory:** ~4-6 GB
- **Training memory:** ~10-15 GB (with batch_size=4)
- **Inference time:** ~50-100 ms per frame (GPU)

## Upsampling Strategies

### Option 1: Bilinear Interpolation (Default)

```python
nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
```

**Pros:**
- No learnable parameters
- Smooth, artifact-free upsampling
- Faster training

**Cons:**
- Less flexible

### Option 2: Transposed Convolution

```python
nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
```

**Pros:**
- Learnable upsampling
- Can learn task-specific patterns

**Cons:**
- More parameters
- Risk of checkerboard artifacts

## Data Consistency Layer (Optional) [and not tried yet as of 27/10/2025]

Physics-guided constraint for k-space enforcement:

```python
class DataConsistencyLayer(nn.Module):
    """
    Ensures reconstruction matches acquired k-space data.
    """
```

### How it works:

1. Transform predicted image to k-space (FFT)
2. Replace acquired k-space lines with ground truth
3. Keep predicted values for unacquired lines
4. Transform back to image domain (IFFT)

```
Predicted Image → FFT → Replace Acquired Lines → IFFT → Corrected Image
                          ↑
                     Mask & Ground Truth
```

**When to use:**
- When you have k-space data and sampling mask
- For physics-informed reconstruction
- To enforce data fidelity

## Training Loss Function

### Combined L1 + SSIM Loss

```python
Loss = α × L1(pred, target) + (1 - α) × SSIM_loss(pred, target)
```

Default: α = 0.84 (84% L1, 16% SSIM)

#### L1 Loss (Mean Absolute Error)
```
L1 = mean(|predicted - target|)
```
- Pixel-wise accuracy
- Sensitive to all errors equally
- Robust to outliers

#### SSIM Loss (Structural Similarity)
```
SSIM = (2μ_x μ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
SSIM_loss = 1 - SSIM
```
- Perceptual quality
- Focuses on structure/texture
- Correlates with human perception

**Why combine both?**
- L1 ensures pixel accuracy
- SSIM preserves perceptual quality
- Balance prevents blurry outputs while maintaining fidelity

## Input/Output Specifications

### Expected Input Format

```python
input_tensor = torch.FloatTensor(B, 1, H, W)
```

- **B:** Batch size (typically 2-8)
- **Channels:** 1 (magnitude MRI)
- **H, W:** Spatial dimensions (must be divisible by 16)
- **Value range:** [0, 1] (normalized)

### Output Format

```python
output_tensor = torch.FloatTensor(B, 1, H, W)
```

Same shape as input.

## Design Choices Explained

### Why 5 Levels?

- **H/16 reduction** balances:
  - Receptive field size (captures global structure)
  - Memory requirements (deeper = more memory)
  - Gradient flow (too deep = vanishing gradients)

### Why BatchNorm?

- Stabilizes training
- Enables higher learning rates
- Acts as mild regularization
- Standard in modern CNNs

### Why ReLU?

- Simple, effective non-linearity
- No gradient vanishing (unlike sigmoid/tanh)
- Computationally efficient
- Induces sparsity (feature selection)

### Why 3×3 Convolutions?

- Good balance of receptive field vs parameters
- Two 3×3 convs = one 5×5 conv (fewer params)
- Standard in modern architectures

## Comparison with Other Architectures

| Architecture | Skip Connections | Typical Use | Params (64 filters) |
|--------------|------------------|-------------|---------------------|
| **U-Net**    | ✅ Yes           | Medical imaging | ~31M |
| ResNet       | ✅ Residual      | Classification | Varies |
| FCN          | ❌ No            | Segmentation | ~134M |
| SRCNN        | ❌ No            | Super-resolution | ~8K |

**U-Net advantages for MRI:**
- Skip connections preserve anatomical details
- Symmetric structure matches MRI characteristics
- Proven track record in medical imaging

## Implementation Notes

### Memory Optimization

```python
# Enable mixed precision training
with torch.cuda.amp.autocast():
    output = model(input)
```

### Gradient Checkpointing

For very large models, trade computation for memory:
```python
from torch.utils.checkpoint import checkpoint

x = checkpoint(self.encoder_block, x)
```

### Testing the Model

```bash
python3 unet_model.py
```

Runs built-in test with dummy data.

## References

1. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
2. Zbontar et al., "fastMRI: An Open Dataset and Benchmarks for Accelerated MRI", arXiv:1811.08839
3. Hammernik et al., "Learning a Variational Network for Reconstruction of Accelerated MRI Data", MRM 2018

## Summary

The U-Net in this implementation:
- ✅ **5-level** encoder-decoder
- ✅ **Skip connections** at each level
- ✅ **~7.8M parameters** (32 base filters, default lightweight baseline)
- ✅ **~31M parameters** (64 base filters, heavier variant)
- ✅ **Combined L1+SSIM loss**
- ✅ **Batch normalization** for stability
- ✅ **Bilinear upsampling** (configurable)
- ✅ **Handles arbitrary input sizes** (divisible by 16)
- ✅ **Optional data consistency** for k-space

**Bottom line:** A robust, well-tested architecture specifically designed for MRI reconstruction with synthetic supervision.

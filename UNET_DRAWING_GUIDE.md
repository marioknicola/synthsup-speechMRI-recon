# U-Net Architecture Drawing Guide

## Architecture Overview
- **Type**: 5-level U-Net with skip connections
- **Base filters**: 32
- **Total parameters**: ~1.9M
- **Upsampling method**: Bilinear interpolation

---

## Layer-by-Layer Dimensions

### INPUT
- **Spatial dimensions**: 312 × 410
- **Channels**: 1 (magnitude MRI image)
- **Shape**: (B, 1, 312, 410)

---

## ENCODER PATH (Downsampling)

### Level 1 - Initial Convolution
- **Input**: 312 × 410 × 1
- **Operation**: DoubleConv (Conv 3×3 + BN + ReLU) × 2
- **Output**: 312 × 410 × 32
- **Filters**: 32

↓ **MaxPool2d (stride=2)**

### Level 2 - Down1
- **Input**: 156 × 205 × 32
- **Operation**: DoubleConv
- **Output**: 156 × 205 × 64
- **Filters**: 64

↓ **MaxPool2d (stride=2)**

### Level 3 - Down2
- **Input**: 78 × 102 × 64
- **Operation**: DoubleConv
- **Output**: 78 × 102 × 128
- **Filters**: 128

↓ **MaxPool2d (stride=2)**

### Level 4 - Down3
- **Input**: 39 × 51 × 128
- **Operation**: DoubleConv
- **Output**: 39 × 51 × 256
- **Filters**: 256

↓ **MaxPool2d (stride=2)**

### Level 5 - Bottleneck (Down4)
- **Input**: 19 × 25 × 256
- **Operation**: DoubleConv
- **Output**: 19 × 25 × 512
- **Filters**: 512
- **Note**: This is the deepest point (bottleneck)

---

## DECODER PATH (Upsampling)

### Level 4 - Up1
- **Upsampled from bottleneck**: 39 × 51 × 256 (after upsampling)
- **Skip connection from**: Level 4 encoder (39 × 51 × 256)
- **Concatenated**: 39 × 51 × 512 (256 + 256)
- **After DoubleConv**: 39 × 51 × 256
- **Filters**: 256

↑ **Bilinear Upsample (scale=2)**

### Level 3 - Up2
- **Upsampled from Up1**: 78 × 102 × 128 (after upsampling)
- **Skip connection from**: Level 3 encoder (78 × 102 × 128)
- **Concatenated**: 78 × 102 × 256 (128 + 128)
- **After DoubleConv**: 78 × 102 × 128
- **Filters**: 128

↑ **Bilinear Upsample (scale=2)**

### Level 2 - Up3
- **Upsampled from Up2**: 156 × 205 × 64 (after upsampling)
- **Skip connection from**: Level 2 encoder (156 × 205 × 64)
- **Concatenated**: 156 × 205 × 128 (64 + 64)
- **After DoubleConv**: 156 × 205 × 64
- **Filters**: 64

↑ **Bilinear Upsample (scale=2)**

### Level 1 - Up4
- **Upsampled from Up3**: 312 × 410 × 32 (after upsampling)
- **Skip connection from**: Level 1 encoder (312 × 410 × 32)
- **Concatenated**: 312 × 410 × 64 (32 + 32)
- **After DoubleConv**: 312 × 410 × 32
- **Filters**: 32

---

### OUTPUT
- **Input**: 312 × 410 × 32
- **Operation**: Conv 1×1 (OutConv)
- **Output**: 312 × 410 × 1
- **Channels**: 1 (reconstructed magnitude image)

---

## Skip Connections (Concatenation)

1. **Level 1** encoder (312 × 410 × 32) → **Up4** decoder
2. **Level 2** encoder (156 × 205 × 64) → **Up3** decoder
3. **Level 3** encoder (78 × 102 × 128) → **Up2** decoder
4. **Level 4** encoder (39 × 51 × 256) → **Up1** decoder

---

## Quick Reference Table

| Level | Spatial Dims | Encoder Filters | Decoder Filters |
|-------|-------------|-----------------|-----------------|
| Input | 312 × 410   | 1               | -               |
| 1     | 312 × 410   | 32              | 32              |
| 2     | 156 × 205   | 64              | 64              |
| 3     | 78 × 102    | 128             | 128             |
| 4     | 39 × 51     | 256             | 256             |
| 5     | 19 × 25     | 512             | -               |
| Output| 312 × 410   | -               | 1               |

---

## Drawing Tips

1. **Layout**: Draw encoder on the left going down, decoder on the right going up
2. **Spatial reduction**: Each MaxPool divides dimensions by 2 (height and width)
3. **Skip connections**: Draw horizontal arrows from encoder to decoder at matching levels
4. **Colors**: Use different colors for encoder, bottleneck, decoder, and skip connections
5. **Annotations**: Label each block with:
   - Spatial dimensions (H × W)
   - Number of filters
   - Operation type (DoubleConv, MaxPool, Upsample)

---

## Block Details

### DoubleConv Block (used in all levels)
```
Conv2d (3×3, padding=1)
↓
BatchNorm2d
↓
ReLU
↓
Conv2d (3×3, padding=1)
↓
BatchNorm2d
↓
ReLU
```

### Down Block (Levels 2-5)
```
MaxPool2d (kernel=2, stride=2)
↓
DoubleConv
```

### Up Block (Levels 1-4 decoder)
```
Bilinear Upsample (scale_factor=2)
↓
Concatenate with skip connection
↓
DoubleConv
```

---

## Key Features
- **Symmetric architecture**: 5 levels down, 4 levels up (plus output)
- **Skip connections**: Preserve high-resolution features
- **Progressive filtering**: 32 → 64 → 128 → 256 → 512 (encoder)
- **Progressive refinement**: 512 → 256 → 128 → 64 → 32 → 1 (decoder)
- **No dropout**: Not used in this implementation
- **Activation**: ReLU throughout, no final activation (linear output)

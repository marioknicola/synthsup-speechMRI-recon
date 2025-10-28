# Data Consistency Guide

Physics-based constraint for MRI reconstruction that ensures predictions respect actual k-space measurements.

---

## Overview

### What is Data Consistency?

Data consistency enforces that your MRI reconstruction matches the actual k-space measurements acquired during scanning. It's a physics-based constraint that improves reconstruction quality.

**The principle:**
- **Acquired k-space lines** = Original measurements (ground truth)
- **Non-acquired lines** = U-Net predictions (filled in by network)

```
┌──────────────┐
│  U-Net       │
│  Prediction  │
└──────┬───────┘
       │
       ↓ FFT (to k-space)
┌──────────────────────────────┐
│  Data Consistency Layer      │
│  Keep: Acquired lines        │
│  Fill: Predicted lines       │
└──────┬───────────────────────┘
       ↓ IFFT (to image)
┌──────────────┐
│  Final       │
│  Consistent  │
│  Image       │
└──────────────┘
```

---

## When to Use Data Consistency

### ✅ Use When:
- You have **raw k-space data** with known sampling pattern
- Working with **undersampled acquisitions** (parallel imaging, compressed sensing)
- Need **physics-guided** reconstruction
- Want to **guarantee** data fidelity to measurements

### ❌ Don't Use When:
- Only have **image-domain** data (no k-space)
- Data has **corruption** or **artifacts** in k-space
- Doing **image-to-image** translation (not reconstruction)
- Want maximum **flexibility** (DC layer constrains predictions)

---

## Implementation Options

### Option 1: Post-Processing (Simplest)

Apply data consistency after U-Net inference:

```python
def apply_data_consistency(predicted_image, kspace_undersampled, mask, num_iterations=5):
    """
    Iteratively enforce data consistency.
    
    Args:
        predicted_image: (H, W) - U-Net prediction
        kspace_undersampled: (H, W) complex - Acquired k-space
        mask: (H, W) binary - Sampling mask (1 = acquired, 0 = not acquired)
        num_iterations: Number of DC iterations
    
    Returns:
        (H, W) - Data-consistent image
    """
    import numpy as np
    
    current_image = predicted_image.copy()
    
    for i in range(num_iterations):
        # Transform to k-space
        current_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(current_image)))
        
        # Replace acquired lines with measurements
        consistent_kspace = mask * kspace_undersampled + (1 - mask) * current_kspace
        
        # Transform back to image
        current_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(consistent_kspace)))
        current_image = np.abs(current_image)
    
    return current_image
```

### Option 2: Integrated DC Layer (Advanced)

Add DC layer to U-Net architecture:

```python
# File: unet_with_dc.py
import torch
import torch.nn as nn
from unet_model import UNet, DataConsistencyLayer

class UNetWithDC(nn.Module):
    """U-Net with integrated Data Consistency layer."""
    
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super().__init__()
        self.unet = UNet(in_channels, out_channels, base_filters)
        self.dc_layer = DataConsistencyLayer()
    
    def forward(self, undersampled_image, kspace_data=None, mask=None):
        # Get U-Net prediction
        prediction = self.unet(undersampled_image)
        
        # Apply data consistency if k-space available
        if kspace_data is not None and mask is not None:
            prediction = self.dc_layer(prediction, kspace_data, mask)
        
        return prediction
```

---

## Quick Start: Test with Your Model

Apply DC to existing U-Net reconstructions:

```python
# quick_dc_test.py
import numpy as np
import nibabel as nib
from scipy.io import loadmat

# Load U-Net reconstruction
recon = nib.load('../reconstructions/recon_Subject0026_sh.nii').get_fdata()

# Load corresponding k-space
mat_data = loadmat('../kspace_mat_US/kspace_Subject0026_sh.mat')
kspace = mat_data['kspace'][:, :, :, 0]  # (Ny, Nx, Nc)

# Multi-coil combination (sum-of-squares)
kspace_combined = np.sqrt(np.sum(np.abs(kspace)**2, axis=2))

# Create undersampling mask (e.g., R=3, every 3rd line)
mask = np.zeros((kspace.shape[0], kspace.shape[1]))
mask[::3, :] = 1.0
kspace_undersampled = kspace_combined * mask

# Apply data consistency (5 iterations)
def apply_dc(image, kspace, mask):
    current = image.copy()
    for _ in range(5):
        k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(current)))
        k = mask * kspace + (1 - mask) * k
        current = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k))))
    return current

dc_recon = apply_dc(recon, kspace_undersampled, mask)

# Save result
nib.save(nib.Nifti1Image(dc_recon, np.eye(4)), '../reconstructions/dc_test.nii')
print("✓ Data consistency applied!")
```

---

## Expected Improvements

### Benefits:
- ✅ **Guaranteed fidelity** to acquired measurements
- ✅ **Reduced aliasing** artifacts
- ✅ **Better edge preservation** in sampled regions
- ✅ **Physics-compliant** reconstructions

### Considerations:
- ⚠️ May **amplify noise** in acquired lines
- ⚠️ **Slightly slower** inference (FFT operations)
- ⚠️ Requires **k-space data** (not just images)
- ⚠️ **5-10 iterations** usually sufficient

---

## Comparison: With vs Without DC

| Aspect | Without DC | With DC |
|--------|------------|---------|
| **Acquired lines** | May differ from measurements | Exactly matches |
| **Artifacts** | Possible aliasing | Reduced aliasing |
| **Noise** | Smoothed by network | Preserved in acquired lines |
| **Training** | Image-only data OK | Needs k-space data |
| **Speed** | Fast | Slightly slower |
| **Flexibility** | Network learns freely | Constrained by physics |

---

## Best Practices

1. **Start without DC** - Train baseline U-Net first
2. **Test post-processing** - Try DC on existing predictions before retraining
3. **Compare metrics** - Check if DC improves PSNR/SSIM
4. **Iteration count** - 5-10 DC iterations usually sufficient
5. **Noise handling** - Consider denoising acquired lines if needed
6. **Coil combination** - Use proper multi-coil reconstruction when available

---

## Further Reading

- **Paper:** "A Deep Cascade of CNNs for Dynamic MR Image Reconstruction" (Schlemper et al., 2018)
- **Paper:** "Learning a Variational Network for Reconstruction of Accelerated MRI Data" (Hammernik et al., 2018)
- **Dataset:** fastMRI - https://fastmri.org/ (benchmark with k-space data)
- **Code:** See `unet_model.py` lines 141-180 for `DataConsistencyLayer` implementation

---

**Status**: Optional advanced feature  
**Last Updated**: 28 October 2025

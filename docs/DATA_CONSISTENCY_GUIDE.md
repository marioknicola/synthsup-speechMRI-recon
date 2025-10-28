# Data Consistency Guide

## What is Data Consistency?

Data consistency is a physics-based constraint that ensures your MRI reconstruction respects the actual k-space measurements acquired during scanning.

### The Problem

Standard U-Net reconstruction can produce images that look good but don't match the acquired k-space data exactly. This violates the physics of MRI acquisition.

### The Solution

The **Data Consistency Layer** enforces that:
- **Acquired k-space lines** = Original measurements (ground truth)
- **Non-acquired lines** = U-Net predictions (filled in by the network)

```
┌──────────────┐
│  U-Net       │
│  Prediction  │
└──────┬───────┘
       │
       ↓ FFT (to k-space)
┌──────────────────────────────┐
│  Data Consistency Layer      │
│                              │
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
1. You have **raw k-space data** with known sampling pattern
2. You want **physics-guided** reconstruction
3. You're working with **undersampled acquisitions** (parallel imaging, compressed sensing)
4. You want to **guarantee** data fidelity

### ❌ Don't Use When:
1. You only have **image-domain** data (no k-space)
2. Your data has **corruption** or **artifacts** in k-space
3. You're doing **image-to-image** translation (not reconstruction)
4. You want maximum **flexibility** (DC layer constrains predictions)

---

## Implementation

### Option 1: Add DC Layer to U-Net (End-to-End)

Create a new model that combines U-Net + Data Consistency:

```python
# File: unet_with_dc.py
import torch
import torch.nn as nn
from unet_model import UNet, DataConsistencyLayer

class UNetWithDC(nn.Module):
    """
    U-Net with integrated Data Consistency layer.
    """
    
    def __init__(self, in_channels=1, out_channels=1, base_filters=32, bilinear=True):
        super().__init__()
        self.unet = UNet(in_channels, out_channels, base_filters, bilinear)
        self.dc_layer = DataConsistencyLayer()
    
    def forward(self, undersampled_image, kspace_data=None, mask=None):
        """
        Args:
            undersampled_image: (B, 1, H, W) - Input undersampled image
            kspace_data: (B, 1, H, W) - Undersampled k-space (complex, optional)
            mask: (B, 1, H, W) - Sampling mask (optional)
        
        Returns:
            If kspace_data and mask provided: data-consistent image
            Otherwise: standard U-Net output
        """
        # Get U-Net prediction
        prediction = self.unet(undersampled_image)
        
        # Apply data consistency if k-space data available
        if kspace_data is not None and mask is not None:
            prediction = self.dc_layer(prediction, kspace_data, mask)
        
        return prediction
```

### Option 2: Post-Processing (Iterative)

Apply data consistency after U-Net inference:

```python
def apply_data_consistency(predicted_image, kspace_undersampled, mask, num_iterations=5):
    """
    Iteratively apply data consistency to improve reconstruction.
    
    Args:
        predicted_image: (H, W) numpy array - U-Net prediction
        kspace_undersampled: (H, W) complex numpy array - Acquired k-space
        mask: (H, W) binary array - Sampling mask
        num_iterations: Number of DC iterations
    
    Returns:
        (H, W) numpy array - Data-consistent image
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

---

## Training with Data Consistency

### Modified Training Script

You need k-space data during training. Update your dataset:

```python
# File: kspace_dataset.py
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from scipy.io import loadmat

class KSpaceReconstructionDataset(Dataset):
    """
    Dataset that loads k-space data for data consistency training.
    """
    
    def __init__(self, kspace_dir, target_dir, undersample_pattern=None):
        """
        Args:
            kspace_dir: Directory with fully-sampled k-space MAT files
            target_dir: Directory with target images
            undersample_pattern: Indices of acquired lines (e.g., [0, 3, 6, 9, ...])
        """
        self.kspace_files = sorted(glob.glob(f"{kspace_dir}/*.mat"))
        self.target_files = sorted(glob.glob(f"{target_dir}/*.nii"))
        self.undersample_pattern = undersample_pattern
    
    def __len__(self):
        return len(self.kspace_files) * 100  # 100 frames per file
    
    def __getitem__(self, idx):
        file_idx = idx // 100
        frame_idx = idx % 100
        
        # Load k-space
        mat_data = loadmat(self.kspace_files[file_idx])
        kspace_full = mat_data['kspace'][:, :, :, frame_idx]  # (Ny, Nx, Nc)
        
        # Multi-coil to single-coil (sum-of-squares)
        kspace_combined = np.sqrt(np.sum(np.abs(kspace_full)**2, axis=2))
        
        # Apply undersampling
        mask = np.zeros_like(kspace_combined)
        mask[self.undersample_pattern, :] = 1.0
        kspace_undersampled = kspace_combined * mask
        
        # Create undersampled image (IFFT)
        undersampled_img = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(kspace_undersampled))
        )
        undersampled_img = np.abs(undersampled_img)
        
        # Load target
        target_img = nib.load(self.target_files[file_idx]).get_fdata()[:, :, frame_idx]
        
        # Normalize
        undersampled_img = undersampled_img / (undersampled_img.max() + 1e-8)
        target_img = target_img / (target_img.max() + 1e-8)
        
        # Convert to torch tensors
        undersampled_img = torch.from_numpy(undersampled_img).float().unsqueeze(0)
        target_img = torch.from_numpy(target_img).float().unsqueeze(0)
        kspace_undersampled = torch.from_numpy(kspace_undersampled).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return {
            'input': undersampled_img,
            'target': target_img,
            'kspace': kspace_undersampled,
            'mask': mask,
            'filename': os.path.basename(self.kspace_files[file_idx]),
            'frame_idx': frame_idx
        }
```

### Training Loop with DC

```python
# File: train_unet_with_dc.py
import torch
import torch.nn as nn
from unet_with_dc import UNetWithDC
from kspace_dataset import KSpaceReconstructionDataset

def train_with_dc():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetWithDC(base_filters=32).to(device)
    
    # Dataset with k-space
    undersample_pattern = list(range(0, 80, 3))  # R=3 undersampling
    dataset = KSpaceReconstructionDataset(
        kspace_dir='../kspace_mat_FS',
        target_dir='../HR_nii',
        undersample_pattern=undersample_pattern
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Loss and optimizer
    criterion = CombinedLoss(alpha=0.84)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    for epoch in range(100):
        model.train()
        epoch_loss = 0
        
        for batch in dataloader:
            input_img = batch['input'].to(device)
            target_img = batch['target'].to(device)
            kspace = batch['kspace'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass with data consistency
            output = model(input_img, kspace, mask)
            
            # Compute loss
            loss = criterion(output, target_img)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss = {epoch_loss/len(dataloader):.4f}")
```

---

## Inference with Data Consistency

### Method 1: Model has DC built-in

If you trained with `UNetWithDC`:

```bash
python3 inference_unet_dc.py \
    --checkpoint ../outputs/checkpoints/best_model_dc.pth \
    --kspace-dir ../kspace_mat_US \
    --output-dir ../reconstructions_dc \
    --undersample-pattern "0,3,6,9,12,..." \
    --compute-metrics
```

### Method 2: Post-processing existing predictions

Apply DC to already-trained U-Net predictions:

```python
# File: apply_dc_postprocess.py
import numpy as np
import nibabel as nib
from scipy.io import loadmat
import glob

def postprocess_with_dc(recon_dir, kspace_dir, output_dir, undersample_pattern):
    """
    Apply data consistency to existing U-Net reconstructions.
    """
    recon_files = sorted(glob.glob(f"{recon_dir}/recon_*.nii"))
    
    for recon_file in recon_files:
        # Load U-Net prediction
        predicted_img = nib.load(recon_file).get_fdata()
        
        # Find corresponding k-space file
        filename = os.path.basename(recon_file).replace('recon_LR_', '')
        kspace_file = os.path.join(kspace_dir, filename.replace('.nii', '.mat'))
        
        if not os.path.exists(kspace_file):
            print(f"Warning: k-space not found for {filename}")
            continue
        
        # Load k-space
        mat_data = loadmat(kspace_file)
        kspace_full = mat_data['kspace'][:, :, :, 0]  # First frame
        kspace_combined = np.sqrt(np.sum(np.abs(kspace_full)**2, axis=2))
        
        # Create mask
        mask = np.zeros_like(kspace_combined)
        mask[undersample_pattern, :] = 1.0
        kspace_undersampled = kspace_combined * mask
        
        # Apply DC
        dc_img = apply_data_consistency(
            predicted_img,
            kspace_undersampled,
            mask,
            num_iterations=10
        )
        
        # Save
        output_path = os.path.join(output_dir, f"dc_{filename}")
        nib.save(nib.Nifti1Image(dc_img, np.eye(4)), output_path)
        print(f"Saved: {output_path}")

# Run
undersample_pattern = list(range(0, 80, 3))  # R=3
postprocess_with_dc(
    recon_dir='../reconstructions',
    kspace_dir='../kspace_mat_US',
    output_dir='../reconstructions_dc',
    undersample_pattern=undersample_pattern
)
```

---

## Expected Improvements

### With Data Consistency:
- ✅ **Guaranteed fidelity** to acquired data
- ✅ **Reduced aliasing** artifacts
- ✅ **Better edge preservation** in sampled regions
- ✅ **Physics-compliant** reconstructions

### Potential Drawbacks:
- ⚠️ May **amplify noise** in acquired lines
- ⚠️ **Slightly slower** inference (FFT operations)
- ⚠️ Requires **k-space data** (not just images)
- ⚠️ Can **overfit** to acquired data if not careful

---

## Quick Start: Post-Process Your Current Results

The easiest way to try DC with your existing model:

```python
# quick_dc_test.py
import numpy as np
import nibabel as nib
from scipy.io import loadmat

# Load your U-Net reconstruction
recon = nib.load('../reconstructions/recon_LR_kspace_Subject0026_sh.nii').get_fdata()

# Load corresponding k-space
kspace_file = '../kspace_mat_US/kspace_Subject0026_sh.mat'
mat_data = loadmat(kspace_file)
kspace = mat_data['kspace'][:, :, :, 0]  # (Ny, Nx, Nc)

# Multi-coil combination (sum-of-squares)
kspace_combined = np.sqrt(np.sum(np.abs(kspace)**2, axis=2))

# Undersampling mask (R=3, every 3rd line)
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
print("Data consistency applied! Check: ../reconstructions/dc_test.nii")
```

---

## Comparison: With vs Without DC

| Metric | Without DC | With DC |
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
2. **Test post-processing** - Try DC on existing predictions
3. **Compare metrics** - See if DC improves PSNR/SSIM
4. **Iterate count** - Usually 5-10 DC iterations sufficient
5. **Noise handling** - Consider denoising acquired lines
6. **Coil combination** - Use proper multi-coil reconstruction if available

---

## Further Reading

- **Paper:** "A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image Reconstruction" (Schlemper et al., 2018)
- **Paper:** "Learning a Variational Network for Reconstruction of Accelerated MRI Data" (Hammernik et al., 2018)
- **fastMRI dataset:** https://fastmri.org/ - Benchmark with k-space data

---

## Need Help?

Check these files:
- `unet_model.py` - `DataConsistencyLayer` implementation (lines 141-180)
- `dataset.py` - `KSpaceDataset` for k-space loading
- `train_unet.py` - Current training script (modify for DC)

The infrastructure is already there - you just need to connect the pieces! 🚀

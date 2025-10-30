#!/usr/bin/env python3
"""
Verify metrics computation by examining one example in detail
"""
import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from scipy import ndimage
import matplotlib.pyplot as plt

def load_nii(filepath):
    """Load NIfTI file"""
    img = nib.load(filepath)
    data = img.get_fdata()
    return data

def resize_to_target(img, target_shape):
    """Resize image to target shape using zoom"""
    scale_factors = [target_shape[i] / img.shape[i] for i in range(len(target_shape))]
    resized = ndimage.zoom(img, scale_factors, order=1)
    return resized

def normalize_to_mri_range(img, target_min, target_max):
    """Normalize image from [img_min, img_max] to [target_min, target_max]."""
    img_min = img.min()
    img_max = img.max()
    
    # Normalize to [0, 1]
    if img_max > img_min:
        normalized = (img - img_min) / (img_max - img_min)
    else:
        normalized = img - img_min
    
    # Scale to target range
    scaled = normalized * (target_max - target_min) + target_min
    
    return scaled

def compute_metrics(pred, target):
    """Compute PSNR, SSIM, MSE"""
    # Ensure same shape
    if pred.shape != target.shape:
        print(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
        pred = resize_to_target(pred, target.shape)
    
    # Flatten for metrics
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Get data range
    data_range = target_flat.max() - target_flat.min()
    
    # Compute metrics
    mse = np.mean((pred_flat - target_flat) ** 2)
    psnr_val = psnr(target, pred, data_range=data_range)
    ssim_val = ssim(target, pred, data_range=data_range)
    
    return psnr_val, ssim_val, mse, data_range

# Test file (using 'ee' phoneme)
test_file = 'ee'
print(f"Analyzing file: Subject0021_{test_file}.nii\n")

# Load ground truth
gt_path = f"../HR_nii/kspace_Subject0021_{test_file}.nii"
gt = load_nii(gt_path)
print(f"Ground Truth: shape={gt.shape}, range=[{gt.min():.2f}, {gt.max():.2f}]")

# Get MRI range from LR file
lr_path = f"../Synth_LR_nii/LR_kspace_Subject0021_{test_file}.nii"
lr = load_nii(lr_path)
mri_min, mri_max = lr.min(), lr.max()
print(f"LR Reference: shape={lr.shape}, range=[{mri_min:.2f}, {mri_max:.2f}]")
print(f"\nUsing MRI intensity range for normalization: [{mri_min:.2f}, {mri_max:.2f}]\n")

# Load U-Net output
unet_path = f"../unet_test_subject0021/recon_Subject0021_{test_file}.nii"
unet = load_nii(unet_path)
print(f"U-Net Output: shape={unet.shape}, range=[{unet.min():.2f}, {unet.max():.2f}]")

# Load Fast-SRGAN output
fastsr_path = f"../fastsr_test_subject0021/sr_LR_kspace_Subject0021_{test_file}.nii"
fastsr = load_nii(fastsr_path)
print(f"Fast-SRGAN (raw):   shape={fastsr.shape}, range=[{fastsr.min():.2f}, {fastsr.max():.2f}]")
fastsr = resize_to_target(fastsr, gt.shape)
fastsr = normalize_to_mri_range(fastsr, mri_min, mri_max)
print(f"Fast-SRGAN (norm):  shape={fastsr.shape}, range=[{fastsr.min():.2f}, {fastsr.max():.2f}]")

# Load SRCNN output
srcnn_path = f"../srcnn_test_subject0021/LR_kspace_Subject0021_{test_file}_upscaled.nii"
srcnn = load_nii(srcnn_path)
print(f"SRCNN (raw):        shape={srcnn.shape}, range=[{srcnn.min():.2f}, {srcnn.max():.2f}]")
srcnn = resize_to_target(srcnn, gt.shape)
srcnn = normalize_to_mri_range(srcnn, mri_min, mri_max)
print(f"SRCNN (norm):       shape={srcnn.shape}, range=[{srcnn.min():.2f}, {srcnn.max():.2f}]")

print("\n" + "="*60)
print("METRICS COMPUTATION")
print("="*60)

# Compute U-Net metrics
psnr_u, ssim_u, mse_u, dr_u = compute_metrics(unet, gt)
print(f"\nU-Net vs Ground Truth:")
print(f"  PSNR: {psnr_u:.2f} dB")
print(f"  SSIM: {ssim_u:.4f}")
print(f"  MSE:  {mse_u:.2e}")
print(f"  Data range: {dr_u:.2f}")

# Compute Fast-SRGAN metrics
psnr_f, ssim_f, mse_f, dr_f = compute_metrics(fastsr, gt)
print(f"\nFast-SRGAN vs Ground Truth (after resize and normalization):")
print(f"  PSNR: {psnr_f:.2f} dB")
print(f"  SSIM: {ssim_f:.4f}")
print(f"  MSE:  {mse_f:.2e}")
print(f"  Data range: {dr_f:.2f}")

# Compute SRCNN metrics
psnr_s, ssim_s, mse_s, dr_s = compute_metrics(srcnn, gt)
print(f"\nSRCNN vs Ground Truth (after resize and normalization):")
print(f"  PSNR: {psnr_s:.2f} dB")
print(f"  SSIM: {ssim_s:.4f}")
print(f"  MSE:  {mse_s:.2e}")
print(f"  Data range: {dr_s:.2f}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Resize pretrained outputs for visualization (already done above)

# Plot images (2D data)
axes[0, 0].imshow(gt, cmap='gray')
axes[0, 0].set_title('Ground Truth')
axes[0, 0].axis('off')

axes[0, 1].imshow(unet, cmap='gray')
axes[0, 1].set_title(f'U-Net\nPSNR: {psnr_u:.2f} dB')
axes[0, 1].axis('off')

axes[0, 2].imshow(fastsr, cmap='gray')
axes[0, 2].set_title(f'Fast-SRGAN (Normalized)\nPSNR: {psnr_f:.2f} dB')
axes[0, 2].axis('off')

axes[1, 0].imshow(srcnn, cmap='gray')
axes[1, 0].set_title(f'SRCNN (Normalized)\nPSNR: {psnr_s:.2f} dB')
axes[1, 0].axis('off')

# Plot difference maps
axes[1, 1].imshow(np.abs(unet - gt), cmap='hot')
axes[1, 1].set_title('U-Net Error Map')
axes[1, 1].axis('off')

axes[1, 2].imshow(np.abs(fastsr - gt), cmap='hot')
axes[1, 2].set_title('Fast-SRGAN Error Map')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('../test_comparison_results/detailed_metrics_verification.png', dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: test_comparison_results/detailed_metrics_verification.png")

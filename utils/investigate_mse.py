#!/usr/bin/env python3
"""
Investigate why MSE values are so high.
"""
import nibabel as nib
import numpy as np

# Load one example file
test_file = 'ee'

# Ground truth
gt_path = f"../HR_nii/kspace_Subject0021_{test_file}.nii"
gt = nib.load(gt_path).get_fdata()

# U-Net output
unet_path = f"../unet_test_subject0021/recon_Subject0021_{test_file}.nii"
unet = nib.load(unet_path).get_fdata()

print("="*70)
print("INVESTIGATING MSE VALUES")
print("="*70)

print(f"\nGround Truth:")
print(f"  Shape: {gt.shape}")
print(f"  Range: [{gt.min():.2f}, {gt.max():.2f}]")
print(f"  Mean: {gt.mean():.2f}")
print(f"  Std: {gt.std():.2f}")

print(f"\nU-Net Output:")
print(f"  Shape: {unet.shape}")
print(f"  Range: [{unet.min():.2f}, {unet.max():.2f}]")
print(f"  Mean: {unet.mean():.2f}")
print(f"  Std: {unet.std():.2f}")

# Compute differences
diff = unet - gt
abs_diff = np.abs(diff)

print(f"\nDifference Statistics:")
print(f"  Mean difference: {diff.mean():.2f}")
print(f"  Mean absolute difference: {abs_diff.mean():.2f}")
print(f"  Max absolute difference: {abs_diff.max():.2f}")
print(f"  Std of difference: {diff.std():.2f}")

# Compute MSE
mse = np.mean(diff ** 2)
rmse = np.sqrt(mse)

print(f"\nMSE Metrics:")
print(f"  MSE: {mse:.2e}")
print(f"  RMSE: {rmse:.2f}")
print(f"  RMSE as % of GT range: {100 * rmse / (gt.max() - gt.min()):.2f}%")
print(f"  RMSE as % of GT mean: {100 * rmse / gt.mean():.2f}%")

# Let's also check normalized MSE (NMSE)
nmse = mse / (gt.std() ** 2)
print(f"  NMSE (normalized by variance): {nmse:.4f}")

# Check if intensities are in comparable ranges
print(f"\nIntensity Scale Analysis:")
print(f"  GT dynamic range: {gt.max() - gt.min():.2f}")
print(f"  U-Net dynamic range: {unet.max() - unet.min():.2f}")
print(f"  GT mean intensity: {gt.mean():.2f}")
print(f"  U-Net mean intensity: {unet.mean():.2f}")
print(f"  Mean intensity difference: {abs(gt.mean() - unet.mean()):.2f}")

print("\n" + "="*70)
print("INTERPRETATION:")
print("="*70)
print(f"MSE is high because MRI intensities are in the range of ~{gt.min():.0f}-{gt.max():.0f}.")
print(f"Even small errors get squared and become large numbers.")
print(f"RMSE of {rmse:.2f} means average pixel error is ~{rmse:.1f} intensity units.")
print(f"This is only {100 * rmse / (gt.max() - gt.min()):.1f}% of the full intensity range.")
print(f"\nFor MRI data with large intensity values, MSE is not as intuitive as PSNR/SSIM.")
print("Consider reporting NMSE or RMSE as % of dynamic range instead.")
print("="*70)

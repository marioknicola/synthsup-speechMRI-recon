#!/usr/bin/env python3
"""
Generate synthetic undersampling pipeline figure for abstract.

Shows the complete pipeline from HR k-space to synthetic LR image.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat
import os

# Set style for publication-quality figures
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

def load_kspace(kspace_file):
    """Load k-space from MAT file."""
    mat_data = loadmat(kspace_file)
    kspace = mat_data['kspace']  # Shape (Ny, Nx, Nc)
    return kspace

def apply_mri_transform(img):
    """Apply rotation and flip to match MRI orientation."""
    # Rotate 90 degrees clockwise
    img_rot = np.rot90(img, k=-1)
    # Flip horizontally
    img_final = np.flip(img_rot, axis=1)
    return img_final


def truncate_kspace_centered_on_max(kspace, target_size=(82, 80)):
    """
    Truncate k-space to target size, centered on k-space maximum value.
    
    Args:
        kspace: (Ny, Nx, Nc) k-space array
        target_size: Target size (Ny_target, Nx_target)
    
    Returns:
        Truncated k-space (Ny_target, Nx_target, Nc)
        Center coordinates (center_y, center_x)
    """
    Ny, Nx, Nc = kspace.shape
    Ny_target, Nx_target = target_size
    
    # Find k-space maximum across all coils
    kspace_magnitude = np.abs(kspace).sum(axis=2)  # Sum over coils
    max_idx = np.unravel_index(np.argmax(kspace_magnitude), kspace_magnitude.shape)
    center_y, center_x = max_idx
    
    # Extract centered region
    start_y = center_y - Ny_target // 2
    end_y = start_y + Ny_target
    start_x = center_x - Nx_target // 2
    end_x = start_x + Nx_target
    
    # Ensure we're within bounds
    start_y = max(0, min(start_y, Ny - Ny_target))
    start_x = max(0, min(start_x, Nx - Nx_target))
    end_y = start_y + Ny_target
    end_x = start_x + Nx_target
    
    kspace_truncated = kspace[start_y:end_y, start_x:end_x, :]
    
    return kspace_truncated, (center_y, center_x), (start_y, end_y, start_x, end_x)

def add_rician_noise(img_complex, snr_db=20):
    """
    Add Rician noise to complex image (simulates MRI noise).
    
    Args:
        img_complex: Complex image array
        snr_db: Signal-to-noise ratio in dB
    
    Returns:
        Noisy magnitude image
    """
    signal_power = np.mean(np.abs(img_complex)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power / 2)  # Divide by 2 for real and imaginary parts
    
    # Add Gaussian noise to real and imaginary parts
    noise_real = np.random.normal(0, noise_std, img_complex.shape)
    noise_imag = np.random.normal(0, noise_std, img_complex.shape)
    
    img_noisy = img_complex + noise_real + 1j * noise_imag
    
    # Return magnitude (Rician distributed)
    return np.abs(img_noisy)

def coil_combine_rss(kspace):
    """Root sum of squares coil combination in image domain."""
    # Transform to image domain
    img_coils = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1)
    )
    
    # RSS combination
    img_combined = np.sqrt(np.sum(np.abs(img_coils)**2, axis=2))
    
    return img_combined

def generate_pipeline_figure():
    """Generate complete pipeline figure with 4 panels."""
    
    # Load data
    print("Loading data...")
    kspace_file = '../kspace_mat_FS/kspace_Subject0026_aa.mat'
    hr_file = '../HR_nii/kspace_Subject0026_aa.nii'
    lr_file = '../Synth_LR_nii/LR_kspace_Subject0026_aa.nii'
    
    # Load k-space
    kspace_full = load_kspace(kspace_file)  # (312, 410, 22)
    print(f"Full k-space shape: {kspace_full.shape}")
    
    # Load HR and LR images
    hr_img = nib.load(hr_file).get_fdata()
    lr_img = nib.load(lr_file).get_fdata()
    
    # Apply MRI transform to images
    hr_img = apply_mri_transform(hr_img)
    lr_img = apply_mri_transform(lr_img)
    
    print(f"HR shape: {hr_img.shape}, LR shape: {lr_img.shape}")
    
    # Pipeline steps
    print("\nProcessing pipeline...")
    
    # Step 1: Truncate k-space centered on maximum
    kspace_truncated, max_coords, bounds = truncate_kspace_centered_on_max(
        kspace_full, target_size=(82, 80)
    )
    center_y, center_x = max_coords
    start_y, end_y, start_x, end_x = bounds
    print(f"1. K-space maximum at: ({center_y}, {center_x})")
    print(f"   Truncated k-space: {kspace_truncated.shape}")
    print(f"   Bounds: Y[{start_y}:{end_y}], X[{start_x}:{end_x}]")
    
    # Step 2: Reconstruct from truncated k-space (no noise)
    img_truncated_clean = coil_combine_rss(kspace_truncated)
    img_truncated_clean = apply_mri_transform(img_truncated_clean)
    print(f"2. Clean truncated image: {img_truncated_clean.shape}")
    
    # Step 3: Add Rician noise
    # Transform truncated image back to complex for noise injection
    img_coils_truncated = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace_truncated, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1)
    )
    # Add noise to each coil
    img_noisy_coils = np.zeros_like(img_coils_truncated)
    snr_db = 25  # SNR in dB
    for c in range(img_coils_truncated.shape[2]):
        img_noisy_coils[:, :, c] = add_rician_noise(img_coils_truncated[:, :, c], snr_db)
    
    # RSS combination of noisy coils
    img_truncated_noisy = np.sqrt(np.sum(img_noisy_coils**2, axis=2))
    img_truncated_noisy = apply_mri_transform(img_truncated_noisy)
    print(f"3. Noisy truncated image (SNR={snr_db} dB): {img_truncated_noisy.shape}")
    
    # For visualization: compute k-space magnitudes
    kspace_full_mag = np.log10(np.abs(kspace_full).sum(axis=2) + 1)
    kspace_truncated_mag = np.log10(np.abs(kspace_truncated).sum(axis=2) + 1)
    
    # Flip k-space vertically for correct display
    kspace_full_mag = np.flipud(kspace_full_mag)
    kspace_truncated_mag = np.flipud(kspace_truncated_mag)
    
    # Create figure with 4 panels
    print("\nGenerating figure...")
    fig = plt.figure(figsize=(20, 5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3,
                  left=0.05, right=0.95, top=0.92, bottom=0.08)
    
    # Panel 1: HR Image
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(hr_img, cmap='gray', aspect='equal', vmin=0, vmax=np.percentile(hr_img, 99.5))
    ax1.axis('off')
    
    # Add arrow
    ax1.annotate('', xy=(1.02, 0.5), xytext=(0.98, 0.5),
                 xycoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                 fontsize=10)
    
    # Panel 2: Full k-space with truncation box
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(kspace_full_mag, cmap='gray', aspect='auto', origin='lower')
    
    # Draw truncation box (need to adjust for flipped coordinates)
    from matplotlib.patches import Rectangle
    # Since we flipped, adjust the y-coordinates
    rect_y = kspace_full_mag.shape[0] - end_y
    rect = Rectangle((start_x, rect_y), end_x - start_x, end_y - start_y,
                     linewidth=3, edgecolor='cyan', facecolor='none')
    ax2.add_patch(rect)
    
    # Mark center (adjust for flip)
    center_y_flipped = kspace_full_mag.shape[0] - center_y
    ax2.plot(center_x, center_y_flipped, 'r+', markersize=15, markeredgewidth=2)
    ax2.text(center_x + 10, center_y_flipped - 10, 'Max', color='red', fontsize=10,
             fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax2.set_xlabel('kx', fontsize=10)
    ax2.set_ylabel('ky', fontsize=10)
    
    # Add arrow
    ax2.annotate('', xy=(1.02, 0.5), xytext=(0.98, 0.5),
                 xycoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                 fontsize=10)
    
    # Panel 3: Truncated k-space with noise
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(kspace_truncated_mag, cmap='gray', aspect='auto', origin='lower')
    ax3.set_xlabel('kx', fontsize=10)
    ax3.set_ylabel('ky', fontsize=10)
    
    # Add arrow
    ax3.annotate('', xy=(1.02, 0.5), xytext=(0.98, 0.5),
                 xycoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                 fontsize=10)
    
    # Panel 4: Final Synthetic LR
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(lr_img, cmap='gray', aspect='equal', vmin=0, vmax=np.percentile(lr_img, 99.5))
    ax4.axis('off')
    
    # Save figure
    output_file = '../pipeline_figure_subject0026_aa.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved: {output_file}")
    
    # Also save high-res version
    output_file_hires = '../pipeline_figure_subject0026_aa_hires.png'
    plt.savefig(output_file_hires, dpi=600, bbox_inches='tight')
    print(f"✓ High-res figure saved: {output_file_hires}")
    
    plt.close()
    
    print("\nImage statistics:")
    print(f"  HR:           mean={hr_img.mean():.1f}, max={hr_img.max():.1f}")
    print(f"  Synth LR:     mean={lr_img.mean():.1f}, max={lr_img.max():.1f}")
    print(f"  Clean 82×80:  mean={img_truncated_clean.mean():.1f}, max={img_truncated_clean.max():.1f}")
    print(f"  Noisy 82×80:  mean={img_truncated_noisy.mean():.1f}, max={img_truncated_noisy.max():.1f}")

if __name__ == '__main__':
    print("=" * 60)
    print("Synthetic Undersampling Pipeline Figure Generator")
    print("=" * 60)
    
    np.random.seed(42)  # For reproducible noise
    generate_pipeline_figure()
    
    print("\n" + "=" * 60)
    print("✓ Figure generated successfully!")
    print("=" * 60)

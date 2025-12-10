"""
Synthetic Undersampling with SENSE Reconstruction

This script demonstrates SENSE reconstruction on synthetically undersampled
fully-sampled k-space data. It performs the following steps:

1. Load fully-sampled k-space data and coil sensitivity maps
2. Crop k-space to target size (80×82) centered on maximum magnitude
3. Resize coil sensitivity maps to match image space dimensions
4. Apply variable-density undersampling mask (R=3 edges, R=1 center)
5. Perform line-by-line SENSE reconstruction
6. Visualize: Original SoS, Aliased, and SENSE reconstructed images

Usage:
    Edit the paths in the __main__ block to point to your k-space and coilmap files:
    - kspace_path: Path to fully-sampled k-space .mat file
    - coilmap_path: Path to coil sensitivity maps .mat file
    - output_path: Path to save the output comparison figure
"""

import numpy as np
import scipy.io
import scipy.fft
import scipy.linalg
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import nibabel as nib
import os


def load_kspace(kspace_path):
    """Load k-space data from .mat file."""
    print(f"Loading k-space from: {kspace_path}")
    matload = scipy.io.loadmat(kspace_path)
    kspace_data = matload['kspace']
    print(f"  K-space shape: {kspace_data.shape}")
    return kspace_data


def load_coil_sensitivity_maps(coilmap_path, average_frames=True):
    """Load coil sensitivity maps from .mat file.
    
    Args:
        coilmap_path: Path to coil sensitivity maps .mat file
        average_frames: If True and maps are 4D (H, W, T, C), average across time dimension
    
    Returns:
        coil_sens_maps: Coil sensitivity maps (H, W, 1, C)
    """
    print(f"Loading coil sensitivity maps from: {coilmap_path}")
    coilmap_mat = scipy.io.loadmat(coilmap_path)
    coil_sens_maps = coilmap_mat['coilmap']
    print(f"  Coil sensitivity maps shape: {coil_sens_maps.shape}")
    
    # Handle 4D dynamic sensitivity maps (H, W, T, C)
    if coil_sens_maps.ndim == 4 and coil_sens_maps.shape[2] > 1 and average_frames:
        print(f"  Averaging across {coil_sens_maps.shape[2]} frames...")
        # Average across time dimension (axis=2)
        coil_sens_maps_avg = np.mean(coil_sens_maps, axis=2, keepdims=True)
        print(f"  Averaged shape: {coil_sens_maps_avg.shape}")
        return coil_sens_maps_avg
    
    return coil_sens_maps


def crop_kspace(kspace, crop_size=(80, 82)):
    """
    Crop k-space around its maximum to specified size.
    
    Args:
        kspace: K-space data (H, W, C) or (H, W, C, T) for dynamic
        crop_size: (crop_x, crop_y) - target size (Ny, Nx)
    
    Returns:
        kspace_cropped: Cropped k-space (or first frame if 4D)
        num_coils: Number of coils
    """
    crop_x, crop_y = crop_size
    
    # Handle 4D k-space (dynamic acquisition) - use first frame
    if kspace.ndim == 4:
        print(f"  4D k-space detected with {kspace.shape[3]} frames, using first frame")
        kspace = kspace[:, :, :, 0]
    
    # Find location of maximum magnitude across coils
    if kspace.ndim == 3:
        mag2d = np.max(np.abs(kspace), axis=2)
        num_coils = kspace.shape[2]
    else:
        mag2d = np.abs(kspace)
        num_coils = 1
    
    max_x, max_y = np.unravel_index(np.argmax(mag2d), mag2d.shape)
    
    # Compute start indices centered on maximum
    start_x = max(0, min(max_x - crop_x // 2, kspace.shape[0] - crop_x))
    start_y = max(0, min(max_y - crop_y // 2, kspace.shape[1] - crop_y))
    
    # Perform crop
    if kspace.ndim == 3:
        kspace_cropped = kspace[start_x:start_x + crop_x, start_y:start_y + crop_y, :]
    else:
        kspace_cropped = kspace[start_x:start_x + crop_x, start_y:start_y + crop_y]
    
    print(f"\nCropping k-space:")
    print(f"  Original shape: {kspace.shape}")
    print(f"  Cropped shape: {kspace_cropped.shape}")
    print(f"  Crop region: start=(x={start_x}, y={start_y}), centered on max=(x={max_x}, y={max_y})")
    
    return kspace_cropped, num_coils


def resize_coil_maps(coil_sens_maps, target_shape, num_coils):
    """
    Resize coil sensitivity maps to match image space dimensions.
    
    Args:
        coil_sens_maps: Original coil sensitivity maps (H, W, 1, C)
        target_shape: (Ny, Nx) - target image space size
        num_coils: Number of coils
    
    Returns:
        coil_sens_maps_resized: Resized sensitivity maps (Ny, Nx, 1, C)
    """
    coil_sens_maps_resized = np.zeros((target_shape[0], target_shape[1], 1, num_coils), 
                                      dtype=coil_sens_maps.dtype)
    
    for c in range(num_coils):
        coil_map = coil_sens_maps[:, :, 0, c]
        zoom_factors = (target_shape[0] / coil_map.shape[0], target_shape[1] / coil_map.shape[1])
        coil_map_resized = zoom(coil_map, zoom_factors, order=3)
        coil_sens_maps_resized[:, :, 0, c] = coil_map_resized
    
    print(f"\nResizing coil sensitivity maps:")
    print(f"  From: {coil_sens_maps.shape[:2]} to {target_shape}")
    print(f"  Resized shape: {coil_sens_maps_resized.shape}")
    
    return coil_sens_maps_resized


def plot_coil_sensitivities(coil_sens_original, coil_sens_resized, coil_indices=[5, 10, 15]):
    """
    Plot selected coil sensitivity maps before and after resizing.
    
    Args:
        coil_sens_original: Original coil sensitivity maps (H, W, 1, C)
        coil_sens_resized: Resized coil sensitivity maps (Ny, Nx, 1, C)
        coil_indices: List of coil indices to visualize
    """
    n_coils = len(coil_indices)
    fig, axes = plt.subplots(2, n_coils, figsize=(5*n_coils, 10))
    
    if n_coils == 1:
        axes = axes.reshape(2, 1)
    
    for idx, coil_idx in enumerate(coil_indices):
        # Original coil map
        coil_map_orig = np.abs(coil_sens_original[:, :, 0, coil_idx])
        axes[0, idx].imshow(coil_map_orig, cmap='gray')
        axes[0, idx].set_title(f'Original Coil {coil_idx+1}\n({coil_map_orig.shape[0]}×{coil_map_orig.shape[1]})', fontsize=12)
        axes[0, idx].axis('off')
        
        # Resized coil map
        coil_map_resized = np.abs(coil_sens_resized[:, :, 0, coil_idx])
        axes[1, idx].imshow(coil_map_resized, cmap='gray')
        axes[1, idx].set_title(f'Resized Coil {coil_idx+1}\n({coil_map_resized.shape[0]}×{coil_map_resized.shape[1]})', fontsize=12)
        axes[1, idx].axis('off')
    
    plt.suptitle('Coil Sensitivity Maps: Before and After Resizing', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_kspace_undersampling(kspace_full, kspace_undersampled, acquired_indices):
    """
    Visualize k-space before and after undersampling, and the sampling mask.
    
    Args:
        kspace_full: Full k-space data (Ny, Nx, Nc)
        kspace_undersampled: Undersampled k-space (Ny, Nx, Nc)
        acquired_indices: Indices of acquired k-space lines
    """
    Ny, Nx, Nc = kspace_full.shape
    
    # Compute sum over coils for visualization
    kspace_full_mag = np.sum(np.abs(kspace_full), axis=2)
    kspace_undersampled_mag = np.sum(np.abs(kspace_undersampled), axis=2)
    
    # Create binary mask visualization
    mask_2d = np.zeros((Ny, Nx))
    mask_2d[:, acquired_indices] = 1.0
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Full k-space (log scale for better visualization)
    im0 = axes[0].imshow(kspace_full_mag, cmap='gray', aspect='auto')
    axes[0].set_title(f'Truncated K-space\n({Ny}×{Nx})', fontsize=12)
    axes[0].set_xlabel('kₓ (Frequency Encoding)')
    axes[0].set_ylabel('kᵧ (Phase Encoding)')
    
    # Undersampling mask
    im1 = axes[1].imshow(mask_2d, cmap='jet', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title(f'Undersampling Mask', fontsize=12)
    axes[1].set_xlabel('kₓ (Frequency Encoding)')
    axes[1].set_ylabel('kᵧ (Phase Encoding)')
    
    # Undersampled k-space
    im2 = axes[2].imshow(kspace_undersampled_mag, cmap='gray', aspect='auto')
    axes[2].set_title(f'Undersampled K-space', fontsize=12)
    axes[2].set_xlabel('kₓ (Frequency Encoding)')
    axes[2].set_ylabel('kᵧ (Phase Encoding)')
    
    plt.suptitle('K-space Undersampling Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def perform_sense_reconstruction(kspace_cropped, coil_sens_resized, acquired_indices):
    """
    Perform SENSE reconstruction on synthetically undersampled k-space.
    
    SENSE Algorithm Overview:
    ------------------------
    The SENSE algorithm solves: E × p = I
    where:
        - p = true unaliased pixels (what we solve for)
        - I = aliased measurements from all coils (what we observe)
        - E = encoding matrix (relates true pixels to measurements via coil sensitivities and aliasing)
    
    The encoding matrix E = A ⊗ S describes two effects:
        - A: aliasing PSF (how undersampling causes spatial folding)
        - S: coil sensitivities (how each coil weights different spatial locations)
    
    With multiple coils (Nc=22) and moderate undersampling (R=2), we have an overdetermined
    system (more equations than unknowns), making the solution robust.
    
    Args:
        kspace_cropped: Cropped k-space data (Ny, Nx, Nc)
        coil_sens_resized: Resized coil sensitivity maps (Ny, Nx, 1, Nc)
        acquired_indices: Indices of acquired k-space lines
    
    Returns:
        recon_normalized: Reconstructed image magnitude, normalized to [0, 1]
    """
    Ny, Nx, Nc = kspace_cropped.shape
    
    print(f"\n=== SENSE RECONSTRUCTION ===")
    print(f"Acquired indices: {len(acquired_indices)} out of {Nx} columns")
    print(f"Undersampling factor: {Nx / len(acquired_indices):.1f}x")
    print(f"Acquisition pattern: R=3 edges + R=1 center (variable density)")
    
    # ========== STEP 1: Create Undersampled K-space ==========
    # Zero out non-acquired k-space lines to simulate undersampling
    kspace_undersampled = np.zeros_like(kspace_cropped)
    kspace_undersampled[:, acquired_indices, :] = kspace_cropped[:, acquired_indices, :]
    
    # Reshape for SENSE reconstruction (add frame dimension)
    k_undersampled = kspace_undersampled[:, :, :, np.newaxis]  # (Ny, Nx, Nc, 1)
    coil_sens = np.transpose(coil_sens_resized, (0, 1, 3, 2))  # (Ny, Nx, Nc, 1)
    
    # ========== STEP 2: Compute Aliasing Point Spread Function (PSF) ==========
    # The PSF describes how undersampling causes aliasing in image space
    # It's the inverse FFT of the k-space sampling mask
    k_space_mask = np.zeros(Nx, dtype=np.complex128)
    k_space_mask[acquired_indices] = 1.0  # Mask: 1 where acquired, 0 elsewhere
    psf_A = scipy.fft.ifftshift(scipy.fft.ifft(scipy.fft.ifftshift(k_space_mask)))
    # psf_A describes how pixels fold onto each other due to undersampling
    
    # ========== STEP 3: Transform to Aliased Image Space ==========
    # Apply inverse FFT to get the aliased (folded) image for each coil
    k_frame = k_undersampled[:, :, :, 0]
    sens_frame = coil_sens[:, :, :, 0]
    
    img_aliased = scipy.fft.ifftshift(
        scipy.fft.ifft2(scipy.fft.ifftshift(k_frame, axes=(0,1)), axes=(0,1)),
        axes=(0,1)
    )
    # img_aliased: (Ny, Nx, Nc) - aliased image with folding artifacts
    
    # ========== STEP 4: SENSE Reconstruction Line-by-Line ==========
    # We can solve for each image row (y-line) independently
    reconstructed_image = np.zeros((Ny, Nx), dtype=np.complex128)
    
    print("Reconstructing...")
    for y_idx in range(Ny):
        if y_idx % 20 == 0:
            print(f"  Line {y_idx+1}/{Ny}")
        
        # --- 4a: Extract aliased data for this row ---
        I_y_slice = img_aliased[y_idx, :, :]  # (Nx, Nc) - aliased pixels from all coils
        I_vec = I_y_slice.T.flatten()  # (Nc*Nx,) - vectorize: [coil0_all_pixels, coil1_all_pixels, ...]
        # I_vec is our measurement vector (what we observe)
        
        # --- 4b: Build the Encoding Matrix E ---
        # E relates true pixels (p) to measurements (I): E × p = I
        # E has shape (Nc*Nx, Nx) - one equation per coil per pixel
        S_y_slice = sens_frame[y_idx, :, :]  # (Nx, Nc) - sensitivities for this row
        E = np.zeros((Nc * Nx, Nx), dtype=np.complex128)
        
        # Create circulant matrix from PSF (describes aliasing pattern)
        # A circulant matrix applies circular convolution
        A_matrix = scipy.linalg.circulant(psf_A)  # (Nx, Nx)
        
        # For each coil, create one block of E
        for c in range(Nc):
            # S_c_diag: diagonal matrix with coil c's sensitivities
            S_c_diag = np.diag(S_y_slice[:, c])  # (Nx, Nx)
            
            # Encoding for coil c: E_c = A × S_c
            # This says: "aliasing pattern A applied to sensitivity-weighted pixels S_c"
            E_c_block = A_matrix @ S_c_diag  # (Nx, Nx)
            
            # Stack blocks vertically: E = [E_0; E_1; ...; E_{Nc-1}]
            E[c*Nx:(c+1)*Nx, :] = E_c_block
        
        # Now E is (Nc*Nx, Nx) = (22*82, 82) = (1804, 82)
        # We have 1804 equations for 82 unknowns → overdetermined system
        
        # --- 4c: Solve the Linear System E × p = I ---
        # Use normal equations: E^H × E × p = E^H × I
        # where E^H is the conjugate transpose (Hermitian)
        E_H = E.conj().T  # (Nx, Nc*Nx)
        E_H_E = E_H @ E   # (Nx, Nx) - square system matrix
        E_H_I = E_H @ I_vec  # (Nx,) - right-hand side
        
        # Add Tikhonov regularization to prevent ill-conditioning
        # reg_param is scaled by the trace to adapt to matrix magnitude
        reg_param = 1e-6 * np.trace(E_H_E) / Nx
        E_H_E_reg = E_H_E + np.eye(Nx, dtype=np.complex128) * reg_param
        
        # Solve the regularized system
        try:
            # assume_a='her' uses efficient Hermitian solver
            p_y_vec = scipy.linalg.solve(E_H_E_reg, E_H_I, assume_a='her')
        except scipy.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular
            p_y_vec = scipy.linalg.pinv(E_H_E_reg) @ E_H_I
        
        # p_y_vec is the unaliased pixel values for this row!
        reconstructed_image[y_idx, :] = p_y_vec
    
    print("Reconstruction complete!")
    
    recon_magnitude = np.abs(reconstructed_image)
    
    # Normalize to [0, 1]
    recon_min, recon_max = recon_magnitude.min(), recon_magnitude.max()
    if recon_max > recon_min:
        recon_normalized = (recon_magnitude - recon_min) / (recon_max - recon_min)
    else:
        recon_normalized = recon_magnitude
    
    print(f"  Before normalization: [{recon_min:.2e}, {recon_max:.2e}], mean={recon_magnitude.mean():.2e}")
    print(f"  After normalization: [0.00e+00, 1.00e+00], mean={recon_normalized.mean():.2e}")
    
    return recon_normalized


def zero_pad_to_original_size(recon_image, target_shape):
    """
    Zero-pad reconstructed image in k-space back to original dimensions.
    
    This function:
    1. Transforms the reconstructed image to k-space
    2. Zero-pads the k-space to target dimensions
    3. Transforms back to image space
    
    Args:
        recon_image: Reconstructed image (Ny_crop, Nx_crop)
        target_shape: (Ny_target, Nx_target) - original k-space dimensions
    
    Returns:
        padded_image: Zero-padded image in image space (Ny_target, Nx_target)
    """
    Ny_crop, Nx_crop = recon_image.shape
    Ny_target, Nx_target = target_shape
    
    print(f"\nZero-padding in k-space:")
    print(f"  From: {recon_image.shape} to {target_shape}")
    
    # Transform to k-space (use axes=(0,1) for proper centering)
    kspace_recon = scipy.fft.fftshift(
        scipy.fft.fft2(scipy.fft.fftshift(recon_image, axes=(0,1)), axes=(0,1)),
        axes=(0,1)
    )
    
    # Create zero-padded k-space
    kspace_padded = np.zeros((Ny_target, Nx_target), dtype=np.complex128)
    
    # Calculate padding offsets to center the cropped k-space
    offset_y = (Ny_target - Ny_crop) // 2
    offset_x = (Nx_target - Nx_crop) // 2
    
    # Place cropped k-space in center of padded k-space
    kspace_padded[offset_y:offset_y + Ny_crop, offset_x:offset_x + Nx_crop] = kspace_recon
    
    # Transform back to image space
    padded_image = scipy.fft.ifftshift(
        scipy.fft.ifft2(scipy.fft.ifftshift(kspace_padded, axes=(0,1)), axes=(0,1)),
        axes=(0,1)
    )
    
    padded_image_mag = np.abs(padded_image)
    
    # Normalize to [0, 1]
    padded_min, padded_max = padded_image_mag.min(), padded_image_mag.max()
    if padded_max > padded_min:
        padded_image_normalized = (padded_image_mag - padded_min) / (padded_max - padded_min)
    else:
        padded_image_normalized = padded_image_mag
    
    print(f"  Padded image shape: {padded_image_normalized.shape}")
    print(f"  Before normalization: [{padded_min:.2e}, {padded_max:.2e}], mean={padded_image_mag.mean():.2e}")
    print(f"  After normalization: [0.00e+00, 1.00e+00], mean={padded_image_normalized.mean():.2e}")
    
    return padded_image_normalized


def compute_sos_image(kspace, use_full_axes=False, normalize=True):
    """
    Compute sum-of-squares image from k-space.
    
    Args:
        kspace: K-space data (Ny, Nx, Nc)
        use_full_axes: If True, use axes=(0,1) for fftshift; if False, use axes=0 only
        normalize: If True, normalize output to [0, 1]
    
    Returns:
        sos_image: Sum-of-squares image (normalized if normalize=True)
    """
    Ny, Nx, Nc = kspace.shape
    sos_image = np.zeros((Ny, Nx))
    
    axes_arg = (0,1) if use_full_axes else 0
    
    for c in range(Nc):
        img_coil = scipy.fft.ifftshift(
            scipy.fft.ifft2(scipy.fft.ifftshift(kspace[:, :, c], axes=axes_arg)),
            axes=axes_arg
        )
        sos_image += np.abs(img_coil)**2
    
    sos_image = np.sqrt(sos_image)
    
    # Normalize to [0, 1] if requested
    if normalize:
        sos_min, sos_max = sos_image.min(), sos_image.max()
        if sos_max > sos_min:
            sos_image = (sos_image - sos_min) / (sos_max - sos_min)
    
    return sos_image


def visualize_results(kspace_cropped, kspace_undersampled, recon_magnitude, acquired_indices, output_path=None, recon_padded=None):
    """
    Visualize original, aliased, and SENSE reconstructed images.
    
    Args:
        kspace_cropped: Full cropped k-space (Ny, Nx, Nc)
        kspace_undersampled: Undersampled k-space (Ny, Nx, Nc)
        recon_magnitude: SENSE reconstructed image magnitude (Ny, Nx)
        acquired_indices: Acquired k-space line indices
        output_path: Path to save figure (optional)
        recon_padded: Zero-padded reconstruction (optional, Ny_orig, Nx_orig)
    """
    print("\n=== VISUALIZATION ===")
    
    Ny, Nx, Nc = kspace_cropped.shape
    
    # Compute original SoS image (use axes=0 for correct orientation, already normalized)
    img_original_norm = compute_sos_image(kspace_cropped, use_full_axes=False, normalize=True)
    print(f"Original SoS: min={img_original_norm.min():.2e}, max={img_original_norm.max():.2e}, mean={img_original_norm.mean():.2e}")
    
    # Compute aliased SoS image (use axes=0 for correct orientation, already normalized)
    img_aliased_norm = compute_sos_image(kspace_undersampled, use_full_axes=False, normalize=True)
    print(f"Aliased SoS: min={img_aliased_norm.min():.2e}, max={img_aliased_norm.max():.2e}, mean={img_aliased_norm.mean():.2e}")
    
    # SENSE reconstruction (already normalized from perform_sense_reconstruction)
    print(f"SENSE recon: min={recon_magnitude.min():.2e}, max={recon_magnitude.max():.2e}, mean={recon_magnitude.mean():.2e}")
    recon_norm = recon_magnitude
    
    # Determine number of subplots
    num_plots = 4 if recon_padded is not None else 3
    
    # Create figure
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    
    axes[0].imshow(img_original_norm, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Truncated kspace', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(img_aliased_norm, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Aliased (40/82 lines)', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(recon_norm, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'SENSE Reconstructed', fontsize=12)
    axes[2].axis('off')
    
    # Add zero-padded reconstruction if available (already normalized in zero_pad_to_original_size)
    if recon_padded is not None:
        print(f"SENSE padded: min={recon_padded.min():.2e}, max={recon_padded.max():.2e}, mean={recon_padded.mean():.2e}")
        
        axes[3].imshow(recon_padded, cmap='gray', vmin=0, vmax=1)
        axes[3].set_title(f'SENSE zero-padded (Synthetic LR v2)', fontsize=12)
        axes[3].axis('off')
    

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    plt.show()


def save_as_nifti(image_data, output_path, subject_id):
    """
    Save image data as NIfTI file.
    
    Args:
        image_data: 2D image array (Ny, Nx), should be normalized to [0, 1]
        output_path: Directory path where to save the file
        subject_id: Subject identifier for filename
    
    Returns:
        filepath: Full path to saved NIfTI file
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Verify normalization
    if image_data.min() < 0 or image_data.max() > 1:
        print(f"Warning: Image not in [0,1] range. Current range: [{image_data.min():.2e}, {image_data.max():.2e}]")
        print("Re-normalizing to [0, 1]...")
        img_min, img_max = image_data.min(), image_data.max()
        if img_max > img_min:
            image_data = (image_data - img_min) / (img_max - img_min)
    
    # Fix orientation: rotate 90 degrees clockwise and flip horizontally
    # np.rot90(image, k=-1) rotates 90 degrees clockwise
    # np.fliplr() flips left-right
    image_data_corrected = np.fliplr(np.rot90(image_data, k=-1))
    
    # Create filename
    filename = f"{subject_id}.nii"
    filepath = os.path.join(output_path, filename)
    
    # Create NIfTI image (2D data)
    nifti_img = nib.Nifti1Image(image_data_corrected, affine=np.eye(4))
    
    # Save to file
    nib.save(nifti_img, filepath)
    
    print(f"\nNIfTI file saved:")
    print(f"  Path: {filepath}")
    print(f"  Shape: {image_data_corrected.shape}")
    print(f"  Data range: [{image_data_corrected.min():.2e}, {image_data_corrected.max():.2e}]")
    
    return filepath


def main(kspace_path, coilmap_path, output_path=None, zero_pad=True, plot=False, save_nifti=False, nifti_output_dir=None):
    """
    Main function for synthetic undersampling with SENSE reconstruction.
    
    Args:
        kspace_path: Path to fully-sampled k-space .mat file
        coilmap_path: Path to coil sensitivity maps .mat file
        output_path: Path to save output figure (optional)
        zero_pad: If True, zero-pad reconstruction back to original k-space size
        plot: If True, show intermediate plots (coil maps and k-space undersampling)
        save_nifti: If True, save zero-padded reconstruction as NIfTI file
        nifti_output_dir: Directory to save NIfTI files (required if save_nifti=True)
    """
    # Variable-density acquisition pattern (from Dynamic SENSE)
    acquired_indices = np.array([
        0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30,  # Left R=3 (11 lines)
        32, 33, 34, 35, 36, 37, 38, 39, 40,      # Center R=1 (18 lines)
        41, 42, 43, 44, 45, 46, 47, 48, 49,
        51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81  # Right R=3 (11 lines)
    ])
    
    # Load data
    kspace_data = load_kspace(kspace_path)
    coil_sens_maps = load_coil_sensitivity_maps(coilmap_path)
    
    # Store original dimensions for zero-padding
    original_shape = kspace_data.shape[:2]  # (Ny_orig, Nx_orig)
    
    # Crop k-space
    crop_size = (80, 82)  # (Ny, Nx)
    kspace_cropped, num_coils = crop_kspace(kspace_data, crop_size)
    
    # Resize coil sensitivity maps
    coil_sens_resized = resize_coil_maps(coil_sens_maps, crop_size, num_coils)
    
    # Plot coil sensitivities before and after resizing (only if resizing was needed)
    if plot and (coil_sens_maps.shape[:2] != crop_size):
        plot_coil_sensitivities(coil_sens_maps, coil_sens_resized, coil_indices=[5, 10, 15])
    
    # Create undersampled k-space
    kspace_undersampled = np.zeros_like(kspace_cropped)
    kspace_undersampled[:, acquired_indices, :] = kspace_cropped[:, acquired_indices, :]
    
    # Plot k-space undersampling visualization
    if plot:
        plot_kspace_undersampling(kspace_cropped, kspace_undersampled, acquired_indices)
    
    # Perform SENSE reconstruction
    recon_magnitude = perform_sense_reconstruction(kspace_cropped, coil_sens_resized, acquired_indices)
    
    # Zero-pad back to original size if requested
    recon_padded = None
    if zero_pad:
        recon_padded = zero_pad_to_original_size(recon_magnitude, original_shape)
        
        # Save as NIfTI if requested
        if save_nifti and recon_padded is not None:
            if nifti_output_dir is None:
                print("Warning: save_nifti=True but nifti_output_dir not specified. Skipping NIfTI save.")
            else:
                # Extract subject ID from kspace filename
                subject_id = os.path.basename(kspace_path).replace('kspace_', '').replace('.mat', '')
                save_as_nifti(recon_padded, nifti_output_dir, subject_id)
    
    # Visualize results
    #visualize_results(kspace_cropped, kspace_undersampled, recon_magnitude, acquired_indices, output_path, recon_padded)


if __name__ == "__main__":
    # Paths to input data
    kspace_path = "/Users/marioknicola/MSc Project/kspace_mat_FS/kspace_Subject0026_vv.mat"
    coilmap_path = "/Users/marioknicola/MSc Project/sensitivity_maps/sens_Subject0026_Exam17853_80x82x100_nC22.mat"
    output_path = "/Users/marioknicola/MSc Project/synthsup-speechMRI-recon/utils/sense_recon_test.png"
    
    # NIfTI output configuration
    nifti_output_dir = "/Users/marioknicola/MSc Project/Synth_LR_v2_nii"
    
    # Configuration
    zero_pad = True      # Zero-pad reconstruction back to original size (312×410)
    plot = False          # Show intermediate plots (coil maps, k-space undersampling)
    save_nifti = True    # Save zero-padded reconstruction as NIfTI file
    
    main(kspace_path, coilmap_path, output_path, zero_pad=zero_pad, plot=plot, 
         save_nifti=save_nifti, nifti_output_dir=nifti_output_dir)


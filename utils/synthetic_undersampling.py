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


def load_kspace(kspace_path):
    """Load k-space data from .mat file."""
    print(f"Loading k-space from: {kspace_path}")
    matload = scipy.io.loadmat(kspace_path)
    kspace_data = matload['kspace']
    print(f"  K-space shape: {kspace_data.shape}")
    return kspace_data


def load_coil_sensitivity_maps(coilmap_path):
    """Load coil sensitivity maps from .mat file."""
    print(f"Loading coil sensitivity maps from: {coilmap_path}")
    coilmap_mat = scipy.io.loadmat(coilmap_path)
    coil_sens_maps = coilmap_mat['coilmap']
    print(f"  Coil sensitivity maps shape: {coil_sens_maps.shape}")
    return coil_sens_maps


def crop_kspace(kspace, crop_size=(80, 82)):
    """
    Crop k-space around its maximum to specified size.
    
    Args:
        kspace: K-space data (H, W, C)
        crop_size: (crop_x, crop_y) - target size (Ny, Nx)
    
    Returns:
        kspace_cropped: Cropped k-space
        num_coils: Number of coils
    """
    crop_x, crop_y = crop_size
    
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


def perform_sense_reconstruction(kspace_cropped, coil_sens_resized, acquired_indices):
    """
    Perform SENSE reconstruction on synthetically undersampled k-space.
    
    Args:
        kspace_cropped: Cropped k-space data (Ny, Nx, Nc)
        coil_sens_resized: Resized coil sensitivity maps (Ny, Nx, 1, Nc)
        acquired_indices: Indices of acquired k-space lines
    
    Returns:
        recon_magnitude: Reconstructed image magnitude
    """
    Ny, Nx, Nc = kspace_cropped.shape
    
    print(f"\n=== SENSE RECONSTRUCTION ===")
    print(f"Acquired indices: {len(acquired_indices)} out of {Nx} columns")
    print(f"Undersampling factor: {Nx / len(acquired_indices):.1f}x")
    print(f"Acquisition pattern: R=3 edges + R=1 center (variable density)")
    
    # Create undersampled k-space
    kspace_undersampled = np.zeros_like(kspace_cropped)
    kspace_undersampled[:, acquired_indices, :] = kspace_cropped[:, acquired_indices, :]
    
    # Reshape for SENSE reconstruction
    k_undersampled = kspace_undersampled[:, :, :, np.newaxis]  # (Ny, Nx, Nc, 1)
    coil_sens = np.transpose(coil_sens_resized, (0, 1, 3, 2))  # (Ny, Nx, Nc, 1)
    
    # Create k-space mask and aliasing PSF
    k_space_mask = np.zeros(Nx, dtype=np.complex128)
    k_space_mask[acquired_indices] = 1.0
    psf_A = scipy.fft.ifftshift(scipy.fft.ifft(scipy.fft.ifftshift(k_space_mask)))
    
    # Transform to image space
    k_frame = k_undersampled[:, :, :, 0]
    sens_frame = coil_sens[:, :, :, 0]
    
    img_aliased = scipy.fft.ifftshift(
        scipy.fft.ifft2(scipy.fft.ifftshift(k_frame, axes=(0,1)), axes=(0,1)),
        axes=(0,1)
    )
    
    # SENSE reconstruction line-by-line
    reconstructed_image = np.zeros((Ny, Nx), dtype=np.complex128)
    
    print("Reconstructing...")
    for y_idx in range(Ny):
        if y_idx % 20 == 0:
            print(f"  Line {y_idx+1}/{Ny}")
        
        # Get aliased image for this line
        I_y_slice = img_aliased[y_idx, :, :]
        I_vec = I_y_slice.T.flatten()
        
        # Build encoding matrix E
        S_y_slice = sens_frame[y_idx, :, :]
        E = np.zeros((Nc * Nx, Nx), dtype=np.complex128)
        A_matrix = scipy.linalg.circulant(psf_A)
        
        for c in range(Nc):
            S_c_diag = np.diag(S_y_slice[:, c])
            E_c_block = A_matrix @ S_c_diag
            E[c*Nx:(c+1)*Nx, :] = E_c_block
        
        # Solve: E * p = I with regularization
        E_H = E.conj().T
        E_H_E = E_H @ E
        E_H_I = E_H @ I_vec
        
        reg_param = 1e-6 * np.trace(E_H_E) / Nx
        E_H_E_reg = E_H_E + np.eye(Nx, dtype=np.complex128) * reg_param
        
        try:
            p_y_vec = scipy.linalg.solve(E_H_E_reg, E_H_I, assume_a='her')
        except scipy.linalg.LinAlgError:
            p_y_vec = scipy.linalg.pinv(E_H_E_reg) @ E_H_I
        
        reconstructed_image[y_idx, :] = p_y_vec
    
    print("Reconstruction complete!")
    
    recon_magnitude = np.abs(reconstructed_image)
    print(f"  Magnitude range: [{recon_magnitude.min():.2e}, {recon_magnitude.max():.2e}], mean={recon_magnitude.mean():.2e}")
    
    return recon_magnitude


def compute_sos_image(kspace, use_full_axes=False):
    """
    Compute sum-of-squares image from k-space.
    
    Args:
        kspace: K-space data (Ny, Nx, Nc)
        use_full_axes: If True, use axes=(0,1) for fftshift; if False, use axes=0 only
    
    Returns:
        sos_image: Sum-of-squares image
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
    
    return np.sqrt(sos_image)


def visualize_results(kspace_cropped, kspace_undersampled, recon_magnitude, acquired_indices, output_path=None):
    """
    Visualize original, aliased, and SENSE reconstructed images.
    
    Args:
        kspace_cropped: Full cropped k-space (Ny, Nx, Nc)
        kspace_undersampled: Undersampled k-space (Ny, Nx, Nc)
        recon_magnitude: SENSE reconstructed image magnitude (Ny, Nx)
        acquired_indices: Acquired k-space line indices
        output_path: Path to save figure (optional)
    """
    print("\n=== VISUALIZATION ===")
    
    Ny, Nx, Nc = kspace_cropped.shape
    
    # Compute original SoS image (use axes=0 for correct orientation)
    img_original = compute_sos_image(kspace_cropped, use_full_axes=False)
    print(f"Original SoS: min={img_original.min():.2e}, max={img_original.max():.2e}, mean={img_original.mean():.2e}")
    img_original_norm = (img_original - img_original.min()) / (img_original.max() - img_original.min())
    
    # Compute aliased SoS image (use axes=0 for correct orientation)
    img_aliased = compute_sos_image(kspace_undersampled, use_full_axes=False)
    print(f"Aliased SoS: min={img_aliased.min():.2e}, max={img_aliased.max():.2e}, mean={img_aliased.mean():.2e}")
    img_aliased_norm = (img_aliased - img_aliased.min()) / (img_aliased.max() - img_aliased.min())
    
    # Normalize SENSE reconstruction
    print(f"SENSE recon: min={recon_magnitude.min():.2e}, max={recon_magnitude.max():.2e}, mean={recon_magnitude.mean():.2e}")
    recon_norm = (recon_magnitude - recon_magnitude.min()) / (recon_magnitude.max() - recon_magnitude.min())
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_original_norm, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original (Full k-space, SoS)', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(img_aliased_norm, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Aliased (Zero-filled, {len(acquired_indices)}/{Nx} lines)', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(recon_norm, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('SENSE Reconstructed', fontsize=12)
    axes[2].axis('off')
    
    plt.suptitle(f'Synthetic Undersampling Reconstruction ({len(acquired_indices)}/{Nx} lines, {Nx/len(acquired_indices):.1f}x)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    plt.show()


def main(kspace_path, coilmap_path, output_path=None):
    """
    Main function for synthetic undersampling with SENSE reconstruction.
    
    Args:
        kspace_path: Path to fully-sampled k-space .mat file
        coilmap_path: Path to coil sensitivity maps .mat file
        output_path: Path to save output figure (optional)
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
    
    # Crop k-space
    crop_size = (80, 82)  # (Ny, Nx)
    kspace_cropped, num_coils = crop_kspace(kspace_data, crop_size)
    
    # Resize coil sensitivity maps
    coil_sens_resized = resize_coil_maps(coil_sens_maps, crop_size, num_coils)
    
    # Create undersampled k-space
    kspace_undersampled = np.zeros_like(kspace_cropped)
    kspace_undersampled[:, acquired_indices, :] = kspace_cropped[:, acquired_indices, :]
    
    # Perform SENSE reconstruction
    recon_magnitude = perform_sense_reconstruction(kspace_cropped, coil_sens_resized, acquired_indices)
    
    # Visualize results
    visualize_results(kspace_cropped, kspace_undersampled, recon_magnitude, acquired_indices, output_path)


if __name__ == "__main__":
    # Paths to input data
    kspace_path = "/Users/marioknicola/MSc Project/kspace_mat_FS/kspace_Subject0026_ee.mat"
    coilmap_path = "/Users/marioknicola/MSc Project/sensitivity_maps/sens_Subject0026_Exam17853_312x410x1_nC22.mat"
    output_path = "/Users/marioknicola/MSc Project/synthsup-speechMRI-recon/utils/sense_recon_test.png"
    
    main(kspace_path, coilmap_path, output_path)


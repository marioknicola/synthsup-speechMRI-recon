import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.fft
import scipy.linalg
import nibabel as nib
import scipy.ndimage
import os
import argparse

def perform_sense_recon_generalized(k_undersampled, coil_sensitivities, acquired_indices, save_nifti=False, output_dir=".", subject_id="default"):
    """
    Performs a generalized SENSE reconstruction using the exact k-space sampling mask.

    Assumes:
    - k_undersampled: (Ny, Nx, N_coils, N_frames) -> (80, 82, 22, 100)
    - coil_sensitivities: (Ny, Nx, N_coils, N_frames)
    - Undersampling is along the Nx (axis 1) dimension.
    - acquired_indices: The list of k-space columns (Nx dim) that were acquired.
    - save_nifti: If True, saves the output as NIfTI files.
    """
    
    # --- 1. Setup ---
    Ny, Nx, Nc, Nf = k_undersampled.shape
    
    # --- 2. Create k-space mask and Aliasing PSF ---
    k_space_mask = np.zeros(Nx, dtype=np.complex128)
    k_space_mask[acquired_indices] = 1.0
    
    psf_A = scipy.fft.ifftshift(scipy.fft.ifft(scipy.fft.ifftshift(k_space_mask)))

    reconstructed_image_all_frames = np.zeros((Ny, Nx, Nf), dtype=np.complex128)
    
    # --- 3. Loop over Frames and Slices ---
    for t in range(Nf):
        print(f"\nReconstructing frame {t+1}/{Nf}...")
        
        k_frame = k_undersampled[:, :, :, t]
        sens_frame = coil_sensitivities[:, :, :, t]
        
        img_aliased_frame = scipy.fft.ifftshift(
            scipy.fft.ifft2(scipy.fft.ifftshift(k_frame, axes=(0,1)), axes=(0,1)), 
            axes=(0,1)
        )
        
        for y_idx in range(Ny):
            if y_idx % 20 == 0:
                print(f"  ...y-slice {y_idx+1}/{Ny}")
            
            # --- 4. Get Aliased Image Data (I_vec) ---
            I_y_slice_all_coils = img_aliased_frame[y_idx, :, :]
            I_vec = I_y_slice_all_coils.T.flatten()

            # --- 5. Build the Encoding Matrix (E) ---
            S_y_slice = sens_frame[y_idx, :, :]
            E = np.zeros((Nc * Nx, Nx), dtype=np.complex128)
            A_matrix = scipy.linalg.circulant(psf_A)
            
            for c in range(Nc):
                S_c_diag = np.diag(S_y_slice[:, c])
                E_c_block = A_matrix @ S_c_diag
                E[c*Nx:(c+1)*Nx, :] = E_c_block

            # --- 6. Solve the System E * p = I ---
            E_H = E.conj().T
            E_H_E = E_H @ E
            E_H_I = E_H @ I_vec
            
            reg_param = 1e-6 * np.trace(E_H_E) / Nx
            E_H_E_reg = E_H_E + np.eye(Nx, dtype=np.complex128) * reg_param
            
            try:
                p_y_vec = scipy.linalg.solve(E_H_E_reg, E_H_I, assume_a='her')
            except scipy.linalg.LinAlgError:
                print(f"Warning: Singular matrix at (t={t}, y={y_idx}). Using pseudo-inverse.")
                p_y_vec = scipy.linalg.pinv(E_H_E_reg) @ E_H_I

            # --- 7. Store Result ---
            reconstructed_image_all_frames[y_idx, :, t] = p_y_vec
                    
    print("\nReconstruction complete.")

    # --- 8. Save NIfTI files if requested ---
    if save_nifti:
        print("Saving NIfTI files...")
        
        # --- 8a. Save original SENSE reconstruction ---
        save_path_orig = os.path.join(output_dir, "Dynamic_SENSE")
        os.makedirs(save_path_orig, exist_ok=True)
        filename_orig = os.path.join(save_path_orig, f"{subject_id}_{Ny}x{Nx}x{Nf}.nii")
        
        # Get magnitude image
        data_to_save_orig_mag = np.abs(reconstructed_image_all_frames) # Shape (Ny, Nx, Nf)
        
        # Apply orientation correction: rot90(k=-1, axes=(0,1)) + flip(axis=1)
        data_rot = np.rot90(data_to_save_orig_mag, k=-1, axes=(0, 1)) # Shape (Nx, Ny, Nf)
        data_to_save_orig = np.flip(data_rot, axis=1) # Flipped along new axis 1
        
        nifti_img_orig = nib.Nifti1Image(data_to_save_orig, affine=np.eye(4))
        nib.save(nifti_img_orig, filename_orig)
        print(f"Saved original SENSE image to {filename_orig}")

        # --- 8b. Save zero-padded SENSE reconstruction ---
        pad_Ny, pad_Nx = 312, 410
        save_path_pad = os.path.join(output_dir, "Dynamic_SENSE_padded")
        os.makedirs(save_path_pad, exist_ok=True)
        filename_pad = os.path.join(save_path_pad, f"{subject_id}_padded_{pad_Ny}x{pad_Nx}x{Nf}.nii")

        # Transform recon image back to k-space
        kspace_recon_full = scipy.fft.fftshift(
            scipy.fft.fft2(scipy.fft.ifftshift(reconstructed_image_all_frames, axes=(0,1)), axes=(0,1)), 
            axes=(0,1)
        )
        
        kspace_padded = np.zeros((pad_Ny, pad_Nx, Nf), dtype=np.complex128)
        
        pad_y_before = (pad_Ny - Ny) // 2
        pad_x_before = (pad_Nx - Nx) // 2
        
        y_start, y_end = pad_y_before, pad_y_before + Ny
        x_start, x_end = pad_x_before, pad_x_before + Nx
        
        kspace_padded[y_start:y_end, x_start:x_end, :] = kspace_recon_full
        
        # Transform padded k-space back to image space
        image_padded = scipy.fft.ifftshift(
            scipy.fft.ifft2(scipy.fft.ifftshift(kspace_padded, axes=(0,1)), axes=(0,1)), 
            axes=(0,1)
        )
        
        # Get magnitude of padded image
        data_to_save_pad_mag = np.abs(image_padded) # Shape (pad_Ny, pad_Nx, Nf)
        
        # Apply orientation correction: rot90(k=-1, axes=(0,1)) + flip(axis=1)
        data_rot_pad = np.rot90(data_to_save_pad_mag, k=-1, axes=(0, 1)) # Shape (pad_Nx, pad_Ny, Nf)
        data_to_save_pad = np.flip(data_rot_pad, axis=1)
        
        nifti_img_pad = nib.Nifti1Image(data_to_save_pad, affine=np.eye(4))
        nib.save(nifti_img_pad, filename_pad)
        print(f"Saved zero-padded SENSE image to {filename_pad}")

    return reconstructed_image_all_frames

def plot_coil_images(kspace, coilmap):
    """
    Plots k-space, aliased image, and sensitivity map for one coil.
    """
    ncoils = kspace.shape[2]
    frame_to_plot = kspace.shape[3] // 2 
    coil_to_plot = ncoils // 2 

    kspace_coil_slice = kspace[:, :, coil_to_plot, frame_to_plot]
    sens_coil = coilmap[:, :, coil_to_plot, frame_to_plot]

    image_coil = scipy.fft.ifftshift(
        scipy.fft.ifft2(scipy.fft.ifftshift(kspace_coil_slice))
    )
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(np.log1p(np.abs(kspace_coil_slice)), cmap='gray', aspect='auto')
    plt.title(f'Coil {coil_to_plot} K-space (log-mag)')
    plt.xlabel('Nx (82)')
    plt.ylabel('Ny (80)')
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(image_coil), cmap='gray', aspect='auto')
    plt.title(f'Coil {coil_to_plot} Aliased Image')
    plt.xlabel('Nx (82)')
    plt.ylabel('Ny (80)')
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(sens_coil), cmap='gray', aspect='auto')
    plt.title(f'Coil {coil_to_plot} Sensitivity Map')
    plt.xlabel('Nx (82)')
    plt.ylabel('Ny (80)')

    plt.tight_layout()
    plt.show()

# --- Main execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SENSE reconstruction with configurable paths')
    parser.add_argument('--kspace', type=str, default='../kspace_mat_US/kspace_Subject0026_vv.mat',
                        help='Path to k-space MAT file')
    parser.add_argument('--coilmap', type=str, default='../sensitivity_maps/sens_Subject0026_Exam17853_80x82x100_nC22.mat',
                        help='Path to coil sensitivity map MAT file')
    parser.add_argument('--output-dir', type=str, default='..',
                        help='Output directory for NIfTI files (default: parent directory ..)')
    parser.add_argument('--subject-id', type=str, default='Subject0026_vv',
                        help='Subject ID for output filenames')
    parser.add_argument('--save-nifti', action='store_true', default=True,
                        help='Save NIfTI outputs (default: True)')
    parser.add_argument('--no-save', dest='save_nifti', action='store_false',
                        help='Disable NIfTI saving')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Display plots after reconstruction')
    parser.add_argument('--smoke-test', action='store_true', default=False,
                        help='Quick test mode: process only first 5 frames and 20 rows')
    args = parser.parse_args()
    
    # 1. Define Acquired K-space Lines
    acquired_indices_list = [
        0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30,  # Left R=3 (11 lines)
        32, 33, 34, 35, 36, 37, 38, 39, 40,      # Center R=1 (18 lines)
        41, 42, 43, 44, 45, 46, 47, 48, 49,
        51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81  # Right R=3 (11 lines)
    ]
    
    # 2. Load Data
    print("Loading data...")
    kspace = loadmat(args.kspace)['kspace'] 
    coilmap = loadmat(args.coilmap)['coilmap'] 

    coilmap = np.transpose(coilmap, (0, 1, 3, 2))
    
    # Smoke test: reduce data size
    if args.smoke_test:
        print("SMOKE TEST MODE: Processing only first 5 frames and 20 y-rows")
        kspace = kspace[:20, :, :, :5]
        coilmap = coilmap[:20, :, :, :5]
    
    print(f"K-space shape (Ny, Nx, Nc, Nf): {kspace.shape}")
    print(f"Coilmap shape (Ny, Nx, Nc, Nf): {coilmap.shape}")
    print(f"Using {len(acquired_indices_list)} acquired k-space lines for Nx (axis 1).")

    # 3. Run SENSE Reconstruction
    reco_image = perform_sense_recon_generalized(
        kspace, 
        coilmap, 
        acquired_indices_list, 
        save_nifti=args.save_nifti,
        output_dir=args.output_dir,
        subject_id=args.subject_id
    )
    
    # 4. Plot Results (optional)
    if args.plot:
        frame_to_plot = reco_image.shape[2] // 2 
        
        aliased_images = scipy.fft.ifftshift(
            scipy.fft.ifft2(scipy.fft.ifftshift(kspace[:, :, :, frame_to_plot], axes=(0,1)), axes=(0,1)), 
            axes=(0,1)
        )
        rss_aliased = np.sqrt(np.sum(np.abs(aliased_images)**2, axis=2))

        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(np.abs(rss_aliased), cmap='gray', aspect='auto')
        plt.title(f'Aliased RSS Image (Frame {frame_to_plot})')
        plt.xlabel('Nx (82)')
        plt.ylabel('Ny (80)')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(np.abs(reco_image[:, :, frame_to_plot]), cmap='gray', aspect='auto')
    plt.title(f'SENSE Reconstructed (Frame {frame_to_plot})')
    plt.xlabel('Nx (82)')
    plt.ylabel('Ny (80)')
    plt.axis('off')

    plt.suptitle("Generalized SENSE Reconstruction Result")
    plt.tight_layout()
    plt.show()
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

def synthetic_undersampling(data_path='Data/Synthetic LR', undersampling_factor=3, save=False):
    """
    Loads images from the specified path, Fourier transforms them,
    undersamples the k-space by removing rows, and plots the results.

    Args:
        data_path (str): Path to the directory containing .nii images.
        undersampling_factor (int): Undersampling factor (e.g., 3 means remove 2 out of 3 rows).
    """

    # Get list of files in the directory
    files = [f for f in os.listdir(data_path) if f.endswith('.nii')]
    files.sort()  # Sort files to ensure consistent order

    for file in files:
        # Load the image
        img = nib.load(os.path.join(data_path, file)).get_fdata()

        # flip img1 horizontally and rotate 90 degrees counter-clockwise
        if file.endswith('.nii'):
            img = np.fliplr(np.rot90(img, k=-1))
        else:
            pass

        # Fourier transform
        kspace = np.fft.fft2(img)
        kspace_shifted = np.fft.fftshift(kspace)

        # Undersample k-space
        rows_to_keep = np.arange(0, kspace_shifted.shape[0], undersampling_factor)
        kspace_undersampled = kspace_shifted[rows_to_keep, :]

        # Create a zero-filled array for the full k-space
        kspace_reconstructed = np.zeros_like(kspace_shifted, dtype=complex)
        kspace_reconstructed[rows_to_keep, :] = kspace_undersampled

        # Inverse Fourier transform to get the undersampled image
        img_undersampled = np.fft.ifft2(np.fft.ifftshift(kspace_reconstructed))
        img_undersampled = np.abs(img_undersampled)  # Take the magnitude

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original Image
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # K-space
        axes[1].imshow(np.log(np.abs(kspace_reconstructed) + 1e-8), cmap='gray')  # Log transform for better visualization
        axes[1].set_title('K-space')
        axes[1].axis('off')

        # Undersampled Image
        axes[2].imshow(img_undersampled, cmap='gray')
        axes[2].set_title('Undersampled Image')
        axes[2].axis('off')

        plt.suptitle(f'File: {file}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
        plt.show()

        # Save the undersampled image if save=True
        if save==True:
            # Create a new NIfTI image with the undersampled data
            img_undersampled_nifti = nib.Nifti1Image(img_undersampled, affine=np.eye(4))
            nib.save(img_undersampled_nifti, os.path.join(data_path, file))

# Example usage:
synthetic_undersampling(data_path='Data/Synthetic LR', undersampling_factor=3, save=False)
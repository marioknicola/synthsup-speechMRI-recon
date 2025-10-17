import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.fft
import nibabel as nib
import os
import glob

def process_kspace_data(file_path, crop_size=40, plot_results=False, save_nifti=True, output_folder_fullres='fullres', output_folder_lowres='lowres'):
    """
    Loads k-space data from a .mat file, crops and zero-pads it, and optionally plots and saves the results.

    Args:
        file_path (str): Path to the .mat file containing the k-space data.
        crop_size (int): Size of the k-space crop.
        plot_results (bool): Whether to plot the original, cropped, and padded k-space and images.
        save_nifti (bool): Whether to save the original and padded images as NIfTI files.
        output_folder_fullres (str): Name of the folder to save full resolution images.
        output_folder_lowres (str): Name of the folder to save low resolution images.

    Returns:
        tuple: sos_image (original), sos_image_padded (cropped and padded)
    """

    # Load the .mat file
    f = loadmat(file_path)

    kspace_data = f['kspace']  # Extract the kspace data

    # initialize array to hold coil images
    coil_images = np.zeros((312,410,22))

    # compute coil images
    for i in range(22):
        coil_images[:,:,i] = scipy.fft.fftshift(np.abs(scipy.fft.ifft2((kspace_data[:,:,i]))), axes=(0,))

    # calculate sum of squares image
    sos_image = np.sqrt(np.sum(coil_images**2, axis=2))

    # Crop k-space from the center
    center_x = np.argmax(np.sum(np.abs(kspace_data), axis=(1, 2))) # note that these display the wrong way around
    center_y = np.argmax(np.sum(np.abs(kspace_data), axis=(0, 2))) 

    kspace_cropped = kspace_data[center_x - crop_size:center_x + crop_size, center_y - crop_size:center_y + crop_size, :]

    # Zero-pad the cropped k-space back to the original size
    kspace_padded = np.zeros_like(kspace_data, dtype=kspace_cropped.dtype)
    padded_center_x = kspace_data.shape[0] // 2
    padded_center_y = kspace_data.shape[1] // 2
    crop_center_x = kspace_cropped.shape[0] // 2
    crop_center_y = kspace_cropped.shape[1] // 2

    kspace_padded[padded_center_x - crop_center_x:padded_center_x + crop_center_x, padded_center_y - crop_center_y:padded_center_y + crop_center_y, :] = kspace_cropped

    # Compute coil images from padded k-space
    coil_images_padded = np.zeros((312,410,22))
    for i in range(22):
        coil_images_padded[:,:,i] = scipy.fft.fftshift(np.abs(scipy.fft.ifft2((kspace_padded[:,:,i]))), axes=(0,))

    # Calculate sum of squares image from padded k-space
    sos_image_padded = np.sqrt(np.sum(coil_images_padded**2, axis=2))

    # Compute coil images from cropped k-space
    coil_images_cropped = np.zeros((crop_size*2,crop_size*2,22))
    for i in range(22):
        coil_images_cropped[:,:,i] = scipy.fft.fftshift(np.abs(scipy.fft.ifft2((kspace_cropped[:,:,i]))), axes=(0,))

    # Calculate sum of squares image from cropped k-space
    sos_image_cropped = np.sqrt(np.sum(coil_images_cropped**2, axis=2))

    if plot_results:
        # Display the original and cropped k-space
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(np.log1p(np.abs(kspace_data[:,:,10])+1e-6), cmap='gray')
        plt.title('Original K-space')

        plt.subplot(1, 3, 2)
        plt.imshow(np.log1p(np.abs(kspace_cropped[:,:,10])+1e-6), cmap='gray')
        plt.title('Cropped K-space')

        plt.subplot(1, 3, 3)
        plt.imshow(np.log1p(np.abs(kspace_padded[:,:,10])+1e-6), cmap='gray')
        plt.title('Cropped and Padded K-space')

        plt.show()

        # Display the original, cropped, and padded images
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(sos_image, cmap='gray')
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(sos_image_padded, cmap='gray')
        plt.title('Cropped & Padded Image')

        plt.show()

    if save_nifti:
        # Create output folders if they don't exist
        os.makedirs(output_folder_fullres, exist_ok=True)
        os.makedirs(output_folder_lowres, exist_ok=True)

        # Rotate and flip images
        sos_image_processed = np.fliplr(np.rot90(sos_image, k=-1))
        sos_image_padded_processed = np.fliplr(np.rot90(sos_image_padded, k=-1))

        # Extract filename without extension
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Save the original and padded images as NIfTI files
        img_original = nib.Nifti1Image(sos_image_processed, np.eye(4))  # Create NIfTI image object
        nib.save(img_original, os.path.join(output_folder_fullres, f'{file_name}.nii'))  # Save as NIfTI file

        img_padded = nib.Nifti1Image(sos_image_padded_processed, np.eye(4))  # Create NIfTI image object
        nib.save(img_padded, os.path.join(output_folder_lowres, f'LR_{file_name}.nii'))  # Save as NIfTI file
    
    return sos_image, sos_image_padded

def process_folder(folder_path, crop_size=40, plot_results=False, save_nifti=True, output_folder_fullres='fullres', output_folder_lowres='lowres'):
    """
    Processes all .mat files in a folder.

    Args:
        folder_path (str): Path to the folder containing .mat files.
        crop_size (int): Size of the k-space crop.
        plot_results (bool): Whether to plot the original, cropped, and padded k-space and images.
        save_nifti (bool): Whether to save the original and padded images as NIfTI files.
        output_folder_fullres (str): Name of the folder to save full resolution images.
        output_folder_lowres (str): Name of the folder to save low resolution images.
    """
    # filepath: /Users/marioknicola/MSc Project/synthsup-speechMRI-recon/dataloading copy.py
    # Find all .mat files in the folder
    mat_files = glob.glob(os.path.join(folder_path, '*.mat'))

    if not mat_files:
        print("No .mat files found in the specified folder.")
        return

    for file_path in mat_files:
        print(f"Processing: {file_path}")
        try:
            process_kspace_data(file_path, crop_size, plot_results, save_nifti, output_folder_fullres, output_folder_lowres)
            print(f"Successfully processed: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == '__main__':
    # Example usage
    folder_path = '/Users/marioknicola/MSc Project/kspace_mat_512x512'  # Replace with your folder path
    output_folder_fullres = 'fullres_images'
    output_folder_lowres = 'lowres_images'
    process_folder(folder_path, crop_size=40, plot_results=True, save_nifti=False, output_folder_fullres=output_folder_fullres, output_folder_lowres=output_folder_lowres)
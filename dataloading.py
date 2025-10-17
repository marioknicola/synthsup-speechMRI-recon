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
    kspace_data_original = np.copy(kspace_data)

    # initialize array to hold coil images
    coil_images = np.zeros((312,410,22))

    # compute coil images
    for i in range(22):
        coil_images[:,:,i] = scipy.fft.fftshift(np.abs(scipy.fft.ifft2((kspace_data[:,:,i]))), axes=(0,))

    # calculate sum of squares image
    sos_image = np.sqrt(np.sum(coil_images**2, axis=2))

    # Calculate signal power
    # Calculate signal power in the original k-space data
    signal_power_original = np.mean(np.abs(kspace_data)**2)

    # Generate complex Gaussian noise with half the SNR
    snr_db = 3  # SNR in dB
    snr = 10 ** (snr_db / 10)  # Linear SNR

    # Reduce SNR by half: SNR_new = SNR/2
    noise_power = signal_power_original / (snr/2)

    # Generate complex Gaussian noise
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*kspace_data.shape) + 1j * np.random.randn(*kspace_data.shape))

    # Add noise to k-space data
    kspace_data_noisy = kspace_data + noise

    # Calculate signal power in the noisy k-space data
    signal_power_noisy = np.mean(np.abs(kspace_data_noisy)**2)

    if plot_results==True:
        # Plot the power spectrum before and after adding noise
        plt.figure(figsize=(12, 6))

        # Power spectrum before adding noise
        plt.subplot(1, 2, 1)
        plt.imshow(np.log1p(np.abs(np.mean(kspace_data, axis=2))), cmap='gray')
        plt.title('Power Spectrum Before Adding Noise')
        plt.colorbar()

        # Power spectrum after adding noise
        plt.subplot(1, 2, 2)
        plt.imshow(np.log1p(np.abs(np.mean(kspace_data_noisy, axis=2))), cmap='gray')
        plt.title('Power Spectrum After Adding Noise')
        plt.colorbar()

        plt.tight_layout()
        plt.show()

    kspace_data = kspace_data_noisy
    # The SNR is halved by doubling the noise power, relative to the signal power.
    # SNR = P_signal / P_noise. If we want SNR' = SNR/2, then
    # SNR' = P_signal / P_noise' = SNR/2
    # P_signal / P_noise' = P_signal / (2*P_noise)
    # Therefore, P_noise' = 2*P_noise.
    signal_power = np.mean(np.abs(kspace_data)**2)

    # Calculate noise power for desired SNR
    snr_db = 3  # SNR in dB
    snr = 10 ** (snr_db / 10)  # Linear SNR
    noise_power = signal_power / snr

    # Generate complex Gaussian noise
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*kspace_data.shape) + 1j * np.random.randn(*kspace_data.shape))

    # Add noise to k-space data
    kspace_data = kspace_data + noise

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

    # initialize array to hold coil images
    coil_images_original = np.zeros((312,410,22))

    # compute coil images
    for i in range(22):
        coil_images_original[:,:,i] = scipy.fft.fftshift(np.abs(scipy.fft.ifft2((kspace_data_original[:,:,i]))), axes=(0,))

    # calculate sum of squares image
    sos_image_original = np.sqrt(np.sum(coil_images_original**2, axis=2))

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
        plt.imshow(sos_image_original, cmap='gray')
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
        sos_image_original_processed = np.fliplr(np.rot90(sos_image_original, k=-1))
        sos_image_padded_processed = np.fliplr(np.rot90(sos_image_padded, k=-1))

        # Extract filename without extension
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Save the original and padded images as NIfTI files
        img_original = nib.Nifti1Image(sos_image_original_processed, np.eye(4))  # Create NIfTI image object
        nib.save(img_original, os.path.join(output_folder_fullres, f'{file_name}.nii'))  # Save as NIfTI file

        img_padded = nib.Nifti1Image(sos_image_padded_processed, np.eye(4))  # Create NIfTI image object
        nib.save(img_padded, os.path.join(output_folder_lowres, f'LR_{file_name}.nii'))  # Save as NIfTI file
    
    return sos_image_original, sos_image_padded

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
    # filepath: /Users/marioknicola/MSc Project/synthsup-speechMRI-recon/dataloading.py
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
    output_folder_fullres = 'HR_nii'
    output_folder_lowres = 'Synth_LR_nii'
    process_folder(folder_path, crop_size=40, plot_results=False, save_nifti=True, output_folder_fullres=output_folder_fullres, output_folder_lowres=output_folder_lowres)

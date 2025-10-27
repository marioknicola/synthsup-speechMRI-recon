import numpy as np

def add_rician_noise(img, mu=0, sigma=0.1):
    """
    Adds Rician noise to an image in the frequency domain.

    Args:
        img (numpy.ndarray): Input image.
        mu (float): Mean of the Rician noise.
        sigma (float): Standard deviation of the Rician noise.

    Returns:
        numpy.ndarray: Image with Rician noise added.
    """
    # Apply Fourier transform
    img_fft = np.fft.fft2(img)
    img_fft_shifted = np.fft.fftshift(img_fft)

    # Add Rician noise in the frequency domain
    real_noise = np.random.normal(mu, sigma, img_fft_shifted.shape)
    imag_noise = np.random.normal(mu, sigma, img_fft_shifted.shape)
    noisy_fft = img_fft_shifted + (real_noise + 1j * imag_noise)

    # Inverse Fourier transform to get back to image domain
    img_noisy = np.abs(np.fft.ifft2(np.fft.ifftshift(noisy_fft)))

    return img_noisy

# Example usage
if __name__ == "__main__":
    import nibabel as nib
    import matplotlib.pyplot as plt

    # Load 2D slice
    img = nib.load("Data/Synthetic LR/LR_000.nii")
    data = img.get_fdata()

    data = np.fliplr(np.rot90(data, k=-1))

    # Define mask: all pixels above 30
    mask = data > 30

    # Add Rician noise
    noisy_data = add_rician_noise(data, mu=0, sigma=0.1)

    # Add noise only where mask is True
    noisy_data_masked = data.copy()
    noisy_data_masked[mask] = noisy_data[mask]

    # Set values lower than 0.01 to 0
    noisy_data_masked[noisy_data_masked < 0.01] = 0

    # Show original and noisy images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(data, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(noisy_data_masked, cmap='gray')
    plt.title('Image with Rician Noise')
    plt.axis('off')
    plt.show()

    # Save back to NIfTI
    noisy_img = nib.Nifti1Image(noisy_data_masked, img.affine, img.header)
    nib.save(noisy_img, "rician_noise_injected.nii")
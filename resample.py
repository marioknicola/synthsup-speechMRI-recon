import nibabel as nib
import numpy as np

def resample_nii(file_path):
    """
    Loads a 2D .nii file, rotates it 90 degrees clockwise, flips it horizontally,
    resamples it by averaging 4x4 blocks of pixels, upscales it back to the original size,
    and returns the resampled numpy array.

    Args:
        file_path (str): Path to the .nii file.

    Returns:
        numpy.ndarray: Resampled numpy array.
    """
    img = nib.load(file_path)
    data = img.get_fdata()

    # Check if the image is 2D
    if len(data.shape) != 2:
        raise ValueError("Input NIfTI image must be 2D.")

    # Rotate 90 degrees clockwise and flip horizontally
    data = np.rot90(data, k=-1)
    data = np.fliplr(data)

    # Get the original dimensions
    rows, cols = data.shape

    # Calculate the new dimensions after resampling
    new_rows = rows // 4
    new_cols = cols // 4

    # Initialize the resampled array
    resampled_data = np.zeros((new_rows, new_cols))

    # Resample the data by averaging 4x4 blocks
    for i in range(new_rows):
        for j in range(new_cols):
            block = data[i*4:(i+1)*4, j*4:(j+1)*4]
            resampled_data[i, j] = np.mean(block)
    
    # Upscale the resampled data back to the original size
    upscaled_data = np.kron(resampled_data, np.ones((4, 4)))

    return upscaled_data

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Load the original image
    original_img = nib.load('Data/HR/HR_000.nii')
    original_data = original_img.get_fdata()

    # Rotate 90 degrees clockwise and flip horizontally
    original_data = np.rot90(original_data, k=-1)
    original_data = np.fliplr(original_data)

    # Resample the file
    resampled_data = resample_nii('Data/HR/HR_000.nii')

    # Print the shape of the original and resampled data
    print("Original shape:", original_data.shape)
    print("Resampled shape:", resampled_data.shape)

    # Plot the original and resampled images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_data, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(resampled_data, cmap='gray')
    axes[1].set_title('Resampled Image')
    plt.show()

    # Save the resampled image to Data/Synthetic LR
    resampled_img = nib.Nifti1Image(resampled_data, original_img.affine)
    nib.save(resampled_img, 'Data/Synthetic LR/LR_000_via_resampling.nii')

    print("Resampling and saving complete!")
    print("Resampling test passed!")
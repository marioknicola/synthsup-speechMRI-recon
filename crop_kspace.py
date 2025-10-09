import os
import numpy as np
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt

# Path to DICOM folder
dicom_dir = "Data"

# Load DICOM files in the directory
dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]

# Read images into list
images = []
for file in dicom_files:
    ds = pydicom.dcmread(file)
    images.append(ds.pixel_array)

images = np.array(images)

if len(images) == 0:
    raise RuntimeError("No DICOM images found!")

# Create output directories if they don't exist
output_dir_lr = os.path.join(dicom_dir, "Synthetic LR")
output_dir_hr = os.path.join(dicom_dir, "HR")
os.makedirs(output_dir_lr, exist_ok=True)
os.makedirs(output_dir_hr, exist_ok=True)

def crop_kspace_center(kspace, crop_shape):
    center = np.array(kspace.shape) // 2
    half_crop0 = crop_shape[0] // 2
    half_crop1 = crop_shape[1] // 2
    return kspace[
        center[0] - half_crop0 : center[0] + half_crop0,
        center[1] - half_crop1 : center[1] + half_crop1
    ]

low_res_images = []
cropped_kspaces = []
zero_padded_kspaces = []
zero_padded_images = []
crop_shape = (128, 102)

for idx, img in enumerate(images):
    img_t = img.T

    # Save original image as NIfTI (HR)
    hr_nifti = nib.Nifti1Image(img_t.astype(np.float32), affine=np.eye(4))
    nib.save(hr_nifti, os.path.join(output_dir_hr, f"HR_{idx:03d}.nii"))

    # Fourier transform
    kspace = np.fft.fftshift(np.fft.fft2(img))
    cropped_kspace = crop_kspace_center(kspace, crop_shape)
    cropped_kspaces.append(cropped_kspace)

    # Inverse FFT on cropped k-space only (no zero-padding)
    low_res_img = np.abs(np.fft.ifft2(np.fft.ifftshift(cropped_kspace)))
    low_res_img_t = low_res_img.T
    low_res_images.append(low_res_img) 

    # Zero-pad cropped k-space back to original size (512x512)
    padded_kspace = np.zeros_like(kspace)
    start0 = (kspace.shape[0] - crop_shape[0]) // 2
    start1 = (kspace.shape[1] - crop_shape[1]) // 2
    padded_kspace[start0:start0+crop_shape[0], start1:start1+crop_shape[1]] = cropped_kspace
    zero_padded_kspaces.append(padded_kspace)

    # Inverse FFT on zero-padded k-space
    zp_img = np.abs(np.fft.ifft2(np.fft.ifftshift(padded_kspace)))
    zero_padded_images.append(zp_img)

# Display images
fig1, axs1 = plt.subplots(1, 2, figsize=(15, 5))

# Original image
axs1[0].imshow(images[0], cmap='gray')
axs1[0].axis('off')

# Zero-padded image (reconstructed)
axs1[1].imshow(zero_padded_images[0], cmap='gray')
axs1[1].axis('off')

plt.tight_layout()
plt.show()

# Display k-spaces: original, centred cropped, zero-padded
fig2, axs2 = plt.subplots(1, 2, figsize=(15, 5))

# Original k-space
kspace_orig = np.fft.fftshift(np.fft.fft2(images[0]))
axs2[0].imshow(np.log1p(np.abs(kspace_orig)), cmap='gray')
axs2[0].axis('off')

# Zero-padded k-space (128x102 in 512x512)
axs2[1].imshow(np.log1p(np.abs(zero_padded_kspaces[0])), cmap='gray')
axs2[1].axis('off')
axs2[1].grid(True)

plt.tight_layout()
plt.show()

# Save low-res image as NIfTI
lr_nifti = nib.Nifti1Image(zero_padded_images[0].T.astype(np.float32), affine=np.eye(4))
nib.save(lr_nifti, os.path.join(output_dir_lr, f"LR_{idx:03d}.nii"))


import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

# Load 2D slice
img = nib.load("Data/Synthetic LR/rician_noise_injected.nii")
data = img.get_fdata()

#data = np.fliplr(np.rot90(data, k=-1))

# Define mask: all pixels above 30
mask = data > 30

# Generate smooth noise field
np.random.seed(0)  # reproducibility
raw_noise = np.random.normal(0, 0.1, size=data.shape)

# Smooth it to make clumpy/patchy noise
smooth_noise = gaussian_filter(raw_noise, sigma=3)  # larger sigma = smoother patches

# Normalise noise
smooth_noise = smooth_noise / np.std(smooth_noise)

# Scale relative to local signal
sigma = 0.1 * np.mean(data[mask])  # adjust multiplier for strength
clumpy_noise = sigma * smooth_noise

# Add noise only where mask is True
noisy_data = data.copy()
noisy_data[mask] += clumpy_noise[mask]

# set values lower than 0.01 to 0
noisy_data[noisy_data < 0.01] = 0

# show original and noisy images
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(data, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(noisy_data, cmap='gray')
plt.title('Image with Clumpy Noise')
plt.axis('off')
plt.show()

# Save back to NIfTI
noisy_img = nib.Nifti1Image(noisy_data, img.affine, img.header)
nib.save(noisy_img, "gaussian_noise_injected.nii")

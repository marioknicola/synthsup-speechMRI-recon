import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.fft

# Load the .mat file
f = loadmat('/Users/marioknicola/MSc Project/kspace_mat_512x512/kspace_Subject0027_oo.mat')

print(f['kspace'].shape)  # Check the shape of the data
kspace_data = f['kspace']  # Extract the kspace data

# initialize array to hold coil images
coil_images = np.zeros((312,410,22))

# compute coil images
for i in range(22):
    coil_images[:,:,i] = scipy.fft.fftshift(np.abs(scipy.fft.ifft2((kspace_data[:,:,i]))), axes=(0,))

# calculate sum of squares image
sos_image = np.sqrt(np.sum(coil_images**2, axis=2))

# plot
plt.figure()
plt.imshow(sos_image, cmap='gray')
plt.show()

# Crop k-space from the center
center_x = 52 # note that these display the wrong way around
center_y = 204
crop_size = 40  # Define the crop size

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
coil_images_cropped = np.zeros((80,80,22))
for i in range(22):
    coil_images_cropped[:,:,i] = scipy.fft.fftshift(np.abs(scipy.fft.ifft2((kspace_cropped[:,:,i]))), axes=(0,))

# Calculate sum of squares image from cropped k-space
sos_image_cropped = np.sqrt(np.sum(coil_images_cropped**2, axis=2))

# Display the original and cropped k-space
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(np.abs(kspace_data[:,:,10])+1e-6, cmap='gray')
plt.title('Original K-space')

plt.subplot(1, 3, 2)
plt.imshow(np.abs(kspace_cropped[:,:,10])+1e-6, cmap='gray')
plt.title('Cropped K-space')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(kspace_padded[:,:,10])+1e-6, cmap='gray')
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
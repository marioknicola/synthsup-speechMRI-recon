import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.fft

# Load the .mat file
f = loadmat('/Users/marioknicola/MSc Project/kspace_mat_512x512/kspace_Subject0026_oo.mat')

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
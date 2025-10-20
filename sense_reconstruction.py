import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.fft 
import nibabel as nib
import scipy.ndimage


kspace = loadmat('kspace_mat_US/kspace_Subject0023_mo.mat')['kspace'] # (80, 82, 22, 100)
coilmap  = loadmat('sensitivity_maps/sens_Subject0023_Exam17698_80x82x100_nC22.mat')['coilmap'] # (80, 82, 100, 22)

# Transpose coilmap to have same dimensions as kspace (x,y,coils,frames)
coilmap = np.transpose(coilmap, (0,1,3,2))  # (80, 82, 22, 100)

def plot_coil_images(kspace, coilmap):
    ncoils = kspace.shape[2]
    for coil in range(ncoils):
        # K-space data for the coil
        kspace_coil = kspace[:, :, coil, :]
        
        # Inverse Fourier transform to get the image
        image_coil = scipy.fft.fftshift(scipy.fft.ifft2(kspace_coil[:,:,50]), axes=(0,))
        
        # Magnitude of the image
        image_coil_abs = np.abs(image_coil)
        
        # Display k-space
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(np.log1p(np.abs(kspace_coil[:, :, 50])), cmap='gray')  # Display middle slice
        plt.title(f'Coil {coil + 1} K-space')
        
        # Display image
        plt.subplot(1, 3, 2)
        plt.imshow(np.abs(image_coil_abs), cmap='gray')  # Display middle slice
        plt.title(f'Coil {coil + 1} Image')
        
        # Display coil sensitivity map
        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(coilmap[:, :, coil, 50]), cmap='gray')
        plt.title(f'Coil {coil + 1} Sensitivity Map')

        plt.tight_layout()
        plt.show()

plot_coil_images(kspace, coilmap)

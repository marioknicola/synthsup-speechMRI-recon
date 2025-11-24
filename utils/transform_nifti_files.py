#!/usr/bin/env python3
"""
Transform NIfTI files: rotate 90 degrees clockwise and flip horizontally.
This applies the standard MRI orientation correction.
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

def apply_mri_transform(img):
    """
    Apply MRI orientation transform:
    1. Rotate 90 degrees clockwise (k=-1 in rot90)
    2. Flip horizontally (axis=1)
    """
    img = np.rot90(img, k=-1, axes=(0, 1))
    img = np.flip(img, axis=1)
    return img

def transform_nifti_file(input_path, output_path):
    """
    Load a NIfTI file, apply transformation, and save.
    
    Args:
        input_path: Path to input NIfTI file
        output_path: Path to save transformed NIfTI file
    """
    # Load the NIfTI file
    nii = nib.load(input_path)
    data = nii.get_fdata()
    
    # Get the number of frames (time dimension)
    if data.ndim == 3:
        num_frames = data.shape[2]
        
        # Transform first frame to get new dimensions
        first_frame = data[:, :, 0]
        first_transformed = apply_mri_transform(first_frame)
        new_shape = (first_transformed.shape[0], first_transformed.shape[1], num_frames)
        
        # Create array with new dimensions
        transformed_data = np.zeros(new_shape, dtype=data.dtype)
        
        # Transform each frame
        for frame_idx in range(num_frames):
            frame = data[:, :, frame_idx]
            transformed_frame = apply_mri_transform(frame)
            transformed_data[:, :, frame_idx] = transformed_frame
    elif data.ndim == 2:
        # Single frame
        transformed_data = apply_mri_transform(data)
    else:
        raise ValueError(f"Unexpected number of dimensions: {data.ndim}")
    
    # Create new NIfTI image with transformed data
    # Keep the same affine and header
    new_nii = nib.Nifti1Image(transformed_data, nii.affine, nii.header)
    
    # Save the transformed file
    nib.save(new_nii, output_path)

def main():
    # Input directory
    input_dir = Path("/Users/marioknicola/MSc Project/recon_dynamic_results")
    
    # Get all NIfTI files
    nifti_files = list(input_dir.glob("*.nii"))
    
    if not nifti_files:
        print(f"No .nii files found in {input_dir}")
        return
    
    print(f"Found {len(nifti_files)} NIfTI files to transform")
    print(f"Input directory: {input_dir}")
    print("Transformation: Rotate 90° clockwise + Flip horizontally")
    print()
    
    # Process each file
    for nifti_file in tqdm(nifti_files, desc="Transforming files"):
        try:
            # Transform in place (overwrite original)
            transform_nifti_file(nifti_file, nifti_file)
        except Exception as e:
            print(f"\nError processing {nifti_file.name}: {e}")
    
    print("\n" + "="*80)
    print("TRANSFORMATION COMPLETE")
    print("="*80)
    print(f"Transformed {len(nifti_files)} files")
    print(f"All files in {input_dir} have been updated")

if __name__ == "__main__":
    main()

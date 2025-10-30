#!/usr/bin/env python3
"""
Crop dynamic MRI volumes using the same ROI as previous comparisons.

Crops the vocal tract region from multi-frame dynamic volumes and saves
the cropped versions to dynamic_cropped folder.
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path


def apply_mri_transform(img):
    """Apply rotation and flip for proper MRI orientation."""
    # Rotate 90 degrees clockwise (k=-1) and flip horizontally
    img_rotated = np.rot90(img, k=-1, axes=(0, 1))
    img_flipped = np.flip(img_rotated, axis=1)
    return img_flipped


def crop_volume(input_path, output_path, crop_params, apply_transform=True):
    """
    Crop a 3D volume (H x W x frames) using specified crop parameters.
    
    Args:
        input_path: Path to input NIfTI file
        output_path: Path to save cropped NIfTI file
        crop_params: Dictionary with crop parameters (x, y, w, h)
        apply_transform: Whether to apply MRI orientation transform before cropping
    
    Returns:
        cropped_data: Cropped numpy array
    """
    print(f"Processing: {os.path.basename(input_path)}")
    
    # Load NIfTI file
    img_nifti = nib.load(input_path)
    img_data = img_nifti.get_fdata()
    
    print(f"  Original shape: {img_data.shape}")
    
    # Handle 2D vs 3D
    if img_data.ndim == 2:
        img_data = img_data[:, :, np.newaxis]
        is_2d = True
    else:
        is_2d = False
    
    H, W, num_frames = img_data.shape
    
    # Extract crop parameters
    x = crop_params['x']
    y = crop_params['y']
    w = crop_params['w']
    h = crop_params['h']
    
    # Apply transformation and crop for each frame
    cropped_volume = np.zeros((h, w, num_frames), dtype=img_data.dtype)
    
    for frame_idx in range(num_frames):
        frame = img_data[:, :, frame_idx]
        
        # Apply MRI transform FIRST (rotate 90 clockwise + flip horizontally)
        if apply_transform:
            frame = apply_mri_transform(frame)
        
        # Then crop the transformed frame
        cropped_frame = frame[y:y+h, x:x+w]
        
        cropped_volume[:, :, frame_idx] = cropped_frame
    
    # Remove extra dimension if input was 2D
    if is_2d:
        cropped_volume = cropped_volume[:, :, 0]
    
    print(f"  Cropped shape before final transform: {cropped_volume.shape}")
    
    # Apply final orientation transform to the entire cropped volume before saving
    if apply_transform:
        if cropped_volume.ndim == 3:
            # Apply to all frames
            final_volume = np.zeros((w, h, num_frames), dtype=cropped_volume.dtype)
            for frame_idx in range(num_frames):
                final_volume[:, :, frame_idx] = apply_mri_transform(cropped_volume[:, :, frame_idx])
            cropped_volume = final_volume
        else:
            # Single 2D image
            cropped_volume = apply_mri_transform(cropped_volume)
    
    print(f"  Final shape after orientation: {cropped_volume.shape}")
    
    # Save cropped volume as NIfTI
    cropped_nifti = nib.Nifti1Image(cropped_volume, affine=img_nifti.affine)
    nib.save(cropped_nifti, output_path)
    print(f"  Saved to: {output_path}")
    
    return cropped_volume


def main():
    # Crop parameters (same as used in visual comparisons)
    crop_params = {
        'x': 40,
        'y': 150,
        'w': 190,
        'h': 130
    }
    
    # Create output directory
    output_dir = Path("/Users/marioknicola/MSc Project/dynamic_cropped")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to crop
    files_to_crop = [
        {
            'input': "/Users/marioknicola/MSc Project/Dynamic_SENSE_padded/Subject0021_speech_padded_312x410x200.nii",
            'output': output_dir / "Subject0021_speech_SENSE_cropped.nii",
            'transform': True
        },
        {
            'input': "/Users/marioknicola/MSc Project/Dynamic_SENSE_padded/unet_speaking.nii",
            'output': output_dir / "Subject0021_speech_unet_cropped.nii",
            'transform': True
        },
        {
            'input': "/Users/marioknicola/MSc Project/HR_nii/kspace_Subject0021_oh.nii",
            'output': output_dir / "Subject0021_oh_HR_cropped.nii",
            'transform': True
        }
    ]
    
    print("="*80)
    print("CROPPING DYNAMIC VOLUMES")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Crop region: x={crop_params['x']}, y={crop_params['y']}, "
          f"w={crop_params['w']}, h={crop_params['h']}")
    print(f"Number of files: {len(files_to_crop)}")
    print("="*80)
    print()
    
    # Process each file
    for file_info in files_to_crop:
        input_path = file_info['input']
        output_path = file_info['output']
        apply_transform = file_info['transform']
        
        if not os.path.exists(input_path):
            print(f"WARNING: File not found: {input_path}")
            print()
            continue
        
        crop_volume(input_path, output_path, crop_params, apply_transform)
        print()
    
    print("="*80)
    print("CROPPING COMPLETE!")
    print("="*80)
    print(f"\nCropped files saved to: {output_dir}")
    
    # List output files with sizes
    print("\nOutput files:")
    for file_path in sorted(output_dir.glob("*.nii")):
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  {file_path.name} ({size_mb:.2f} MB)")
    print()


if __name__ == "__main__":
    main()

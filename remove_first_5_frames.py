#!/usr/bin/env python3
"""
Remove first 5 frames from dynamic SENSE data and sensitivity maps.

The first 5 frames are much brighter due to the dynamic sequence nature.
This script removes them to ensure consistent intensity across frames.

Processes:
1. Dynamic_SENSE_padded/*.nii (100 frames → 95 frames)
2. sensitivity_maps/*_80x82x100_nC22.mat (100 frames → 95 frames)
"""

import os
import numpy as np
import nibabel as nib
import scipy.io as sio
import glob
from tqdm import tqdm
import shutil


def process_dynamic_nii(input_dir, output_dir):
    """Remove first 5 frames from NIfTI files."""
    os.makedirs(output_dir, exist_ok=True)
    
    nii_files = sorted(glob.glob(os.path.join(input_dir, "*.nii")))
    
    print(f"Processing {len(nii_files)} NIfTI files...")
    for nii_file in tqdm(nii_files):
        # Load data
        img = nib.load(nii_file)
        data = img.get_fdata()
        
        if data.ndim != 3:
            print(f"  Skipping {os.path.basename(nii_file)} (not 3D)")
            continue
        
        H, W, num_frames = data.shape
        
        if num_frames != 100:
            print(f"  Skipping {os.path.basename(nii_file)} (expected 100 frames, got {num_frames})")
            continue
        
        # Remove first 5 frames
        data_trimmed = data[:, :, 5:]  # Keep frames 5-99 (95 frames)
        
        # Save
        output_path = os.path.join(output_dir, os.path.basename(nii_file))
        img_trimmed = nib.Nifti1Image(data_trimmed, affine=img.affine)
        nib.save(img_trimmed, output_path)
    
    print(f"Saved {len(nii_files)} files to {output_dir}")


def process_sensitivity_maps(input_dir, output_dir):
    """Remove first 5 frames from sensitivity map MAT files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find sensitivity maps with 100 frames
    mat_files = sorted(glob.glob(os.path.join(input_dir, "*_80x82x100_nC22.mat")))
    
    if len(mat_files) == 0:
        print("No 80x82x100 sensitivity maps found - skipping")
        return
    
    print(f"\nProcessing {len(mat_files)} sensitivity map files...")
    for mat_file in tqdm(mat_files):
        # Load data
        data = sio.loadmat(mat_file)
        
        if 'coilmap' not in data:
            print(f"  Warning: 'coilmap' not found in {os.path.basename(mat_file)}")
            continue
        
        coilmap = data['coilmap']  # (Ny, Nx, Nf, Nc) = (80, 82, 100, 22)
        
        if coilmap.ndim != 4 or coilmap.shape[2] != 100:
            print(f"  Skipping {os.path.basename(mat_file)} (unexpected shape: {coilmap.shape})")
            continue
        
        # Remove first 5 frames
        coilmap_trimmed = coilmap[:, :, 5:, :]  # (80, 82, 95, 22)
        
        # Create output filename with updated shape
        basename = os.path.basename(mat_file)
        new_basename = basename.replace('80x82x100', '80x82x95')
        output_path = os.path.join(output_dir, new_basename)
        
        # Save
        sio.savemat(output_path, {'coilmap': coilmap_trimmed})
    
    print(f"Saved {len(mat_files)} files to {output_dir}")


def copy_other_files(input_dir, output_dir):
    """Copy non-100-frame sensitivity maps unchanged."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find files that don't have 100 frames
    all_files = glob.glob(os.path.join(input_dir, "*.mat"))
    frame100_files = set(glob.glob(os.path.join(input_dir, "*_80x82x100_nC22.mat")))
    other_files = [f for f in all_files if f not in frame100_files]
    
    if other_files:
        print(f"\nCopying {len(other_files)} unchanged files...")
        for file_path in other_files:
            basename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, basename)
            shutil.copy2(file_path, output_path)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Remove first 5 frames from dynamic data and sensitivity maps'
    )
    parser.add_argument('--dynamic-dir', type=str, default='../Dynamic_SENSE_padded',
                        help='Input directory with dynamic NIfTI files')
    parser.add_argument('--sens-dir', type=str, default='../sensitivity_maps',
                        help='Input directory with sensitivity maps')
    parser.add_argument('--dynamic-output', type=str, default='../Dynamic_SENSE_padded_95f',
                        help='Output directory for trimmed dynamic files')
    parser.add_argument('--sens-output', type=str, default='../sensitivity_maps_95f',
                        help='Output directory for trimmed sensitivity maps')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("REMOVE FIRST 5 FRAMES FROM DYNAMIC DATA")
    print("=" * 70)
    print()
    print(f"Dynamic input: {args.dynamic_dir}")
    print(f"Dynamic output: {args.dynamic_output}")
    print(f"Sensitivity input: {args.sens_dir}")
    print(f"Sensitivity output: {args.sens_output}")
    print()
    print("Reason: First 5 frames are much brighter due to sequence dynamics")
    print("Action: Remove frames 0-4, keep frames 5-99 (95 frames total)")
    print("=" * 70)
    print()
    
    # Process dynamic data
    process_dynamic_nii(args.dynamic_dir, args.dynamic_output)
    
    # Process sensitivity maps
    process_sensitivity_maps(args.sens_dir, args.sens_output)
    
    # Copy other sensitivity maps unchanged
    copy_other_files(args.sens_dir, args.sens_output)
    
    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print()
    print(f"Trimmed dynamic data: {args.dynamic_output}")
    print(f"Trimmed sensitivity maps: {args.sens_output}")
    print()
    print("Next steps:")
    print("  1. Check the trimmed data")
    print("  2. Update sense_reconstruction.py to use 95 frames")
    print("  3. Generate SENSE synthetic data")
    print("=" * 70)


if __name__ == '__main__':
    main()

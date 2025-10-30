#!/usr/bin/env python3
"""
Create side-by-side visual comparisons of input vs U-Net output for visual scoring.
Crops a specific region of interest for detailed comparison.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def load_nii(filepath):
    """Load NIfTI file and return data."""
    img = nib.load(filepath)
    return img.get_fdata()

def apply_mri_transform(img):
    """Apply rotation and flip for proper MRI orientation."""
    # Rotate 90 degrees clockwise (k=-1) and flip horizontally
    img_rotated = np.rot90(img, k=-1)
    img_flipped = np.fliplr(img_rotated)
    return img_flipped

def create_comparison(input_path, output_path, hr_path, phoneme, save_path, crop_params):
    """
    Create a side-by-side comparison of input, output, and HR images
    with cropped regions.
    
    Args:
        input_path: Path to input NIfTI file
        output_path: Path to output NIfTI file
        hr_path: Path to HR (ground truth) NIfTI file
        phoneme: Phoneme identifier for the title
        save_path: Path to save the comparison image
        crop_params: Dictionary with crop parameters (x, y, w, h)
    """
    # Load NIfTI files
    input_data = load_nii(input_path)
    output_data = load_nii(output_path)
    hr_data = load_nii(hr_path)
    
    # Apply MRI transform (rotation and flip for correct orientation)
    input_img = apply_mri_transform(input_data)
    output_img = apply_mri_transform(output_data)
    hr_img = apply_mri_transform(hr_data)
    
    # Extract crop parameters
    x = crop_params['x']
    y = crop_params['y']
    w = crop_params['w']
    h = crop_params['h']
    
    # Extract cropped regions
    input_crop = input_img[y:y+h, x:x+w]
    output_crop = output_img[y:y+h, x:x+w]
    hr_crop = hr_img[y:y+h, x:x+w]
    
    # Create figure with three columns (cropped regions)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Display cropped input
    ax1.imshow(input_crop, cmap='gray', interpolation='nearest')
    ax1.set_title('Input (Synthetic LR)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Display cropped output
    ax2.imshow(output_crop, cmap='gray', interpolation='nearest')
    ax2.set_title('Output (U-Net)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Display cropped HR
    ax3.imshow(hr_crop, cmap='gray', interpolation='nearest')
    ax3.set_title('Ground Truth (HR)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Add overall title with phoneme
    fig.suptitle(f'Subject0021 - {phoneme.upper()}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Created: {os.path.basename(save_path)}")

def main():
    # Paths
    input_dir = "/Users/marioknicola/MSc Project/Test_Subject0021"
    output_dir = "/Users/marioknicola/MSc Project/unet_test_subject0021"
    hr_dir = "/Users/marioknicola/MSc Project/HR_nii"
    save_dir = "/Users/marioknicola/MSc Project/visual_scoring"
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Crop parameters
    crop_params = {
        'x': 40,
        'y': 150,
        'w': 190,
        'h': 130
    }
    
    print("="*70)
    print("CREATING VISUAL COMPARISONS FOR SCORING")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"HR directory: {hr_dir}")
    print(f"Save directory: {save_dir}")
    print(f"Crop region: x={crop_params['x']}, y={crop_params['y']}, "
          f"w={crop_params['w']}, h={crop_params['h']}")
    print("="*70)
    
    # Get list of input files
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.nii')])
    
    print(f"\nProcessing {len(input_files)} files...\n")
    
    for input_file in input_files:
        # Construct paths
        input_path = os.path.join(input_dir, input_file)
        
        # Find corresponding output and HR files
        # Input: LR_kspace_Subject0021_*.nii
        # Output: recon_Subject0021_*.nii
        # HR: kspace_Subject0021_*.nii
        phoneme = input_file.replace('LR_kspace_Subject0021_', '').replace('.nii', '')
        output_file = f'recon_Subject0021_{phoneme}.nii'
        hr_file = f'kspace_Subject0021_{phoneme}.nii'
        output_path = os.path.join(output_dir, output_file)
        hr_path = os.path.join(hr_dir, hr_file)
        
        if not os.path.exists(output_path):
            print(f"  Warning: Output file not found for {input_file}")
            continue
        
        if not os.path.exists(hr_path):
            print(f"  Warning: HR file not found for {input_file}")
            continue
        
        # Create save path
        save_filename = f'comparison_Subject0021_{phoneme}.png'
        save_path = os.path.join(save_dir, save_filename)
        
        # Create comparison
        create_comparison(input_path, output_path, hr_path, phoneme, save_path, crop_params)
    
    print("\n" + "="*70)
    print(f"COMPLETE! All comparisons saved to: {save_dir}")
    print("="*70)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create a high-resolution figure showing Input vs Output vs Ground Truth
for 3 examples - perfect for abstract figure.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

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

def create_abstract_figure(examples, crop_params, save_path):
    """
    Create high-resolution figure with 3 rows x 3 columns.
    Each row: Input | Output | Ground Truth
    
    Args:
        examples: List of tuples (input_path, output_path, hr_path, label)
        crop_params: dict with 'x', 'y', 'w', 'h' for crop region
        save_path: Path to save figure
    """
    n_examples = len(examples)
    
    # Create figure - high resolution for publication
    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 3*n_examples))
    
    # Extract crop parameters
    x, y, w, h = crop_params['x'], crop_params['y'], crop_params['w'], crop_params['h']
    
    for row, (input_path, output_path, hr_path, label) in enumerate(examples):
        # Load data
        input_data = load_nii(input_path)
        output_data = load_nii(output_path)
        hr_data = load_nii(hr_path)
        
        # Apply MRI transform for proper orientation
        input_img = apply_mri_transform(input_data)
        output_img = apply_mri_transform(output_data)
        hr_img = apply_mri_transform(hr_data)
        
        # Crop regions
        input_crop = input_img[y:y+h, x:x+w]
        output_crop = output_img[y:y+h, x:x+w]
        hr_crop = hr_img[y:y+h, x:x+w]
        
        # Plot Input
        axes[row, 0].imshow(input_crop, cmap='gray', interpolation='nearest')
        if row == 0:
            axes[row, 0].set_title('Input (Synthetic LR)', fontsize=14, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Add phoneme label to the left of the first column
        axes[row, 0].text(-0.15, 0.5, label, transform=axes[row, 0].transAxes,
                          fontsize=13, fontweight='bold', va='center', ha='right')
        
        # Plot Output
        axes[row, 1].imshow(output_crop, cmap='gray', interpolation='nearest')
        if row == 0:
            axes[row, 1].set_title('U-Net Output', fontsize=14, fontweight='bold')
        axes[row, 1].axis('off')
        
        # Plot Ground Truth
        axes[row, 2].imshow(hr_crop, cmap='gray', interpolation='nearest')
        if row == 0:
            axes[row, 2].set_title('Ground Truth (HR)', fontsize=14, fontweight='bold')
        axes[row, 2].axis('off')
    
    plt.suptitle('Super-Resolution Reconstruction Results', 
                 fontsize=16, fontweight='bold', y=0.99)
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # Reduce spacing between rows and columns
    
    # Save at very high resolution (600 DPI for publication quality)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"\nHigh-resolution figure saved to: {save_path}")
    print(f"Resolution: 600 DPI")
    plt.close()

def main():
    # Paths
    input_dir = "/Users/marioknicola/MSc Project/Test_Subject0021"
    output_dir = "/Users/marioknicola/MSc Project/unet_test_subject0021"
    hr_dir = "/Users/marioknicola/MSc Project/HR_nii"
    save_path = "/Users/marioknicola/MSc Project/visual_scoring/scoring_figure.png"
    
    # Crop parameters
    crop_params = {
        'x': 40,
        'y': 150,
        'w': 190,
        'h': 130
    }
    
    # Select 3 representative examples (choose phonemes with good contrast)
    # Format: (filename_suffix, display_label)
    phoneme_labels = [
        ('ee', 'eee [i:]'),      # Change '/EE/' to your custom label
        ('oh', 'ooh [oʊ:]'),      # Change '/MC/' to your custom label
        ('th', 'th [θ:]')       # Change '/MO/' to your custom label
    ]
    
    examples = []
    for phoneme, label in phoneme_labels:
        input_path = os.path.join(input_dir, f'LR_kspace_Subject0021_{phoneme}.nii')
        output_path = os.path.join(output_dir, f'recon_Subject0021_{phoneme}.nii')
        hr_path = os.path.join(hr_dir, f'kspace_Subject0021_{phoneme}.nii')
        
        # Check all files exist
        if all(os.path.exists(p) for p in [input_path, output_path, hr_path]):
            examples.append((input_path, output_path, hr_path, label))
        else:
            print(f"Warning: Files not found for phoneme {phoneme}")
    
    if len(examples) != 3:
        print(f"Error: Expected 3 examples, found {len(examples)}")
        return
    
    print("="*70)
    print("CREATING HIGH-RESOLUTION ABSTRACT FIGURE")
    print("="*70)
    print(f"Phonemes: {', '.join([e[3] for e in examples])}")
    print(f"Crop region: x={crop_params['x']}, y={crop_params['y']}, "
          f"w={crop_params['w']}, h={crop_params['h']}")
    print(f"Output: {save_path}")
    print("="*70)
    
    create_abstract_figure(examples, crop_params, save_path)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run inference on test subject (Subject 0021) and visualize results.
"""

import os
import argparse
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from unet_model import get_model


def apply_mri_transform(img):
    """Apply MRI orientation transform."""
    img = np.rot90(img, k=-1, axes=(0, 1))
    img = np.flip(img, axis=1)
    return img


def run_inference(model, input_dir, output_dir, device):
    """Run inference on all files in input directory."""
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.nii')])
    
    print(f"Found {len(input_files)} files to process")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in tqdm(input_files, desc="Processing"):
        # Load input
        input_path = os.path.join(input_dir, filename)
        input_nii = nib.load(input_path)
        input_data = input_nii.get_fdata()
        
        # Normalize
        data_min = input_data.min()
        data_max = input_data.max()
        input_normalized = (input_data - data_min) / (data_max - data_min) if data_max > data_min else input_data
        
        # Process each frame
        num_frames = input_data.shape[2] if input_data.ndim == 3 else 1
        output_frames = []
        
        for frame_idx in range(num_frames):
            if input_data.ndim == 3:
                frame = input_normalized[:, :, frame_idx]
            else:
                frame = input_normalized
            
            # Convert to tensor
            frame_tensor = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                output = model(frame_tensor)
            
            # Convert back
            output_frame = output.squeeze().cpu().numpy()
            
            # Denormalize
            output_frame = output_frame * (data_max - data_min) + data_min
            
            output_frames.append(output_frame)
        
        # Stack frames
        if len(output_frames) > 1:
            output_data = np.stack(output_frames, axis=2)
        else:
            output_data = output_frames[0]
        
        # Save
        output_filename = filename.replace('LR_kspace_', 'recon_')
        output_path = os.path.join(output_dir, output_filename)
        output_nii = nib.Nifti1Image(output_data, input_nii.affine, input_nii.header)
        nib.save(output_nii, output_path)
    
    print(f"\nInference complete! Saved to {output_dir}")


def visualize_samples(input_dir, output_dir, viz_dir, num_samples=3):
    """Create visualization comparing input and output."""
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.nii')])
    
    os.makedirs(viz_dir, exist_ok=True)
    
    # Select random samples
    sample_files = np.random.choice(input_files, min(num_samples, len(input_files)), replace=False)
    
    print(f"\nCreating visualizations for {len(sample_files)} samples...")
    
    for filename in sample_files:
        # Load input and output
        input_path = os.path.join(input_dir, filename)
        output_filename = filename.replace('LR_kspace_', 'recon_')
        output_path = os.path.join(output_dir, output_filename)
        
        input_data = nib.load(input_path).get_fdata()
        output_data = nib.load(output_path).get_fdata()
        
        # Get middle frame
        frame_idx = input_data.shape[2] // 2 if input_data.ndim == 3 else 0
        
        if input_data.ndim == 3:
            input_frame = input_data[:, :, frame_idx]
            output_frame = output_data[:, :, frame_idx]
        else:
            input_frame = input_data
            output_frame = output_data
        
        # Apply transform
        input_frame = apply_mri_transform(input_frame)
        output_frame = apply_mri_transform(output_frame)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(input_frame, cmap='gray')
        axes[0].set_title('Input (Undersampled)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(output_frame, cmap='gray')
        axes[1].set_title('U-Net Output', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.suptitle(f'{filename} - Frame {frame_idx}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        viz_filename = filename.replace('.nii', '_comparison.png')
        viz_path = os.path.join(viz_dir, viz_filename)
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {viz_filename}")
    
    print(f"\nVisualizations saved to {viz_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run inference on test subject')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory with test subject input files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save reconstructions')
    parser.add_argument('--viz-dir', type=str, default=None,
                        help='Directory to save visualizations (default: output_dir/visualizations)')
    parser.add_argument('--base-filters', type=int, default=32,
                        help='Number of base filters (must match training)')
    parser.add_argument('--num-viz-samples', type=int, default=5,
                        help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}")
    model = get_model(
        in_channels=1,
        out_channels=1,
        base_filters=args.base_filters,
        bilinear=True
    )
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Run inference
    print(f"\nRunning inference on {args.input_dir}")
    run_inference(model, args.input_dir, args.output_dir, device)
    
    # Visualize
    viz_dir = args.viz_dir if args.viz_dir else os.path.join(args.output_dir, 'visualizations')
    visualize_samples(args.input_dir, args.output_dir, viz_dir, args.num_viz_samples)
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE!")
    print(f"Reconstructions: {args.output_dir}")
    print(f"Visualizations: {viz_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
Inference script for U-Net based MRI reconstruction.

Loads a trained U-Net model and performs reconstruction on test data.
Saves reconstructed images as NIfTI files and computes metrics.

Usage:
    python inference_unet.py --checkpoint outputs/checkpoints/best_model.pth \
                             --input-dir Dynamic_SENSE \
                             --output-dir reconstructions \
                             --compute-metrics --target-dir Dynamic_SENSE_padded
"""

import os
import argparse
from pathlib import Path
import glob

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from unet_model import get_model
from dataset import MRIReconstructionDataset


def compute_psnr(pred, target):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    max_pixel = 1.0  # Assuming normalized images
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_ssim(pred, target, window_size=11):
    """Compute Structural Similarity Index (simple implementation)."""
    from scipy.ndimage import uniform_filter
    
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    mu1 = uniform_filter(pred, window_size)
    mu2 = uniform_filter(target, window_size)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = uniform_filter(pred ** 2, window_size) - mu1_sq
    sigma2_sq = uniform_filter(target ** 2, window_size) - mu2_sq
    sigma12 = uniform_filter(pred * target, window_size) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)


def load_model(checkpoint_path, device, in_channels=1, out_channels=1, base_filters=32):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    model = get_model(in_channels, out_channels, base_filters, bilinear=True)
    model = model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Model loaded from epoch {epoch}")
    
    return model


def reconstruct_single_file(model, input_file, device, output_path=None, 
                            normalize=True, apply_orientation=False):
    """
    Reconstruct a single NIfTI file frame-by-frame.
    
    Args:
        model: Trained U-Net model
        input_file: Path to input NIfTI file
        device: torch device
        output_path: Where to save reconstructed NIfTI
        normalize: Whether to normalize input
        apply_orientation: Apply the same orientation correction as SENSE code
    
    Returns:
        reconstructed_volume: numpy array (H, W, num_frames)
    """
    # Load input
    img_nifti = nib.load(input_file)
    img_data = img_nifti.get_fdata()
    
    # Handle 2D vs 3D
    if img_data.ndim == 2:
        img_data = img_data[:, :, np.newaxis]
    
    H, W, num_frames = img_data.shape
    reconstructed_volume = np.zeros_like(img_data)
    
    print(f"Reconstructing {num_frames} frames from {os.path.basename(input_file)}")
    
    with torch.no_grad():
        for frame_idx in tqdm(range(num_frames), desc="Processing frames"):
            frame = img_data[:, :, frame_idx]
            
            # Normalize
            if normalize:
                frame_min, frame_max = frame.min(), frame.max()
                if frame_max - frame_min > 0:
                    frame = (frame - frame_min) / (frame_max - frame_min)
            
            # Convert to tensor
            frame_tensor = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0)
            frame_tensor = frame_tensor.to(device)
            
            # Reconstruct
            recon_tensor = model(frame_tensor)
            
            # Convert back to numpy
            recon_frame = recon_tensor.cpu().squeeze().numpy()
            
            # Denormalize if needed
            if normalize:
                recon_frame = recon_frame * (frame_max - frame_min) + frame_min
            
            reconstructed_volume[:, :, frame_idx] = recon_frame
    
    # Apply orientation correction (match SENSE code convention)
    if apply_orientation:
        # rot90(k=-1, axes=(0,1)) then flip(axis=1)
        recon_rot = np.rot90(reconstructed_volume, k=-1, axes=(0, 1))
        reconstructed_volume = np.flip(recon_rot, axis=1)
    
    # Save if output path provided
    if output_path:
        recon_nifti = nib.Nifti1Image(reconstructed_volume, affine=img_nifti.affine)
        nib.save(recon_nifti, output_path)
        print(f"Saved reconstruction to {output_path}")
    
    return reconstructed_volume


def visualize_comparison(input_img, recon_img, target_img=None, frame_idx=None, save_path=None):
    """Visualize input, reconstruction, and optionally target with SENSE orientation."""
    if frame_idx is not None:
        input_slice = input_img[:, :, frame_idx] if input_img.ndim == 3 else input_img
        recon_slice = recon_img[:, :, frame_idx] if recon_img.ndim == 3 else recon_img
        target_slice = target_img[:, :, frame_idx] if target_img is not None and target_img.ndim == 3 else target_img
    else:
        input_slice = input_img
        recon_slice = recon_img
        target_slice = target_img
    
    # Apply SENSE orientation convention: rot90(k=-1) + flip(axis=1)
    input_slice = np.flip(np.rot90(input_slice, k=-1), axis=1)
    recon_slice = np.flip(np.rot90(recon_slice, k=-1), axis=1)
    if target_slice is not None:
        target_slice = np.flip(np.rot90(target_slice, k=-1), axis=1)
    
    num_plots = 3 if target_img is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    axes[0].imshow(input_slice, cmap='gray', aspect='auto')
    axes[0].set_title('Input (Undersampled)')
    axes[0].axis('off')
    
    axes[1].imshow(recon_slice, cmap='gray', aspect='auto')
    axes[1].set_title('Reconstruction (U-Net)')
    axes[1].axis('off')
    
    if target_img is not None:
        axes[2].imshow(target_slice, cmap='gray', aspect='auto')
        axes[2].set_title('Target (Ground Truth)')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='U-Net inference for MRI reconstruction')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--in-channels', type=int, default=1,
                        help='Number of input channels')
    parser.add_argument('--out-channels', type=int, default=1,
                        help='Number of output channels')
    parser.add_argument('--base-filters', type=int, default=32,
                        help='Number of filters in first layer (default: 32 for lightweight baseline)')
    
    # Data arguments
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory with input NIfTI files')
    parser.add_argument('--target-dir', type=str, default=None,
                        help='Directory with target files (for metrics computation)')
    parser.add_argument('--output-dir', type=str, default='../reconstructions',
                        help='Output directory for reconstructed NIfTI files (default: ../reconstructions)')
    parser.add_argument('--file-pattern', type=str, default='*.nii*',
                        help='Glob pattern for input files')
    parser.add_argument('--test-indices', type=str, default=None,
                        help='Path to test_indices.txt file (to test only on held-out test set)')
    
    # Processing arguments
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize inputs')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                        help='Disable normalization')
    parser.add_argument('--apply-orientation', action='store_true', default=False,
                        help='Apply orientation correction (rot90 + flip)')
    parser.add_argument('--compute-metrics', action='store_true', default=False,
                        help='Compute PSNR/SSIM metrics (requires --target-dir)')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Generate visualization plots')
    parser.add_argument('--num-vis', type=int, default=5,
                        help='Number of frames to visualize per file')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.visualize:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
    
    # Load model
    model = load_model(
        args.checkpoint, 
        device, 
        args.in_channels, 
        args.out_channels, 
        args.base_filters
    )
    
    # Find input files
    input_files = sorted(glob.glob(str(Path(args.input_dir) / args.file_pattern)))
    
    if len(input_files) == 0:
        print(f"No files found matching {args.input_dir}/{args.file_pattern}")
        return
    
    # Filter by test indices if provided
    if args.test_indices:
        print(f"Loading test indices from {args.test_indices}")
        with open(args.test_indices, 'r') as f:
            test_indices = []
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    try:
                        test_indices.append(int(line))
                    except ValueError:
                        # Skip lines that aren't integers
                        continue
        
        # Filter input files to only test set
        if len(test_indices) > 0:
            filtered_files = [input_files[i] for i in test_indices if i < len(input_files)]
            print(f"Filtering to {len(filtered_files)} test set files (from {len(input_files)} total)")
            print(f"Test set indices: {sorted(test_indices)}")
            input_files = filtered_files
        else:
            print("Warning: No valid test indices found")
    
    print(f"Found {len(input_files)} files to reconstruct\n")
    
    # Process each file
    all_metrics = []
    
    for input_file in input_files:
        filename = os.path.basename(input_file)
        output_path = output_dir / f"recon_{filename}"
        
        print(f"\n{'='*80}")
        print(f"Processing: {filename}")
        print(f"{'='*80}")
        
        # Reconstruct
        recon_volume = reconstruct_single_file(
            model, 
            input_file, 
            device, 
            output_path=output_path,
            normalize=args.normalize,
            apply_orientation=args.apply_orientation
        )
        
        # Compute metrics if target provided
        if args.compute_metrics and args.target_dir:
            # Handle filename mapping: LR_kspace_* -> kspace_*
            target_filename = filename.replace('LR_', '') if filename.startswith('LR_') else filename
            target_file = os.path.join(args.target_dir, target_filename)
            
            if os.path.exists(target_file):
                target_data = nib.load(target_file).get_fdata()
                
                # Handle 2D vs 3D data
                if len(target_data.shape) == 2:
                    target_data = target_data[:, :, np.newaxis]
                if len(recon_volume.shape) == 2:
                    recon_volume = recon_volume[:, :, np.newaxis]
                
                # Check if shapes match, transpose target if needed
                if target_data.shape[:2] != recon_volume.shape[:2]:
                    if target_data.shape[:2] == recon_volume.shape[:2][::-1]:
                        # Transpose spatial dimensions
                        target_data = np.transpose(target_data, (1, 0, 2))
                        print(f"  Transposed target to match reconstruction shape: {target_data.shape}")
                    else:
                        print(f"  Warning: Shape mismatch! Recon: {recon_volume.shape}, Target: {target_data.shape}")
                
                # Compute frame-wise metrics
                num_frames = min(recon_volume.shape[2], target_data.shape[2])
                psnr_values = []
                ssim_values = []
                
                for frame_idx in range(num_frames):
                    recon_frame = recon_volume[:, :, frame_idx]
                    target_frame = target_data[:, :, frame_idx]
                    
                    # Normalize both to [0, 1] for fair comparison
                    recon_norm = (recon_frame - recon_frame.min()) / (recon_frame.max() - recon_frame.min() + 1e-8)
                    target_norm = (target_frame - target_frame.min()) / (target_frame.max() - target_frame.min() + 1e-8)
                    
                    psnr = compute_psnr(recon_norm, target_norm)
                    ssim = compute_ssim(recon_norm, target_norm)
                    
                    psnr_values.append(psnr)
                    ssim_values.append(ssim)
                
                avg_psnr = np.mean(psnr_values)
                avg_ssim = np.mean(ssim_values)
                
                print(f"\nMetrics for {filename}:")
                print(f"  Avg PSNR: {avg_psnr:.2f} dB")
                print(f"  Avg SSIM: {avg_ssim:.4f}")
                
                all_metrics.append({
                    'filename': filename,
                    'psnr': avg_psnr,
                    'ssim': avg_ssim,
                    'num_frames': num_frames
                })
            else:
                print(f"Warning: Target file not found: {target_file}")
        
        # Visualize
        if args.visualize:
            input_data = nib.load(input_file).get_fdata()
            # Handle filename mapping for target: LR_kspace_* -> kspace_*
            target_filename = filename.replace('LR_', '') if filename.startswith('LR_') else filename
            target_file_vis = os.path.join(args.target_dir, target_filename) if args.target_dir else None
            target_data = nib.load(target_file_vis).get_fdata() if target_file_vis and os.path.exists(target_file_vis) else None
            
            num_frames = recon_volume.shape[2]
            vis_indices = np.linspace(0, num_frames - 1, min(args.num_vis, num_frames), dtype=int)
            
            for vis_idx in vis_indices:
                vis_path = vis_dir / f"{Path(filename).stem}_frame{vis_idx}.png"
                visualize_comparison(
                    input_data, recon_volume, target_data,
                    frame_idx=vis_idx, save_path=vis_path
                )
    
    # Summary
    print(f"\n{'='*80}")
    print("RECONSTRUCTION COMPLETE")
    print(f"{'='*80}")
    print(f"Processed {len(input_files)} files")
    print(f"Outputs saved to: {output_dir}")
    
    if all_metrics:
        print(f"\n{'='*80}")
        print("OVERALL METRICS")
        print(f"{'='*80}")
        avg_psnr = np.mean([m['psnr'] for m in all_metrics])
        avg_ssim = np.mean([m['ssim'] for m in all_metrics])
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare U-Net, Fast-SRGAN, and SRCNN on test subject using boxplots.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import json


def compute_metrics(pred, target):
    """Compute PSNR, SSIM, MSE, and NMSE."""
    data_range = max(pred.max(), target.max()) - min(pred.min(), target.min())
    
    psnr_val = psnr(target, pred, data_range=data_range)
    ssim_val = ssim(target, pred, data_range=data_range)
    mse_val = np.mean((pred - target) ** 2)
    
    # Normalized MSE (divide by variance of target)
    target_var = np.var(target)
    nmse_val = mse_val / target_var if target_var > 0 else 0
    
    return psnr_val, ssim_val, mse_val, nmse_val


def resize_to_target(img, target_shape):
    """Resize image to target shape."""
    if img.shape == target_shape:
        return img
    
    if img.ndim == 3:
        scale_factors = (
            target_shape[0] / img.shape[0],
            target_shape[1] / img.shape[1],
            1.0
        )
    else:
        scale_factors = (
            target_shape[0] / img.shape[0],
            target_shape[1] / img.shape[1]
        )
    
    return zoom(img, scale_factors, order=1)


def normalize_to_mri_range(img, target_min, target_max):
    """Normalize image from [img_min, img_max] to [target_min, target_max]."""
    img_min = img.min()
    img_max = img.max()
    
    # Normalize to [0, 1]
    if img_max > img_min:
        normalized = (img - img_min) / (img_max - img_min)
    else:
        normalized = img - img_min
    
    # Scale to target range
    scaled = normalized * (target_max - target_min) + target_min
    
    return scaled


def evaluate_method(method_dir, target_dir, method_name, mri_min=None, mri_max=None, normalize=False):
    """Evaluate a single method against ground truth."""
    print(f"\nEvaluating {method_name}...")
    if normalize and mri_min is not None and mri_max is not None:
        print(f"  Normalizing to MRI range: [{mri_min:.2f}, {mri_max:.2f}]")
    
    # Get file lists
    method_files = sorted([f for f in os.listdir(method_dir) if f.endswith('.nii')])
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.nii')])
    
    print(f"Found {len(method_files)} files")
    
    metrics = {
        'psnr': [],
        'ssim': [],
        'mse': [],
        'nmse': []
    }
    
    for mf in tqdm(method_files, desc=f"{method_name}"):
        # Find corresponding target file
        # Method files: 
        #   - Input: LR_kspace_Subject0021_*.nii
        #   - U-Net: recon_Subject0021_*.nii
        #   - Fast-SRGAN: sr_LR_kspace_Subject0021_*.nii or *_upscaled.nii
        #   - SRCNN: similar patterns
        # Target files: kspace_Subject0021_*.nii
        
        # Extract subject and phoneme
        if 'Subject0021' in mf:
            parts = mf.split('Subject0021_')
            if len(parts) > 1:
                phoneme = parts[1].replace('.nii', '').replace('_upscaled', '')
                target_file = f'kspace_Subject0021_{phoneme}.nii'
        else:
            continue
        
        target_path = os.path.join(target_dir, target_file)
        
        if not os.path.exists(target_path):
            print(f"Warning: Target not found for {mf}")
            continue
        
        # Load files
        method_path = os.path.join(method_dir, mf)
        method_data = nib.load(method_path).get_fdata()
        target_data = nib.load(target_path).get_fdata()
        
        # Resize if needed
        if method_data.shape[:2] != target_data.shape[:2]:
            method_data = resize_to_target(method_data, target_data.shape)
        
        # Normalize to MRI range if requested
        if normalize and mri_min is not None and mri_max is not None:
            method_data = normalize_to_mri_range(method_data, mri_min, mri_max)
        
        # Compute metrics for each frame
        num_frames = method_data.shape[2] if method_data.ndim == 3 else 1
        
        for frame_idx in range(num_frames):
            if method_data.ndim == 3:
                pred_frame = method_data[:, :, frame_idx]
                target_frame = target_data[:, :, frame_idx]
            else:
                pred_frame = method_data
                target_frame = target_data
            
            psnr_val, ssim_val, mse_val, nmse_val = compute_metrics(pred_frame, target_frame)
            
            metrics['psnr'].append(psnr_val)
            metrics['ssim'].append(ssim_val)
            metrics['mse'].append(mse_val)
            metrics['nmse'].append(nmse_val)
    
    # Compute statistics
    stats = {
        'psnr_mean': np.mean(metrics['psnr']),
        'psnr_std': np.std(metrics['psnr']),
        'ssim_mean': np.mean(metrics['ssim']),
        'ssim_std': np.std(metrics['ssim']),
        'mse_mean': np.mean(metrics['mse']),
        'mse_std': np.std(metrics['mse']),
        'nmse_mean': np.mean(metrics['nmse']),
        'nmse_std': np.std(metrics['nmse']),
        'num_frames': len(metrics['psnr'])
    }
    
    print(f"{method_name} Results:")
    print(f"  PSNR: {stats['psnr_mean']:.2f} ± {stats['psnr_std']:.2f} dB")
    print(f"  SSIM: {stats['ssim_mean']:.3f} ± {stats['ssim_std']:.3f}")
    print(f"  MSE:  {stats['mse_mean']:.2f} ± {stats['mse_std']:.2f}")
    print(f"  NMSE: {stats['nmse_mean']:.4f} ± {stats['nmse_std']:.4f}")
    print(f"  Frames: {stats['num_frames']}")
    
    return metrics, stats


def create_boxplots(results, output_path):
    """Create boxplots comparing all methods."""
    methods = list(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Color boxes - add gray for Input
    colors = ['#808080', '#2ecc71', '#e74c3c', '#f39c12']
    
    # PSNR
    psnr_data = [results[method]['metrics']['psnr'] for method in methods]
    psnr_medians = [np.median(data) for data in psnr_data]
    
    bp1 = axes[0].boxplot(psnr_data, tick_labels=methods, patch_artist=True,
                          showmeans=False,
                          medianprops=dict(color='black', linewidth=1.5))
    axes[0].set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    axes[0].set_title('Peak Signal-to-Noise Ratio', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].locator_params(axis='y', nbins=10)
    
    for i, (patch, color) in enumerate(zip(bp1['boxes'], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        # Add median text on left side of box to avoid clipping
        median_y = psnr_medians[i]
        axes[0].text(i+0.75, median_y, f'{psnr_medians[i]:.1f}', 
                     ha='right', va='center', fontsize=10, fontweight='bold')
    
    # SSIM
    ssim_data = [results[method]['metrics']['ssim'] for method in methods]
    ssim_medians = [np.median(data) for data in ssim_data]
    
    bp2 = axes[1].boxplot(ssim_data, tick_labels=methods, patch_artist=True,
                          showmeans=False,
                          medianprops=dict(color='black', linewidth=1.5))
    axes[1].set_ylabel('SSIM', fontsize=12, fontweight='bold')
    axes[1].set_title('Structural Similarity Index', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].locator_params(axis='y', nbins=10)
    
    for i, (patch, color) in enumerate(zip(bp2['boxes'], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        # Add median text next to box
        median_y = ssim_medians[i]
        # Last box: put text on left side to avoid going outside
        if i == len(colors) - 1:
            axes[1].text(i+0.75, median_y, f'{ssim_medians[i]:.3f}', 
                         ha='right', va='center', fontsize=11, fontweight='bold')
        else:
            axes[1].text(i+1.25, median_y, f'{ssim_medians[i]:.3f}', 
                         ha='left', va='center', fontsize=11, fontweight='bold')
    
    # NMSE (normalized MSE)
    nmse_data = [results[method]['metrics']['nmse'] for method in methods]
    nmse_medians = [np.median(data) for data in nmse_data]
    
    bp3 = axes[2].boxplot(nmse_data, tick_labels=methods, patch_artist=True,
                          showmeans=False,
                          medianprops=dict(color='black', linewidth=1.5))
    axes[2].set_ylabel('NMSE', fontsize=12, fontweight='bold')
    axes[2].set_title('Normalised Mean Squared Error', fontsize=13, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3, linestyle='--')
    axes[2].locator_params(axis='y', nbins=10)
    
    for i, (patch, color) in enumerate(zip(bp3['boxes'], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        # Add median text next to box
        median_y = nmse_medians[i]
        # Last box: put text on left side to avoid going outside
        if i == len(colors) - 1:
            axes[2].text(i+0.75, median_y, f'{nmse_medians[i]:.4f}', 
                         ha='right', va='center', fontsize=11, fontweight='bold')
        else:
            axes[2].text(i+1.25, median_y, f'{nmse_medians[i]:.4f}', 
                         ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.suptitle('Test Set Performance Comparison - Subject 0021', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nBoxplots saved to: {output_path}")
    plt.close()


def main():
    # Paths
    input_dir = "/Users/marioknicola/MSc Project/Test_Subject0021"  # LR input images
    unet_dir = "/Users/marioknicola/MSc Project/unet_test_subject0021"
    fastsr_dir = "/Users/marioknicola/MSc Project/fastsr_test_subject0021"
    srcnn_dir = "/Users/marioknicola/MSc Project/srcnn_test_subject0021"
    target_dir = "/Users/marioknicola/MSc Project/HR_nii"
    lr_dir = "/Users/marioknicola/MSc Project/Synth_LR_nii"  # Reference for intensity range
    output_dir = "/Users/marioknicola/MSc Project/test_comparison_results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("COMPARING METHODS ON TEST SET (SUBJECT 0021)")
    print("=" * 80)
    
    # Get MRI intensity range from a reference LR file
    print("\nDetermining MRI intensity range from reference LR file...")
    lr_files = [f for f in os.listdir(lr_dir) if 'Subject0021' in f and f.endswith('.nii')]
    if lr_files:
        ref_file = os.path.join(lr_dir, lr_files[0])
        ref_data = nib.load(ref_file).get_fdata()
        mri_min = float(ref_data.min())
        mri_max = float(ref_data.max())
        print(f"  Reference file: {lr_files[0]}")
        print(f"  MRI intensity range: [{mri_min:.2f}, {mri_max:.2f}]")
    else:
        print("  Warning: No reference LR file found, using default range")
        mri_min, mri_max = 0.0, 10000.0
    
    # Evaluate each method
    results = {}
    
    # Input (LR) - baseline for comparison (no normalization needed)
    input_metrics, input_stats = evaluate_method(
        input_dir, target_dir, "Input (LR)", 
        normalize=False
    )
    results['Input'] = {'metrics': input_metrics, 'stats': input_stats}
    
    # U-Net (no normalization needed - already in correct range)
    unet_metrics, unet_stats = evaluate_method(
        unet_dir, target_dir, "U-Net (Ours)", 
        normalize=False
    )
    results['U-Net'] = {'metrics': unet_metrics, 'stats': unet_stats}
    
    # Fast-SRGAN (normalize from 0-255 range to MRI range)
    fastsr_metrics, fastsr_stats = evaluate_method(
        fastsr_dir, target_dir, "Fast-SRGAN", 
        mri_min=mri_min, mri_max=mri_max, normalize=True
    )
    results['Fast-SRGAN'] = {'metrics': fastsr_metrics, 'stats': fastsr_stats}
    
    # SRCNN (normalize from 0-1 range to MRI range)
    srcnn_metrics, srcnn_stats = evaluate_method(
        srcnn_dir, target_dir, "SRCNN", 
        mri_min=mri_min, mri_max=mri_max, normalize=True
    )
    results['SRCNN'] = {'metrics': srcnn_metrics, 'stats': srcnn_stats}
    
    # Create boxplots
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    boxplot_path = os.path.join(output_dir, 'method_comparison_boxplots.png')
    create_boxplots(results, boxplot_path)
    
    # Save statistics to JSON
    stats_output = {method: results[method]['stats'] for method in results}
    json_path = os.path.join(output_dir, 'comparison_statistics.json')
    with open(json_path, 'w') as f:
        json.dump(stats_output, f, indent=2)
    print(f"Statistics saved to: {json_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Method':<15} {'PSNR (dB)':<20} {'SSIM':<20} {'NMSE':<20}")
    print("-" * 80)
    for method in results:
        stats = results[method]['stats']
        print(f"{method:<15} {stats['psnr_mean']:.2f} ± {stats['psnr_std']:.2f}        "
              f"{stats['ssim_mean']:.3f} ± {stats['ssim_std']:.3f}      "
              f"{stats['nmse_mean']:.4f} ± {stats['nmse_std']:.4f}")
    print("=" * 80)
    
    print(f"\nNote: NMSE (Normalized MSE) = MSE / variance of ground truth")
    print(f"      NMSE is scale-independent and more interpretable for MRI data.")
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()

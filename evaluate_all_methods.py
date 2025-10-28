"""
Comprehensive evaluation script comparing multiple reconstruction methods:
- U-Net (our method)
- Fast-SRGAN (pretrained)
- SRCNN (pretrained)
- Bilinear interpolation (baseline)

Generates boxplots and comparison figures for abstract.
"""

import os
import argparse
from pathlib import Path
import glob
import json

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.ndimage import uniform_filter
import seaborn as sns
import pandas as pd

from unet_model import get_model
from dataset import MRIReconstructionDataset


def apply_mri_transform(img):
    """Apply MRI visualization transform: rotate 90 clockwise + flip horizontally."""
    # Rotate 90 degrees clockwise (k=-1 means 90 degrees clockwise)
    img_rot = np.rot90(img, k=-1)
    # Flip horizontally
    img_flip = np.flip(img_rot, axis=1)
    return img_flip


def crop_roi(img, x_start=30, x_end=230, y_start=130, y_end=280):
    """Crop image to region of interest."""
    return img[y_start:y_end, x_start:x_end]


def compute_psnr(pred, target):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    max_pixel = 1.0  # Normalized images
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_ssim(pred, target, window_size=11):
    """Compute Structural Similarity Index."""
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


def compute_mse(pred, target):
    """Compute Mean Squared Error."""
    return np.mean((pred - target) ** 2)


def compute_mae(pred, target):
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(pred - target))


def load_test_indices(indices_file):
    """Load test indices from file."""
    indices = []
    with open(indices_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                indices.append(int(line))
    return indices


def bilinear_interpolation(img, target_shape):
    """Apply bilinear interpolation to match target shape."""
    if len(img.shape) == 2:
        zoom_factors = [target_shape[0] / img.shape[0], 
                       target_shape[1] / img.shape[1]]
        return zoom(img, zoom_factors, order=1)  # order=1 is bilinear
    elif len(img.shape) == 3:
        zoom_factors = [target_shape[0] / img.shape[0], 
                       target_shape[1] / img.shape[1],
                       1.0]  # Don't zoom time dimension
        return zoom(img, zoom_factors, order=1)
    return img


def load_unet_model(checkpoint_path, device, base_filters=32):
    """Load trained U-Net model."""
    print(f"Loading U-Net model from {checkpoint_path}")
    
    model = get_model(in_channels=1, out_channels=1, 
                     base_filters=base_filters, bilinear=True)
    model = model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def run_unet_inference(model, dataset, test_indices, device):
    """Run U-Net inference on test data."""
    print("\n" + "="*60)
    print("Running U-Net inference...")
    print("="*60)
    
    results = []
    
    test_subset = Subset(dataset, test_indices)
    
    with torch.no_grad():
        for idx in tqdm(test_indices, desc="U-Net"):
            sample = dataset[idx]
            input_img = sample['input'].unsqueeze(0).to(device)
            target_img = sample['target'].cpu().numpy().squeeze()
            
            # Reconstruct
            output = model(input_img)
            pred_img = output.cpu().numpy().squeeze()
            
            # Compute metrics
            psnr = compute_psnr(pred_img, target_img)
            ssim = compute_ssim(pred_img, target_img)
            mse = compute_mse(pred_img, target_img)
            mae = compute_mae(pred_img, target_img)
            
            results.append({
                'method': 'U-Net (Ours)',
                'index': idx,
                'filename': sample.get('filename', f'frame_{idx}'),
                'PSNR': psnr,
                'SSIM': ssim,
                'MSE': mse,
                'MAE': mae,
                'prediction': pred_img,
                'target': target_img,
                'input': sample['input'].cpu().numpy().squeeze()
            })
    
    return results


def evaluate_bilinear(dataset, test_indices):
    """Evaluate bilinear interpolation baseline."""
    print("\n" + "="*60)
    print("Evaluating Bilinear Interpolation...")
    print("="*60)
    
    results = []
    
    for idx in tqdm(test_indices, desc="Bilinear"):
        sample = dataset[idx]
        input_img = sample['input'].cpu().numpy().squeeze()
        target_img = sample['target'].cpu().numpy().squeeze()
        
        # Apply bilinear interpolation (assume input is already correct size)
        # For this dataset, input and target are same size, so bilinear acts as identity
        # But we'll apply it anyway for consistency
        pred_img = input_img.copy()  # No actual upsampling needed
        
        # Compute metrics
        psnr = compute_psnr(pred_img, target_img)
        ssim = compute_ssim(pred_img, target_img)
        mse = compute_mse(pred_img, target_img)
        mae = compute_mae(pred_img, target_img)
        
        results.append({
            'method': 'Bilinear',
            'index': idx,
            'filename': sample.get('filename', f'frame_{idx}'),
            'PSNR': psnr,
            'SSIM': ssim,
            'MSE': mse,
            'MAE': mae,
            'prediction': pred_img,
            'target': target_img,
            'input': input_img
        })
    
    return results


def load_pretrained_results(pretrained_dir, dataset, test_indices, method_name):
    """Load and evaluate pretrained method results."""
    print(f"\n" + "="*60)
    print(f"Evaluating {method_name}...")
    print("="*60)
    
    results = []
    
    # Find corresponding output files
    output_files = sorted(glob.glob(os.path.join(pretrained_dir, "*.nii")))
    
    if not output_files:
        print(f"Warning: No .nii files found in {pretrained_dir}")
        return results
    
    print(f"Found {len(output_files)} output files")
    
    # Create mapping of base filenames to output files
    output_map = {}
    for out_file in output_files:
        basename = os.path.basename(out_file)
        # Extract the key part (e.g., "Subject0026_aa" from various formats)
        if 'Subject' in basename:
            # Extract Subject####_xx pattern
            import re
            match = re.search(r'(Subject\d+_\w+)', basename)
            if match:
                key = match.group(1)
                output_map[key] = out_file
    
    for idx in tqdm(test_indices, desc=method_name):
        sample = dataset[idx]
        target_img = sample['target'].cpu().numpy().squeeze()
        input_img = sample['input'].cpu().numpy().squeeze()
        
        # Get the input filename to match
        input_filename = sample.get('filename', '')
        
        # Extract key from input filename
        pred_img = None
        if 'Subject' in input_filename:
            import re
            match = re.search(r'(Subject\d+_\w+)', input_filename)
            if match:
                key = match.group(1)
                if key in output_map:
                    out_file = output_map[key]
                    try:
                        out_nifti = nib.load(out_file)
                        out_data = out_nifti.get_fdata()
                        
                        # Use first frame if 3D
                        if len(out_data.shape) == 3:
                            pred_img = out_data[:, :, 0]
                        else:
                            pred_img = out_data
                    except Exception as e:
                        print(f"Error loading {out_file}: {e}")
        
        if pred_img is None:
            print(f"Warning: Could not find matching output for index {idx} ({input_filename})")
            # Use input as fallback
            pred_img = input_img
        
        # Resize pred_img to match target if needed
        if pred_img.shape != target_img.shape:
            print(f"Resizing {method_name} output from {pred_img.shape} to {target_img.shape}")
            pred_img = zoom(pred_img, 
                          [target_img.shape[0]/pred_img.shape[0], 
                           target_img.shape[1]/pred_img.shape[1]], 
                          order=1)
        
        # Normalize pred_img to [0, 1] if needed
        if pred_img.max() > 1.0:
            pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min())
        
        # Compute metrics
        psnr = compute_psnr(pred_img, target_img)
        ssim = compute_ssim(pred_img, target_img)
        mse = compute_mse(pred_img, target_img)
        mae = compute_mae(pred_img, target_img)
        
        results.append({
            'method': method_name,
            'index': idx,
            'filename': input_filename,
            'PSNR': psnr,
            'SSIM': ssim,
            'MSE': mse,
            'MAE': mae,
            'prediction': pred_img,
            'target': target_img,
            'input': input_img
        })
    
    return results


def plot_boxplots(all_results, output_dir):
    """Generate scatter plots with mean bars for all metrics."""
    print("\n" + "="*60)
    print("Generating metrics comparison plots...")
    print("="*60)
    
    # Convert to DataFrame
    df_data = []
    for result in all_results:
        df_data.append({
            'Method': result['method'],
            'PSNR': result['PSNR'],
            'SSIM': result['SSIM'],
            'MSE': result['MSE'],
            'MAE': result['MAE']
        })
    
    df = pd.DataFrame(df_data)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Reconstruction Quality Comparison on Unseen Test Data (n=3)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    metrics = ['PSNR', 'SSIM', 'MSE', 'MAE']
    metric_labels = ['PSNR (dB) ↑', 'SSIM ↑', 'MSE ↓', 'MAE ↓']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        methods = df['Method'].unique()
        x_pos = np.arange(len(methods))
        
        # Plot mean bars
        means = [df[df['Method'] == method][metric].mean() for method in methods]
        stds = [df[df['Method'] == method][metric].std() for method in methods]
        
        bars = ax.bar(x_pos, means, color=colors[idx], alpha=0.6, 
                     edgecolor='black', linewidth=2)
        
        # Add error bars (std)
        ax.errorbar(x_pos, means, yerr=stds, fmt='none', 
                   ecolor='black', capsize=5, linewidth=2)
        
        # Add individual points as X markers
        for i, method in enumerate(methods):
            method_data = df[df['Method'] == method][metric].values
            x_scatter = np.full(len(method_data), i)
            ax.scatter(x_scatter, method_data, marker='x', s=200, 
                      color='black', linewidths=3, zorder=10)
        
        ax.set_title(label, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('')
        ax.set_ylabel(label, fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean values as text on bars
        for i, (mean_val, std_val) in enumerate(zip(means, stds)):
            ax.text(i, mean_val + std_val + ax.get_ylim()[1] * 0.02, 
                   f'{mean_val:.3f}±{std_val:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics comparison saved to: {output_file}")
    
    # High-res version
    output_file_hires = os.path.join(output_dir, 'metrics_comparison_hires.png')
    plt.savefig(output_file_hires, dpi=600, bbox_inches='tight')
    print(f"✓ High-res metrics comparison saved to: {output_file_hires}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    summary = df.groupby('Method')[metrics].agg(['mean', 'std', 'min', 'max'])
    print(summary.to_string())
    
    return df


def plot_visual_comparison(all_results, output_dir, num_samples=3):
    """Create visual comparison of reconstruction results."""
    print("\n" + "="*60)
    print("Generating visual comparisons...")
    print("="*60)
    
    # Group by index
    by_index = {}
    for result in all_results:
        idx = result['index']
        if idx not in by_index:
            by_index[idx] = []
        by_index[idx].append(result)
    
    # Select samples to visualize
    indices_to_plot = sorted(by_index.keys())[:num_samples]
    
    for sample_idx in indices_to_plot:
        methods_results = by_index[sample_idx]
        
        # Sort by method name for consistent ordering
        methods_results = sorted(methods_results, key=lambda x: x['method'])
        
        num_methods = len(methods_results)
        fig, axes = plt.subplots(2, num_methods + 1, figsize=(4*(num_methods+1), 8))
        
        # Get input and target (same for all methods)
        input_img = methods_results[0]['input']
        target_img = methods_results[0]['target']
        
        # First column: Input and Target
        im0 = axes[0, 0].imshow(input_img, cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('Input\n(Synthetic LR)', fontsize=11, fontweight='bold')
        axes[0, 0].axis('off')
        
        im1 = axes[1, 0].imshow(target_img, cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title('Ground Truth\n(HR)', fontsize=11, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Remaining columns: Each method
        for col_idx, result in enumerate(methods_results, start=1):
            pred_img = result['prediction']
            method = result['method']
            
            # Top row: Reconstruction
            im = axes[0, col_idx].imshow(pred_img, cmap='gray', vmin=0, vmax=1)
            axes[0, col_idx].set_title(f"{method}\nPSNR: {result['PSNR']:.2f} dB", 
                                      fontsize=11, fontweight='bold')
            axes[0, col_idx].axis('off')
            
            # Bottom row: Error map
            error_map = np.abs(pred_img - target_img)
            im_err = axes[1, col_idx].imshow(error_map, cmap='hot', vmin=0, vmax=0.1)
            axes[1, col_idx].set_title(f"Error Map\nSSIM: {result['SSIM']:.4f}", 
                                      fontsize=11, fontweight='bold')
            axes[1, col_idx].axis('off')
        
        # Add colorbars
        fig.colorbar(im0, ax=axes[0, :], orientation='horizontal', 
                    pad=0.02, fraction=0.046, label='Intensity')
        fig.colorbar(im_err, ax=axes[1, :], orientation='horizontal', 
                    pad=0.02, fraction=0.046, label='Absolute Error')
        
        plt.suptitle(f'Reconstruction Comparison - Test Sample {sample_idx}', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save
        output_file = os.path.join(output_dir, f'visual_comparison_sample_{sample_idx}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Visual comparison saved: {output_file}")
        
        plt.close()


def plot_side_by_side_outputs(all_results, output_dir):
    """Create clean side-by-side output comparison."""
    print("\n" + "="*60)
    print("Generating side-by-side output comparisons...")
    print("="*60)
    
    # Group by index
    by_index = {}
    for result in all_results:
        idx = result['index']
        if idx not in by_index:
            by_index[idx] = []
        by_index[idx].append(result)
    
    # Get all test indices
    indices_to_plot = sorted(by_index.keys())
    
    # Create one large figure with all samples
    for sample_idx in indices_to_plot:
        methods_results = by_index[sample_idx]
        
        # Sort by method name for consistent ordering: Bilinear, Fast-SRGAN, SRCNN, U-Net
        method_order = ['Bilinear', 'Fast-SRGAN', 'SRCNN', 'U-Net (Ours)']
        methods_results_sorted = []
        for method_name in method_order:
            for result in methods_results:
                if result['method'] == method_name:
                    methods_results_sorted.append(result)
                    break
        
        # Get images
        input_img = methods_results_sorted[0]['input']
        target_img = methods_results_sorted[0]['target']
        
        # Apply MRI transform for visualization
        input_img = apply_mri_transform(input_img)
        target_img = apply_mri_transform(target_img)
        
        # Crop to region of interest
        input_img = crop_roi(input_img)
        target_img = crop_roi(target_img)
        
        # Create figure: Input, Target, then all methods
        num_cols = 2 + len(methods_results_sorted)  # Input + Target + methods
        fig, axes = plt.subplots(1, num_cols, figsize=(4*num_cols, 4.5))
        
        # Column 0: Input
        axes[0].imshow(input_img, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Input\n(Synthetic LR)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Column 1: Target
        axes[1].imshow(target_img, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth\n(Fully Sampled)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Remaining columns: Methods
        for col_idx, result in enumerate(methods_results_sorted, start=2):
            pred_img = result['prediction']
            method = result['method']
            
            # Apply MRI transform for visualization
            pred_img = apply_mri_transform(pred_img)
            
            # Crop to region of interest
            pred_img = crop_roi(pred_img)
            
            axes[col_idx].imshow(pred_img, cmap='gray', vmin=0, vmax=1)
            title = f"{method}\nPSNR: {result['PSNR']:.2f} dB | SSIM: {result['SSIM']:.3f}"
            axes[col_idx].set_title(title, fontsize=12, fontweight='bold')
            axes[col_idx].axis('off')
        
        # Extract filename for title
        filename = methods_results_sorted[0]['filename']
        plt.suptitle(f'Side-by-Side Comparison: {filename}', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save
        output_file = os.path.join(output_dir, f'sidebyside_sample_{sample_idx}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Side-by-side comparison saved: {output_file}")
        
        plt.close()
    
    # Also create a combined figure with all 3 samples in rows
    print("\nCreating combined all-samples figure...")
    fig, axes = plt.subplots(len(indices_to_plot), 6, figsize=(24, 4*len(indices_to_plot)))
    
    if len(indices_to_plot) == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, sample_idx in enumerate(indices_to_plot):
        methods_results = by_index[sample_idx]
        
        # Sort by method order
        methods_results_sorted = []
        for method_name in method_order:
            for result in methods_results:
                if result['method'] == method_name:
                    methods_results_sorted.append(result)
                    break
        
        input_img = methods_results_sorted[0]['input']
        target_img = methods_results_sorted[0]['target']
        filename = methods_results_sorted[0]['filename']
        
        # Apply MRI transform for visualization
        input_img = apply_mri_transform(input_img)
        target_img = apply_mri_transform(target_img)
        
        # Crop to region of interest
        input_img = crop_roi(input_img)
        target_img = crop_roi(target_img)
        
        # Column 0: Input
        axes[row_idx, 0].imshow(input_img, cmap='gray', vmin=0, vmax=1)
        if row_idx == 0:
            axes[row_idx, 0].set_title('Input', fontsize=11, fontweight='bold')
        axes[row_idx, 0].set_ylabel(f'{filename}', fontsize=10, fontweight='bold')
        axes[row_idx, 0].axis('off')
        
        # Column 1: Target
        axes[row_idx, 1].imshow(target_img, cmap='gray', vmin=0, vmax=1)
        if row_idx == 0:
            axes[row_idx, 1].set_title('Ground Truth', fontsize=11, fontweight='bold')
        axes[row_idx, 1].axis('off')
        
        # Columns 2-5: Methods
        for col_idx, result in enumerate(methods_results_sorted, start=2):
            pred_img = result['prediction']
            method = result['method']
            
            # Apply MRI transform for visualization
            pred_img = apply_mri_transform(pred_img)
            
            # Crop to region of interest
            pred_img = crop_roi(pred_img)
            
            axes[row_idx, col_idx].imshow(pred_img, cmap='gray', vmin=0, vmax=1)
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(method, fontsize=11, fontweight='bold')
            
            # Add PSNR text on image
            axes[row_idx, col_idx].text(0.5, 0.05, f"PSNR: {result['PSNR']:.2f}",
                                       transform=axes[row_idx, col_idx].transAxes,
                                       ha='center', va='bottom', fontsize=9,
                                       bbox=dict(boxstyle='round,pad=0.3', 
                                               facecolor='white', alpha=0.8),
                                       fontweight='bold')
            axes[row_idx, col_idx].axis('off')
    
    plt.suptitle('Complete Reconstruction Comparison - All Test Samples', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'all_samples_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ All-samples comparison saved: {output_file}")
    
    output_file_hires = os.path.join(output_dir, 'all_samples_comparison_hires.png')
    plt.savefig(output_file_hires, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"✓ High-res all-samples comparison saved: {output_file_hires}")
    
    plt.close()


def save_results_json(all_results, output_dir):
    """Save all results to JSON file."""
    output_file = os.path.join(output_dir, 'evaluation_results.json')
    
    # Remove numpy arrays from results for JSON serialization
    json_results = []
    for result in all_results:
        json_result = {}
        for k, v in result.items():
            if k not in ['prediction', 'target', 'input']:
                # Convert numpy types to Python types
                if isinstance(v, (np.floating, np.integer)):
                    json_result[k] = float(v)
                else:
                    json_result[k] = v
        json_results.append(json_result)
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate all reconstruction methods')
    
    parser.add_argument('--checkpoint', type=str, 
                       default='../new_outputs/best_model.pth',
                       help='Path to U-Net checkpoint')
    parser.add_argument('--test-indices', type=str,
                       default='../new_outputs/test_indices.txt',
                       help='Path to test indices file')
    parser.add_argument('--input-dir', type=str,
                       default='../Synth_LR_nii',
                       help='Input directory')
    parser.add_argument('--target-dir', type=str,
                       default='../HR_nii',
                       help='Target directory')
    parser.add_argument('--output-dir', type=str,
                       default='../evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--base-filters', type=int, default=32,
                       help='Base filters for U-Net')
    parser.add_argument('--srgan-dir', type=str,
                       default='../pretrained/Fast-SRGAN/output',
                       help='Fast-SRGAN output directory')
    parser.add_argument('--srcnn-dir', type=str,
                       default='../pretrained/SRCNN/image-super-resolution/results',
                       help='SRCNN output directory')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test indices
    test_indices = load_test_indices(args.test_indices)
    print(f"\nTest set: {len(test_indices)} samples")
    print(f"Indices: {test_indices}")
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = MRIReconstructionDataset(
        input_dir=args.input_dir,
        target_dir=args.target_dir,
        normalize=True
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Run evaluations
    all_results = []
    
    # 1. U-Net (our method)
    model = load_unet_model(args.checkpoint, device, args.base_filters)
    unet_results = run_unet_inference(model, dataset, test_indices, device)
    all_results.extend(unet_results)
    
    # 2. Bilinear interpolation baseline
    bilinear_results = evaluate_bilinear(dataset, test_indices)
    all_results.extend(bilinear_results)
    
    # 3. Fast-SRGAN (if available)
    if os.path.exists(args.srgan_dir):
        srgan_results = load_pretrained_results(
            args.srgan_dir, dataset, test_indices, 'Fast-SRGAN'
        )
        all_results.extend(srgan_results)
    else:
        print(f"\nWarning: Fast-SRGAN directory not found: {args.srgan_dir}")
    
    # 4. SRCNN (if available)
    if os.path.exists(args.srcnn_dir):
        srcnn_results = load_pretrained_results(
            args.srcnn_dir, dataset, test_indices, 'SRCNN'
        )
        all_results.extend(srcnn_results)
    else:
        print(f"\nWarning: SRCNN directory not found: {args.srcnn_dir}")
    
    # Generate plots
    df = plot_boxplots(all_results, output_dir)
    plot_visual_comparison(all_results, output_dir, num_samples=min(3, len(test_indices)))
    plot_side_by_side_outputs(all_results, output_dir)
    
    # Save results
    save_results_json(all_results, output_dir)
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

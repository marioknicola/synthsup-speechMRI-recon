#!/usr/bin/env python3
"""
Evaluate all folds from 5-fold sliding window cross-validation.

This script:
1. Loads each fold's best model
2. Runs inference on that fold's test subject
3. Computes PSNR, SSIM, NMSE metrics
4. Generates summary statistics across all folds
5. Identifies best performing fold

Usage:
    python evaluate_cv_sliding_folds.py --models-dir cv_results_sliding \
                                        --input-dir ../Synth_LR_unpadded_nii \
                                        --target-dir ../Dynamic_SENSE_padded \
                                        --output-dir evaluation_results_sliding
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

from dataset import MRIReconstructionDataset
from unet_model import get_model


# Fold test subject mapping
FOLD_TEST_SUBJECTS = {
    1: ['0027'],
    2: ['0021'],
    3: ['0022'],
    4: ['0023'],
    5: ['0024']
}


def compute_psnr(pred, target, data_range=1.0):
    """Compute PSNR between prediction and target."""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    return psnr


def compute_ssim(pred, target, data_range=1.0, window_size=11):
    """Compute SSIM between prediction and target."""
    from scipy.ndimage import uniform_filter
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    mu1 = uniform_filter(pred, size=window_size)
    mu2 = uniform_filter(target, size=window_size)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = uniform_filter(pred ** 2, size=window_size) - mu1_sq
    sigma2_sq = uniform_filter(target ** 2, size=window_size) - mu2_sq
    sigma12 = uniform_filter(pred * target, size=window_size) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)


def compute_nmse(pred, target):
    """Compute Normalized Mean Squared Error."""
    mse = np.mean((pred - target) ** 2)
    target_energy = np.mean(target ** 2)
    if target_energy == 0:
        return 0.0
    nmse = mse / target_energy
    return nmse


def evaluate_fold(fold_num, args, device):
    """Evaluate a single fold on its test subject."""
    print(f"\n{'='*80}")
    print(f"EVALUATING FOLD {fold_num}")
    print(f"{'='*80}")
    
    # Get test subject for this fold
    test_subject = FOLD_TEST_SUBJECTS[fold_num][0]
    print(f"Test subject: {test_subject}")
    
    # Load model checkpoint
    fold_dir = Path(args.models_dir) / f'fold{fold_num}'
    model_path = fold_dir / f'fold{fold_num}_best.pth'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = get_model(
        args.in_channels,
        args.out_channels,
        args.base_filters,
        bilinear=args.bilinear
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded (trained epoch: {checkpoint['epoch']})")
    print(f"Training val loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    
    # Create test dataset
    test_dataset = MRIReconstructionDataset(
        args.input_dir,
        args.target_dir,
        subject_ids=[test_subject],
        normalize=True
    )
    
    print(f"Test dataset: {len(test_dataset)} frames")
    
    # Run inference
    all_psnr = []
    all_ssim = []
    all_nmse = []
    all_mse = []
    per_file_metrics = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc=f"Fold {fold_num}"):
            sample = test_dataset[idx]
            input_tensor = sample['input'].unsqueeze(0).to(device)
            target_tensor = sample['target'].unsqueeze(0).to(device)
            
            # Run model
            output = model(input_tensor)
            
            # Convert to numpy
            input_np = input_tensor.cpu().numpy()[0, 0]
            output_np = output.cpu().numpy()[0, 0]
            target_np = target_tensor.cpu().numpy()[0, 0]
            
            # Compute metrics
            psnr = compute_psnr(output_np, target_np)
            ssim = compute_ssim(output_np, target_np)
            nmse = compute_nmse(output_np, target_np)
            mse = np.mean((output_np - target_np) ** 2)
            
            all_psnr.append(psnr)
            all_ssim.append(ssim)
            all_nmse.append(nmse)
            all_mse.append(mse)
            
            per_file_metrics.append({
                'file_index': idx,
                'input_file': sample.get('input_file', f'frame_{idx}'),
                'psnr': float(psnr),
                'ssim': float(ssim),
                'nmse': float(nmse),
                'mse': float(mse)
            })
    
    # Compute statistics
    results = {
        'fold': fold_num,
        'test_subject': test_subject,
        'num_frames': len(test_dataset),
        'model_epoch': checkpoint['epoch'],
        'training_val_loss': float(checkpoint.get('val_loss', 0)),
        'test_metrics': {
            'psnr': {
                'mean': float(np.mean(all_psnr)),
                'std': float(np.std(all_psnr)),
                'min': float(np.min(all_psnr)),
                'max': float(np.max(all_psnr))
            },
            'ssim': {
                'mean': float(np.mean(all_ssim)),
                'std': float(np.std(all_ssim)),
                'min': float(np.min(all_ssim)),
                'max': float(np.max(all_ssim))
            },
            'nmse': {
                'mean': float(np.mean(all_nmse)),
                'std': float(np.std(all_nmse)),
                'min': float(np.min(all_nmse)),
                'max': float(np.max(all_nmse))
            },
            'mse': {
                'mean': float(np.mean(all_mse)),
                'std': float(np.std(all_mse)),
                'min': float(np.min(all_mse)),
                'max': float(np.max(all_mse))
            }
        },
        'per_file_metrics': per_file_metrics
    }
    
    # Save fold results
    fold_output_dir = Path(args.output_dir) / f'fold{fold_num}'
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = fold_output_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"FOLD {fold_num} RESULTS (Test Subject: {test_subject})")
    print(f"{'='*80}")
    print(f"PSNR: {results['test_metrics']['psnr']['mean']:.2f} ± {results['test_metrics']['psnr']['std']:.2f} dB")
    print(f"SSIM: {results['test_metrics']['ssim']['mean']:.4f} ± {results['test_metrics']['ssim']['std']:.4f}")
    print(f"NMSE: {results['test_metrics']['nmse']['mean']:.4f} ± {results['test_metrics']['nmse']['std']:.4f}")
    print(f"MSE:  {results['test_metrics']['mse']['mean']:.6f} ± {results['test_metrics']['mse']['std']:.6f}")
    print(f"\nResults saved to: {results_path}")
    print(f"{'='*80}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate 5-fold CV results')
    
    # Model arguments
    parser.add_argument('--models-dir', type=str, default='cv_results_sliding',
                        help='Directory containing fold subdirectories with models')
    parser.add_argument('--output-dir', type=str, default='evaluation_results_sliding',
                        help='Directory to save evaluation results')
    
    # Data arguments
    parser.add_argument('--input-dir', type=str, default='../Synth_LR_unpadded_nii',
                        help='Directory with input (LR) images')
    parser.add_argument('--target-dir', type=str, default='../Dynamic_SENSE_padded',
                        help='Directory with target (HR) images')
    
    # Model architecture arguments
    parser.add_argument('--in-channels', type=int, default=1)
    parser.add_argument('--out-channels', type=int, default=1)
    parser.add_argument('--base-filters', type=int, default=32)
    parser.add_argument('--bilinear', action='store_true', default=True)
    
    # Fold selection
    parser.add_argument('--fold', type=int, choices=[1, 2, 3, 4, 5],
                        help='Evaluate a specific fold only')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine which folds to evaluate
    if args.fold:
        folds_to_evaluate = [args.fold]
    else:
        folds_to_evaluate = [1, 2, 3, 4, 5]
    
    print(f"\n{'='*80}")
    print(f"5-FOLD CROSS-VALIDATION EVALUATION")
    print(f"{'='*80}")
    print(f"Models directory: {args.models_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Folds to evaluate: {folds_to_evaluate}")
    print(f"{'='*80}")
    
    # Evaluate each fold
    all_results = []
    for fold_num in folds_to_evaluate:
        results = evaluate_fold(fold_num, args, device)
        if results:
            all_results.append(results)
    
    # Generate overall summary
    if len(all_results) == 5:
        print(f"\n{'='*80}")
        print(f"OVERALL CROSS-VALIDATION SUMMARY")
        print(f"{'='*80}")
        
        # Aggregate metrics across folds
        all_psnr_means = [r['test_metrics']['psnr']['mean'] for r in all_results]
        all_ssim_means = [r['test_metrics']['ssim']['mean'] for r in all_results]
        all_nmse_means = [r['test_metrics']['nmse']['mean'] for r in all_results]
        all_mse_means = [r['test_metrics']['mse']['mean'] for r in all_results]
        
        summary = {
            'total_folds': len(all_results),
            'fold_results': all_results,
            'aggregate_metrics': {
                'psnr': {
                    'mean': float(np.mean(all_psnr_means)),
                    'std': float(np.std(all_psnr_means)),
                    'min': float(np.min(all_psnr_means)),
                    'max': float(np.max(all_psnr_means))
                },
                'ssim': {
                    'mean': float(np.mean(all_ssim_means)),
                    'std': float(np.std(all_ssim_means)),
                    'min': float(np.min(all_ssim_means)),
                    'max': float(np.max(all_ssim_means))
                },
                'nmse': {
                    'mean': float(np.mean(all_nmse_means)),
                    'std': float(np.std(all_nmse_means)),
                    'min': float(np.min(all_nmse_means)),
                    'max': float(np.max(all_nmse_means))
                },
                'mse': {
                    'mean': float(np.mean(all_mse_means)),
                    'std': float(np.std(all_mse_means)),
                    'min': float(np.min(all_mse_means)),
                    'max': float(np.max(all_mse_means))
                }
            }
        }
        
        # Identify best fold
        best_fold_idx = np.argmax(all_psnr_means)
        best_fold = all_results[best_fold_idx]
        summary['best_fold'] = {
            'fold_number': best_fold['fold'],
            'test_subject': best_fold['test_subject'],
            'psnr': best_fold['test_metrics']['psnr']['mean'],
            'ssim': best_fold['test_metrics']['ssim']['mean'],
            'nmse': best_fold['test_metrics']['nmse']['mean']
        }
        
        # Save summary
        summary_path = Path(args.output_dir) / 'cv_evaluation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Print summary
        print(f"\nAggregate Performance (mean ± std across 5 folds):")
        print(f"  PSNR: {summary['aggregate_metrics']['psnr']['mean']:.2f} ± {summary['aggregate_metrics']['psnr']['std']:.2f} dB")
        print(f"  SSIM: {summary['aggregate_metrics']['ssim']['mean']:.4f} ± {summary['aggregate_metrics']['ssim']['std']:.4f}")
        print(f"  NMSE: {summary['aggregate_metrics']['nmse']['mean']:.4f} ± {summary['aggregate_metrics']['nmse']['std']:.4f}")
        print(f"  MSE:  {summary['aggregate_metrics']['mse']['mean']:.6f} ± {summary['aggregate_metrics']['mse']['std']:.6f}")
        
        print(f"\nBest performing fold: Fold {summary['best_fold']['fold_number']} (Test Subject: {summary['best_fold']['test_subject']})")
        print(f"  PSNR: {summary['best_fold']['psnr']:.2f} dB")
        print(f"  SSIM: {summary['best_fold']['ssim']:.4f}")
        print(f"  NMSE: {summary['best_fold']['nmse']:.4f}")
        
        print(f"\nPer-fold breakdown:")
        for r in all_results:
            print(f"  Fold {r['fold']} (Subject {r['test_subject']}): PSNR={r['test_metrics']['psnr']['mean']:.2f} dB, SSIM={r['test_metrics']['ssim']['mean']:.4f}")
        
        print(f"\nSummary saved to: {summary_path}")
        print(f"{'='*80}")
    else:
        print(f"\n⚠️  Only {len(all_results)} fold(s) evaluated. Run all 5 folds for complete summary.")


if __name__ == "__main__":
    main()

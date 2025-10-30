#!/usr/bin/env python3
"""
Batch evaluation of all CV folds on their test subjects.

This script runs inference for all folds and collects metrics to help select the best model.

Usage:
    # Evaluate all folds
    python utils/evaluate_all_folds.py \
        --models-dir ./cv_models_from_colab \
        --input-dir ../Synth_LR_nii \
        --target-dir ../HR_nii \
        --output-dir ./evaluation_results
    
    # Evaluate specific folds
    python utils/evaluate_all_folds.py \
        --models-dir ./cv_models_from_colab \
        --folds fold1 fold3 fold5 \
        --input-dir ../Synth_LR_nii \
        --target-dir ../HR_nii \
        --output-dir ./evaluation_results
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np
from collections import defaultdict


def load_fold_config(fold_dir):
    """Load configuration for a fold."""
    config_file = fold_dir / 'config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    return None


def run_inference_for_fold(fold_name, fold_dir, input_dir, target_dir, output_dir):
    """Run inference for a single fold on its test subject(s)."""
    
    # Load config to get test subjects
    config = load_fold_config(fold_dir)
    if not config:
        print(f"  ❌ No config found for {fold_name}")
        return None
    
    test_subjects = config.get('test_subjects', [])
    if not test_subjects:
        print(f"  ❌ No test subjects in config for {fold_name}")
        return None
    
    # Check if model exists
    model_path = fold_dir / f"{fold_name}_best.pth"
    if not model_path.exists():
        print(f"  ❌ Model not found: {model_path}")
        return None
    
    print(f"\n{'='*70}")
    print(f"FOLD: {fold_name}")
    print(f"{'='*70}")
    print(f"Test subjects: {test_subjects}")
    print(f"Model: {model_path}")
    
    # Create output directory for this fold
    fold_output_dir = output_dir / fold_name / 'test_inference'
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build file pattern for test subjects
    # Handles both single subject and multiple subjects
    file_patterns = [f"*Subject{subj}*.nii" for subj in test_subjects]
    
    # Run inference for each test subject
    all_metrics = []
    for subject in test_subjects:
        print(f"\n  Running inference on Subject {subject}...")
        
        subject_output = fold_output_dir / f"Subject{subject}"
        subject_output.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'python', 'inference_unet.py',
            '--checkpoint', str(model_path),
            '--input-dir', str(input_dir),
            '--target-dir', str(target_dir),
            '--output-dir', str(subject_output),
            '--file-pattern', f'*Subject{subject}*.nii',
            '--compute-metrics'
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  ✅ Inference complete for Subject {subject}")
            
            # Look for metrics file
            metrics_file = subject_output / 'metrics_summary.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    all_metrics.append(metrics)
                    
                    # Print key metrics
                    if 'average' in metrics:
                        avg = metrics['average']
                        print(f"     PSNR: {avg.get('psnr', 'N/A'):.2f} dB")
                        print(f"     SSIM: {avg.get('ssim', 'N/A'):.4f}")
                        print(f"     MSE:  {avg.get('mse', 'N/A'):.6f}")
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Inference failed for Subject {subject}")
            print(f"     Error: {e.stderr}")
            continue
    
    # Compute average metrics across all test subjects for this fold
    if all_metrics:
        avg_fold_metrics = {}
        for key in ['psnr', 'ssim', 'mse']:
            values = [m['average'][key] for m in all_metrics if 'average' in m and key in m['average']]
            if values:
                avg_fold_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
        
        # Save fold summary
        summary_file = fold_output_dir / 'fold_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'fold_name': fold_name,
                'test_subjects': test_subjects,
                'metrics': avg_fold_metrics,
                'individual_results': all_metrics
            }, f, indent=4)
        
        return avg_fold_metrics
    
    return None


def print_summary_table(results):
    """Print a summary table of all fold results."""
    
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Fold':<10} {'Test Subject(s)':<20} {'PSNR (dB)':<15} {'SSIM':<15} {'MSE':<15}")
    print("-"*80)
    
    all_psnr = []
    all_ssim = []
    all_mse = []
    
    for fold_name, metrics in sorted(results.items()):
        if metrics:
            test_subjects = ', '.join(metrics.get('test_subjects', ['N/A']))
            psnr = metrics['metrics'].get('psnr', {})
            ssim = metrics['metrics'].get('ssim', {})
            mse = metrics['metrics'].get('mse', {})
            
            psnr_str = f"{psnr.get('mean', 0):.2f} ± {psnr.get('std', 0):.2f}"
            ssim_str = f"{ssim.get('mean', 0):.4f} ± {ssim.get('std', 0):.4f}"
            mse_str = f"{mse.get('mean', 0):.6f}"
            
            print(f"{fold_name:<10} {test_subjects:<20} {psnr_str:<15} {ssim_str:<15} {mse_str:<15}")
            
            all_psnr.append(psnr.get('mean', 0))
            all_ssim.append(ssim.get('mean', 0))
            all_mse.append(mse.get('mean', 0))
    
    print("-"*80)
    
    if all_psnr:
        print(f"{'OVERALL':<10} {'(across folds)':<20} "
              f"{np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f}"
              f"{'':>3}{np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}"
              f"{'':>3}{np.mean(all_mse):.6f}")
    
    print("="*80)
    
    # Identify best fold
    if all_psnr:
        best_psnr_idx = np.argmax(all_psnr)
        best_ssim_idx = np.argmax(all_ssim)
        best_mse_idx = np.argmin(all_mse)
        
        fold_names = sorted(results.keys())
        
        print("\n" + "="*80)
        print("BEST MODELS")
        print("="*80)
        print(f"Highest PSNR: {fold_names[best_psnr_idx]} ({all_psnr[best_psnr_idx]:.2f} dB)")
        print(f"Highest SSIM: {fold_names[best_ssim_idx]} ({all_ssim[best_ssim_idx]:.4f})")
        print(f"Lowest MSE:   {fold_names[best_mse_idx]} ({all_mse[best_mse_idx]:.6f})")
        
        # Recommend best overall
        # Use PSNR as primary metric
        print(f"\n🎯 RECOMMENDED: {fold_names[best_psnr_idx]} (best PSNR)")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Batch evaluation of cross-validation folds'
    )
    parser.add_argument('--models-dir', type=str, required=True,
                        help='Directory containing fold subdirectories with models')
    parser.add_argument('--folds', nargs='+', default=None,
                        help='Specific folds to evaluate (e.g., fold1 fold2). If not specified, evaluates all.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory with input (LR) images')
    parser.add_argument('--target-dir', type=str, required=True,
                        help='Directory with target (HR) images')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    input_dir = Path(args.input_dir)
    target_dir = Path(args.target_dir)
    output_dir = Path(args.output_dir)
    
    # Validate directories
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        sys.exit(1)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)
    
    if not target_dir.exists():
        print(f"❌ Target directory not found: {target_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find folds to evaluate
    if args.folds:
        fold_dirs = [(models_dir / fold, fold) for fold in args.folds]
    else:
        fold_dirs = [(d, d.name) for d in models_dir.iterdir() if d.is_dir()]
    
    fold_dirs = [(d, name) for d, name in fold_dirs if d.exists()]
    
    if not fold_dirs:
        print("❌ No fold directories found")
        sys.exit(1)
    
    print("="*80)
    print("BATCH EVALUATION OF CV FOLDS")
    print("="*80)
    print(f"Models directory: {models_dir}")
    print(f"Input directory:  {input_dir}")
    print(f"Target directory: {target_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Folds to evaluate: {len(fold_dirs)}")
    print("="*80)
    
    # Run inference for each fold
    results = {}
    for fold_dir, fold_name in fold_dirs:
        metrics = run_inference_for_fold(
            fold_name, fold_dir, input_dir, target_dir, output_dir
        )
        
        if metrics:
            config = load_fold_config(fold_dir)
            results[fold_name] = {
                'metrics': metrics,
                'test_subjects': config.get('test_subjects', []) if config else []
            }
    
    # Print summary
    if results:
        print_summary_table(results)
        
        # Save overall summary
        summary_file = output_dir / 'cv_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        print(f"\n✅ Evaluation complete!")
        print(f"   Summary saved to: {summary_file}")
        print(f"   Individual results in: {output_dir}/")
    else:
        print("\n❌ No successful evaluations")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download trained models from Google Drive after Colab training.

This script helps you download the best models from each fold after training in Colab.
The models are saved in Google Drive under: MyDrive/MSc_Project/cross_validation_results/

Usage:
    # Download all fold models
    python utils/download_colab_models.py --output-dir ./cv_models_from_colab
    
    # Download specific folds only
    python utils/download_colab_models.py --folds fold1 fold3 fold5 --output-dir ./cv_models
    
After downloading, you can run inference locally using inference_unet.py
"""

import os
import argparse
from pathlib import Path


def print_instructions(output_dir, folds=None):
    """Print instructions for manual download since we can't directly access Google Drive."""
    
    print("\n" + "="*80)
    print("DOWNLOAD MODELS FROM GOOGLE DRIVE")
    print("="*80)
    print("\nAfter training in Colab, your models are saved in Google Drive at:")
    print("  📂 MyDrive/MSc_Project/cross_validation_results/")
    print("\nEach fold directory contains:")
    print("  • fold_name_best.pth     - Best model checkpoint")
    print("  • config.json            - Training configuration")
    print("  • training_history.json  - Training metrics per epoch")
    
    print("\n" + "="*80)
    print("MANUAL DOWNLOAD STEPS")
    print("="*80)
    print("\n1. Open Google Drive in your browser:")
    print("   https://drive.google.com/")
    
    print("\n2. Navigate to:")
    print("   MyDrive → MSc_Project → cross_validation_results")
    
    if folds:
        print(f"\n3. Download these fold directories:")
        for fold in folds:
            print(f"   • {fold}/")
    else:
        print("\n3. Download all fold directories (fold1, fold2, fold3, etc.)")
    
    print(f"\n4. Extract/move to your local directory:")
    print(f"   {output_dir.resolve()}/")
    
    print("\n" + "="*80)
    print("EXPECTED DIRECTORY STRUCTURE")
    print("="*80)
    print(f"\n{output_dir}/")
    if folds:
        for fold in folds:
            print(f"  {fold}/")
            print(f"    ├── {fold}_best.pth")
            print(f"    ├── config.json")
            print(f"    └── training_history.json")
    else:
        print("  fold1/")
        print("    ├── fold1_best.pth")
        print("    ├── config.json")
        print("    └── training_history.json")
        print("  fold2/")
        print("    ├── fold2_best.pth")
        print("    ├── config.json")
        print("    └── training_history.json")
        print("  ...")
    
    print("\n" + "="*80)
    print("ALTERNATIVE: USE GOOGLE DRIVE DESKTOP APP")
    print("="*80)
    print("\nIf you have Google Drive Desktop app installed:")
    print("1. The files are synced to: ~/Google Drive/My Drive/...")
    print("2. Copy directly from there:")
    print(f"   cp -r ~/Google\\ Drive/My\\ Drive/MSc_Project/cross_validation_results/* {output_dir}/")
    
    print("\n" + "="*80)
    print("NEXT STEPS: RUN INFERENCE LOCALLY")
    print("="*80)
    print("\nOnce models are downloaded, run inference:")
    print("  python inference_unet.py \\")
    print(f"    --checkpoint {output_dir}/fold1/fold1_best.pth \\")
    print("    --input-dir /path/to/Synth_LR_nii \\")
    print("    --target-dir /path/to/HR_nii \\")
    print("    --output-dir ./inference_results/fold1 \\")
    print("    --file-pattern '*Subject0026*.nii' \\")
    print("    --compute-metrics \\")
    print("    --visualize")
    
    print("\nOr use the batch evaluation script:")
    print("  python utils/evaluate_all_folds.py \\")
    print(f"    --models-dir {output_dir} \\")
    print("    --input-dir /path/to/Synth_LR_nii \\")
    print("    --target-dir /path/to/HR_nii \\")
    print("    --output-dir ./evaluation_results")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Instructions for downloading models from Colab/Google Drive'
    )
    parser.add_argument('--output-dir', type=str, default='./cv_models_from_colab',
                        help='Local directory to download models to')
    parser.add_argument('--folds', nargs='+', default=None,
                        help='Specific fold names to download (e.g., fold1 fold2)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_instructions(output_dir, args.folds)
    
    # Check if Google Drive Desktop is available
    possible_paths = [
        Path.home() / "Google Drive" / "My Drive" / "MSc_Project" / "cross_validation_results",
        Path.home() / "GoogleDrive" / "My Drive" / "MSc_Project" / "cross_validation_results",
        "/Volumes/GoogleDrive/My Drive/MSc_Project/cross_validation_results"
    ]
    
    found_path = None
    for path in possible_paths:
        if path.exists():
            found_path = path
            break
    
    if found_path:
        print("\n" + "="*80)
        print("✅ GOOGLE DRIVE FOUND")
        print("="*80)
        print(f"\nFound Google Drive at: {found_path}")
        print("\nAvailable folds:")
        
        try:
            folds_to_copy = args.folds if args.folds else [
                d.name for d in found_path.iterdir() if d.is_dir()
            ]
            
            for fold_name in folds_to_copy:
                fold_path = found_path / fold_name
                if fold_path.exists():
                    print(f"  • {fold_name}/")
                    model_file = fold_path / f"{fold_name}_best.pth"
                    if model_file.exists():
                        size_mb = model_file.stat().st_size / (1024 * 1024)
                        print(f"    └── {model_file.name} ({size_mb:.1f} MB)")
            
            print(f"\nCopy command:")
            print(f"  cp -r {found_path}/* {output_dir}/")
            
            # Ask if user wants to copy now
            try:
                response = input("\nCopy models now? [y/N]: ")
                if response.lower() == 'y':
                    import shutil
                    for fold_name in folds_to_copy:
                        src = found_path / fold_name
                        dst = output_dir / fold_name
                        if src.exists():
                            print(f"  Copying {fold_name}...")
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                    print(f"\n✅ Models copied to: {output_dir}")
            except KeyboardInterrupt:
                print("\n\nCancelled.")
        except Exception as e:
            print(f"Error: {e}")
        
        print("="*80)


if __name__ == "__main__":
    main()

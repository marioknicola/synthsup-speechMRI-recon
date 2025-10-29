#!/usr/bin/env python3
"""
Quick script to verify data is ready for training with subject-based split.
"""

import os
from pathlib import Path
from collections import defaultdict

def check_data():
    # Paths
    input_dir = Path("../Synth_LR_nii")
    target_dir = Path("../HR_nii")
    
    print("=" * 80)
    print("Data Readiness Check for Subject-Based Training")
    print("=" * 80)
    
    # Check directories exist
    if not input_dir.exists():
        print(f"❌ ERROR: Input directory not found: {input_dir}")
        return False
    
    if not target_dir.exists():
        print(f"❌ ERROR: Target directory not found: {target_dir}")
        return False
    
    print(f"✓ Input directory found: {input_dir}")
    print(f"✓ Target directory found: {target_dir}")
    print()
    
    # Count files by subject
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.nii')])
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.nii')])
    
    print(f"Total files: {len(input_files)} input, {len(target_files)} target")
    print()
    
    # Group by subject
    input_by_subject = defaultdict(list)
    target_by_subject = defaultdict(list)
    
    for f in input_files:
        for part in f.split('_'):
            if 'Subject' in part:
                subject = part.replace('Subject', '')
                input_by_subject[subject].append(f)
                break
    
    for f in target_files:
        for part in f.split('_'):
            if 'Subject' in part:
                subject = part.replace('Subject', '')
                target_by_subject[subject].append(f)
                break
    
    # Print split breakdown
    train_subjects = ['0025', '0026', '0027']
    val_subjects = ['0023']
    test_subjects = ['0024']
    
    print("TRAINING SET (Subjects 0025, 0026, 0027):")
    print("-" * 80)
    train_total_input = 0
    train_total_target = 0
    for subject in train_subjects:
        n_input = len(input_by_subject.get(subject, []))
        n_target = len(target_by_subject.get(subject, []))
        train_total_input += n_input
        train_total_target += n_target
        status = "✓" if n_input > 0 and n_target > 0 else "❌"
        print(f"  {status} Subject {subject}: {n_input} input files, {n_target} target files")
    print(f"  Total: {train_total_input} input files, {train_total_target} target files")
    print()
    
    print("VALIDATION SET (Subject 0023):")
    print("-" * 80)
    val_total_input = 0
    val_total_target = 0
    for subject in val_subjects:
        n_input = len(input_by_subject.get(subject, []))
        n_target = len(target_by_subject.get(subject, []))
        val_total_input += n_input
        val_total_target += n_target
        status = "✓" if n_input > 0 and n_target > 0 else "❌"
        print(f"  {status} Subject {subject}: {n_input} input files, {n_target} target files")
    print(f"  Total: {val_total_input} input files, {val_total_target} target files")
    print()
    
    print("TEST SET (Subject 0024):")
    print("-" * 80)
    test_total_input = 0
    test_total_target = 0
    for subject in test_subjects:
        n_input = len(input_by_subject.get(subject, []))
        n_target = len(target_by_subject.get(subject, []))
        test_total_input += n_input
        test_total_target += n_target
        status = "✓" if n_input > 0 and n_target > 0 else "❌"
        print(f"  {status} Subject {subject}: {n_input} input files, {n_target} target files")
    print(f"  Total: {test_total_input} input files, {test_total_target} target files")
    print()
    
    # Check for mismatches
    print("=" * 80)
    print("VERIFICATION:")
    print("-" * 80)
    
    all_ok = True
    
    # Check all subjects have matching files
    all_subjects = set(input_by_subject.keys()) | set(target_by_subject.keys())
    for subject in sorted(all_subjects):
        input_set = set(input_by_subject.get(subject, []))
        target_set = set([f.replace('LR_', '') for f in target_by_subject.get(subject, [])])
        
        if len(input_set) != len(target_set):
            print(f"⚠️  Warning: Subject {subject} has mismatched file counts")
            all_ok = False
    
    if train_total_input == 0 or train_total_target == 0:
        print("❌ ERROR: No training data found!")
        all_ok = False
    else:
        print(f"✓ Training data ready: {train_total_input} files")
    
    if val_total_input == 0 or val_total_target == 0:
        print("❌ ERROR: No validation data found!")
        all_ok = False
    else:
        print(f"✓ Validation data ready: {val_total_input} files")
    
    if test_total_input == 0 or test_total_target == 0:
        print("❌ ERROR: No test data found!")
        all_ok = False
    else:
        print(f"✓ Test data ready: {test_total_input} files")
    
    print()
    
    if all_ok:
        print("=" * 80)
        print("✓ ALL CHECKS PASSED - Ready for training!")
        print("=" * 80)
        print("\nTo start training, run:")
        print("  python train_unet_subject_split.py --input-dir ../Synth_LR_nii --target-dir ../HR_nii --epochs 200")
        print()
    else:
        print("=" * 80)
        print("❌ SOME CHECKS FAILED - Please review issues above")
        print("=" * 80)
    
    return all_ok


if __name__ == "__main__":
    check_data()

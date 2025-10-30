"""
Helper script to generate cross-validation fold configurations.

Provides common CV split strategies for subject-level cross-validation.
"""

import json
from pathlib import Path


def leave_one_out_cv(subjects):
    """
    Generate leave-one-out cross-validation folds.
    Each subject is used as test set once.
    
    Args:
        subjects: List of subject IDs (e.g., ['0023', '0024', '0025', '0026'])
    
    Returns:
        List of fold configurations
    """
    folds = []
    for i, test_subject in enumerate(subjects):
        train_subjects = [s for s in subjects if s != test_subject]
        folds.append({
            'fold_name': f'fold_{test_subject}',
            'train_subjects': train_subjects,
            'val_subjects': None,
            'test_subjects': [test_subject],
            'description': f'Leave-one-out: test on {test_subject}'
        })
    return folds


def k_fold_cv(subjects, k=4):
    """
    Generate k-fold cross-validation splits.
    
    Args:
        subjects: List of subject IDs
        k: Number of folds
    
    Returns:
        List of fold configurations
    """
    import numpy as np
    
    if len(subjects) < k:
        raise ValueError(f"Cannot split {len(subjects)} subjects into {k} folds")
    
    subjects = np.array(subjects)
    fold_size = len(subjects) // k
    
    folds = []
    for i in range(k):
        # Test set for this fold
        test_start = i * fold_size
        test_end = test_start + fold_size if i < k - 1 else len(subjects)
        test_subjects = subjects[test_start:test_end].tolist()
        
        # Training set is all other subjects
        train_mask = np.ones(len(subjects), dtype=bool)
        train_mask[test_start:test_end] = False
        train_subjects = subjects[train_mask].tolist()
        
        folds.append({
            'fold_name': f'fold{i+1}',
            'train_subjects': train_subjects,
            'val_subjects': None,
            'test_subjects': test_subjects,
            'description': f'{k}-fold CV: fold {i+1}/{k}'
        })
    
    return folds


def stratified_cv_with_val(subjects, test_subject, val_subject=None):
    """
    Create a single fold with explicit test and optional validation subject.
    
    Args:
        subjects: List of all subject IDs
        test_subject: Subject ID for testing
        val_subject: Subject ID for validation (optional)
    
    Returns:
        Fold configuration dictionary
    """
    if val_subject:
        train_subjects = [s for s in subjects if s not in [test_subject, val_subject]]
        val_subjects = [val_subject]
    else:
        train_subjects = [s for s in subjects if s != test_subject]
        val_subjects = None
    
    fold = {
        'fold_name': f'test_{test_subject}' + (f'_val_{val_subject}' if val_subject else ''),
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': [test_subject],
        'description': f'Test: {test_subject}' + (f', Val: {val_subject}' if val_subject else '')
    }
    
    return fold


def save_cv_config(folds, output_path='cv_config.json'):
    """Save cross-validation configuration to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(folds, f, indent=4)
    print(f"Saved CV configuration to {output_path}")
    print(f"Total folds: {len(folds)}")


def print_cv_summary(folds):
    """Print summary of cross-validation folds."""
    print("="*80)
    print("CROSS-VALIDATION FOLD SUMMARY")
    print("="*80)
    for fold in folds:
        print(f"\n{fold['fold_name']}: {fold['description']}")
        print(f"  Train: {fold['train_subjects']}")
        if fold['val_subjects']:
            print(f"  Val:   {fold['val_subjects']}")
        print(f"  Test:  {fold['test_subjects']}")
    print("\n" + "="*80)


def generate_training_commands(folds, base_command=None):
    """
    Generate training commands for each fold.
    
    Args:
        folds: List of fold configurations
        base_command: Base command template (optional)
    
    Returns:
        List of training commands
    """
    if base_command is None:
        base_command = "python train_cross_validation.py"
    
    commands = []
    for fold in folds:
        cmd = f"{base_command} \\\n"
        cmd += f"    --train-subjects {' '.join(fold['train_subjects'])} \\\n"
        if fold['val_subjects']:
            cmd += f"    --val-subjects {' '.join(fold['val_subjects'])} \\\n"
        cmd += f"    --test-subjects {' '.join(fold['test_subjects'])} \\\n"
        cmd += f"    --fold-name {fold['fold_name']}"
        commands.append(cmd)
    
    return commands


# Example configurations for common scenarios
def main():
    """Generate example CV configurations."""
    
    # Available subjects (modify based on your dataset)
    all_subjects = ['0023', '0024', '0025', '0026', '0022', '0021']
    
    print("="*80)
    print("CROSS-VALIDATION CONFIGURATION GENERATOR")
    print("="*80)
    print(f"Available subjects: {all_subjects}")
    print()
    
    # Example 1: Leave-one-out CV (no validation set)
    print("\n1. LEAVE-ONE-OUT CV (No Validation Set)")
    print("-" * 80)
    loo_folds = leave_one_out_cv(all_subjects)
    print_cv_summary(loo_folds)
    save_cv_config(loo_folds, 'cv_leave_one_out.json')
    
    print("\nTraining commands:")
    for cmd in generate_training_commands(loo_folds[:2]):  # Show first 2
        print(f"\n{cmd}\n")
    
    # Example 2: 3-fold CV
    print("\n2. 3-FOLD CV")
    print("-" * 80)
    k3_folds = k_fold_cv(all_subjects, k=3)
    print_cv_summary(k3_folds)
    save_cv_config(k3_folds, 'cv_3fold.json')
    
    # Example 3: Custom split with validation
    print("\n3. CUSTOM SPLIT WITH VALIDATION")
    print("-" * 80)
    custom_fold = stratified_cv_with_val(
        subjects=all_subjects,
        test_subject='0021',
        val_subject='0022'
    )
    print_cv_summary([custom_fold])
    
    print("\nTraining command:")
    print(generate_training_commands([custom_fold])[0])
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()

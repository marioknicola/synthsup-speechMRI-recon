"""
5-Fold Cross-Validation with Sliding Window Approach for U-Net MRI Super-Resolution.

Strategy: Train/Val/Test = 5/1/1 subjects per fold using sliding window
For 7 subjects (0021-0027), create 5 folds where each fold uses:
- 5 consecutive subjects for training
- 1 subject for validation (next in sequence)
- 1 subject for testing (next after validation)

Example folds:
Fold 1: Train=[0021,0022,0023,0024,0025], Val=[0026], Test=[0027]
Fold 2: Train=[0022,0023,0024,0025,0026], Val=[0027], Test=[0021]
Fold 3: Train=[0023,0024,0025,0026,0027], Val=[0021], Test=[0022]
Fold 4: Train=[0024,0025,0026,0027,0021], Val=[0022], Test=[0023]
Fold 5: Train=[0025,0026,0027,0021,0022], Val=[0023], Test=[0024]

This ensures every subject appears in train/val/test splits across folds.

Usage:
    # Train all 5 folds sequentially
    python train_cross_validation_sliding.py --all-folds
    
    # Or train a specific fold
    python train_cross_validation_sliding.py --fold 1
"""

import os
import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MRIReconstructionDataset
from unet_model import get_model


# Define sliding window splits for 7 subjects
ALL_SUBJECTS = ['0021', '0022', '0023', '0024', '0025', '0026', '0027']

FOLD_SPLITS = {
    1: {
        'train': ['0021', '0022', '0023', '0024', '0025'],
        'val': ['0026'],
        'test': ['0027']
    },
    2: {
        'train': ['0022', '0023', '0024', '0025', '0026'],
        'val': ['0027'],
        'test': ['0021']
    },
    3: {
        'train': ['0023', '0024', '0025', '0026', '0027'],
        'val': ['0021'],
        'test': ['0022']
    },
    4: {
        'train': ['0024', '0025', '0026', '0027', '0021'],
        'val': ['0022'],
        'test': ['0023']
    },
    5: {
        'train': ['0025', '0026', '0027', '0021', '0022'],
        'val': ['0023'],
        'test': ['0024']
    }
}


class CombinedLoss(nn.Module):
    """
    Combined MSE and SSIM loss for better perceptual quality.
    Loss = alpha * MSE + (1 - alpha) * (1 - SSIM)
    """
    def __init__(self, alpha=0.7, window_size=11):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.window_size = window_size
        
    def gaussian_window(self, window_size, sigma=1.5):
        """Create Gaussian window for SSIM calculation."""
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window(self, window_size, channel=1):
        """Create 2D Gaussian window."""
        _1D_window = self.gaussian_window(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(self, img1, img2, window, window_size, channel=1):
        """Calculate SSIM between two images."""
        mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = torch.nn.functional.conv2d(
            img1 * img1, window, padding=window_size//2, groups=channel
        ) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(
            img2 * img2, window, padding=window_size//2, groups=channel
        ) - mu2_sq
        sigma12 = torch.nn.functional.conv2d(
            img1 * img2, window, padding=window_size//2, groups=channel
        ) - mu1_mu2
        
        C1 = 0.0001
        C2 = 0.0009
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def forward(self, pred, target):
        """Calculate combined loss."""
        mse = self.mse_loss(pred, target)
        
        channel = pred.size(1)
        window = self.create_window(self.window_size, channel).to(pred.device)
        ssim_val = self.ssim(pred, target, window, self.window_size, channel)
        ssim_loss = 1 - ssim_val
        
        combined = self.alpha * mse + (1 - self.alpha) * ssim_loss
        
        return combined, mse, ssim_val


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, test_loss, fold_name, output_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'fold_name': fold_name
    }
    
    if is_best:
        best_path = os.path.join(output_dir, f'{fold_name}_best.pth')
        torch.save(checkpoint, best_path)
        print(f"    ✅ Saved best model: {best_path} (epoch {epoch}, val_loss: {val_loss:.6f})")


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_ssim = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss, mse, ssim = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_mse += mse.item()
        running_ssim += ssim.item()
        
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'mse': running_mse / (batch_idx + 1),
            'ssim': running_ssim / (batch_idx + 1)
        })
    
    avg_loss = running_loss / len(train_loader)
    avg_mse = running_mse / len(train_loader)
    avg_ssim = running_ssim / len(train_loader)
    
    return avg_loss, avg_mse, avg_ssim


def validate_epoch(model, val_loader, criterion, device, epoch, split_name='Val'):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    running_ssim = 0.0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [{split_name}]')
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(inputs)
            loss, mse, ssim = criterion(outputs, targets)
            
            running_loss += loss.item()
            running_mse += mse.item()
            running_ssim += ssim.item()
            
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'mse': running_mse / (batch_idx + 1),
                'ssim': running_ssim / (batch_idx + 1)
            })
    
    avg_loss = running_loss / len(val_loader)
    avg_mse = running_mse / len(val_loader)
    avg_ssim = running_ssim / len(val_loader)
    
    return avg_loss, avg_mse, avg_ssim


def train_single_fold(fold_num, args):
    """Train a single fold."""
    print("\n" + "="*80)
    print(f"FOLD {fold_num}/{len(FOLD_SPLITS)}")
    print("="*80)
    
    # Get subject splits for this fold
    split = FOLD_SPLITS[fold_num]
    train_subjects = split['train']
    val_subjects = split['val']
    test_subjects = split['test']
    
    print(f"Train subjects: {train_subjects}")
    print(f"Val subject:    {val_subjects}")
    print(f"Test subject:   {test_subjects}")
    
    # Create output directory for this fold
    fold_name = f'fold{fold_num}'
    fold_output_dir = Path(args.output_dir) / fold_name
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save fold configuration
    config = {
        'fold_num': fold_num,
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
        **vars(args)
    }
    with open(fold_output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = MRIReconstructionDataset(
        args.input_dir, 
        args.target_dir,
        subject_ids=train_subjects,
        normalize=True
    )
    
    val_dataset = MRIReconstructionDataset(
        args.input_dir,
        args.target_dir,
        subject_ids=val_subjects,
        normalize=True
    )
    
    test_dataset = MRIReconstructionDataset(
        args.input_dir,
        args.target_dir,
        subject_ids=test_subjects,
        normalize=True
    )
    
    print(f"Train: {len(train_dataset)} samples from subjects {train_subjects}")
    print(f"Val:   {len(val_dataset)} samples from subjects {val_subjects}")
    print(f"Test:  {len(test_dataset)} samples from subjects {test_subjects}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = get_model(
        args.in_channels,
        args.out_channels,
        args.base_filters,
        bilinear=args.bilinear
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = CombinedLoss(alpha=args.loss_alpha)
    
    # Training loop
    print("\n" + "="*80)
    print(f"TRAINING FOLD {fold_num}")
    print("="*80)
    
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    history = {
        'train_loss': [],
        'train_mse': [],
        'train_ssim': [],
        'val_loss': [],
        'val_mse': [],
        'val_ssim': [],
        'test_loss': [],
        'test_mse': [],
        'test_ssim': [],
        'epoch_times': [],
        'early_stopped': False,
        'best_epoch': 0
    }
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_mse, train_ssim = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_mse, val_ssim = validate_epoch(
            model, val_loader, criterion, device, epoch, split_name='Val'
        )
        
        # Test (evaluation only, not used for model selection)
        test_loss, test_mse, test_ssim = validate_epoch(
            model, test_loader, criterion, device, epoch, split_name='Test'
        )
        
        epoch_time = time.time() - epoch_start
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_mse'].append(train_mse)
        history['train_ssim'].append(train_ssim)
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_mse)
        history['val_ssim'].append(val_ssim)
        history['test_loss'].append(test_loss)
        history['test_mse'].append(test_mse)
        history['test_ssim'].append(test_ssim)
        history['epoch_times'].append(epoch_time)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train - Loss: {train_loss:.6f}, MSE: {train_mse:.6f}, SSIM: {train_ssim:.4f}")
        print(f"  Val   - Loss: {val_loss:.6f}, MSE: {val_mse:.6f}, SSIM: {val_ssim:.4f}")
        print(f"  Test  - Loss: {test_loss:.6f}, MSE: {test_mse:.6f}, SSIM: {test_ssim:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Check for improvement (only save best model)
        improvement = best_val_loss - val_loss
        is_best = improvement > args.early_stopping_min_delta
        
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            print(f"  🎯 New best validation loss: {best_val_loss:.6f}")
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, test_loss,
                fold_name, fold_output_dir, is_best=True
            )
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
        
        # Save history
        history['best_epoch'] = best_epoch
        with open(fold_output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=4)
        
        # Early stopping check
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"\n⏹️  Early stopping triggered after {epoch} epochs")
            print(f"   No improvement for {args.early_stopping_patience} consecutive epochs")
            print(f"   Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
            history['early_stopped'] = True
            history['stopped_epoch'] = epoch
            with open(fold_output_dir / 'training_history.json', 'w') as f:
                json.dump(history, f, indent=4)
            break
    
    print("\n" + "="*80)
    print(f"FOLD {fold_num} COMPLETE")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"Final test loss: {test_loss:.6f}")
    if history['early_stopped']:
        print(f"Early stopped at epoch {history['stopped_epoch']}/{args.epochs}")
    else:
        print(f"Completed all {args.epochs} epochs")
    print(f"Results saved to: {fold_output_dir}")
    print("="*80)
    
    return {
        'fold': fold_num,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'final_test_loss': test_loss,
        'early_stopped': history['early_stopped'],
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects
    }


def main():
    parser = argparse.ArgumentParser(description='5-Fold Cross-Validation with Sliding Window')
    
    # Fold selection
    parser.add_argument('--fold', type=int, choices=[1, 2, 3, 4, 5],
                        help='Train a specific fold (1-5)')
    parser.add_argument('--all-folds', action='store_true',
                        help='Train all 5 folds sequentially')
    
    # Data arguments
    parser.add_argument('--input-dir', type=str, 
                        default='../Synth_LR_unpadded_nii',
                        help='Directory with input (LR) images')
    parser.add_argument('--target-dir', type=str,
                        default='../Dynamic_SENSE_padded',
                        help='Directory with target (HR) images')
    
    # Model arguments
    parser.add_argument('--in-channels', type=int, default=1)
    parser.add_argument('--out-channels', type=int, default=1)
    parser.add_argument('--base-filters', type=int, default=32)
    parser.add_argument('--bilinear', action='store_true', default=True)
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--loss-alpha', type=float, default=0.7)
    
    # Early stopping arguments
    parser.add_argument('--early-stopping-patience', type=int, default=20,
                        help='Number of epochs with no validation improvement before stopping')
    parser.add_argument('--early-stopping-min-delta', type=float, default=1e-6,
                        help='Minimum change in validation loss to qualify as improvement')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='cv_results_sliding')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.fold and not args.all_folds:
        parser.error("Must specify either --fold or --all-folds")
    
    # Print overall configuration
    print("="*80)
    print("5-FOLD CROSS-VALIDATION WITH SLIDING WINDOW")
    print("="*80)
    print(f"Strategy: Train/Val/Test = 5/1/1 subjects per fold")
    print(f"All subjects: {ALL_SUBJECTS}")
    print(f"\nFold splits:")
    for fold_num, split in FOLD_SPLITS.items():
        print(f"  Fold {fold_num}: Train={split['train']}, Val={split['val']}, Test={split['test']}")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Epochs per fold: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*80)
    
    # Train folds
    if args.all_folds:
        folds_to_train = list(range(1, 6))
    else:
        folds_to_train = [args.fold]
    
    results = []
    for fold_num in folds_to_train:
        result = train_single_fold(fold_num, args)
        results.append(result)
    
    # Save overall summary
    if args.all_folds:
        summary = {
            'total_folds': len(results),
            'results': results,
            'avg_val_loss': np.mean([r['best_val_loss'] for r in results]),
            'avg_test_loss': np.mean([r['final_test_loss'] for r in results]),
            'std_val_loss': np.std([r['best_val_loss'] for r in results]),
            'std_test_loss': np.std([r['final_test_loss'] for r in results])
        }
        
        summary_path = Path(args.output_dir) / 'cv_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print("\n" + "="*80)
        print("ALL FOLDS COMPLETE")
        print("="*80)
        print(f"Average validation loss: {summary['avg_val_loss']:.6f} ± {summary['std_val_loss']:.6f}")
        print(f"Average test loss: {summary['avg_test_loss']:.6f} ± {summary['std_test_loss']:.6f}")
        print(f"Summary saved to: {summary_path}")
        print("="*80)


if __name__ == "__main__":
    main()

"""
Cross-validation training script for U-Net MRI super-resolution.

Designed for N-fold cross-validation on N subjects, with optional held-out subject
for final testing. This enables proper model selection:
1. Run N-fold CV on N subjects (e.g., 6-fold on subjects 0023-0026, 0022, 0024)
2. Select best fold based on average test performance
3. Test that fold's model on held-out subject (e.g., 0021)

Usage:
    # 6-fold CV (one subject held out for final test)
    python train_cross_validation.py --train-subjects 0023 0024 0025 \
                                     --test-subjects 0026 \
                                     --fold-name fold1
    
    # Held-out subject is NOT included in any fold during CV
    # After CV, test best fold's model on held-out subject using inference script
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
        """
        Calculate combined loss.
        
        Args:
            pred: Predicted image (B, C, H, W)
            target: Ground truth image (B, C, H, W)
        
        Returns:
            Combined loss value
        """
        # MSE loss
        mse = self.mse_loss(pred, target)
        
        # SSIM loss
        channel = pred.size(1)
        window = self.create_window(self.window_size, channel).to(pred.device)
        ssim_val = self.ssim(pred, target, window, self.window_size, channel)
        ssim_loss = 1 - ssim_val
        
        # Combined loss
        combined = self.alpha * mse + (1 - self.alpha) * ssim_loss
        
        return combined, mse, ssim_val


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, fold_name, output_dir, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss (None if no validation set)
        fold_name: Name of the fold (e.g., 'fold1', 'fold2')
        output_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'fold_name': fold_name
    }
    
    # Only save best model to save storage space
    if is_best:
        best_path = os.path.join(output_dir, f'{fold_name}_best.pth')
        torch.save(checkpoint, best_path)
        print(f"    ✅ Saved best model: {best_path} (epoch {epoch}, loss: {val_loss if val_loss else train_loss:.6f})")


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
        
        # Update progress bar
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


def main():
    parser = argparse.ArgumentParser(description='Cross-validation training for U-Net')
    
    # Cross-validation arguments
    parser.add_argument('--train-subjects', nargs='+', required=True,
                        help='Subject IDs for training (e.g., 0023 0024 0025)')
    parser.add_argument('--test-subjects', nargs='+', required=True,
                        help='Subject IDs for testing (e.g., 0026)')
    parser.add_argument('--val-subjects', nargs='+', default=None,
                        help='Subject IDs for validation (optional, e.g., 0022)')
    parser.add_argument('--fold-name', type=str, required=True,
                        help='Name for this fold (e.g., fold1, fold2)')
    
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
    parser.add_argument('--base-filters', type=int, default=32,
                        help='Number of filters in first layer')
    parser.add_argument('--bilinear', action='store_true', default=True)
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--loss-alpha', type=float, default=0.7,
                        help='Weight for MSE in combined loss (0.7 = 70% MSE + 30% SSIM)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='cross_validation_results')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Create output directory for this fold
    fold_output_dir = Path(args.output_dir) / args.fold_name
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config_path = fold_output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Print configuration
    print("="*80)
    print(f"CROSS-VALIDATION TRAINING: {args.fold_name}")
    print("="*80)
    print(f"Train subjects: {args.train_subjects}")
    if args.val_subjects:
        print(f"Val subjects:   {args.val_subjects}")
    print(f"Test subjects:  {args.test_subjects}")
    print(f"Output dir:     {fold_output_dir}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Learning rate:  {args.lr}")
    print(f"Loss alpha:     {args.loss_alpha} (MSE weight)")
    print(f"Base filters:   {args.base_filters}")
    print("="*80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = MRIReconstructionDataset(
        args.input_dir, 
        args.target_dir,
        subject_ids=args.train_subjects,
        normalize=True
    )
    
    if args.val_subjects:
        val_dataset = MRIReconstructionDataset(
            args.input_dir,
            args.target_dir,
            subject_ids=args.val_subjects,
            normalize=True
        )
    else:
        val_dataset = None
    
    print(f"Train dataset: {len(train_dataset)} samples")
    if val_dataset:
        print(f"Val dataset:   {len(val_dataset)} samples")
    print(f"Test subjects: {args.test_subjects} (not loaded during training)")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    else:
        val_loader = None
    
    # Initialize model
    print("\nInitializing model...")
    model = get_model(
        args.in_channels,
        args.out_channels,
        args.base_filters,
        bilinear=args.bilinear
    )
    model = model.to(device)
    
    # Count parameters
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
    print("STARTING TRAINING")
    print("="*80)
    
    best_loss = float('inf')
    history = {
        'train_loss': [],
        'train_mse': [],
        'train_ssim': [],
        'val_loss': [],
        'val_mse': [],
        'val_ssim': [],
        'epoch_times': []
    }
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_mse, train_ssim = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate (if validation set exists)
        if val_loader:
            val_loss, val_mse, val_ssim = validate_epoch(
                model, val_loader, criterion, device, epoch, split_name='Val'
            )
            monitor_loss = val_loss  # Use validation loss for model selection
        else:
            val_loss, val_mse, val_ssim = None, None, None
            monitor_loss = train_loss  # Use training loss if no validation set
        
        epoch_time = time.time() - epoch_start
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_mse'].append(train_mse)
        history['train_ssim'].append(train_ssim)
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_mse)
        history['val_ssim'].append(val_ssim)
        history['epoch_times'].append(epoch_time)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train - Loss: {train_loss:.6f}, MSE: {train_mse:.6f}, SSIM: {train_ssim:.4f}")
        if val_loader:
            print(f"  Val   - Loss: {val_loss:.6f}, MSE: {val_mse:.6f}, SSIM: {val_ssim:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save checkpoint (only if best)
        is_best = monitor_loss < best_loss
        if is_best:
            best_loss = monitor_loss
            print(f"  🎯 New best {'validation' if val_loader else 'training'} loss: {best_loss:.6f}")
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                args.fold_name, fold_output_dir, is_best=True
            )
        
        # Save history
        history_path = fold_output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best {'validation' if val_loader else 'training'} loss: {best_loss:.6f}")
    print(f"Results saved to: {fold_output_dir}")
    print(f"Best model: {fold_output_dir / f'{args.fold_name}_best.pth'}")
    print("="*80)


if __name__ == "__main__":
    main()

"""
Training script for U-Net based synthetically supervised MRI reconstruction.

This script trains a U-Net to map from undersampled/aliased MRI images to
fully-sampled reconstructions using synthetically generated training pairs.

Usage:
    python train_unet.py --input-dir Dynamic_SENSE --target-dir Dynamic_SENSE_padded --epochs 100
"""

import os
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from unet_model import get_model
from dataset import get_dataloaders


class CombinedLoss(nn.Module):
    """
    Combined loss function: L2 (MSE) + SSIM loss.
    
    L2 loss for pixel-wise accuracy + SSIM for structural similarity.
    """
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.l2_loss = nn.MSELoss()
    
    def ssim_loss(self, pred, target, window_size=11):
        """Compute SSIM loss (1 - SSIM)."""
        # Simple SSIM implementation
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = torch.nn.functional.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu_y = torch.nn.functional.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = torch.nn.functional.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_x_sq
        sigma_y_sq = torch.nn.functional.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_y_sq
        sigma_xy = torch.nn.functional.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_xy
        
        ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        return 1 - ssim.mean()
    
    def forward(self, pred, target):
        l2 = self.l2_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        return self.alpha * l2 + (1 - self.alpha) * ssim


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def compute_metrics(pred, target):
    """Compute PSNR and SSIM metrics."""
    # PSNR
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    # Simple SSIM (using the loss function)
    loss_fn = CombinedLoss()
    ssim = 1 - loss_fn.ssim_loss(pred, target).item()
    
    return psnr.item(), ssim


def save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best model with val_loss={best_loss:.4f}")


def train(args):
    """Main training function."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    checkpoint_dir = Path(args.output_dir) / 'checkpoints'
    log_dir = Path(args.output_dir) / 'logs'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Create model
    print("\nCreating model...")
    model = get_model(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        base_filters=args.base_filters,
        bilinear=args.bilinear
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
    
    # Loss function and optimizer
    criterion = CombinedLoss(alpha=args.loss_alpha)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5 # removed verbose=True as Colab didn't like it
    )
    
    # Data loaders
    print("\nLoading data...")
    
    # Standard training approach: 80/10/10 split from Synth_LR_nii and HR_nii
    # Dynamic_SENSE data is reserved for final testing after training
    print("Using 80/10/10 split from training data")
    print(f"Training data: {args.input_dir} -> {args.target_dir}")
    
    from dataset import MRIReconstructionDataset
    
    # Create full dataset from synthetic pairs
    full_dataset = MRIReconstructionDataset(
        input_dir=args.input_dir,
        target_dir=args.target_dir,
        normalize=True,
        frame_range=None
    )
    
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    print(f"\nDataset split:")
    print(f"  Total frames: {dataset_size}")
    print(f"  Train: {train_size} frames (80%)")
    print(f"  Val: {val_size} frames (10%)")
    print(f"  Test: {test_size} frames (10%, saved for later)")
    print(f"  Note: Dynamic_SENSE data reserved for final testing\n")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    # Save test indices for later use
    test_indices_path = Path(args.output_dir) / 'test_indices.txt'
    test_indices_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_indices_path, 'w') as f:
        f.write(f"# Test set indices (10% of data, seed=42)\n")
        f.write(f"# Total dataset size: {dataset_size}\n")
        f.write(f"# Test indices: {test_size} frames\n")
        for idx in test_dataset.indices:
            f.write(f"{idx}\n")
    print(f"Test set indices saved to: {test_indices_path}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"Warning: Checkpoint {args.resume} not found. Starting from scratch.")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log metrics
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.1f}s")
        print("-" * 80)
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir, is_best=False)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir, is_best=True)
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train U-Net for MRI reconstruction')
    
    # Data arguments
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory with undersampled input images (training + val + test)')
    parser.add_argument('--target-dir', type=str, required=True,
                        help='Directory with fully-sampled target images (training + val + test)')
    parser.add_argument('--output-dir', type=str, default='../outputs',
                        help='Output directory for checkpoints and logs (default: ../outputs)')
    
    # Model arguments
    parser.add_argument('--in-channels', type=int, default=1,
                        help='Number of input channels')
    parser.add_argument('--out-channels', type=int, default=1,
                        help='Number of output channels')
    parser.add_argument('--base-filters', type=int, default=32,
                        help='Number of filters in first layer (default: 32 for lightweight baseline)')
    parser.add_argument('--bilinear', action='store_true', default=True,
                        help='Use bilinear upsampling')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--loss-alpha', type=float, default=0.7,
                        help='Weight for L2 loss in combined loss (1-alpha for SSIM, default: 0.7 for 30%% SSIM)')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Fraction of data to use for training (only used if val-*-dir not provided)')
    
    # System arguments
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "=" * 80)
    print("Training Configuration")
    print("=" * 80)
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    print("=" * 80 + "\n")
    
    train(args)


if __name__ == "__main__":
    main()

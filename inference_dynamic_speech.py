"""
Inference script for applying trained U-Net to dynamic speech reconstruction.

Processes the SENSE-reconstructed dynamic speech data (200 frames) through
the trained U-Net model and saves the result as "unet_speaking.nii".

Usage:
    python inference_dynamic_speech.py --checkpoint "2910 result/best_model.pth" \
                                       --input-file ../Dynamic_SENSE_padded/Subject0021_speech_padded_312x410x200.nii \
                                       --output-file ../Dynamic_SENSE_padded/unet_speaking.nii
"""

import os
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

from unet_model import get_model


def load_model(checkpoint_path, device, in_channels=1, out_channels=1, base_filters=32):
    """
    Load trained U-Net model from checkpoint.
    
    Args:
        checkpoint_path: Path to best_model.pth checkpoint
        device: torch device (cuda or cpu)
        in_channels: Number of input channels (default: 1 for grayscale MRI)
        out_channels: Number of output channels (default: 1)
        base_filters: Number of filters in first layer (default: 32)
    
    Returns:
        model: Loaded U-Net model in eval mode
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Initialize model architecture
    model = get_model(in_channels, out_channels, base_filters, bilinear=True)
    model = model.to(device)
    
    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Display checkpoint info
    epoch = checkpoint.get('epoch', 'unknown')
    val_loss = checkpoint.get('val_loss', 'unknown')
    print(f"Model loaded from epoch {epoch}, val_loss: {val_loss}")
    
    return model


def process_dynamic_volume(model, input_file, device, output_path, normalize=True):
    """
    Process a multi-frame dynamic MRI volume through U-Net frame-by-frame.
    
    Args:
        model: Trained U-Net model in eval mode
        input_file: Path to input NIfTI file (H x W x num_frames)
        device: torch device
        output_path: Where to save reconstructed NIfTI
        normalize: Whether to normalize each frame to [0, 1] before processing
    
    Returns:
        reconstructed_volume: numpy array (H, W, num_frames)
    """
    # Load input NIfTI file
    print(f"\nLoading input: {input_file}")
    img_nifti = nib.load(input_file)
    img_data = img_nifti.get_fdata()
    
    # Validate dimensions
    if img_data.ndim == 2:
        img_data = img_data[:, :, np.newaxis]
        print(f"  Input shape: {img_data.shape} (2D, added frame dimension)")
    elif img_data.ndim == 3:
        print(f"  Input shape: {img_data.shape} (H x W x frames)")
    else:
        raise ValueError(f"Unexpected input dimensions: {img_data.shape}")
    
    H, W, num_frames = img_data.shape
    print(f"  Processing {num_frames} frames of size {H}x{W}")
    
    # Allocate output volume
    reconstructed_volume = np.zeros_like(img_data, dtype=np.float32)
    
    # Process frame-by-frame
    print(f"\nRunning U-Net inference on {num_frames} frames...")
    with torch.no_grad():
        for frame_idx in tqdm(range(num_frames), desc="Processing frames"):
            frame = img_data[:, :, frame_idx]
            
            # Normalize frame to [0, 1] if requested
            original_min, original_max = None, None
            if normalize:
                original_min = frame.min()
                original_max = frame.max()
                if original_max - original_min > 0:
                    frame = (frame - original_min) / (original_max - original_min)
            
            # Convert to PyTorch tensor: (1, 1, H, W)
            frame_tensor = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0)
            frame_tensor = frame_tensor.to(device)
            
            # Run U-Net forward pass
            recon_tensor = model(frame_tensor)
            
            # Convert back to numpy: (H, W)
            recon_frame = recon_tensor.cpu().squeeze().numpy()
            
            # Denormalize back to original intensity range if normalized
            if normalize and original_max is not None:
                recon_frame = recon_frame * (original_max - original_min) + original_min
            
            # Store reconstructed frame
            reconstructed_volume[:, :, frame_idx] = recon_frame
    
    # Save reconstructed volume as NIfTI
    print(f"\nSaving reconstruction to: {output_path}")
    recon_nifti = nib.Nifti1Image(reconstructed_volume, affine=img_nifti.affine)
    nib.save(recon_nifti, output_path)
    print(f"Successfully saved {num_frames} frames to {output_path}")
    
    # Display intensity statistics
    print(f"\nIntensity Statistics:")
    print(f"  Input  - Min: {img_data.min():.2f}, Max: {img_data.max():.2f}, Mean: {img_data.mean():.2f}")
    print(f"  Output - Min: {reconstructed_volume.min():.2f}, Max: {reconstructed_volume.max():.2f}, Mean: {reconstructed_volume.mean():.2f}")
    
    return reconstructed_volume


def main():
    parser = argparse.ArgumentParser(
        description='Apply trained U-Net to dynamic speech MRI reconstruction'
    )
    
    # Model arguments
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default='2910 result/best_model.pth',
        help='Path to trained model checkpoint (default: 2910 result/best_model.pth)'
    )
    parser.add_argument(
        '--in-channels', 
        type=int, 
        default=1,
        help='Number of input channels (default: 1)'
    )
    parser.add_argument(
        '--out-channels', 
        type=int, 
        default=1,
        help='Number of output channels (default: 1)'
    )
    parser.add_argument(
        '--base-filters', 
        type=int, 
        default=32,
        help='Number of filters in first layer (default: 32)'
    )
    
    # Data arguments
    parser.add_argument(
        '--input-file', 
        type=str, 
        default='../Dynamic_SENSE_padded/Subject0021_speech_padded_312x410x200.nii',
        help='Path to input NIfTI file (default: ../Dynamic_SENSE_padded/Subject0021_speech_padded_312x410x200.nii)'
    )
    parser.add_argument(
        '--output-file', 
        type=str, 
        default='../Dynamic_SENSE_padded/unet_speaking.nii',
        help='Path for output NIfTI file (default: ../Dynamic_SENSE_padded/unet_speaking.nii)'
    )
    
    # Processing arguments
    parser.add_argument(
        '--normalize', 
        action='store_true', 
        default=True,
        help='Normalize each frame to [0, 1] before processing (default: True)'
    )
    parser.add_argument(
        '--no-normalize', 
        dest='normalize', 
        action='store_false',
        help='Disable frame-wise normalization'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*80}")
    print(f"U-Net Dynamic Speech Inference")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"Normalize: {args.normalize}")
    print(f"{'='*80}\n")
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file not found: {args.input_file}")
        print(f"Please wait for SENSE reconstruction to complete.")
        return
    
    # Create output directory if needed
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trained model
    model = load_model(
        args.checkpoint,
        device,
        args.in_channels,
        args.out_channels,
        args.base_filters
    )
    
    # Process the dynamic volume
    reconstructed_volume = process_dynamic_volume(
        model,
        args.input_file,
        device,
        args.output_file,
        normalize=args.normalize
    )
    
    print(f"\n{'='*80}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*80}")
    print(f"Reconstructed volume saved to: {args.output_file}")
    print(f"Shape: {reconstructed_volume.shape}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

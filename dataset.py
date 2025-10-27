"""
Dataset loader for synthetically supervised MRI reconstruction.

Loads pairs of:
- Input: Undersampled/aliased MRI frames (from SENSE reconstruction or synthetic undersampling)
- Target: Fully-sampled ground truth frames

Supports both NIfTI files and direct numpy arrays.
"""

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob


class MRIReconstructionDataset(Dataset):
    """
    Dataset for MRI reconstruction with synthetic supervision.
    
    Loads paired undersampled (input) and fully-sampled (target) MRI data.
    Each sample is a single 2D frame from a dynamic MRI sequence.
    
    Args:
        input_dir: Directory containing undersampled/aliased NIfTI files
        target_dir: Directory containing fully-sampled ground truth NIfTI files
        transform: Optional transforms to apply to both input and target
        normalize: Whether to normalize images to [0, 1] range (default: True)
        frame_range: Optional tuple (start, end) to load only specific frames
    """
    
    def __init__(self, input_dir, target_dir, transform=None, normalize=True, frame_range=None):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.transform = transform
        self.normalize = normalize
        self.frame_range = frame_range
        
        # Find all NIfTI files
        self.input_files = sorted(glob.glob(str(self.input_dir / "*.nii*")))
        self.target_files = sorted(glob.glob(str(self.target_dir / "*.nii*")))
        
        if len(self.input_files) == 0:
            raise ValueError(f"No NIfTI files found in {input_dir}")
        if len(self.target_files) == 0:
            raise ValueError(f"No NIfTI files found in {target_dir}")
        
        print(f"Found {len(self.input_files)} input files")
        print(f"Found {len(self.target_files)} target files")
        
        # Build frame index: list of (file_idx, frame_idx) tuples
        self.frame_index = []
        for file_idx, input_file in enumerate(self.input_files):
            # Load to get number of frames
            img = nib.load(input_file).get_fdata()
            
            # Handle both 2D and 3D (with frames) data
            if img.ndim == 2:
                num_frames = 1
            elif img.ndim == 3:
                num_frames = img.shape[2]
            else:
                raise ValueError(f"Unexpected image dimensionality: {img.ndim}")
            
            # Apply frame range filter if specified
            if self.frame_range is not None:
                start, end = self.frame_range
                frame_indices = range(max(0, start), min(num_frames, end))
            else:
                frame_indices = range(num_frames)
            
            for frame_idx in frame_indices:
                self.frame_index.append((file_idx, frame_idx))
        
        print(f"Total frames in dataset: {len(self.frame_index)}")
    
    def __len__(self):
        return len(self.frame_index)
    
    def __getitem__(self, idx):
        file_idx, frame_idx = self.frame_index[idx]
        
        # Load input (undersampled) frame
        input_img = nib.load(self.input_files[file_idx]).get_fdata()
        if input_img.ndim == 3:
            input_frame = input_img[:, :, frame_idx]
        else:
            input_frame = input_img
        
        # Load target (fully-sampled) frame
        target_img = nib.load(self.target_files[file_idx]).get_fdata()
        if target_img.ndim == 3:
            target_frame = target_img[:, :, frame_idx]
        else:
            target_frame = target_img
        
        # Normalize to [0, 1] if requested
        if self.normalize:
            input_frame = self._normalize(input_frame)
            target_frame = self._normalize(target_frame)
        
        # Convert to torch tensors with channel dimension: (1, H, W)
        input_tensor = torch.from_numpy(input_frame).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_frame).float().unsqueeze(0)
        
        # Apply transforms if provided
        if self.transform is not None:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)
        
        return {
            'input': input_tensor,
            'target': target_tensor,
            'file_idx': file_idx,
            'frame_idx': frame_idx,
            'filename': os.path.basename(self.input_files[file_idx])
        }
    
    def _normalize(self, img):
        """Normalize image to [0, 1] range."""
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max - img_min > 0:
            return (img - img_min) / (img_max - img_min)
        else:
            return img


class KSpaceDataset(Dataset):
    """
    Dataset that works directly with k-space data (MAT files).
    
    Performs synthetic undersampling on-the-fly from fully-sampled k-space.
    Useful when you have fully-sampled data and want to train with various
    undersampling patterns.
    
    Args:
        kspace_dir: Directory containing k-space MAT files
        sensitivity_dir: Directory containing coil sensitivity MAT files
        acquired_indices: List of k-space column indices that are "acquired"
        normalize: Whether to normalize images to [0, 1] range
        frame_range: Optional tuple (start, end) to load only specific frames
    """
    
    def __init__(self, kspace_dir, sensitivity_dir=None, acquired_indices=None, 
                 normalize=True, frame_range=None):
        from scipy.io import loadmat
        
        self.kspace_dir = Path(kspace_dir)
        self.sensitivity_dir = Path(sensitivity_dir) if sensitivity_dir else None
        self.acquired_indices = acquired_indices
        self.normalize = normalize
        self.frame_range = frame_range
        
        # Find MAT files
        self.kspace_files = sorted(glob.glob(str(self.kspace_dir / "*.mat")))
        
        if len(self.kspace_files) == 0:
            raise ValueError(f"No MAT files found in {kspace_dir}")
        
        print(f"Found {len(self.kspace_files)} k-space files")
        
        # Build frame index
        self.frame_index = []
        for file_idx, kspace_file in enumerate(self.kspace_files):
            # Load to get shape
            data = loadmat(kspace_file)
            kspace = data['kspace']
            
            # Expected shape: (Ny, Nx, Nc, Nf)
            num_frames = kspace.shape[3] if kspace.ndim == 4 else 1
            
            if self.frame_range is not None:
                start, end = self.frame_range
                frame_indices = range(max(0, start), min(num_frames, end))
            else:
                frame_indices = range(num_frames)
            
            for frame_idx in frame_indices:
                self.frame_index.append((file_idx, frame_idx))
        
        print(f"Total frames in dataset: {len(self.frame_index)}")
    
    def __len__(self):
        return len(self.frame_index)
    
    def __getitem__(self, idx):
        from scipy.io import loadmat
        import scipy.fft
        
        file_idx, frame_idx = self.frame_index[idx]
        
        # Load k-space
        data = loadmat(self.kspace_files[file_idx])
        kspace = data['kspace']
        
        # Get specific frame: (Ny, Nx, Nc)
        if kspace.ndim == 4:
            kspace_frame = kspace[:, :, :, frame_idx]
        else:
            kspace_frame = kspace
        
        # Generate fully-sampled image (RSS over coils)
        img_full_coils = scipy.fft.ifftshift(
            scipy.fft.ifft2(scipy.fft.ifftshift(kspace_frame, axes=(0,1)), axes=(0,1)),
            axes=(0,1)
        )
        target_frame = np.sqrt(np.sum(np.abs(img_full_coils)**2, axis=2))
        
        # Generate undersampled image if acquired_indices provided
        if self.acquired_indices is not None:
            # Create mask
            mask = np.zeros(kspace_frame.shape[1])
            mask[self.acquired_indices] = 1.0
            
            # Apply mask to k-space
            kspace_undersampled = kspace_frame * mask[np.newaxis, :, np.newaxis]
            
            # Reconstruct undersampled image (RSS)
            img_under_coils = scipy.fft.ifftshift(
                scipy.fft.ifft2(scipy.fft.ifftshift(kspace_undersampled, axes=(0,1)), axes=(0,1)),
                axes=(0,1)
            )
            input_frame = np.sqrt(np.sum(np.abs(img_under_coils)**2, axis=2))
        else:
            # No undersampling, use full image as both input and target
            input_frame = target_frame.copy()
        
        # Normalize
        if self.normalize:
            input_frame = self._normalize(input_frame)
            target_frame = self._normalize(target_frame)
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_frame).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_frame).float().unsqueeze(0)
        
        return {
            'input': input_tensor,
            'target': target_tensor,
            'file_idx': file_idx,
            'frame_idx': frame_idx,
            'filename': os.path.basename(self.kspace_files[file_idx])
        }
    
    def _normalize(self, img):
        """Normalize image to [0, 1] range."""
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max - img_min > 0:
            return (img - img_min) / (img_max - img_min)
        else:
            return img


def get_dataloaders(input_dir, target_dir, batch_size=4, num_workers=4, 
                   train_split=0.8, frame_range=None, shuffle=True):
    """
    Create train and validation dataloaders.
    
    Args:
        input_dir: Directory with undersampled inputs
        target_dir: Directory with fully-sampled targets
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        train_split: Fraction of data to use for training (rest for validation)
        frame_range: Optional (start, end) frame range
        shuffle: Whether to shuffle training data
    
    Returns:
        train_loader, val_loader
    """
    # Create full dataset
    full_dataset = MRIReconstructionDataset(
        input_dir=input_dir,
        target_dir=target_dir,
        normalize=True,
        frame_range=frame_range
    )
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Train set: {len(train_dataset)} frames")
    print(f"Validation set: {len(val_dataset)} frames")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test of the dataset loader
    print("Testing MRIReconstructionDataset...")
    
    # Example paths (adjust as needed)
    input_dir = "Dynamic_SENSE"
    target_dir = "Dynamic_SENSE_padded"
    
    if os.path.exists(input_dir) and os.path.exists(target_dir):
        dataset = MRIReconstructionDataset(
            input_dir=input_dir,
            target_dir=target_dir,
            normalize=True
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        # Test loading a sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Input shape: {sample['input'].shape}")
        print(f"Target shape: {sample['target'].shape}")
        print(f"Input range: [{sample['input'].min():.3f}, {sample['input'].max():.3f}]")
        print(f"Target range: [{sample['target'].min():.3f}, {sample['target'].max():.3f}]")
        
        print("\n✓ Dataset test passed!")
    else:
        print(f"Directories not found. Please ensure {input_dir} and {target_dir} exist.")
        print("This is a test script - adjust paths as needed.")

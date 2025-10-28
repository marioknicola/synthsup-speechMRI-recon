"""
U-Net with integrated Data Consistency layer.

This model combines the U-Net architecture with a data consistency layer
that enforces k-space measurements in the reconstruction.
"""

import torch
import torch.nn as nn
from unet_model import UNet, DataConsistencyLayer


class UNetWithDC(nn.Module):
    """
    U-Net with integrated Data Consistency layer.
    
    This model performs:
    1. U-Net prediction (image-to-image)
    2. Data consistency enforcement (k-space correction)
    
    Args:
        in_channels: Number of input channels (default: 1)
        out_channels: Number of output channels (default: 1)
        base_filters: Number of filters in first conv layer (default: 32)
        bilinear: Use bilinear upsampling (default: True)
        dc_iterations: Number of data consistency iterations (default: 1)
    """
    
    def __init__(self, in_channels=1, out_channels=1, base_filters=32, 
                 bilinear=True, dc_iterations=1):
        super().__init__()
        self.unet = UNet(in_channels, out_channels, base_filters, bilinear)
        self.dc_layer = DataConsistencyLayer()
        self.dc_iterations = dc_iterations
    
    def forward(self, undersampled_image, kspace_data=None, mask=None):
        """
        Forward pass with optional data consistency.
        
        Args:
            undersampled_image: (B, 1, H, W) - Input undersampled image
            kspace_data: (B, 1, H, W) - Undersampled k-space complex tensor (optional)
            mask: (B, 1, H, W) - Sampling mask (1=acquired, 0=not) (optional)
        
        Returns:
            (B, 1, H, W) - Reconstructed image (data-consistent if k-space provided)
        """
        # Get U-Net prediction
        prediction = self.unet(undersampled_image)
        
        # Apply data consistency if k-space data available
        if kspace_data is not None and mask is not None:
            for _ in range(self.dc_iterations):
                prediction = self.dc_layer(prediction, kspace_data, mask)
        
        return prediction


def get_model_with_dc(in_channels=1, out_channels=1, base_filters=32, 
                      bilinear=True, dc_iterations=1):
    """
    Factory function to create a U-Net model with data consistency.
    
    Example usage:
        model = get_model_with_dc(base_filters=32, dc_iterations=5)
        model = model.to(device)
    """
    return UNetWithDC(in_channels, out_channels, base_filters, bilinear, dc_iterations)


if __name__ == "__main__":
    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = get_model_with_dc(base_filters=32, dc_iterations=5)
    model = model.to(device)
    
    # Test with dummy data
    batch_size = 2
    height, width = 312, 410
    
    dummy_input = torch.randn(batch_size, 1, height, width).to(device)
    dummy_kspace = torch.randn(batch_size, 1, height, width).to(device) + \
                   1j * torch.randn(batch_size, 1, height, width).to(device)
    dummy_mask = torch.ones(batch_size, 1, height, width).to(device)
    dummy_mask[:, :, ::3, :] = 0  # Simulate R=3 undersampling
    
    print(f"\nModel architecture: UNetWithDC")
    print(f"Input shape: {dummy_input.shape}")
    print(f"K-space shape: {dummy_kspace.shape}")
    print(f"Mask shape: {dummy_mask.shape}")
    
    # Test forward pass without DC
    with torch.no_grad():
        output_no_dc = model(dummy_input)
    print(f"\nOutput (no DC): {output_no_dc.shape}")
    
    # Test forward pass with DC
    with torch.no_grad():
        output_with_dc = model(dummy_input, dummy_kspace, dummy_mask)
    print(f"Output (with DC): {output_with_dc.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {num_params:,}")
    
    print("\n✓ Model test passed!")

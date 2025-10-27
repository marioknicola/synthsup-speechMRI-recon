"""
U-Net model for MRI reconstruction with synthetically supervised learning.

This implementation is designed for:
- Input: Undersampled/aliased MRI images (magnitude)
- Output: Fully-sampled reconstructed MRI images
- Expected input shape: (B, 1, H, W) where H, W are spatial dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive 3x3 convolutions with BatchNorm and ReLU."""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block: MaxPool followed by DoubleConv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block: Upsample, concatenate with skip connection, then DoubleConv."""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch between x1 and x2 (encoder skip connection)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution to map to output channels."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for MRI reconstruction.
    
    Args:
        in_channels: Number of input channels (default: 1 for magnitude images)
        out_channels: Number of output channels (default: 1)
        base_filters: Number of filters in the first conv layer (default: 32 for lightweight baseline)
        bilinear: Use bilinear upsampling instead of transposed conv (default: True)
    """
    
    def __init__(self, in_channels=1, out_channels=1, base_filters=32, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Encoder path
        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor)
        
        # Decoder path
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, bilinear)
        
        # Output layer
        self.outc = OutConv(base_filters, out_channels)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits


class DataConsistencyLayer(nn.Module):
    """
    Data consistency layer for k-space enforcement.
    Replaces acquired k-space lines in the network output with ground truth.
    
    This is useful for physics-guided reconstruction where we want to enforce
    that the reconstruction matches the acquired k-space data.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predicted_image, kspace_undersampled, mask, sensitivity_maps=None):
        """
        Args:
            predicted_image: (B, 1, H, W) predicted image from U-Net
            kspace_undersampled: (B, 1, H, W) undersampled k-space (complex)
            mask: (B, 1, H, W) sampling mask (1 = acquired, 0 = not acquired)
            sensitivity_maps: Optional coil sensitivity maps for multi-coil data
        
        Returns:
            (B, 1, H, W) data-consistent image
        """
        # Transform predicted image to k-space
        predicted_kspace = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(predicted_image, dim=(-2, -1))),
            dim=(-2, -1)
        )
        
        # Apply data consistency: keep acquired k-space, use predicted for rest
        consistent_kspace = mask * kspace_undersampled + (1 - mask) * predicted_kspace
        
        # Transform back to image domain
        consistent_image = torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(consistent_kspace, dim=(-2, -1))),
            dim=(-2, -1)
        )
        
        return torch.abs(consistent_image)


def get_model(in_channels=1, out_channels=1, base_filters=64, bilinear=True):
    """
    Factory function to create a U-Net model.
    
    Example usage:
        model = get_model(in_channels=1, out_channels=1, base_filters=64)
        model = model.to(device)
    """
    return UNet(in_channels, out_channels, base_filters, bilinear)


if __name__ == "__main__":
    # Quick test of the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = get_model(in_channels=1, out_channels=1, base_filters=64)
    model = model.to(device)
    
    # Test with dummy input (typical MRI size after padding: 312x410)
    dummy_input = torch.randn(2, 1, 312, 410).to(device)
    
    print(f"\nModel architecture:")
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {num_params:,}")
    
    print("\n✓ Model test passed!")

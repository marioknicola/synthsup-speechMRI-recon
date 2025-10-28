"""
Generate U-Net Architecture Diagram
Visualizes the default U-Net architecture used for MRI reconstruction.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

def draw_conv_block(ax, x, y, width, height, filters, label, color='lightblue'):
    """Draw a convolutional block."""
    rect = FancyBboxPatch((x, y), width, height, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='black', facecolor=color, 
                          linewidth=2)
    ax.add_patch(rect)
    
    # Add text
    ax.text(x + width/2, y + height/2, f'{filters}', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(x + width/2, y + height + 0.3, label, 
            ha='center', va='bottom', fontsize=8)
    
    return rect

def draw_arrow(ax, x1, y1, x2, y2, color='black', style='->', linewidth=2, label=''):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, color=color, 
                           linewidth=linewidth, zorder=1)
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, label, fontsize=7, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    return arrow

def generate_unet_architecture():
    """Generate the U-Net architecture diagram."""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Architecture parameters (base_filters=32)
    base_f = 32
    encoder_filters = [base_f, base_f*2, base_f*4, base_f*8, base_f*16]
    decoder_filters = [base_f*8, base_f*4, base_f*2, base_f]
    
    # Block dimensions
    block_width = 1.2
    block_height = 0.8
    
    # Encoder path (left side, going down)
    encoder_x = 2
    encoder_y_positions = [8, 6.5, 5, 3.5, 2]
    encoder_blocks = []
    
    print("Drawing encoder path...")
    for i, (y_pos, filters) in enumerate(zip(encoder_y_positions, encoder_filters)):
        label = 'Input\nConv' if i == 0 else f'Down {i}'
        color = 'lightgreen' if i == 0 else 'lightblue'
        block = draw_conv_block(ax, encoder_x, y_pos, block_width, block_height, 
                               filters, label, color)
        encoder_blocks.append((encoder_x + block_width/2, y_pos + block_height/2))
        
        # Draw downsampling arrow
        if i < len(encoder_y_positions) - 1:
            arrow_start_y = y_pos
            arrow_end_y = encoder_y_positions[i+1] + block_height
            draw_arrow(ax, encoder_x + block_width/2, arrow_start_y, 
                      encoder_x + block_width/2, arrow_end_y, 
                      color='red', linewidth=2.5, label='MaxPool2d')
    
    # Bottleneck
    bottleneck_x = encoder_x
    bottleneck_y = encoder_y_positions[-1]
    bottleneck_center = (bottleneck_x + block_width/2, bottleneck_y + block_height/2)
    
    # Decoder path (right side, going up)
    decoder_x = 13
    decoder_y_positions = [3.5, 5, 6.5, 8]
    decoder_blocks = []
    
    print("Drawing decoder path...")
    for i, (y_pos, filters) in enumerate(zip(decoder_y_positions, decoder_filters)):
        label = f'Up {i+1}'
        color = 'lightyellow'
        block = draw_conv_block(ax, decoder_x, y_pos, block_width, block_height, 
                               filters, label, color)
        decoder_blocks.append((decoder_x + block_width/2, y_pos + block_height/2))
    
    # Output block
    output_x = decoder_x
    output_y = 9
    draw_conv_block(ax, output_x, output_y, block_width, block_height, 
                   1, 'Output\n1×1 Conv', color='lightcoral')
    
    # Draw upsampling arrows (from bottleneck to first decoder)
    print("Drawing upsampling connections...")
    # From bottleneck to first decoder
    draw_arrow(ax, bottleneck_center[0] + block_width/2, bottleneck_center[1],
              decoder_blocks[0][0] - block_width/2, decoder_blocks[0][1],
              color='blue', linewidth=2.5, label='Upsample')
    
    # Between decoder blocks
    for i in range(len(decoder_blocks) - 1):
        draw_arrow(ax, decoder_blocks[i][0], decoder_blocks[i][1] + block_height/2,
                  decoder_blocks[i+1][0], decoder_blocks[i+1][1] - block_height/2,
                  color='blue', linewidth=2.5, label='Upsample')
    
    # From last decoder to output
    draw_arrow(ax, decoder_blocks[-1][0], decoder_blocks[-1][1] + block_height/2,
              output_x + block_width/2, output_y,
              color='black', linewidth=2)
    
    # Draw skip connections
    print("Drawing skip connections...")
    skip_colors = ['green', 'green', 'green', 'green']
    for i in range(4):
        encoder_pos = encoder_blocks[i]
        decoder_pos = decoder_blocks[3-i]
        
        # Draw curved skip connection
        ax.annotate('', xy=(decoder_pos[0] - block_width/2, decoder_pos[1]),
                   xytext=(encoder_pos[0] + block_width/2, encoder_pos[1]),
                   arrowprops=dict(arrowstyle='->', color=skip_colors[i], 
                                 linewidth=2, linestyle='--',
                                 connectionstyle="arc3,rad=0.3"))
        
        # Add label for first skip connection
        if i == 0:
            mid_x = (encoder_pos[0] + decoder_pos[0]) / 2
            mid_y = encoder_pos[1] + 0.5
            ax.text(mid_x, mid_y, 'Skip\nConnections', fontsize=8,
                   ha='center', color='green', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           alpha=0.9, edgecolor='green'))
    
    # Add dimension annotations
    print("Adding annotations...")
    
    # Input/Output dimensions
    ax.text(encoder_x + block_width/2, 9.2, '312×410×1', 
           ha='center', fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.7))
    
    ax.text(output_x + block_width/2, 9.8, '312×410×1', 
           ha='center', fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.7))
    
    # Spatial dimension annotations (left side)
    dimensions = ['312×410', '156×205', '78×102', '39×51', '19×25']
    for i, (y_pos, dim) in enumerate(zip(encoder_y_positions, dimensions)):
        ax.text(encoder_x - 0.5, y_pos + block_height/2, dim, 
               ha='right', va='center', fontsize=8, style='italic', color='navy')
    
    # Add legend
    print("Adding legend...")
    legend_elements = [
        mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='Input Layer'),
        mpatches.Patch(facecolor='lightblue', edgecolor='black', label='Encoder (Down)'),
        mpatches.Patch(facecolor='lightyellow', edgecolor='black', label='Decoder (Up)'),
        mpatches.Patch(facecolor='lightcoral', edgecolor='black', label='Output Layer'),
        mpatches.FancyArrow(0, 0, 0.3, 0, width=0.1, color='red', label='MaxPool (↓2×)'),
        mpatches.FancyArrow(0, 0, 0.3, 0, width=0.1, color='blue', label='Upsample (↑2×)'),
        mpatches.FancyArrow(0, 0, 0.3, 0, width=0.1, color='green', 
                           linestyle='--', label='Skip Connection (Concat)')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, 
             framealpha=0.95, bbox_to_anchor=(0.02, 0.98))
    
    # Add title and description
    ax.text(8, 9.7, 'U-Net Architecture for MRI Reconstruction', 
           ha='center', fontsize=16, fontweight='bold')
    
    description = (
        'Base filters: 32 | Depth: 5 levels | Parameters: ~1.9M\n'
        'DoubleConv: [Conv3×3 + BN + ReLU] × 2 | Bilinear upsampling'
    )
    ax.text(8, 0.3, description, ha='center', fontsize=9, 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    # Add operation details
    ops_x = 7.5
    ops_y = 1.5
    operations = [
        'Encoder: Progressive downsampling (5→19 pixels)',
        'Skip Connections: Preserve spatial details',
        'Decoder: Progressive upsampling (19→312 pixels)',
        'Output: 1×1 Conv for final reconstruction'
    ]
    
    for i, op in enumerate(operations):
        ax.text(ops_x, ops_y - i*0.3, f'• {op}', 
               ha='left', fontsize=8, family='monospace')
    
    # Save figure
    plt.tight_layout()
    
    output_file = '../unet_architecture.png'
    output_file_hires = '../unet_architecture_hires.png'
    
    print("\nSaving figures...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure saved: {output_file}")
    
    plt.savefig(output_file_hires, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"✓ High-res figure saved: {output_file_hires}")
    
    plt.close()
    
    print("\n" + "="*60)
    print("Architecture Summary:")
    print("="*60)
    print(f"Input shape:          312×410×1 (magnitude MRI)")
    print(f"Output shape:         312×410×1 (reconstructed MRI)")
    print(f"Base filters:         {base_f}")
    print(f"Encoder filters:      {encoder_filters}")
    print(f"Decoder filters:      {decoder_filters}")
    print(f"Downsampling:         MaxPool2d (stride=2)")
    print(f"Upsampling:           Bilinear interpolation")
    print(f"Skip connections:     4 concatenations")
    print(f"Depth:                5 levels")
    print(f"Estimated parameters: ~1.9M (trainable)")
    print("="*60)
    print("\n✓ U-Net architecture diagram generated successfully!")

if __name__ == "__main__":
    print("="*60)
    print("          U-Net Architecture Diagram Generator")
    print("="*60)
    print()
    generate_unet_architecture()

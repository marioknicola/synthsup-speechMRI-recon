import os
import argparse
import nibabel as nib
import numpy as np
from PIL import Image

def convert_nii_to_png(input_dir, output_dir):
    """
    Converts 2D .nii or .nii.gz files from an input directory to .png
    files in an output directory.
    
    It handles files that are strictly 2D or 3D with a single slice.
    It skips 3D volumes (e.g., 256x256x128) with a warning.
    
    Saves images as 16-bit grayscale PNGs.
    """
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        # Check for .nii or .nii.gz extensions
        if filename.endswith((".nii", ".nii.gz")):
            
            # Construct full file paths
            input_path = os.path.join(input_dir, filename)
            
            # Create output filename by replacing the extension
            output_filename = filename.replace(".nii.gz", ".png").replace(".nii", ".png")
            output_path = os.path.join(output_dir, output_filename)

            try:
                print(f"Processing: {filename}")
                
                # Load the NIfTI file
                nii_img = nib.load(input_path)
                
                # Reorient to a standard (RAS) orientation for consistency
                nii_img = nib.as_closest_canonical(nii_img)

                # Get the image data as a numpy array
                data = nii_img.get_fdata()

                # Check the shape of the data
                slice_data = None
                if data.ndim == 2:
                    # This is a pure 2D file
                    slice_data = data
                elif data.ndim == 3 and data.shape[2] == 1:
                    # This is a 2D file saved as 3D (e.g., shape 256, 256, 1)
                    # Squeeze out the last dimension
                    slice_data = np.squeeze(data, axis=2)
                elif data.ndim == 3 and data.shape[2] > 1:
                    # This is a 3D file (a volume), not a 2D image
                    print(f"  WARNING: {filename} is a 3D volume (shape {data.shape}). Skipping.")
                    continue # Skip to the next file
                else:
                    # Unsupported shape (e.g., 4D fMRI)
                    print(f"  WARNING: {filename} has an unsupported shape {data.shape}. Skipping.")
                    continue

                # --- Image Normalization ---
                # Rotate the image 90 degrees counter-clockwise and flip vertically
                # This often corrects the orientation from NIfTI to standard image
                slice_data = np.flipud(np.rot90(slice_data))
                
                # Normalize the data to 0-1
                # This is a simple min-max scaling
                min_val = np.min(slice_data)
                max_val = np.max(slice_data)
                
                if max_val - min_val == 0:
                    # Handle flat image (all pixels are the same value)
                    # Create a 16-bit zero array
                    uint16_data = np.zeros(slice_data.shape, dtype=np.uint16)
                else:
                    # Normalize to 0.0 - 1.0
                    normalized_data = (slice_data - min_val) / (max_val - min_val)
                    # Scale to 0-65535 and convert to 16-bit unsigned integer
                    uint16_data = (normalized_data * 65535).astype(np.uint16)

                # --- Saving the Image ---
                # Create a PIL image from the numpy array
                # PIL will automatically use 16-bit mode ('I;16') from the np.uint16 dtype
                pil_img = Image.fromarray(uint16_data)
                
                # Save the image as a PNG
                # This will now be a 16-bit grayscale PNG
                pil_img.save(output_path)
                
                print(f"  Successfully saved: {output_path}")

            except Exception as e:
                # Handle corrupted files or other errors
                print(f"  ERROR processing {filename}: {e}")

    print("\nConversion complete.")

def main():
    """
    Main function to parse command-line arguments and run the conversion.
    """
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Convert 2D NIfTI (.nii, .nii.gz) files to 16-bit PNG.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "input_folder",
        type=str,
        help="The path to the folder containing your .nii files."
    )
    
    parser.add_argument(
        "output_folder",
        type=str,
        help="The path to the folder where .png files will be saved.\n(Will be created if it doesn't exist)."
    )
    
    args = parser.parse_args()
    
    # Run the conversion
    convert_nii_to_png(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()


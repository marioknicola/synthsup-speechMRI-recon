import os
import nibabel as nib
import numpy as np

def normalise_nifti(folder_path):
    """
    Normalises NIfTI images in a folder by dividing pixel values by the maximum value in each image.
    Overwrites the original files with the normalised images.

    Args:
        folder_path (str): Path to the folder containing NIfTI files.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            file_path = os.path.join(folder_path, filename)
            try:
                # Load the NIfTI image
                img = nib.load(file_path)
                data = img.get_fdata()

                # Calculate the maximum value in the image
                max_value = np.max(data)

                if max_value == 0:
                    print(f"Warning: Maximum value is 0 in {filename}. Skipping normalisation.")
                    continue

                # Normalise the image data
                normalised_data = data / max_value

                # Create a new NIfTI image with the normalised data
                normalised_img = nib.Nifti1Image(normalised_data, img.affine, img.header)

                # Save the normalised image, overwriting the original
                nib.save(normalised_img, file_path)

                print(f"Normalised and saved: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing NIfTI files: ")
    normalise_nifti(folder_path)
    print("Normalisation process complete.")
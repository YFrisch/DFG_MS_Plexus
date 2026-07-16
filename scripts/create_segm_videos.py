import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import os


def load_nifti(file_path):
    """Load a NIfTI file."""
    img = nib.load(file_path)
    return img.get_fdata(), img.affine


def normalize_volume(volume):
    """Normalize the volume to have values between 0 and 255."""
    min_val = np.min(volume)
    max_val = np.max(volume)
    normalized_volume = (volume - min_val) / (max_val - min_val) * 255
    return normalized_volume.astype(np.uint8)


def create_video_from_slices(image, mask, output_filename, axis=0, fps=10):
    """Create a video from slices of the image and overlay the mask."""
    height, width = image.shape[1], image.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # Determine the number of slices based on the specified axis
    if axis == 0:
        slices = image.shape[0]
        for i in range(slices):
            img_slice = image[i, :, :]
            mask_slice = mask[i, :, :]
            combined = overlay_slices(img_slice, mask_slice)
            out.write(combined)

    elif axis == 1:
        slices = image.shape[1]
        for i in range(slices):
            img_slice = image[:, i, :]
            mask_slice = mask[:, i, :]
            # Rotate 90 degrees counter-clockwise
            img_slice = cv2.rotate(img_slice, cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask_slice = cv2.rotate(mask_slice, cv2.ROTATE_90_COUNTERCLOCKWISE)
            combined = overlay_slices(img_slice, mask_slice)
            out.write(combined)

    elif axis == 2:
        slices = image.shape[2]
        for i in range(slices):
            img_slice = image[:, :, i]
            mask_slice = mask[:, :, i]
            # Rotate 90 degrees clockwise
            img_slice = cv2.rotate(img_slice, cv2.ROTATE_90_CLOCKWISE)
            mask_slice = cv2.rotate(mask_slice, cv2.ROTATE_90_CLOCKWISE)
            combined = overlay_slices(img_slice, mask_slice)
            out.write(combined)

    out.release()


def overlay_slices(img_slice, mask_slice):
    """Overlay the segmentation mask on the image slice."""

    # Convert the mask to a binary image
    mask_slice = mask_slice.astype(np.uint8) * 255

    # Create a color overlay for the mask (e.g., red)
    colored_mask = np.zeros((*mask_slice.shape, 3), dtype=np.uint8)  # Create a 3-channel color mask
    colored_mask[mask_slice == 255] = [0, 0, 255]  # Set mask areas to red (BGR format)

    # Manually convert grayscale image to BGR without color conversion
    img_slice_bgr = np.zeros((*img_slice.shape, 3), dtype=np.uint8)
    img_slice_bgr[..., 0] = img_slice  # Blue channel
    img_slice_bgr[..., 1] = img_slice  # Green channel
    img_slice_bgr[..., 2] = img_slice  # Red channel

    # Combine the images: use an appropriate alpha for the mask overlay
    alpha = 0.5  # Adjust transparency as needed
    overlay = cv2.addWeighted(img_slice_bgr, 1, colored_mask, alpha, 0)

    return overlay


def main(mri_path, mask_path, output_dir):
    """Main function to create videos for each axis."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load MRI and mask
    image, _ = load_nifti(mri_path)
    image = normalize_volume(image)
    mask, _ = load_nifti(mask_path)

    # Create videos for each axis
    create_video_from_slices(image, mask, os.path.join(output_dir, 'axial_video.mp4'), axis=0)
    create_video_from_slices(image, mask, os.path.join(output_dir, 'coronal_video.mp4'), axis=1)
    create_video_from_slices(image, mask, os.path.join(output_dir, 'sagittal_video.mp4'), axis=2)


if __name__ == "__main__":
    mri_path = "/home/yfrisch_locale/DATA/dfg_plexus/T1s/Subject010_T1.nii"  # Update with your MRI scan path
    mask_path = "/home/yfrisch_locale/DATA/dfg_plexus/plexus_segmentation_resampled/Subject010.nii.gz"  # Update with your mask path
    output_dir = "../segmentation_resampled_subject010_videos"  # Directory to save the output videos
    main(mri_path, mask_path, output_dir)

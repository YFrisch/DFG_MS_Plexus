import sys
from pathlib import Path

import click
import nibabel as nib
import numpy as np
from nibabel import Nifti1Image
from scipy.ndimage import zoom
from tqdm import tqdm

# SOURCE_IMG_DIR = "F:/DATA/dfg_plexus/HC_T1s___freesurfer___reorientated_padded/"
# SOURCE_IMG_DIR = '/media/yannik/Intenso/DATA/dfg_plexus/T1s/'
# SOURCE_IMG_DIR = "/mnt/Intenso/DATA/dfg_plexus/T1s___float32___harmonized/"
SOURCE_IMG_DIR = "/mnt/Intenso/DATA/dfg_plexus/HC_T1s___freesurfer___reorientated_padded___harmonized/"

# SOURCE_MASK_DIR = "F:/DATA/dfg_plexus/nnUNet/nnUNet_inference_results/Dataset002_DFGFinetuned_3d_fullres_full_new_hc"
# SOURCE_MASK_DIR = '/media/yannik/Intenso/DATA/dfg_plexus/nnUNet/nnUNet_inference_results/Dataset002_DFGFinetuned_3d_fullres_full_ms'
# SOURCE_MASK_DIR = "/mnt/Intenso/DATA/dfg_plexus/nnUNet/nnUNet_inference_results/Dataset002_DFGFinetuned_3d_fullres_full_ms/"
SOURCE_MASK_DIR = "/mnt/Intenso/DATA/dfg_plexus/nnUNet/nnUNet_inference_results/Dataset002_DFGFinetuned_3d_fullres_full_new_hc/"

# OUTPUT_IMG_DIR = "F:/DATA/dfg_plexus/HC_T1s___freesurfer___reorientated_padded___cropped/"
# OUTPUT_IMG_DIR = '/media/yannik/Intenso/DATA/dfg_plexus/T1s___cropped/'
# OUTPUT_IMG_DIR = "/mnt/Intenso/DATA/dfg_plexus/T1s___float32___harmonized___cropped/"
OUTPUT_IMG_DIR = "/mnt/Intenso/DATA/dfg_plexus/HC_T1s___freesurfer___reorientated_padded___harmonized___cropped/"

# OUTPUT_MASK_DIR = "F:/DATA/dfg_plexus/nnUNet/nnUNet_inference_results/Dataset002_DFGFinetuned_3d_fullres_full_new_hc_cropped"
# OUTPUT_MASK_DIR = '/media/yannik/Intenso/DATA/dfg_plexus/nnUNet/nnUNet_inference_results/Dataset002_DFGFinetuned_3d_fullres_full_ms_cropped'
# OUTPUT_MASK_DIR = "/mnt/Intenso/DATA/dfg_plexus/nnUNet/nnUNet_inference_results/Dataset002_DFGFinetuned_3d_fullres_full_ms_cropped/"
OUTPUT_MASK_DIR = "/mnt/Intenso/DATA/dfg_plexus/nnUNet/nnUNet_inference_results/Dataset002_DFGFinetuned_3d_fullres_full_new_hc_cropped/"

TARGET_SIZE = (32, 32, 32)


def try_file_matching(img_path: Path, mask_path: Path):

    print("\nThis script will match the following files for cropping the images with masks:")
    print(f"Image: {img_path}")
    print(f"Mask: {mask_path}")

    if not click.confirm("\nIs this the intended behaviour? Continue?"):
        print("Operation cancelled by user.")
        sys.exit(0)


def resample_to_target(img: nib.Nifti1Image,
                       target_size: tuple[int, int, int],
                       order: int = 3) -> nib.Nifti1Image:
    data = img.get_fdata()
    current_shape = np.array(data.shape)
    target_size = np.array(target_size)

    zoom_factors = target_size / current_shape
    resampled = zoom(data, zoom_factors, order=order)

    # Original voxel spacing (mm/voxel)
    old_spacing = nib.affines.voxel_sizes(img.affine)
    # New spacing should shrink according to zoom factor
    new_spacing = old_spacing / zoom_factors

    # Keep same orientation and origin, but update scale (voxel spacing)
    new_affine = img.affine.copy()
    for i in range(3):
        new_affine[:3, i] = (
            new_affine[:3, i] / np.linalg.norm(new_affine[:3, i]) * new_spacing[i]
        )

    return nib.Nifti1Image(resampled, new_affine, img.header)


@click.command()
@click.option('--source_img_dir', default=SOURCE_IMG_DIR,
              help='Directory containing source image NIfTI files.')
@click.option('--source_mask_dir', default=SOURCE_MASK_DIR,
              help='Directory containing source mask NIfTI files for cropping.')
@click.option('--output_img_dir', default=OUTPUT_IMG_DIR,
              help='Directory to save cropped NIfTI image files.')
@click.option('--output_mask_dir', default=OUTPUT_MASK_DIR,
              help='Directory to save cropped NIfTI mask files.')
@click.option('--target_size', default=TARGET_SIZE, type=(int, int, int),
              help='Target (D, H, W) for resampling cropped NIfTI images.')
def main(source_img_dir, source_mask_dir, output_img_dir, output_mask_dir, target_size):

    source_img_dir = Path(source_img_dir)
    source_mask_dir = Path(source_mask_dir)
    output_img_dir = Path(output_img_dir)
    output_mask_dir = Path(output_mask_dir)

    assert source_img_dir.exists(), f"\n{source_img_dir} does not exist."
    assert source_mask_dir.exists(), f"\n{source_mask_dir} does not exist."

    source_img_paths = sorted(source_img_dir.glob("*.nii*"))
    assert len(source_img_paths) > 0, f"\n{source_img_dir} does not contain any NIfTI files."

    source_mask_paths = sorted(source_mask_dir.glob("*.nii*"))
    assert len(source_mask_paths) > 0, f"\n{source_mask_dir} does not contain any NIfTI files."

    if len(source_img_paths) != len(source_mask_paths):
        print('\nWarning: Number of image and mask files do not match.')

    try_file_matching(source_img_paths[0], source_mask_paths[0])

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Global BB for cropping, instead of per-sample BBs?

    for (source_img_path, source_mask_path) in tqdm(zip(source_img_paths, source_mask_paths),
                                                    total=len(source_img_paths)):

        img = nib.load(str(source_img_path))
        mask = nib.load(str(source_mask_path))
        img_data = img.get_fdata()
        mask_data = mask.get_fdata()

        nonzero = np.nonzero(mask_data)
        if len(nonzero[0]) == 0:
            print(f"Warning:⚠️ Empty mask for {source_mask_path.name}, skipping.")
            continue

        min_coords = np.min(nonzero, axis=1)
        max_coords = np.max(nonzero, axis=1) + 1  # include endpoint

        cropped_img = img_data[
            min_coords[0]:max_coords[0],
            min_coords[1]:max_coords[1],
            min_coords[2]:max_coords[2]
        ]

        cropped_mask = mask_data[
            min_coords[0]:max_coords[0],
            min_coords[1]:max_coords[1],
            min_coords[2]:max_coords[2]
        ]

        # Adjust affine to account for cropping
        new_affine = img.affine.copy()
        new_affine[:3, 3] += np.dot(img.affine[:3, :3], min_coords)

        cropped_img_nii = Nifti1Image(cropped_img, new_affine, img.header)
        cropped_mask_nii = Nifti1Image(cropped_mask, new_affine, img.header)

        if target_size is not None:
            cropped_resampled_img_nii = resample_to_target(cropped_img_nii, target_size, order=3)
            cropped_resampled_mask_nii = resample_to_target(cropped_mask_nii, target_size, order=3)
        else:
            cropped_resampled_img_nii = cropped_img_nii
            cropped_resampled_mask_nii = cropped_mask_nii

        out_img_path = output_img_dir / source_img_path.name
        out_mask_path = output_mask_dir / source_mask_path.name
        nib.save(cropped_resampled_img_nii, str(out_img_path))
        nib.save(cropped_resampled_mask_nii, str(out_mask_path))

    print(f"\n✅\t Cropped NIfTI images saved to: {output_img_dir}")
    print(f"\t Cropped NIfTI masks saved to: {output_mask_dir}")

if __name__ == '__main__':
    main()



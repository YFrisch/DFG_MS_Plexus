""" This script converts NIfTI files to MGZ format,
    applies intensity normalization,
    and converts the corrected files back to NIfTI format.

    The output are normalized NIfTI files with 8-bit depth.
"""


import os
import subprocess

import click

from tqdm import tqdm

SOURCE_DIR = "/media/yannik/Seagate Portable Drive/dfg_plexus/HC_T1s/"
OUTPUT_DIR = "/media/yannik/Seagate Portable Drive/dfg_plexus/HC_T1s___freesurfer/"

@click.command()
@click.option('--source_dir', default=SOURCE_DIR, help='Directory containing source NIfTI files.')
@click.option('--output_dir', default=OUTPUT_DIR, help='Directory to save converted NIfTI files.')
@click.option('--tmp_dir', default="/tmp/mgz_conversion/" , help='Temporary directory for MGZ files.')
def convert_freesurfer(source_dir, output_dir, tmp_dir):

    # Make sure output and tmp directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # Loop through all NIfTI files
    for fname in tqdm(os.listdir(source_dir)):
        if fname.endswith(".nii") or fname.endswith(".nii.gz"):
            base = os.path.splitext(fname)[0].replace(".nii", "")  # handle .nii.gz too

            nii_path = os.path.join(source_dir, fname)
            mgz_path = os.path.join(tmp_dir, f"{base}.mgz")
            norm_mgz_path = os.path.join(tmp_dir, f"{base}_nu.mgz")
            norm_nii_path = os.path.join(output_dir, f"{base}.nii")

            print(f"Processing: {fname}")

            # Step 1: Convert to MGZ
            subprocess.run(["mri_convert", nii_path, mgz_path], check=True)

            """
            # Step 2: Run mri_nu_correct.mni
            subprocess.run([
                "mri_nu_correct.mni",
                # "--no-rescale",
                "--i", mgz_path,
                "--o", norm_mgz_path,
                "--ants-n4",
                "--n", "1",
                "--proto-iters", "1000",
                "--distance", "50"
            ], check=True)
            """
            subprocess.run([
                "mri_normalize",
                mgz_path,
                norm_mgz_path,
            ])

            # Step 3: Convert back to NIfTI
            subprocess.run(["mri_convert", norm_mgz_path, norm_nii_path], check=True)

            os.remove(mgz_path)
            os.remove(norm_mgz_path)

    print("✅ Done! All files processed.")


if __name__ == "__main__":
    convert_freesurfer()

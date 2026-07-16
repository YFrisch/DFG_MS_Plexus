import os
from pathlib import Path

import click
import SimpleITK as sitk
from tqdm import tqdm


@click.command()
@click.option('--src', type=click.Path(exists=True))
@click.option('--tgt', type=click.Path(exists=False))
def cast_volumes_to_float32(src, tgt):
    
    """ Casts all nifti files in 'src' to Float32. """

    Path(tgt).mkdir(parents=True, exist_ok=True)

    filepaths = [os.path.join(src, f) for f in os.listdir(src) 
                 if f.endswith('.nii') or f.endswith('.nii.gz')]

    for idx, path in enumerate(tqdm(filepaths, desc="Converting to Float32: ")):

        img = sitk.ReadImage(path)
        
        # sitk.Cast retains all spatial metadata (spacing, origin, direction)
        img_float = sitk.Cast(img, sitk.sitkFloat32)
        
        out_path = os.path.join(tgt, os.path.basename(path))
        sitk.WriteImage(img_float, out_path)

if __name__ == "__main__":
    cast_volumes_to_float32()

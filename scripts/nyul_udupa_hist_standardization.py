import os
from pathlib import Path

import click
import torchio as tio
from tqdm import tqdm

@click.command()
@click.option('--site1', type=click.Path(exists=True))  # 
@click.option('--site2', type=click.Path(exists=True))
@click.option('--tgt1', type=click.Path(exists=False))
@click.option('--tgt2', type=click.Path(exists=False))
def run_hist_st(site1, site2, tgt1, tgt2):
    
    Path(tgt1).mkdir(parents=True, exist_ok=True)
    Path(tgt2).mkdir(parents=True, exist_ok=True)

    site1_paths = [os.path.join(site1, f) for f in os.listdir(site1) if f.endswith(('.nii', '.nii.gz'))]
    site2_paths = [os.path.join(site2, f) for f in os.listdir(site2) if f.endswith(('.nii', '.nii.gz'))]

    print("\nTraining landmarks on site 1...")

    landmarks_file = "site1_landmarks.npy"
    landmarks = tio.HistogramStandardization.train(
        images_paths=site1_paths,
        output_path=landmarks_file,
        masking_function=lambda x: x > 0

    )
    print(f"Landmarks trained!")

    print("\nApplying Nyúl-Udupa Standardization to all images...")

    transform = tio.HistogramStandardization({'t1': landmarks})

    for path in tqdm(site1_paths, desc="Site 1: "):
        subject = tio.Subject(t1=tio.ScalarImage(path))
        harmonized_subject = transform(subject)

        filename = os.path.basename(path)
        out_path = os.path.join(tgt1, filename)
        harmonized_subject.t1.save(out_path)

    for path in tqdm(site2_paths, desc="Site 2: "):
        subject = tio.Subject(t1=tio.ScalarImage(path))
        harmonized_subject = transform(subject)

        filename = os.path.basename(path)
        out_path = os.path.join(tgt2, filename)
        harmonized_subject.t1.save(out_path)

    print("Dataset harmonization complete.")

if __name__ == "__main__":
    run_hist_st()

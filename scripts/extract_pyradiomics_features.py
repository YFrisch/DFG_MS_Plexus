from pathlib import Path

import click
import pandas as pd
from radiomics import featureextractor
from natsort import natsorted
from tqdm import tqdm


# ROOT = Path("/media/yannik/Intenso/DATA/dfg_plexus/")
ROOT = Path("/mnt/Intenso/DATA/dfg_plexus/")
# IMG_DIR_HC =  ROOT / "HC_T1s___freesurfer___reorientated_padded/"
IMG_DIR_HC = ROOT / "HC_T1s___freesurfer___reorientated_padded___harmonized/"
# IMG_DIR_MS = ROOT / "T1s/"
IMG_DIR_MS = ROOT / "T1s___float32___harmonized/"
MASK_DIR_HC = ROOT / "nnUNet/nnUNet_inference_results/Dataset002_DFGFinetuned_3d_fullres_full_new_hc/"
MASK_DIR_MS = ROOT / "nnUNet/nnUNet_inference_results/Dataset002_DFGFinetuned_3d_fullres_full_ms/"

@click.command()
@click.option('--img_dir', type=click.Path(exists=True), default=IMG_DIR_MS)
@click.option('--mask_dir', type=click.Path(exists=True), default=MASK_DIR_MS)
@click.option('--pyradiomics_conf', type=click.Path(exists=True))
@click.option('--target_filename', type=click.Path())
def extract_pyradiomics_features(
    img_dir,
    mask_dir,
    pyradiomics_conf,
    target_filename
):

    print(f"{img_dir=}")
    print(f"{mask_dir=}")

    extractor = featureextractor.RadiomicsFeatureExtractor(pyradiomics_conf)

    all_features = pd.DataFrame()

    for img_path, mask_path in tqdm(
        zip(
            natsorted(img_dir.glob("*.nii")),
            natsorted(mask_dir.glob("*.nii.gz")),
        ),
        desc=f"Extracting PyRadiomics features from '{img_dir.name}/'",
        total=len(list(img_dir.glob("*.nii"))),
    ):
        """
        subject_id = mask_path.stem.removesuffix(".nii")
        features = extractor.execute(str(img_path), str(mask_path))

        features_row = pd.DataFrame([features], index=[subject_id])
        all_features = pd.concat([all_features, features_row], axis=0)
        """
        subject_id = mask_path.stem.removesuffix(".nii")

        # 1. Get the raw dictionary containing 0-d arrays and diagnostics
        raw_features = extractor.execute(str(img_path), str(mask_path))

        # 2. THE FIX: Filter and cast to standard Python floats immediately
        clean_features = {}
        for key, value in raw_features.items():
            if key.startswith("original_"):
                clean_features[key] = float(value)

        # 3. Pandas now receives perfect, pure float data
        features_row = pd.DataFrame([clean_features], index=[subject_id])
        
        # Note: pd.concat in a loop can get slow for huge datasets, but for ~600 it's fine.
        all_features = pd.concat([all_features, features_row], axis=0)

    """ 
    df = all_features.copy()

    # Drop diagnostics / metadata columns
    diag_cols = [c for c in df.columns if c.startswith("diagnostics_")]
    df = df.drop(columns=diag_cols, errors="ignore")

    # Keep only radiomics feature columns (for Original image type)
    # They start with 'original_' if we use `imageType: Original`
    feature_cols = [c for c in df.columns if c.startswith("original_")]
    radiomics_df = df[feature_cols].copy()

    # Force everything to numeric (anything non-numeric becomes NaN)
    radiomics_df = radiomics_df.apply(
        pd.to_numeric,
        errors="coerce",  # invalid values -> NaN instead of breaking
    )

    print(f"Extracted {radiomics_df.shape} features")
    radiomics_df.to_csv(target_filename, sep=";")
    """

    print(f"Extracted {all_features.shape} features")
    all_features.to_csv(target_filename, sep=";")


if __name__ == "__main__":
    extract_pyradiomics_features()

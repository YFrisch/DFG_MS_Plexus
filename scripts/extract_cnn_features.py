from pathlib import Path

import torch
import click
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

from dfg_ms_plexus.model import ResNet3DMRI


CROPPED_IMG_DIR_MS = '/media/yannik/Intenso/DATA/dfg_plexus/T1s___cropped/'

CROPPED_IMG_DIR_HC = "/media/yannik/Intenso/DATA/dfg_plexus/HC_T1s___freesurfer___reorientated_padded___cropped/"

CNN_CHCKPT = '/media/yannik/Intenso/DATA/dfg_plexus/cnn_clf___best_val_f1.pth'

FT_CSV_PATH = '/media/yannik/Intenso/DATA/dfg_plexus/radiomics_features___combined.csv'

TARGET_NAME = '/media/yannik/Intenso/DATA/dfg_plexus/radiomics_and_cnn_features___combined.csv'


@click.command()
@click.option('--cropped_hc', default=CROPPED_IMG_DIR_HC,
              help='Directory containing the cropped HC image NIfTI files.')
@click.option('--cropped_ms', default=CROPPED_IMG_DIR_MS,
              help='Directory containing the cropped MS image NIfTI files.')
@click.option('--cnn', default=CNN_CHCKPT,
              help='Path to CNN checkpoint (.pth).')
@click.option('--ft_csv_path', default=FT_CSV_PATH,
              help='Path to existing features (will be combined with the cnn features).')
@click.option('--target_name', default=TARGET_NAME,
              help='Target name (path) for output dataframe file (.csv).')
@click.option('--device', default='cpu',
              help='Device to use (cpu, cuda).')
def main(cropped_hc, cropped_ms, cnn, ft_csv_path, target_name, device):

    cropped_hc = Path(cropped_hc)
    assert cropped_hc.exists(), f"\n{cropped_hc} does not exist."
    cropped_ms = Path(cropped_ms)
    assert cropped_ms.exists(), f"\n{cropped_ms} does not exist."
    cnn_path = Path(cnn)
    assert cnn_path.exists(), f"\n{cnn_path} does not exist."
    target_path = Path(target_name)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # TODO: Save and load model config
    cnn_model = ResNet3DMRI(
        num_classes=3,
        base_filters=8,
        dropout=0.2
    ).to(device)
    state_dict = torch.load(cnn_path, map_location='cpu', weights_only=True)
    cnn_model.load_state_dict(state_dict)
    cnn_model.eval()

    cropped_img_paths = sorted(cropped_hc.glob("*.nii*")) + sorted(cropped_ms.glob("*.nii*"))

    data = []
    for cropped_img_path in tqdm(cropped_img_paths):

        id = cropped_img_path.stem.split("_")[0]

        img_data = nib.load(cropped_img_path).get_fdata()

        # TODO: Use DS class instead for consistent pre-processing etc.
        vol_mean = np.mean(img_data)
        vol_std = np.std(img_data)
        img_data_normalized = (img_data - vol_mean) / (vol_std + 1e-6)

        img_tensor = torch.from_numpy(img_data_normalized).float()
        img_tensor = img_tensor.view((1, 1, *img_tensor.shape)).to(device)

        with torch.no_grad():
            # pred = cnn_model(img_tensor).argmax(dim=1).squeeze().cpu().numpy()
            ft = cnn_model.extract_features(img_tensor).squeeze().cpu().numpy()
            data.append([id, *ft.T])

    n_features = len(data[0]) - 1  # minus patID
    cnn_cols = [f"cnn_{i + 1}" for i in range(n_features)]
    cols = ["patID"] + cnn_cols
    cnn_df = pd.DataFrame(data, columns=cols)
    cnn_df["patID"] = cnn_df["patID"].astype(str)

    if ft_csv_path is not None:
        ft_dataframe = pd.read_csv(ft_csv_path, delimiter=";")
        ft_dataframe["patID"] = ft_dataframe["patID"].astype(str)
        ft_merged = ft_dataframe.merge(cnn_df, on="patID", how="left")
    else:
        ft_merged = cnn_df

    print(f"{ft_merged.head()=}")
    print(f"{ft_merged.shape=}")

    ft_merged.to_csv(target_path, index=False, sep=";")


if __name__ == "__main__":
    main()

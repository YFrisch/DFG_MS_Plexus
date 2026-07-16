import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

from .labels import get_labels_hc_ms


class MRIDataset(Dataset):

    def __init__(self,
                 hc_mri_dir: Path = Path('/media/yannik/Intenso/DATA/dfg_plexus/HC_T1s___freesurfer___reorientated_padded___cropped'),
                 ms_mri_dir: Path = Path('/media/yannik/Intenso/DATA/dfg_plexus/T1s___cropped'),
                 annotations: Path = Path('/media/yannik/Intenso/DATA/dfg_plexus/radiomics_features___combined.csv'),
                 sample_ids: Path | None = None,  # Path('/media/yannik/Intenso/DATA/dfg_plexus/train_idx.pkl')
                 normalize_per_volume: bool = False,
                 class_mapping=get_labels_hc_ms,
                 return_fts: bool = False
                 ):

        super().__init__()

        self.normalize_per_volume = normalize_per_volume
        self.class_mapping = class_mapping
        self.return_fts = return_fts

        self.mri_paths = list(hc_mri_dir.glob('*.nii*')) + list(ms_mri_dir.glob('*.nii*'))
        self.annotations = pd.read_csv(annotations, delimiter=";")
        # Truncate mri path list for missing annotations
        pat_col = next((c for c in self.annotations.columns if c.lower() == "patid"), None)
        ann_ids = set(self.annotations[pat_col].astype(str).str.strip().tolist())
        if pat_col is None:
            raise KeyError("patID column not found in annotations")
        self.mri_paths = [p for p in self.mri_paths if str(self.get_pat_id(p)).strip() in ann_ids]

        raw_labels = self.annotations['label'].astype(int)
        mapped_labels, self.class_mapping = class_mapping(raw_labels)
        self.annotations['label_mapped'] = pd.Series(mapped_labels, index=self.annotations.index).astype(int)

        # TODO: split by patient ids instead
        # Subset by indices (Assuming order of mri_paths == order of annotations)
        if sample_ids is not None:
            with open(sample_ids, 'rb') as f:
                sample_idx = list(map(int, pickle.load(f)))

            self.annotations = self.annotations.iloc[sample_idx].reset_index(drop=True)

            # Lookup dictionary mapping patID -> file path
            path_dict = {str(self.get_pat_id(p)).strip(): p for p in self.mri_paths}

            # Rebuilding the mri_paths list so it perfectly matches the annotations order
            ordered_paths = []
            for pid in self.annotations['patID'].astype(str).str.strip():
                if pid in path_dict:
                    ordered_paths.append(path_dict[pid])
                else:
                    raise FileNotFoundError(f"Missing MRI file for patID: {pid}")

            self.mri_paths = ordered_paths

    @staticmethod
    def get_pat_id(path: Path) -> str:
        return str(path.stem.split('_')[0])

    def __len__(self):
        return len(self.mri_paths)

    def __getitem__(self, idx):

        # TODO: (Optionally) return feature vector too

        mri_path = self.mri_paths[idx]

        pat_id = self.get_pat_id(mri_path)

        match = self.annotations.loc[self.annotations['patID'].astype(str) == pat_id]
        if self.return_fts:
            fts = match.drop(
                columns=['label', 'label_mapped', 'patID', 'DWI (0no,1yes)', 'LesionVolume', 'DiseaseDuration', 'EDSS']
            ).iloc[0].astype(np.float32).to_numpy()
            fts = torch.from_numpy(fts)

        if match.empty:
            raise ValueError(f"No annotations for patID={pat_id} (file: {mri_path.name})")

        class_id = int(match['label_mapped'].iloc[0])
        class_id = torch.tensor(class_id, dtype=torch.long)

        mri_img = nib.load(str(mri_path))
        mri_img_data = mri_img.get_fdata()[None, :]

        """ Data stats from nnUNet training:
        
            "max": 105.0,
            "mean": 54.49951171875,
            "median": 53.0,
            "min": 17.0,
            "percentile_00_5": 33.0,
            "percentile_99_5": 86.0,
            "std": 11.935002326965332
        
        """

        if self.normalize_per_volume:
            vol_mean = np.mean(mri_img_data)
            vol_std = np.std(mri_img_data)
            mri_img_data_normalized = (mri_img_data - vol_mean) / (vol_std + 1e-6)
        else:
            mri_img_data_cliped = np.clip(mri_img_data, 33, 86)
            mri_img_data_normalized = (mri_img_data_cliped - 54.4995) / 11.9350

        mri_img_data = torch.from_numpy(mri_img_data_normalized).float()

        if not self.return_fts:
            return mri_img_data, class_id
        else:
            return mri_img_data, class_id, fts

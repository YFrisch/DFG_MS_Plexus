import time
import os
from glob import glob

import click
import torch
import nibabel as nib
from natsort import natsorted
from tqdm import tqdm
from torchmetrics.functional import jaccard_index, average_precision
from torchmetrics.functional.segmentation import dice_score

# gt_dir = '/home/yfrisch_locale/nnUNet/nnUNet_raw/Dataset002_DFGFinetuned/labelsTs/'
# gt_dir = '/media/yannik/Seagate Portable Drive/dfg_plexus/plexus_segmentation___resampled_finetuned/'
GT_DIR = '/media/yannik/Intenso/DATA/dfg_plexus/nnUNet/nnUNet_raw/Dataset002_DFGFinetuned/labelsTs/'

# pred_dir = '/home/yfrisch_locale/nnUNet/nnUNet_inference_results/Dataset002_DFGFinetuned_old_2d_pp/'
# pred_dir = '/home/yfrisch_locale/nnUNet/nnUNet_inference_results/Dataset002_DFGFinetuned_old_2d/'  # (Less samples)
# pred_dir = '/home/yfrisch_locale/nnUNet/nnUNet_inference_results/Dataset002_DFGFinetuned_2d/'
# pred_dir = '/home/yfrisch_locale/nnUNet/nnUNet_inference_results/Dataset002_DFGFinetuned_3d_fullres/'
# pred_dir = '/home/yfrisch_locale/nnUNet/nnUNet_inference_results/Dataset002_DFGFinetuned_3d_fullres_pp/'
# pred_dir = '/media/yannik/Seagate Portable Drive/dfg_plexus/nnUNet_inference_results//'
PRED_DIR = '/media/yannik/Intenso/DATA/dfg_plexus/nnUNet/nnUNet_inference_results/Dataset002_DFGFinetuned_3d_fullres_pp/'


@click.command()
@click.argument('gt_dir', type=click.Path(exists=True), default=GT_DIR)
@click.argument('pred_dir', type=click.Path(exists=True), default=PRED_DIR)
def eval_nnunet(gt_dir, pred_dir):

    assert os.path.isdir(gt_dir), "gt_dir does not exist"
    assert os.path.isdir(pred_dir), "pred_dir does not exist"

    gt_paths = natsorted(glob(os.path.join(gt_dir, '*.nii.gz')))
    pred_paths = natsorted(glob(os.path.join(pred_dir, '*.nii.gz')))

    assert len(gt_paths) == len(pred_paths)

    print("---------- Loading ground truth masks ----------")
    time.sleep(0.1)
    gt_masks = []
    for gt_path in tqdm(gt_paths):
        gt_mask = nib.load(gt_path)
        gt_mask = torch.from_numpy(gt_mask.get_fdata()).unsqueeze(0).to(torch.uint8)
        gt_masks.append(gt_mask)
    gt_masks = torch.stack(gt_masks, dim=0)

    print("---------- Loading predicted masks ----------")
    time.sleep(0.1)
    pred_masks = []
    for pred_path in tqdm(pred_paths):
        pred_mask = nib.load(pred_path)
        pred_mask = torch.from_numpy(pred_mask.get_fdata()).unsqueeze(0).to(torch.float32)
        pred_masks.append(pred_mask)
    pred_masks = torch.stack(pred_masks, dim=0)

    print(f"\n{gt_masks.shape=}, {gt_masks.dtype=}")
    print(f"{pred_masks.shape=}, {pred_masks.dtype=}")

    print("\nDice: ", dice_score(pred_masks.to(torch.uint8), gt_masks,
                                 num_classes=1, average='macro').tolist())
    print("\nmDice: ", dice_score(pred_masks.to(torch.uint8), gt_masks,
                                  num_classes=1, average='macro', aggregation_level='global').item())
    print("mIoU: ", jaccard_index(pred_masks, gt_masks, task='binary').item())
    print("mAP: ", average_precision(pred_masks, gt_masks, task='binary').item())


if __name__ == "__main__":
    eval_nnunet()

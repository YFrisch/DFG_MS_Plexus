""" Script to evaluate 3D .nii.gz segmentation masks."""

import os
import argparse

import nibabel as nib
import torch
import numpy as np
from torchmetrics.functional import dice, jaccard_index, average_precision
from glob import glob
from natsort import natsorted
from tqdm import tqdm


def read_nii(file_path: str) -> np.ndarray:
    """Read a .nii.gz file and return a numpy array."""
    nii = nib.load(file_path)
    return nii.get_fdata()


def evaluate_metrics(pred: torch.Tensor, gt: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """Evaluate Dice Score, Mean IoU, and Mean AP for 3D binary segmentation masks.

    Args:
        pred (torch.Tensor): The predicted segmentation mask.
        gt (torch.Tensor): The ground truth segmentation mask.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Dice Score, Mean IoU, Mean AP.
    """
    gt = gt.long()

    dice_scores = dice(pred, gt)
    mean_iou = jaccard_index(pred, gt, task='binary')
    mean_ap = average_precision(pred, gt, task='binary')

    return dice_scores, mean_iou, mean_ap


def main(args):
    prediction_files = natsorted(glob(os.path.join(args.predictions, '*.nii.gz')))
    ground_truth_files = natsorted(glob(os.path.join(args.ground_truth, '*.nii.gz')))

    # Initialize lists to store metric values
    all_dice_scores = []
    all_mean_iou = []
    all_mean_ap = []

    # Loop through your dataset with a progress bar
    for pred_file, gt_file in tqdm(zip(prediction_files, ground_truth_files), total=len(prediction_files),
                                   desc="Evaluating"):
        pred = read_nii(pred_file)
        gt = read_nii(gt_file)

        pred_tensor = torch.from_numpy(pred).unsqueeze(0).float()
        gt_tensor = torch.from_numpy(gt).unsqueeze(0).float()

        dice_score, mean_iou, mean_ap = evaluate_metrics(pred_tensor, gt_tensor)

        # Append metric values to the lists
        all_dice_scores.append(dice_score.item())
        all_mean_iou.append(mean_iou.item())
        all_mean_ap.append(mean_ap.item())

    # Calculate mean values of the metrics
    mean_dice = np.mean(all_dice_scores)
    mean_iou = np.mean(all_mean_iou)
    mean_ap = np.mean(all_mean_ap)

    # Print mean metric values
    print(f"Mean Dice Score: {mean_dice}")
    print(f"Mean IoU: {mean_iou}")
    print(f"Mean AP: {mean_ap}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate 3D binary segmentation masks.')
    parser.add_argument('--predictions', required=True, help='Folder with predicted .nii.gz files')
    parser.add_argument('--ground_truth', required=True, help='Folder with ground truth .nii.gz files')
    args = parser.parse_args()
    main(args)

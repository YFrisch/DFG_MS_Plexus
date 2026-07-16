""" Compact script to evaluate nnUNetv2 gt vs. predicted segmentation mask folders. """
# TODO: replace argparse with click

import os
import argparse
from glob import glob

import torch
from natsort import natsorted
from torchvision.io import read_image
from torchmetrics.functional import jaccard_index, dice, average_precision


def convert_to_binary_mask(mask: torch.Tensor, num_classes: int, ignore_index: int | None = None) -> torch.Tensor:
    """
    Convert an integer segmentation mask to a binary segmentation mask.

    Parameters:
    mask (torch.Tensor): The integer segmentation mask, shape (B, H, W).
    num_classes (int): The number of classes K.
    ignore_index (int|None):

    Returns:
    binary_mask (torch.Tensor): The binary segmentation mask, shape (B, K, H, W).
    """
    B, H, W = mask.shape
    binary_mask = torch.zeros((B, num_classes, H, W), dtype=mask.dtype, device=mask.device)

    if ignore_index is not None:
        valid_mask = (mask != ignore_index)
        binary_mask = binary_mask.to(torch.long)
        valid_labels = mask * valid_mask.long()  # Ignore the ignore_index for now
        binary_mask.scatter_(1, valid_labels.unsqueeze(1), valid_mask.long().unsqueeze(1))
        binary_mask[:, -1] = (mask == ignore_index).long()  # Handle ignore_index separately
    else:
        binary_mask.scatter_(1, mask.unsqueeze(1), 1)

    return binary_mask


def eval_iou(pred: torch.Tensor, gt: torch.Tensor, ncls: int, ignore_index: int = 255):
    return jaccard_index(pred, gt, task='multiclass', num_classes=ncls, ignore_index=ignore_index)


def eval_ap(pred: torch.Tensor, gt: torch.Tensor, ncls: int, ignore_index: int = 255):
    # Convert predictions from integer (N, ...) to binary (N, C, ...)
    binary_pred = convert_to_binary_mask(pred, num_classes=ncls, ignore_index=ignore_index).float()
    return average_precision(binary_pred, gt, task='multiclass', num_classes=ncls, ignore_index=ignore_index)


def eval_dice(pred: torch.Tensor, gt: torch.Tensor, ncls: int, ignore_index: int = 255):
    return dice(pred, gt, average='macro', num_classes=ncls)


def main(gt: str, pred: str, ncls: int, ignore_index: int|None = None):

    assert os.path.isdir(gt), f"{gt} not a dir."
    assert os.path.isdir(pred), f"{pred} not a dir."

    gt_paths = natsorted(glob(os.path.join(gt, '*.png')))
    pred_paths = natsorted(glob(os.path.join(pred, '*.png')))

    assert len(gt_paths) == len(pred_paths), f"{len(gt_paths)} GT labels != {len(pred_paths)} pred labels"

    gt_masks = []
    for gt_path in gt_paths:
        gt_mask = read_image(gt_path).squeeze(0).to(torch.int64)  # (H, W)
        gt_masks.append(gt_mask)
    gt_masks = torch.stack(gt_masks, dim=0)

    pred_masks = []
    for pred_path in pred_paths:
        pred_mask = read_image(pred_path).squeeze(0).to(torch.int64)  # (H, W)
        pred_masks.append(pred_mask)
    pred_masks = torch.stack(pred_masks, dim=0)

    assert pred_masks.shape[0] == gt_masks.shape[0]
    assert pred_masks.shape[-2:] == gt_masks.shape[-2:]

    print("mDice: ", eval_dice(pred_masks, gt_masks, ncls, ignore_index))
    print("mIoU: ", eval_iou(pred_masks, gt_masks, ncls, ignore_index))
    print("mAP: ", eval_ap(pred_masks, gt_masks, ncls, ignore_index))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str)
    parser.add_argument('--pred', type=str)
    parser.add_argument('--ncls', type=int)
    parser.add_argument('--ignore', type=int, default=None)
    args = parser.parse_args()
    main(gt=args.gt, pred=args.pred, ncls=args.ncls, ignore_index=args.ignore)

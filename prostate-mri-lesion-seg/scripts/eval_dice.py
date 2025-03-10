# Script that loads two nifti files and evaluates them using the DICE metric.
# Usage: python eval.py <path_to_ground_truth> <path_to_prediction>

import os
import nibabel as nib
import argparse
import torch
from monai.metrics import DiceMetric

def main():
    parser = argparse.ArgumentParser(description="Evaluate two nifti files using the DICE metric.")
    parser.add_argument("ground_truth", type=str, help="Path to the ground truth nifti file")
    parser.add_argument("prediction", type=str, help="Path to the prediction nifti file")

    args = parser.parse_args()

    gt_path = os.path.abspath(os.path.expanduser(os.path.expandvars(args.ground_truth)))
    pred_path = os.path.abspath(os.path.expanduser(os.path.expandvars(args.prediction)))

    gt = nib.load(gt_path).get_fdata()
    pred = nib.load(pred_path).get_fdata()

    # Print the dimensions of the loaded images
    print(f"Ground truth dimensions: {gt.shape}")
    print(f"Prediction dimensions: {pred.shape}")

    # Convert numpy arrays to PyTorch tensors and add batch and channel dimensions
    gt_tensor = torch.tensor(gt).unsqueeze(0).unsqueeze(0)
    pred_tensor = torch.tensor(pred).unsqueeze(0).unsqueeze(0)

    # Compute the DICE metric
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice = dice_metric(pred_tensor, gt_tensor)
    print(f"DICE: {dice.item()}")

if __name__ == "__main__":
    main()

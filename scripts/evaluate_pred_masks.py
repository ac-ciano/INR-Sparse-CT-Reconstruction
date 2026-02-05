"""Evaluate geometric segmentation metrics for predicted tumour masks.

This script computes standard 3D segmentation evaluation metrics comparing
predicted tumour masks against ground truth consensus masks. Evaluation is
performed for both SIREN-based and interpolation-based predictions.

Computed metrics:
    - Volume (mm^3) for predicted and ground truth masks
    - Absolute Relative Volume Error (ARVE)
    - Dice Similarity Coefficient (DSC)
    - Hausdorff Distance (HD) in physical units

Usage:
    python evaluate_pred_masks.py

Inputs:
    - Predicted masks: data/pred_masks_{siren|interpolation}/{nodule_id}_pred_mask.npy
    - Ground truth masks: data/processed/{nodule_id}_mask.npy
    - Metadata: data/processed/{nodule_id}_metadata.json

Outputs:
    - logs/vol_{SIREN|interpolation}_{mode}/volume_metrics_{threshold}.csv
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import csv
from datetime import datetime
import numpy as np
from pathlib import Path

from tumour_tomography.geometric_metrics import (
    absolute_relative_volume_error,
    dice_coefficient,
    hausdorff_distance,
    compute_volume,
)
from tumour_tomography import config as cfg

# Define specific log directories using config base path
LOG_SIREN_DIR = cfg.LOGS_DIR / f"vol_SIREN_{cfg.ROI_GENERATION_MODE}"
LOG_INTERP_DIR = cfg.LOGS_DIR / f"vol_interpolation_{cfg.ROI_GENERATION_MODE}"


def _load_metadata(nodule_id: str) -> dict:
    """Load nodule metadata from JSON file.
    
    Args:
        nodule_id: Unique identifier for the nodule.
        
    Returns:
        Dictionary containing nodule metadata including slice_thickness,
        pixel_spacing, shape, and malignancy information.
    """
    meta_path = cfg.PROCESSED_DIR / f"{nodule_id}_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _evaluate_set(pred_dir: Path, log_dir: Path):
    """Evaluate all predicted masks in a directory against ground truth.
    
    Computes geometric segmentation metrics (volume, ARVE, Dice, Hausdorff)
    for each predicted mask and saves results to a CSV file.
    
    Args:
        pred_dir: Path to directory containing predicted mask files
            (*_pred_mask.npy format).
        log_dir: Path to directory where evaluation CSV will be saved.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"volume_metrics_{cfg.HU_DENSITY_THRESHOLD}.csv"

    pred_files = sorted(pred_dir.glob("*_pred_mask.npy"))

    rows = []
    for pred_path in pred_files:
        nodule_id = pred_path.name.replace("_pred_mask.npy", "")
        gt_path = cfg.PROCESSED_DIR / f"{nodule_id}_mask.npy"
        meta_path = cfg.PROCESSED_DIR / f"{nodule_id}_metadata.json"

        if not gt_path.exists():
            print(f"✗  Missing GT mask: {gt_path}")
            continue
        if not meta_path.exists():
            print(f"✗  Missing metadata: {meta_path}")
            continue

        pred_mask = np.load(pred_path)
        gt_mask = np.load(gt_path)
        meta = _load_metadata(nodule_id)

        slice_thickness = float(meta["slice_thickness"])
        pixel_spacing = float(meta["pixel_spacing"])
        voxel_volume = slice_thickness * (pixel_spacing ** 2)
        spacing = (slice_thickness, pixel_spacing, pixel_spacing)

        v_pred = compute_volume(pred_mask, voxel_volume)
        v_gt = compute_volume(gt_mask, voxel_volume)

        arve = absolute_relative_volume_error(pred_mask, gt_mask, voxel_volume)
        dice = dice_coefficient(pred_mask, gt_mask)
        hd = hausdorff_distance(pred_mask, gt_mask, spacing)

        rows.append(
            [
                nodule_id,
                v_gt,
                v_pred,
                arve,
                dice,
                hd,
            ]
        )

    # Write CSV
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "nodule_id",
                "gt_volume",
                "pred_volume",
                "arve",
                "dice",
                "hausdorff_distance",
            ]
        )
        writer.writerows(rows)

    print(f"Saved log: {log_path}")


def main():
    """Run evaluation pipeline for both SIREN and interpolation predictions.
    
    Evaluates predicted tumour masks from both reconstruction methods
    against ground truth consensus masks, saving separate CSV logs
    for each method.
    """
    print("Evaluating SIREN predictions...")
    _evaluate_set(cfg.PRED_MASKS_SIREN_DIR, LOG_SIREN_DIR)

    print("Evaluating interpolation predictions...")
    _evaluate_set(cfg.PRED_MASKS_INTERPOLATION_DIR, LOG_INTERP_DIR)


if __name__ == "__main__":
    main()
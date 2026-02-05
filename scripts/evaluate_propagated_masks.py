"""Evaluate propagated tumour masks on validation slices.

This script computes geometric segmentation metrics for propagated masks
against ground truth consensus masks. Evaluation is performed only on
validation slices (those not used during training) to assess the quality
of mask propagation from training slices to held-out slices.

Both SIREN-based and interpolation-based propagated masks are evaluated
with metrics saved to separate CSV files for comparison.

Computed metrics:
    - Dice Similarity Coefficient (DSC)
    - Hausdorff Distance (HD) in physical units
    - Volume and Absolute Relative Volume Error (ARVE)

Usage:
    python evaluate_propagated_masks.py

Inputs:
    - Ground truth masks: data/processed/{nodule_id}_mask.npy
    - SIREN propagated: data/propagated_masks_siren/{nodule_id}_propagated_mask.npy
    - Interpolation propagated: data/propagated_masks_interpolation/{nodule_id}_propagated_mask.npy

Outputs:
    - logs/propagated_evaluation/siren_propagated_{std_mult}.csv
    - logs/propagated_evaluation/interp_propagated_{std_mult}.csv
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

import numpy as np
import csv
from datetime import datetime
from pathlib import Path
from tumour_tomography.geometric_metrics import dice_coefficient, hausdorff_distance, compute_volume, absolute_relative_volume_error
from tumour_tomography import config as cfg


std_mult = cfg.MASK_PROPAGATION_STD_MULTIPLIER

def evaluate_on_validation_slices_only():
    """Evaluate propagated mask quality on validation slices only.
    
    Computes segmentation metrics (Dice, Hausdorff Distance, Volume, ARVE)
    comparing propagated masks to ground truth on the 4/5 of slices that
    were not used for training (validation slices are all slices except
    every 5th slice).
    
    Processes all nodules with both SIREN and interpolation propagated masks
    and saves per-nodule metrics to CSV files in the propagated evaluation
    logs directory.
    """
    siren_log = cfg.PROPAGATED_EVAL_LOGS_DIR / f"siren_propagated_{std_mult}.csv"
    interp_log = cfg.PROPAGATED_EVAL_LOGS_DIR / f"interp_propagated_{std_mult}.csv"
    
    siren_rows = []
    interp_rows = []
    
    for mask_path in sorted(cfg.PROCESSED_DIR.glob("*_mask.npy")):
        nodule_id = mask_path.stem.replace("_mask", "")
        
        # Load ground truth
        gt_mask = np.load(mask_path)
        D = gt_mask.shape[0]
        
        # Validation slices only (not every 5th)
        val_mask_1d = np.ones(D, dtype=bool)
        val_mask_1d[::5] = False
        
        # Load propagated predictions
        siren_pred_path = cfg.PROPAGATED_MASKS_SIREN_DIR / f"{nodule_id}_propagated_mask.npy"
        interp_pred_path = cfg.PROPAGATED_MASKS_INTERPOLATION_DIR / f"{nodule_id}_propagated_mask.npy"
        
        if not siren_pred_path.exists() or not interp_pred_path.exists():
            continue
        
        siren_pred = np.load(siren_pred_path)
        interp_pred = np.load(interp_pred_path)
        
        # Extract validation slices only
        gt_val = gt_mask[val_mask_1d]
        siren_val = siren_pred[val_mask_1d]
        interp_val = interp_pred[val_mask_1d]
        
        # Load metadata for spacing
        meta_path = cfg.PROCESSED_DIR / f"{nodule_id}_metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        
        slice_thickness = float(meta["slice_thickness"])
        pixel_spacing = float(meta["pixel_spacing"])
        spacing = (slice_thickness, pixel_spacing, pixel_spacing)
        voxel_volume = slice_thickness * (pixel_spacing ** 2)
        
        # Compute metrics on validation slices only
        siren_dice = dice_coefficient(siren_val, gt_val)
        interp_dice = dice_coefficient(interp_val, gt_val)
        
        siren_hd = hausdorff_distance(siren_val, gt_val, spacing)
        interp_hd = hausdorff_distance(interp_val, gt_val, spacing)
        
        siren_vol = compute_volume(siren_val, voxel_volume)
        interp_vol = compute_volume(interp_val, voxel_volume)
        gt_vol = compute_volume(gt_val, voxel_volume)

        siren_arve = absolute_relative_volume_error(siren_val, gt_val, voxel_volume)
        interp_arve = absolute_relative_volume_error(interp_val, gt_val, voxel_volume)
        
        siren_rows.append([nodule_id, gt_vol, siren_vol, siren_dice, siren_hd, siren_arve])
        interp_rows.append([nodule_id, gt_vol, interp_vol, interp_dice, interp_hd, interp_arve])
    
    # Write CSVs
    for path, rows in [(siren_log, siren_rows), (interp_log, interp_rows)]:
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['nodule_id', 'gt_volume', 'pred_volume', 'dice', 'hausdorff_distance', 'arve'])
            writer.writerows(rows)
        print(f"Saved: {path}")


if __name__ == "__main__":
    evaluate_on_validation_slices_only()
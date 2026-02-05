"""Generate and evaluate simple morphological mask propagation.

This script evaluates the simple morphological interpolation approach for
mask propagation. It generates propagated masks using linear interpolation
between training slices and computes segmentation metrics against ground truth.

The evaluation is restricted to validation slices (those not used in training)
to assess how well the propagation method generalizes mask shapes to unseen
slices.

Computed metrics:
    - Dice Similarity Coefficient (DSC)
    - Hausdorff Distance (HD) in physical units
    - Absolute Relative Volume Error (ARVE)

Usage:
    python propagate_simple_gen-eval.py

Inputs:
    - Ground truth masks: data/processed/{nodule_id}_mask.npy
    - Metadata: data/processed/{nodule_id}_metadata.json
    - Training logs: logs/training/*.csv (to identify trained nodules)

Outputs:
    - Console output with per-nodule and aggregate metrics
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from pathlib import Path
from tumour_tomography.geometric_metrics import dice_coefficient, hausdorff_distance, absolute_relative_volume_error
from tumour_tomography import config as cfg
from scripts.propagate_mask import propagate_mask_simple_morph


def main():
    """Evaluate simple morphological mask propagation on trained nodules.
    
    Generates propagated masks using linear interpolation between training
    slices and evaluates segmentation quality on validation slices against
    ground truth. Computes Dice, Hausdorff Distance, and ARVE metrics.
    
    Only evaluates nodules that have been trained (identified by presence
    of training CSV logs).
    """
    # Storage for metrics
    dice_scores = []
    hd_scores = []
    arve_scores = []
    
    TRAINING_LOG_DIR = Path("/home/noxiusk/Desktop/Oncology/logs/training")

    def training_nodule_id(name: str) -> str:
        return name.split("_20", 1)[0]

    trained_nodule_ids = {
        training_nodule_id(p.stem)
        for p in TRAINING_LOG_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv"
    }
    trained_nodule_ids.discard("")  # drop empty strings from failed parses

    all_vol_files = sorted(cfg.PROCESSED_DIR.glob("*_vol.npy"))
    vol_files = [
        v for v in all_vol_files
        if v.stem.replace("_vol", "") in trained_nodule_ids
    ]
    
    print(f"Found {len(vol_files)} nodules to process\n")
    
    for vol_path in vol_files:
        nodule_id = vol_path.stem.replace("_vol", "")
        
        # Load ground truth mask
        mask_path = cfg.PROCESSED_DIR / f"{nodule_id}_mask.npy"
        if not mask_path.exists():
            print(f"Skipping {nodule_id}: missing mask")
            continue
        
        consensus_mask = np.load(mask_path)
        D = consensus_mask.shape[0]
        
        # Same training slices as in propagate_mask.py
        train_slices = np.arange(0, D, 5)
        
        # Generate propagated mask using simple morph
        propagated = propagate_mask_simple_morph(consensus_mask, train_slices, iterations=1)
        
        # Validation slices only (not every 5th)
        val_mask_1d = np.ones(D, dtype=bool)
        val_mask_1d[::5] = False
        
        gt_val = consensus_mask[val_mask_1d]
        pred_val = propagated[val_mask_1d]
        
        # Load metadata for spacing
        meta_path = cfg.PROCESSED_DIR / f"{nodule_id}_metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        
        slice_thickness = float(meta["slice_thickness"])
        pixel_spacing = float(meta["pixel_spacing"])
        spacing = (slice_thickness, pixel_spacing, pixel_spacing)
        voxel_volume = slice_thickness * (pixel_spacing ** 2)
        
        # Compute metrics
        dice = dice_coefficient(pred_val, gt_val)
        hd = hausdorff_distance(pred_val, gt_val, spacing)
        arve = absolute_relative_volume_error(pred_val, gt_val, voxel_volume)
        
        # Check for infinite HD (empty masks)
        if np.isinf(hd):
            print(f"  {nodule_id}: Infinite HD (skipping)")
            continue
        
        dice_scores.append(dice)
        hd_scores.append(hd)
        arve_scores.append(arve)
        
        print(f"  {nodule_id}: Dice={dice:.4f}, HD={hd:.4f}, ARVE={arve:.4f}")
    
    # Calculate and print final statistics
    print("\n" + "="*60)
    print("FINAL RESULTS - Simple Morphological Propagation")
    print("="*60)
    print(f"Nodules evaluated: {len(dice_scores)}")
    print(f"\nDice Coefficient:        {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"Hausdorff Distance (mm): {np.mean(hd_scores):.4f} ± {np.std(hd_scores):.4f}")
    print(f"ARVE:                    {np.mean(arve_scores):.4f} ± {np.std(arve_scores):.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
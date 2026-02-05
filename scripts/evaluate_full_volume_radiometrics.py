"""Evaluate full volume reconstruction quality for SIREN and Interpolation.

This script computes radiometric image quality metrics (MSE, MAE, RMSE, PSNR, SSIM)
comparing SIREN and cubic interpolation reconstructions against ground truth
CT volumes. Evaluation is restricted to test slices (slices NOT used for training)
but includes ALL voxels in those slices (not restricted to tumor regions).

Results are printed to console only - no files are written.

Usage:
    python evaluate_full_volume_metrics.py

Inputs:
    - Ground truth volumes: data/processed/{nodule}_vol.npy
    - SIREN reconstructions: data/reconstructed-SIREN/{nodule}_siren_recon.npy
    - Interpolation reconstructions: data/reconstructed-interpolation/{nodule}_cubic_z_recon.npy
"""
import numpy as np
import pandas as pd
import torch
import os
import glob
import sys
from pathlib import Path

# --- Setup paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

from tumour_tomography.radiometric_metrics import calculate_all_metrics

# --- Configuration ---
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SIREN_DIR = PROJECT_ROOT / "data" / "reconstructed-SIREN"
INTERP_DIR = PROJECT_ROOT / "data" / "reconstructed-interpolation"


def main():
    # Get list of nodules from SIREN reconstructions
    recon_files = list(SIREN_DIR.glob("*_siren_recon.npy"))
    nodule_ids = [f.stem.replace("_siren_recon", "") for f in recon_files]
    
    print(f"Found {len(nodule_ids)} nodules to evaluate\n")
    
    if len(nodule_ids) == 0:
        print("No reconstructed volumes found. Exiting.")
        return
    
    # Storage for results
    results = []
    
    # Process each nodule
    for nodule in nodule_ids:
        # Define file paths
        vol_gt_path = PROCESSED_DIR / f"{nodule}_vol.npy"
        vol_siren_path = SIREN_DIR / f"{nodule}_siren_recon.npy"
        vol_interp_path = INTERP_DIR / f"{nodule}_cubic_z_recon.npy"
        
        # Check all files exist
        if not vol_gt_path.exists():
            print(f"Skipping {nodule}: missing ground truth volume")
            continue
        if not vol_siren_path.exists():
            print(f"Skipping {nodule}: missing SIREN reconstruction")
            continue
        if not vol_interp_path.exists():
            print(f"Skipping {nodule}: missing interpolation reconstruction")
            continue
        
        # Load full volumes
        vol_gt = np.load(vol_gt_path)
        vol_siren = np.load(vol_siren_path)
        vol_interp = np.load(vol_interp_path)
        
        # Verify shapes match
        if vol_gt.shape != vol_siren.shape or vol_gt.shape != vol_interp.shape:
            print(f"Skipping {nodule}: shape mismatch - GT:{vol_gt.shape}, SIREN:{vol_siren.shape}, Interp:{vol_interp.shape}")
            continue
        
        # Create train/test split (same as in data_loader.py)
        n_slices = vol_gt.shape[2]
        train_mask = np.zeros(n_slices, dtype=bool)
        train_mask[::5] = True  # Every 5th slice used for training
        test_mask = ~train_mask  # Slices NOT used for training
        
        test_slice_indices = np.where(test_mask)[0]
        
        # Collect voxels from test slices only (all voxels in those slices)
        gt_voxels = []
        siren_voxels = []
        interp_voxels = []
        
        for z in test_slice_indices:
            gt_voxels.append(vol_gt[:, :, z].flatten())
            siren_voxels.append(vol_siren[:, :, z].flatten())
            interp_voxels.append(vol_interp[:, :, z].flatten())
        
        if len(gt_voxels) == 0:
            print(f"Skipping {nodule}: no test slices available")
            continue
        
        # Concatenate all test voxels
        gt_flat = np.concatenate(gt_voxels)
        siren_flat = np.concatenate(siren_voxels)
        interp_flat = np.concatenate(interp_voxels)
        
        # Convert to tensors
        gt_tensor = torch.from_numpy(gt_flat).float()
        siren_tensor = torch.from_numpy(siren_flat).float()
        interp_tensor = torch.from_numpy(interp_flat).float()
        
        # Normalize to [0, 1] for PSNR/SSIM calculation
        min_val = gt_tensor.min()
        max_val = gt_tensor.max()
        data_range = max_val - min_val
        
        # Handle edge case of constant volume
        if data_range == 0:
            print(f"Skipping {nodule}: constant volume (no intensity variation)")
            continue
        
        gt_norm = (gt_tensor - min_val) / data_range
        siren_norm = (siren_tensor - min_val) / data_range
        interp_norm = (interp_tensor - min_val) / data_range
        
        # Calculate all metrics
        siren_metrics = calculate_all_metrics(siren_norm, gt_norm, max_val=1.0)
        interp_metrics = calculate_all_metrics(interp_norm, gt_norm, max_val=1.0)
        
        # Store per-nodule results
        results.append({
            'nodule_id': nodule,
            'n_voxels': len(gt_flat),
            'n_test_slices': len(test_slice_indices),
            'volume_shape': vol_gt.shape,
            'siren_mse': siren_metrics['mse'],
            'siren_mae': siren_metrics['mae'],
            'siren_rmse': siren_metrics['rmse'],
            'siren_psnr': siren_metrics['psnr'],
            'siren_ssim': siren_metrics['ssim'],
            'interp_mse': interp_metrics['mse'],
            'interp_mae': interp_metrics['mae'],
            'interp_rmse': interp_metrics['rmse'],
            'interp_psnr': interp_metrics['psnr'],
            'interp_ssim': interp_metrics['ssim'],
        })
        
        print(f"Processed {nodule}: {len(test_slice_indices)} test slices, {len(gt_flat):,} voxels")
    
    # Check if we have any results
    if len(results) == 0:
        print("\nNo nodules were successfully processed. Exiting.")
        return
    
    # --- Summary Calculation and Printing ---
    metrics = ['mse', 'mae', 'rmse', 'psnr', 'ssim']
    summary_data = []
    
    for metric in metrics:
        siren_values = [r[f'siren_{metric}'] for r in results]
        interp_values = [r[f'interp_{metric}'] for r in results]
        
        # Filter out inf values for mean/std calculation (can happen with PSNR for perfect reconstruction)
        siren_finite = [v for v in siren_values if np.isfinite(v)]
        interp_finite = [v for v in interp_values if np.isfinite(v)]
        
        if len(siren_finite) > 0:
            siren_mean = np.mean(siren_finite)
            siren_std = np.std(siren_finite, ddof=1) if len(siren_finite) > 1 else 0.0
        else:
            siren_mean = float('nan')
            siren_std = float('nan')
        
        if len(interp_finite) > 0:
            interp_mean = np.mean(interp_finite)
            interp_std = np.std(interp_finite, ddof=1) if len(interp_finite) > 1 else 0.0
        else:
            interp_mean = float('nan')
            interp_std = float('nan')
        
        pct_diff = ((siren_mean - interp_mean) / interp_mean * 100) if interp_mean != 0 else float('nan')
        
        summary_data.append({
            "Metric": metric.upper(),
            "SIREN Mean": siren_mean,
            "SIREN Std": siren_std,
            "Interp Mean": interp_mean,
            "Interp Std": interp_std,
            "Improvement % (SIREN vs Interp)": pct_diff
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Determine which metrics are better when lower
    lower_is_better = ['mse', 'mae', 'rmse']
    for idx, row in summary_df.iterrows():
        metric_name = row['Metric'].lower()
        improvement = row['Improvement % (SIREN vs Interp)']
        
        is_lower_better = metric_name in lower_is_better
        
        if pd.notna(improvement):
            if (is_lower_better and improvement < 0) or (not is_lower_better and improvement > 0):
                summary_df.loc[idx, 'Result'] = 'SIREN Better'
            else:
                summary_df.loc[idx, 'Result'] = 'Interp Better'
    
    print("\n" + "=" * 80)
    print("FULL VOLUME RECONSTRUCTION METRICS (Test Slices Only)")
    print("=" * 80)
    print(f"Based on {len(results)} nodule volumes")
    print(f"Evaluated on slices NOT in training set (excluded every 5th slice)")
    print(f"All voxels in test slices included (no tumor mask filtering)\n")
    print(summary_df.to_string(index=False, float_format="%.6f"))
    print("=" * 80)


if __name__ == "__main__":
    main()

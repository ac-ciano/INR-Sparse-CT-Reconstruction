"""Evaluate intra-tumour radiometric reconstruction quality.

This script computes radiometric image quality metrics (MSE, MAE, RMSE, PSNR, SSIM)
comparing SIREN and cubic interpolation reconstructions against ground truth
CT volumes. Evaluation is restricted to voxels within the tumour mask on
held-out test slices (slices not used for training).

The script produces:
    - Per-patient metrics with mean and std saved to CSV
    - Summary statistics with mean/std across all patients

Usage:
    python evaluate_intra_tumour_radiometrics.py

Inputs:
    - Ground truth volumes: data/processed/{patient_id}_vol.npy
    - Tumour masks: data/processed/{patient_id}_mask.npy
    - SIREN reconstructions: data/reconstructed-SIREN/{patient_id}_siren_recon.npy
    - Interpolation reconstructions: data/reconstructed-interpolation/{patient_id}_cubic_z_recon.npy

Outputs:
    - logs/intra_tumour_radiometric_metrics.csv
"""
import numpy as np
import pandas as pd
import torch
import os
import glob
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Add project root to path
sys.path.insert(0, str(PROJECT_ROOT))
from tumour_tomography.radiometric_metrics import calculate_all_metrics

# --- Configuration ---
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SIREN_DIR = PROJECT_ROOT / "data" / "reconstructed-SIREN"
INTERP_DIR = PROJECT_ROOT / "data" / "reconstructed-interpolation"
OUTPUT_CSV = PROJECT_ROOT / "logs" / "intra_tumour_radiometric_metrics.csv"


# Find all reconstructed nodules
recon_files = glob.glob(os.path.join(INTERP_DIR, "*_cubic_z_recon.npy"))
nodule_ids = [os.path.basename(f).replace("_cubic_z_recon.npy", "") for f in recon_files]

print(f"Found {len(nodule_ids)} nodules to evaluate\n")

# --- Storage for results ---
results = []

for nodule in nodule_ids:
    # File paths
    vol_path = PROCESSED_DIR / f"{nodule}_vol.npy"
    mask_path = PROCESSED_DIR / f"{nodule}_mask.npy"
    siren_path = SIREN_DIR / f"{nodule}_siren_recon.npy"
    interp_path = INTERP_DIR / f"{nodule}_cubic_z_recon.npy"
    
    # Check all files exist
    if not all(p.exists() for p in [vol_path, mask_path, siren_path, interp_path]):
        print(f"Skipping {nodule}: missing files")
        continue
    
    # Load data
    vol_gt = np.load(vol_path)
    mask = np.load(mask_path)
    vol_siren = np.load(siren_path)
    vol_interp = np.load(interp_path)
    
    n_slices = vol_gt.shape[2]
    train_mask = np.zeros(n_slices, dtype=bool)
    train_mask[::5] = True
    test_mask = ~train_mask  # Slices NOT used for training
    
    test_slice_indices = np.where(test_mask)[0]
    
    # Collect voxels from test slices within tumor mask
    gt_voxels = []
    siren_voxels = []
    interp_voxels = []
    
    for z in test_slice_indices:
        slice_mask = mask[:, :, z] > 0
        if np.sum(slice_mask) == 0:
            continue
        
        gt_voxels.append(vol_gt[:, :, z][slice_mask])
        siren_voxels.append(vol_siren[:, :, z][slice_mask])
        interp_voxels.append(vol_interp[:, :, z][slice_mask])
    
    if len(gt_voxels) == 0:
        print(f"Skipping {pid}: no valid test voxels in tumor region")
        continue
    
    # Concatenate all voxels
    gt_all = np.concatenate(gt_voxels)
    siren_all = np.concatenate(siren_voxels)
    interp_all = np.concatenate(interp_voxels)
    
    # Convert to torch tensors
    gt_tensor = torch.tensor(gt_all, dtype=torch.float32)
    siren_tensor = torch.tensor(siren_all, dtype=torch.float32)
    interp_tensor = torch.tensor(interp_all, dtype=torch.float32)
    
    # Normalize to [0, 1] for PSNR calculation
    min_val = gt_tensor.min()
    max_val = gt_tensor.max()
    data_range = max_val - min_val
    
    gt_norm = (gt_tensor - min_val) / data_range
    siren_norm = (siren_tensor - min_val) / data_range
    interp_norm = (interp_tensor - min_val) / data_range
    
    # Calculate metrics
    siren_metrics = calculate_all_metrics(siren_norm, gt_norm, max_val=1.0)
    interp_metrics = calculate_all_metrics(interp_norm, gt_norm, max_val=1.0)

    
    # Store results
    results.append({
        'nodule_id': nodule,
        'n_test_voxels': len(gt_all),
        'n_test_slices': len(gt_voxels),
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
    
    print(f"Processed {nodule}: {len(gt_voxels)} test slices, {len(gt_all)} voxels")

# --- Create DataFrame and save ---
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nResults saved to {OUTPUT_CSV}")

# --- Summary Calculation and Printing ---
metrics = ['mse', 'mae', 'rmse', 'psnr', 'ssim']
summary_data = []

for m in metrics:
    siren_col = f'siren_{m}'
    interp_col = f'interp_{m}'
    
    siren_series = df[siren_col]
    interp_series = df[interp_col]
    
    siren_mean = siren_series.mean()
    siren_std = siren_series.std(ddof=1)
    interp_mean = interp_series.mean()
    interp_std = interp_series.std(ddof=1)
    
    pct_diff = ((siren_mean - interp_mean) / interp_mean * 100) if interp_mean != 0 else float('nan')
    
    row = {
        "Metric": m.upper(),
        "SIREN Mean": siren_mean,
        "SIREN Std": siren_std,
        "Interp Mean": interp_mean,
        "Interp Std": interp_std,
        "Improvement % (SIREN vs Interp)": pct_diff
    }
    
    if f'{siren_col}_std' in df.columns:
        row["SIREN Intra-Patient Std"] = df[f'{siren_col}_std'].mean()
        row["Interp Intra-Patient Std"] = df[f'{interp_col}_std'].mean()
    
    summary_data.append(row)

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
print("INTRA-TUMOUR RADIOMETRIC METRICS (Test Slices Only)")
print("=" * 80)
print(f"Based on {len(df)} patient models.")
print(f"Evaluated on slices NOT in training set (excluded every 5th slice)")
print(f"Only voxels inside ground truth tumor mask considered\n")
print(summary_df.to_string(index=False, float_format="%.6f"))
print("=" * 80)
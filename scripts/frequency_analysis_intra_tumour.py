"""Intra-tumour frequency analysis for SIREN and interpolation reconstructions.

This script computes frequency-domain statistics and visualizations comparing
SIREN and cubic interpolation reconstructions against ground truth CT volumes.
Analysis is restricted to voxels within tumour masks (intra-tumour analysis).

The script produces:
    - Aggregate statistics (mean, std, kurtosis) across all nodules
    - Statistical significance tests (paired t-tests)
    - Two-panel visualization:
        A. Hounsfield Unit distribution (KDE)
        B. Power Spectral Density (radial average)

Usage:
    python frequency_analysis_intra_tumour.py

Inputs:
    - Ground truth volumes: data/processed/{nodule_id}_vol.npy
    - Tumour masks: data/processed/{nodule_id}_mask.npy
    - SIREN reconstructions: data/reconstructed-SIREN/{nodule_id}_siren_recon.npy
    - Interpolation reconstructions: data/reconstructed-interpolation/{nodule_id}_cubic_z_recon.npy

Outputs:
    - intra_tumour_frequency_analysis.pdf
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import stats
import pandas as pd
from scipy.stats import gaussian_kde
from numpy.fft import fft2, fftshift
from tumour_tomography import config as cfg


def main():
    """Main execution function for intra-tumour frequency analysis."""
    # Find all reconstructed nodules
    recon_files = glob.glob(os.path.join(cfg.RECONSTRUCTED_INTERPOLATION_DIR, "*_cubic_z_recon.npy"))
    nodule_ids = [os.path.basename(f).replace("_cubic_z_recon.npy", "") for f in recon_files]

    print(f"Found {len(nodule_ids)} nodules to analyze\n")

    # Storage for statistics
    results = {
        'nodule_id': [],
        'gt_mean': [], 'gt_std': [], 'gt_kurtosis': [],
        'interp_mean': [], 'interp_std': [], 'interp_kurtosis': [],
        'siren_mean': [], 'siren_std': [], 'siren_kurtosis': [],
        'interp_std_ratio': [], 'siren_std_ratio': []
    }

    # Process each nodule
    for nodule in nodule_ids:
        vol_path = os.path.join(cfg.PROCESSED_DIR, f"{nodule}_vol.npy")
        mask_path = os.path.join(cfg.PROCESSED_DIR, f"{nodule}_mask.npy")
        recon_interp_path = os.path.join(cfg.RECONSTRUCTED_INTERPOLATION_DIR, f"{nodule}_cubic_z_recon.npy")
        recon_siren_path = os.path.join(cfg.RECONSTRUCTED_SIREN_DIR, f"{nodule}_siren_recon.npy")

        # Check if all files exist
        if not all(os.path.exists(p) for p in [vol_path, mask_path, recon_interp_path, recon_siren_path]):
            print(f"Skipping {nodule}: missing files")
            continue
        
        # Load data
        vol = np.load(vol_path)
        mask = np.load(mask_path)
        recon_interpolation = np.load(recon_interp_path)
        recon_SIREN = np.load(recon_siren_path)
        
        # Extract masked regions
        mask_bool = mask > 0
        vol_masked = vol[mask_bool]
        recon_interp_masked = recon_interpolation[mask_bool]
        recon_siren_masked = recon_SIREN[mask_bool]
        
        # Calculate statistics
        gt_kurt = np.mean((vol_masked - np.mean(vol_masked))**4) / np.std(vol_masked)**4
        interp_kurt = np.mean((recon_interp_masked - np.mean(recon_interp_masked))**4) / np.std(recon_interp_masked)**4
        siren_kurt = np.mean((recon_siren_masked - np.mean(recon_siren_masked))**4) / np.std(recon_siren_masked)**4
        
        # Store results
        results['nodule_id'].append(nodule)
        results['gt_mean'].append(np.mean(vol_masked))
        results['gt_std'].append(np.std(vol_masked))
        results['gt_kurtosis'].append(gt_kurt)
        results['interp_mean'].append(np.mean(recon_interp_masked))
        results['interp_std'].append(np.std(recon_interp_masked))
        results['interp_kurtosis'].append(interp_kurt)
        results['siren_mean'].append(np.mean(recon_siren_masked))
        results['siren_std'].append(np.std(recon_siren_masked))
        results['siren_kurtosis'].append(siren_kurt)
        results['interp_std_ratio'].append(np.std(recon_interp_masked) / np.std(vol_masked))
        results['siren_std_ratio'].append(np.std(recon_siren_masked) / np.std(vol_masked))

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Statistical analysis
    print("=" * 80)
    print("AGGREGATE STATISTICS (Mean ± Std across all nodules)")
    print("=" * 80)

    print("\nGround Truth:")
    print(f"  Mean Intensity: {df['gt_mean'].mean():.2f} ± {df['gt_mean'].std():.2f} HU")
    print(f"  Mean Std Dev: {df['gt_std'].mean():.2f} ± {df['gt_std'].std():.2f} HU")
    print(f"  Mean Kurtosis: {df['gt_kurtosis'].mean():.2f} ± {df['gt_kurtosis'].std():.2f}")

    print("\nInterpolation:")
    print(f"  Mean Intensity: {df['interp_mean'].mean():.2f} ± {df['interp_mean'].std():.2f} HU")
    print(f"  Mean Std Dev: {df['interp_std'].mean():.2f} ± {df['interp_std'].std():.2f} HU")
    print(f"  Mean Kurtosis: {df['interp_kurtosis'].mean():.2f} ± {df['interp_kurtosis'].std():.2f}")

    print("\nSIREN:")
    print(f"  Mean Intensity: {df['siren_mean'].mean():.2f} ± {df['siren_mean'].std():.2f} HU")
    print(f"  Mean Std Dev: {df['siren_std'].mean():.2f} ± {df['siren_std'].std():.2f} HU")
    print(f"  Mean Kurtosis: {df['siren_kurtosis'].mean():.2f} ± {df['siren_kurtosis'].std():.2f}")

    print("\n" + "=" * 80)
    print("VARIANCE PRESERVATION")
    print("=" * 80)
    print(f"Interpolation Std / GT Std: {df['interp_std_ratio'].mean():.3f} ± {df['interp_std_ratio'].std():.3f}")
    print(f"SIREN Std / GT Std: {df['siren_std_ratio'].mean():.3f} ± {df['siren_std_ratio'].std():.3f}")

    # Paired statistical tests
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS (Paired t-tests)")
    print("=" * 80)

    # Test if std ratios differ from 1.0
    t_stat_interp, p_val_interp = stats.ttest_1samp(df['interp_std_ratio'], 1.0)
    t_stat_siren, p_val_siren = stats.ttest_1samp(df['siren_std_ratio'], 1.0)

    print(f"\nInterpolation std ratio vs 1.0: t={t_stat_interp:.3f}, p={p_val_interp:.4f}")
    print(f"SIREN std ratio vs 1.0: t={t_stat_siren:.3f}, p={p_val_siren:.4f}")

    # Compare interpolation vs SIREN
    t_stat_comp, p_val_comp = stats.ttest_rel(df['interp_std_ratio'], df['siren_std_ratio'])
    print(f"\nInterpolation vs SIREN std ratio: t={t_stat_comp:.3f}, p={p_val_comp:.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ============================================================================
    # PANEL A: Kernel Density Estimation (KDE) Plot
    # ============================================================================
    # Concatenate all HU values across all nodules for smoother KDE
    all_gt_hu = []
    all_interp_hu = []
    all_siren_hu = []

    for nodule in nodule_ids:
        vol_path = os.path.join(cfg.PROCESSED_DIR, f"{nodule}_vol.npy")
        mask_path = os.path.join(cfg.PROCESSED_DIR, f"{nodule}_mask.npy")
        recon_interp_path = os.path.join(cfg.RECONSTRUCTED_INTERPOLATION_DIR, f"{nodule}_cubic_z_recon.npy")
        recon_siren_path = os.path.join(cfg.RECONSTRUCTED_SIREN_DIR, f"{nodule}_siren_recon.npy")

        if not all(os.path.exists(p) for p in [vol_path, mask_path, recon_interp_path, recon_siren_path]):
            continue
        
        vol = np.load(vol_path)
        mask = np.load(mask_path)
        recon_interpolation = np.load(recon_interp_path)
        recon_SIREN = np.load(recon_siren_path)
        
        mask_bool = mask > 0
        all_gt_hu.extend(vol[mask_bool].flatten())
        all_interp_hu.extend(recon_interpolation[mask_bool].flatten())
        all_siren_hu.extend(recon_SIREN[mask_bool].flatten())

    # Convert to arrays
    all_gt_hu = np.array(all_gt_hu)
    all_interp_hu = np.array(all_interp_hu)
    all_siren_hu = np.array(all_siren_hu)

    # Create KDE
    hu_range = np.linspace(-600, 200, 500)
    kde_gt = gaussian_kde(all_gt_hu, bw_method='scott')
    kde_interp = gaussian_kde(all_interp_hu, bw_method='scott')
    kde_siren = gaussian_kde(all_siren_hu, bw_method='scott')

    # Plot KDE
    axes[0].plot(hu_range, kde_gt(hu_range), color='black', linewidth=2, label='Ground Truth', linestyle='--', alpha=0.8)
    axes[0].plot(hu_range, kde_interp(hu_range), color='blue', linewidth=2, label='Interpolation', alpha=0.8)
    axes[0].plot(hu_range, kde_siren(hu_range), color='red', linewidth=2, label='SIREN', alpha=0.8)
    axes[0].set_xlabel('Hounsfield Units (HU)', fontsize=11)
    axes[0].set_ylabel('Probability density', fontsize=11)
    axes[0].set_title('A. HU distribution', fontsize=12, fontweight='bold')
    axes[0].legend(frameon=True, fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(-600, 200)

    # ============================================================================
    # PANEL B: Power Spectral Density (PSD)
    # ============================================================================
    # Compute 2D FFT for each slice and average across all nodules
    psd_gt_all = []
    psd_interp_all = []
    psd_siren_all = []

    for nodule in nodule_ids:
        vol_path = os.path.join(cfg.PROCESSED_DIR, f"{nodule}_vol.npy")
        mask_path = os.path.join(cfg.PROCESSED_DIR, f"{nodule}_mask.npy")
        recon_interp_path = os.path.join(cfg.RECONSTRUCTED_INTERPOLATION_DIR, f"{nodule}_cubic_z_recon.npy")
        recon_siren_path = os.path.join(cfg.RECONSTRUCTED_SIREN_DIR, f"{nodule}_siren_recon.npy")

        if not all(os.path.exists(p) for p in [vol_path, mask_path, recon_interp_path, recon_siren_path]):
            continue
        
        vol = np.load(vol_path)
        mask = np.load(mask_path)
        recon_interpolation = np.load(recon_interp_path)
        recon_SIREN = np.load(recon_siren_path)
        
        # Process each slice where mask exists
        for z in range(vol.shape[2]):
            if np.sum(mask[:, :, z]) == 0:
                continue
                
            # Apply mask to extract ROI
            slice_gt = vol[:, :, z] * (mask[:, :, z] > 0)
            slice_interp = recon_interpolation[:, :, z] * (mask[:, :, z] > 0)
            slice_siren = recon_SIREN[:, :, z] * (mask[:, :, z] > 0)
            
            # Compute 2D FFT
            fft_gt = fftshift(fft2(slice_gt))
            fft_interp = fftshift(fft2(slice_interp))
            fft_siren = fftshift(fft2(slice_siren))
            
            # Power spectrum
            psd_gt = np.abs(fft_gt) ** 2
            psd_interp = np.abs(fft_interp) ** 2
            psd_siren = np.abs(fft_siren) ** 2
            
            # Radial average
            center = np.array(psd_gt.shape) // 2
            y, x = np.indices(psd_gt.shape)
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
            
            max_r = min(center)
            radial_psd_gt = np.bincount(r.ravel(), psd_gt.ravel()) / np.bincount(r.ravel())
            radial_psd_interp = np.bincount(r.ravel(), psd_interp.ravel()) / np.bincount(r.ravel())
            radial_psd_siren = np.bincount(r.ravel(), psd_siren.ravel()) / np.bincount(r.ravel())
            
            psd_gt_all.append(radial_psd_gt[:max_r])
            psd_interp_all.append(radial_psd_interp[:max_r])
            psd_siren_all.append(radial_psd_siren[:max_r])

    # Average PSDs
    min_len = min([len(p) for p in psd_gt_all])
    psd_gt_avg = np.mean([p[:min_len] for p in psd_gt_all], axis=0)
    psd_interp_avg = np.mean([p[:min_len] for p in psd_interp_all], axis=0)
    psd_siren_avg = np.mean([p[:min_len] for p in psd_siren_all], axis=0)

    # Frequency axis (normalized)
    freqs = np.arange(len(psd_gt_avg))

    # Plot PSD on log scale
    axes[1].plot(freqs, np.log10(psd_gt_avg + 1e-10), color='black', linewidth=2, label='Ground Truth', linestyle='--', alpha=0.8)
    axes[1].plot(freqs, np.log10(psd_interp_avg + 1e-10), color='blue', linewidth=2, label='Interpolation', alpha=0.8)
    axes[1].plot(freqs, np.log10(psd_siren_avg + 1e-10), color='red', linewidth=2, label='SIREN', alpha=0.8)
    axes[1].set_xlabel('Spatial frequency $k$ (cycles/pixel)', fontsize=11)
    axes[1].set_ylabel('$\log_{10}$(power)', fontsize=11)
    axes[1].set_title('B. Power Spectral Density', fontsize=12, fontweight='bold')
    axes[1].legend(frameon=True, fontsize=9)
    axes[1].grid(alpha=0.3)

    # ============================================================================
    # Save figure
    # ============================================================================
    plt.tight_layout()
    file_name = 'intra_tumour_frequency_analysis.pdf'
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n[OK] Figure saved as {file_name}")


if __name__ == "__main__":
    main()
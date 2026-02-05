"""Geometric analysis for voxel ROI expansion evaluation.

This script evaluates mask reconstruction quality using HU-based thresholding
across different threshold values. It compares SIREN and cubic interpolation
reconstructions against morphological baseline performance using dilated ROI analysis.

The script produces:
    - Aggregate statistics (mean, SEM, std) for ARVE, Dice, and Hausdorff distance
    - Three-panel visualization comparing methods across threshold values:
        A. Absolute Relative Volume Error (ARVE)
        B. Dice Similarity Coefficient
        C. Hausdorff Distance

Usage:
    python geometric_analysis_voxel_expansion.py

Inputs:
    - SIREN metrics: logs/vol_SIREN_dilation/volume_metrics_*.csv
    - Interpolation metrics: logs/vol_interpolation_dilation/volume_metrics_*.csv

Outputs:
    - geometric_ROI_expansion_plot.pdf
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import glob
import re


# --- Configuration ---
SIREN_DIR = "logs/vol_SIREN_dilation"
INTERP_DIR = "logs/vol_interpolation_dilation"

# Morphological baseline values
MORPHO_ARVE = 0.0393
MORPHO_ARVE_SEM = 0.0480
MORPHO_DICE = 0.8367
MORPHO_DICE_SEM = 0.0610
MORPHO_HAUSDORFF = 1.9830
MORPHO_HAUSDORFF_SEM = 0.5498

# Plot styling
SIREN_COLOR = 'red'
INTERP_COLOR = 'blue'
MORPHO_COLOR = 'black'
BAND_ALPHA = 0.12


def extract_threshold(filename):
    """Extract the HU threshold number from filename.
    
    Args:
        filename: CSV filename like 'volume_metrics_-300.csv'
    
    Returns:
        int: Threshold value, or None if not found
    """
    match = re.search(r'_(-?\d+)\.csv$', filename)
    if match:
        return int(match.group(1))
    return None


def load_and_average_metrics(directory):
    """Load all CSV files and compute average metrics with SEM.
    
    Args:
        directory: Path to directory containing volume_metrics_*.csv files
    
    Returns:
        pd.DataFrame: DataFrame with threshold-averaged metrics
    """
    pattern = os.path.join(directory, "volume_metrics_*.csv")
    files = glob.glob(pattern)
    
    results = []
    
    for file_path in files:
        threshold = extract_threshold(os.path.basename(file_path))
        if threshold is None:
            continue
            
        df = pd.read_csv(file_path)
        
        # Calculate mean, SEM, and std for each metric
        avg_metrics = {
            'threshold': threshold,
            'arve_mean': df['arve'].mean(),
            'arve_sem': stats.sem(df['arve']),
            'arve_std': df['arve'].std(),
            'dice_mean': df['dice'].mean(),
            'dice_sem': stats.sem(df['dice']),
            'dice_std': df['dice'].std(),
            'hausdorff_mean': df['hausdorff_distance'].mean(),
            'hausdorff_sem': stats.sem(df['hausdorff_distance']),
            'hausdorff_std': df['hausdorff_distance'].std(),
        }
        results.append(avg_metrics)
    
    results = sorted(results, key=lambda x: x['threshold'])
    return pd.DataFrame(results)


def main():
    """Main execution function for geometric ROI expansion analysis."""
    # ============================================================================
    # Load metrics data
    # ============================================================================
    siren_metrics = load_and_average_metrics(SIREN_DIR)
    interp_metrics = load_and_average_metrics(INTERP_DIR)

    print("SIREN Metrics:")
    print(siren_metrics)
    print("\nInterpolation Metrics:")
    print(interp_metrics)

    # ============================================================================
    # Create visualization
    # ============================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # ===== Plot 1: ARVE =====
    ax = axes[0]
    # SIREN
    ax.plot(siren_metrics['threshold'], siren_metrics['arve_mean'], 
            marker='o', linewidth=2, label='SIREN', color=SIREN_COLOR)

    # Interpolation
    ax.plot(interp_metrics['threshold'], interp_metrics['arve_mean'], 
            marker='s', linewidth=2, label='Interpolation', color=INTERP_COLOR)

    # Morphological baseline (mean only, no shaded band)
    ax.axhline(y=MORPHO_ARVE, color=MORPHO_COLOR, linestyle='--', 
               linewidth=2, label=f"Morphological Baseline")

    ax.set_xlabel('Segmentation threshold (HU)', fontsize=11)
    ax.set_ylabel('Absolute relative volume error', fontsize=11)
    ax.set_title('A. Volume error', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ===== Plot 2: Dice Coefficient =====
    ax = axes[1]
    ax.plot(siren_metrics['threshold'], siren_metrics['dice_mean'], 
            marker='o', linewidth=2, label='SIREN', color=SIREN_COLOR)

    ax.plot(interp_metrics['threshold'], interp_metrics['dice_mean'], 
            marker='s', linewidth=2, label='Interpolation', color=INTERP_COLOR)

    ax.axhline(y=MORPHO_DICE, color=MORPHO_COLOR, linestyle='--', 
               linewidth=2, label=f"Morphological Baseline")

    ax.set_xlabel('Segmentation threshold (HU)', fontsize=11)
    ax.set_ylabel('Dice similarity coefficient', fontsize=11)
    ax.set_title('B. Overlap', fontsize=12, fontweight='bold')
    ax.set_ylim(0.4, 0.9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ===== Plot 3: Hausdorff Distance =====
    ax = axes[2]
    ax.plot(siren_metrics['threshold'], siren_metrics['hausdorff_mean'], 
            marker='o', linewidth=2, label='SIREN', color=SIREN_COLOR)

    ax.plot(interp_metrics['threshold'], interp_metrics['hausdorff_mean'], 
            marker='s', linewidth=2, label='Interpolation', color=INTERP_COLOR)

    ax.axhline(y=MORPHO_HAUSDORFF, color=MORPHO_COLOR, linestyle='--', 
               linewidth=2, label=f"Morphological Baseline")

    ax.set_xlabel('Segmentation threshold (HU)', fontsize=11)
    ax.set_ylabel('Hausdorff distance (mm)', fontsize=11)
    ax.set_title('C. Surface distance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Single, horizontal legend above the subplots
    handles_labels = [ax.get_legend_handles_labels() for ax in axes]
    handles = sum((hl[0] for hl in handles_labels), [])
    labels = sum((hl[1] for hl in handles_labels), [])

    # Deduplicate while preserving order
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_handles.append(h)
        uniq_labels.append(l)

    fig.legend(uniq_handles, uniq_labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))

    # ============================================================================
    # Save figure
    # ============================================================================
    plt.tight_layout(rect=[0, 0, 1, 1.02])
    file_name = 'geometric_ROI_expansion_plot.pdf'
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()

    # ============================================================================
    # Print summary statistics
    # ============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nSIREN - Mean Dice: {siren_metrics['dice_mean'].mean():.4f} ± {siren_metrics['dice_mean'].std():.4f}")
    print(f"SIREN - Mean ARVE: {siren_metrics['arve_mean'].mean():.4f} ± {siren_metrics['arve_mean'].std():.4f}")
    print(f"SIREN - Mean Hausdorff Distance: {siren_metrics['hausdorff_mean'].mean():.4f} ± {siren_metrics['hausdorff_mean'].std():.4f}")

    print(f"\nInterpolation - Mean Dice: {interp_metrics['dice_mean'].mean():.4f} ± {interp_metrics['dice_mean'].std():.4f}")
    print(f"Interpolation - Mean ARVE: {interp_metrics['arve_mean'].mean():.4f} ± {interp_metrics['arve_mean'].std():.4f}")
    print(f"Interpolation - Mean Hausdorff Distance: {interp_metrics['hausdorff_mean'].mean():.4f} ± {interp_metrics['hausdorff_mean'].std():.4f}")

    print(f"\n[OK] Figure saved as {file_name}")


if __name__ == "__main__":
    main()

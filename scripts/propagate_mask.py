"""Propagate tumour masks from training slices to validation slices.

This script implements mask propagation strategies to generate full 3D tumour
masks from sparse training slice annotations. Two propagation methods are provided:

1. Intensity-based propagation: Uses reconstructed volume intensities to refine
   mask boundaries based on similarity to known tumour regions in training slices.

2. Simple morphological interpolation: Linear interpolation of mask shapes
   between adjacent training slices.

The intensity-based method leverages the reconstructed volumes from SIREN or
cubic interpolation to guide mask propagation using intensity similarity.

Usage:
    python propagate_mask.py

Inputs:
    - Consensus masks: data/processed/{nodule_id}_mask.npy
    - SIREN reconstructions: data/reconstructed-SIREN/{nodule_id}_siren_recon.npy
    - Interpolation reconstructions: data/reconstructed-interpolation/{nodule_id}_cubic_z_recon.npy

Outputs:
    - data/propagated_masks_siren/{nodule_id}_propagated_mask.npy
    - data/propagated_masks_interpolation/{nodule_id}_propagated_mask.npy
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import glob
from scipy.ndimage import binary_dilation, distance_transform_edt
from pathlib import Path
from tumour_tomography import config as cfg

std_mult = cfg.MASK_PROPAGATION_STD_MULTIPLIER # Standard deviation multiplier for intensity similarity

def propagate_mask_intensity(recon_vol, consensus_mask, train_slices, std_mult=1.5):
    """
    Propagate consensus mask to validation slices based on intensity similarity to training slices.
    
    Strategy:
    1. On training slices: use consensus mask directly
    2. On validation slices: merge masks from neighboring training slices and threshold using intensity similarity to training slices
    """
    D, H, W = recon_vol.shape
    
    # Start with training slice masks
    propagated = np.zeros_like(consensus_mask)
    propagated[train_slices] = consensus_mask[train_slices]
    
    # For each validation slice, find nearest training slices
    val_slices = np.setdiff1d(np.arange(D), train_slices)
    
    for z in val_slices:
        # Find nearest training slices above and below
        train_below = train_slices[train_slices < z]
        train_above = train_slices[train_slices > z]
        
        if len(train_below) == 0 and len(train_above) == 0:
            continue
        
        # Get masks from neighboring training slices
        masks_to_merge = []
        if len(train_below) > 0:
            z_below = train_below[-1]
            masks_to_merge.append(consensus_mask[z_below])
        if len(train_above) > 0:
            z_above = train_above[0]
            masks_to_merge.append(consensus_mask[z_above])
        
        # Average the masks and threshold
        merged_mask = np.mean(masks_to_merge, axis=0) > 0.5
        
        # Refine using intensity similarity to training slices
        current_slice = recon_vol[z]
        
        # Only keep regions with similar intensity to tumor in training slices
        tumor_intensities = recon_vol[train_slices][consensus_mask[train_slices] > 0]
        mean_tumor = np.mean(tumor_intensities)
        std_tumor = np.std(tumor_intensities)
        
        # Keep voxels within std_mult std of tumor intensity
        intensity_mask = np.abs(current_slice - mean_tumor) < std_mult * std_tumor
        
        propagated[z] = merged_mask & intensity_mask
    
    return propagated


def propagate_mask_simple_morph(consensus_mask, train_slices, iterations=1):
    """
    Simple approach: interpolate mask shape between training slices.
    """
    D = consensus_mask.shape[0]
    propagated = consensus_mask.copy()
    
    # Fill gaps between training slices
    for i in range(len(train_slices) - 1):
        z_start = train_slices[i]
        z_end = train_slices[i + 1]
            
        if z_end - z_start == 1:
            continue  # Adjacent slices, no gap
        
        # Linear interpolation of masks
        for z in range(z_start + 1, z_end):
            alpha = (z - z_start) / (z_end - z_start)
            mask_interp = (1 - alpha) * consensus_mask[z_start] + alpha * consensus_mask[z_end]
            propagated[z] = mask_interp > 0.5
    
    return propagated


def main():
    """Run mask propagation pipeline for all processed nodules.
    
    Loads consensus masks and reconstructed volumes, then applies
    intensity-based mask propagation to generate full 3D tumour masks.
    Processes both SIREN and interpolation reconstructions.
    
    Outputs:
        - PROPAGATED_MASKS_SIREN_DIR/{nodule_id}_propagated_mask.npy
        - PROPAGATED_MASKS_INTERPOLATION_DIR/{nodule_id}_propagated_mask.npy
    """
    vol_files = sorted(cfg.PROCESSED_DIR.glob("*_vol.npy"))
    
    for vol_path in vol_files:
        nodule_id = vol_path.stem.replace("_vol", "")
        
        # Load data
        mask_path = cfg.PROCESSED_DIR / f"{nodule_id}_mask.npy"
        siren_path = cfg.RECONSTRUCTED_SIREN_DIR / f"{nodule_id}_siren_recon.npy"
        interp_path = cfg.RECONSTRUCTED_INTERPOLATION_DIR / f"{nodule_id}_cubic_z_recon.npy"
        
        if not all(p.exists() for p in [mask_path, siren_path, interp_path]):
            print(f"Skipping {nodule_id}: missing files")
            continue
        
        consensus_mask = np.load(mask_path)
        siren_recon = np.load(siren_path)
        interp_recon = np.load(interp_path)
        
        # Define training slices (every 5th)
        D = consensus_mask.shape[0]
        train_slices = np.arange(0, D, 5)
        
        # Gradient-based propagation
        siren_prop = propagate_mask_intensity(siren_recon, consensus_mask, train_slices, std_mult=std_mult)
        interp_prop = propagate_mask_intensity(interp_recon, consensus_mask, train_slices, std_mult=std_mult)
        
        # Save
        np.save(cfg.PROPAGATED_MASKS_SIREN_DIR / f"{nodule_id}_propagated_mask.npy", siren_prop.astype(np.uint8))
        np.save(cfg.PROPAGATED_MASKS_INTERPOLATION_DIR / f"{nodule_id}_propagated_mask.npy", interp_prop.astype(np.uint8))
        
        print(f"✓ Processed {nodule_id}")


if __name__ == "__main__":
    main()
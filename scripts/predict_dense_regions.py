"""
Script to predict dense regions in reconstructed CT scans using ROI masks.
Applies a Hounsfield Unit (HU) threshold within the ROI to generate binary prediction masks
"""
import sys
import os
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tumour_tomography import config as cfg

def load_nodule_id(roi_path: str) -> str:
    """Extract nodule identifier from ROI mask file path.
    
    Args:
        roi_path: Full path to an ROI mask file (*_roi_mask.npy).
        
    Returns:
        The nodule identifier string (e.g., 'LIDC-IDRI-0001_nodule_0').
    """
    base = os.path.basename(roi_path)
    return base.replace("_roi_mask.npy", "")


def main():
    """Generate predicted tumour masks using HU thresholding within ROI.
    
    For each nodule with reconstructed volumes, applies Hounsfield Unit
    thresholding within the ROI mask to generate binary tumour predictions.
    Processes both SIREN and interpolation reconstructions.
    
    Outputs:
        - PRED_MASKS_INTERPOLATION_DIR/{nodule_id}_pred_mask.npy
        - PRED_MASKS_SIREN_DIR/{nodule_id}_pred_mask.npy
    """
    os.makedirs(cfg.PRED_MASKS_INTERPOLATION_DIR, exist_ok=True)
    os.makedirs(cfg.PRED_MASKS_SIREN_DIR, exist_ok=True)

    roi_files = sorted(glob.glob(os.path.join(cfg.PROCESSED_DIR, "*_roi_mask.npy")))
    if not roi_files:
        raise FileNotFoundError(f"No ROI masks found at: {cfg.PROCESSED_DIR}")

    print(f"Found {len(roi_files)} ROI masks.")

    skipped = 0
    for roi_path in roi_files:
        nodule_id = load_nodule_id(roi_path)

        interp_path = os.path.join(cfg.RECONSTRUCTED_INTERPOLATION_DIR, f"{nodule_id}_cubic_z_recon.npy")
        siren_path = os.path.join(cfg.RECONSTRUCTED_SIREN_DIR, f"{nodule_id}_siren_recon.npy")

        if not (os.path.exists(interp_path) and os.path.exists(siren_path)):
            print(f"✗  Missing recon for {nodule_id}. Skipping.")
            skipped += 1
            continue

        roi = np.load(roi_path)
        interp = np.load(interp_path)
        siren = np.load(siren_path)

        # Ensure shapes match
        if not (roi.shape == interp.shape == siren.shape):
            print(f"✗  Shape mismatch for {nodule_id}. Skipping.")
            skipped += 1
            continue

        # ROI filter, HU threshold
        roi_mask = (roi == 1)

        pred_interp = (roi_mask & (interp > cfg.HU_DENSITY_THRESHOLD)).astype(np.uint8)
        pred_siren = (roi_mask & (siren > cfg.HU_DENSITY_THRESHOLD)).astype(np.uint8)

        out_interp_path = os.path.join(cfg.PRED_MASKS_INTERPOLATION_DIR, f"{nodule_id}_pred_mask.npy")
        out_siren_path = os.path.join(cfg.PRED_MASKS_SIREN_DIR, f"{nodule_id}_pred_mask.npy")

        np.save(out_interp_path, pred_interp)
        np.save(out_siren_path, pred_siren)

    print(f"Done. Skipped: {skipped}")


if __name__ == "__main__":
    main()
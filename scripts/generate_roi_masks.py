"""
Script to generate expanded ROI masks for existing processed nodules.
Expands the existing consensus mask by margin_mm using 3D morphological dilation 
accounting for voxel spacing.
"""
import sys
import os
import glob
import json
import argparse
import numpy as np
from scipy import ndimage

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tumour_tomography import config as cfg


def generate_roi_masks(margin_mm=cfg.ROI_GENERATION_MARGIN_MM, mode=cfg.ROI_GENERATION_MODE):
    """Generate expanded ROI masks for all processed nodules.
    
    Creates region-of-interest masks by expanding consensus tumour masks
    using either isotropic 3D dilation or rectangular bounding box expansion.
    The expansion accounts for anisotropic voxel spacing.
    
    Args:
        margin_mm: Expansion margin in millimeters (default from config).
        mode: Expansion mode - 'dilation' for 3D morphological dilation
            with ellipsoidal structuring element, or 'box' for simple
            rectangular bounding box expansion.
    
    Outputs:
        Saves ROI masks as {nodule_id}_roi_mask.npy in PROCESSED_DIR.
    """
    print("=" * 80)
    title = "ROI Mask Generation (3D Isotropic Dilation)" if mode == "dilation" else "ROI Mask Generation (Box Expansion)"
    print(title)
    print("=" * 80)
    print(f"Input Directory: {cfg.PROCESSED_DIR}")
    print(f"Margin: {margin_mm} mm")
    print(f"Mode: {mode}")
    print("=" * 80)

    # Find all metadata files
    metadata_files = glob.glob(os.path.join(cfg.PROCESSED_DIR, "*_metadata.json"))
    
    if not metadata_files:
        print("No metadata files found.")
        return

    processed_count = 0
    skipped_count = 0

    for meta_path in metadata_files:
        try:
            # Load metadata
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            # Construct paths
            base_name = os.path.basename(meta_path).replace("_metadata.json", "")
            mask_path = os.path.join(cfg.PROCESSED_DIR, f"{base_name}_mask.npy")
            roi_output_path = os.path.join(cfg.PROCESSED_DIR, f"{base_name}_roi_mask.npy")

            if not os.path.exists(mask_path):
                print(f"⚠ Mask not found for {base_name}, skipping.")
                skipped_count += 1
                continue

            # Load mask and dimensions
            mask = np.load(mask_path)
            
            # Extract spacing info
            z_spacing = metadata.get('slice_thickness')
            xy_spacing = metadata.get('pixel_spacing')

            if z_spacing is None or xy_spacing is None:
                print(f"⚠ Missing spacing information for {base_name}, skipping.")
                skipped_count += 1
                continue

            # Find bounding box of current mask
            locs = np.where(mask > 0)
            
            if len(locs[0]) == 0:
                print(f"⚠ Empty mask for {base_name}, skipping.")
                skipped_count += 1
                continue

            if mode == "dilation":
                # Build ellipsoidal structuring element in voxel units
                rz = int(np.ceil(margin_mm / z_spacing))
                rxy = int(np.ceil(margin_mm / xy_spacing))

                if rz <= 0 and rxy <= 0:
                    roi_mask = (mask > 0).astype(np.uint8)
                    np.save(roi_output_path, roi_mask)
                    processed_count += 1
                    continue

                zz, yy, xx = np.ogrid[-rz:rz+1, -rxy:rxy+1, -rxy:rxy+1]
                dist = (zz * z_spacing) ** 2 + (yy * xy_spacing) ** 2 + (xx * xy_spacing) ** 2
                selem = dist <= (margin_mm ** 2)

                # 3D dilation
                roi_mask = ndimage.binary_dilation(mask > 0, structure=selem).astype(np.uint8)
            else:
                z_min, z_max = np.min(locs[0]), np.max(locs[0])
                y_min, y_max = np.min(locs[1]), np.max(locs[1])
                x_min, x_max = np.min(locs[2]), np.max(locs[2])

                # 2. Calculate voxel padding
                pad_z = int(np.ceil(margin_mm / z_spacing))
                pad_xy = int(np.ceil(margin_mm / xy_spacing))

                # 3. Define new bounding box limits (clipped to volume shape)
                D, H, W = mask.shape

                z_start = max(0, z_min - pad_z)
                z_stop = min(D, z_max + pad_z + 1) # +1 for slicing inclusive top

                y_start = max(0, y_min - pad_xy)
                y_stop = min(H, y_max + pad_xy + 1)

                x_start = max(0, x_min - pad_xy)
                x_stop = min(W, x_max + pad_xy + 1)

                # 4. Create Rectangular ROI Mask
                roi_mask = np.zeros_like(mask, dtype=np.uint8)
                roi_mask[z_start:z_stop, y_start:y_stop, x_start:x_stop] = 1
                # -- ROI LOGIC END --

            np.save(roi_output_path, roi_mask)
            processed_count += 1
            
        except Exception as e:
            print(f"✗ Error processing {os.path.basename(meta_path)}: {e}")
            skipped_count += 1

    print("\n" + "=" * 80)
    print(f"Generation Complete.")
    print(f"Generated: {processed_count}")
    print(f"Skipped/Error: {skipped_count}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ROI masks with optional isotropic 3D dilation.")
    parser.add_argument("--margin-mm", type=float, default=cfg.ROI_GENERATION_MARGIN_MM, help="Margin in millimeters.")
    parser.add_argument(
        "--mode",
        choices=["box", "dilation"],
        default=cfg.ROI_GENERATION_MODE,
        help="ROI expansion mode: 'box' for rectangular expansion, 'dilation' for isotropic 3D dilation.",
    )
    args = parser.parse_args()
    generate_roi_masks(margin_mm=args.margin_mm, mode=args.mode)
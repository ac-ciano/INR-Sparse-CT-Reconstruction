"""
Script to extract high-malignancy lung nodules from LIDC-IDRI dataset.
This script queries the database, downloads DICOM files, and saves processed nodules.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pylidc as pl
import numpy as np
from tumour_tomography import config as cfg
from tumour_tomography.processor import process_scan_nodules


def main():
    """Main execution function for nodule extraction."""
    print("=" * 80)
    print("LIDC-IDRI Nodule Extraction Pipeline")
    print("=" * 80)
    print(f"Target nodule count: {cfg.TARGET_NODULE_COUNT}")
    print(f"Malignancy threshold: > {cfg.MALIGNANCY_THRESHOLD}")
    print(f"Slice thickness limit: ≤ {cfg.SLICE_THICKNESS_LIMIT} mm")
    print(f"Output directory: {cfg.PROCESSED_DIR}")
    print("=" * 80)

    np.random.seed(42)
    
    # Query scans with thickness filter
    print("\nQuerying database...")
    scans = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= cfg.SLICE_THICKNESS_LIMIT).all()
    np.random.shuffle(scans)
    
    print(f"Found {len(scans)} candidate scans.\n")

    total_nodules_saved = 0
    scans_processed = 0

    for scan in scans:
        if total_nodules_saved >= cfg.TARGET_NODULE_COUNT:
            print(f"\n✓ Target nodule count ({cfg.TARGET_NODULE_COUNT}) reached!")
            break
        
        # Pre-check: Does this scan have high-malignancy nodules?
        try:
            clusters = scan.cluster_annotations()
            has_target_nodule = any(
                np.mean([a.malignancy for a in cluster]) > cfg.MALIGNANCY_THRESHOLD 
                for cluster in clusters
            )
        except Exception as e:
            print(f"⚠ Skipping scan {scan.patient_id} (annotation error): {e}")
            continue

        if has_target_nodule:
            scans_processed += 1
            print(f"\n[{scans_processed}] Processing {scan.patient_id}...")
            n_saved = process_scan_nodules(scan, total_nodules_saved, cfg.TARGET_NODULE_COUNT)
            total_nodules_saved += n_saved
            print(f"  Progress: {total_nodules_saved}/{cfg.TARGET_NODULE_COUNT} nodules saved")
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total nodules saved: {total_nodules_saved}")
    print(f"Scans processed: {scans_processed}")
    print(f"Output location: {cfg.PROCESSED_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
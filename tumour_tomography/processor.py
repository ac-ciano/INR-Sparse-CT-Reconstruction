"""CT scan processing and nodule extraction utilities.

This module provides functions for extracting and processing pulmonary nodules
from the LIDC-IDRI dataset. It handles DICOM data downloading, consensus mask
generation from multiple radiologist annotations, and nodule volume extraction
with configurable padding.

Key functionalities:
    - Consensus mask computation from multiple annotations
    - Padded volume extraction with boundary handling
    - Automatic DICOM download via TCIA API
    - Metadata generation (malignancy scores, dimensions, spacing)

The processing pipeline filters nodules based on malignancy threshold and
saves standardized numpy arrays for downstream SIREN training.
"""
import os
import json
import numpy as np
import pylidc as pl
from pylidc.utils import consensus
from tcia_utils import nbia
from . import config as cfg

def get_padded_nodule(cluster, scan, vol_full, pad_mm=cfg.PAD_MM, consensus_threshold=cfg.CONSENSUS_THRESHOLD):
    """
    Computes consensus mask and extracts padded volume for a nodule cluster.
    
    Args:
        cluster: List of annotations for a single nodule
        scan: pylidc Scan object
        vol_full: Full CT volume array
        pad_mm: Padding in millimeters around nodule
        consensus_threshold: Threshold for consensus mask generation
        
    Returns:
        tuple: (vol_padded, mask_padded) - Padded volume and mask arrays
    """
    cmask, cbbox = pl.utils.consensus(cluster, clevel=consensus_threshold, ret_masks=False)

    spacing = np.array([scan.slice_thickness, scan.pixel_spacing, scan.pixel_spacing])
    pad_voxels = np.ceil(pad_mm / spacing).astype(int)

    starts = np.array([s.start for s in cbbox])
    stops = np.array([s.stop for s in cbbox])
    
    ideal_starts = starts - pad_voxels
    ideal_stops = stops + pad_voxels
    
    vol_shape = np.array(vol_full.shape)
    valid_starts = np.maximum(0, ideal_starts)
    valid_stops = np.minimum(vol_shape, ideal_stops)
    
    slices = tuple(slice(s, e) for s, e in zip(valid_starts, valid_stops))
    vol_crop = vol_full[slices]

    pad_before = valid_starts - ideal_starts 
    pad_after = ideal_stops - valid_stops
    
    pad_width_vol = list(zip(pad_before, pad_after))
    
    # Pad volume with minimum intensity (air) to handle boundaries
    min_val = np.min(vol_full)
    vol_padded = np.pad(vol_crop, pad_width_vol, mode='constant', constant_values=min_val)
    
    # Pad mask to match the full volume size
    pad_width_mask = list(zip(pad_voxels, pad_voxels))
    mask_padded = np.pad(cmask, pad_width_mask, mode='constant', constant_values=0)
    
    mask_padded = mask_padded.astype(np.uint8)
    
    if vol_padded.shape != mask_padded.shape:
        raise ValueError(f"Shape mismatch: {vol_padded.shape} vs {mask_padded.shape}")

    return vol_padded, mask_padded


def download_scan_if_needed(scan):
    """
    Downloads DICOM data for a scan if not already present.
    
    Args:
        scan: pylidc Scan object
        
    Returns:
        str or None: Path to downloaded directory, or None if download failed
    """
    patient_id = scan.patient_id
    series_uid = scan.series_instance_uid
    expected_dir = os.path.join(cfg.RAW_DATA_DIR, patient_id)

    if not os.path.exists(expected_dir):
        print(f"  Downloading DICOM data for {patient_id}...")
        try:
            nbia.downloadSeries([series_uid], input_type="list", path=expected_dir)
        except Exception as e:
            print(f"  Download failed: {e}")
            return None
    
    if not os.path.isdir(expected_dir) or not os.listdir(expected_dir):
        return None
        
    return expected_dir


def save_nodule_data(nodule_id, vol, mask, scan, cluster_index, malignancy):
    """
    Saves nodule volume, mask, and metadata to disk.
    
    Args:
        nodule_id: Unique identifier for the nodule
        vol: Volume array
        mask: Mask array
        scan: pylidc Scan object
        cluster_index: Index of the cluster/nodule
        malignancy: Average malignancy score
    """
    np.save(os.path.join(cfg.PROCESSED_DIR, f"{nodule_id}_vol.npy"), vol)
    np.save(os.path.join(cfg.PROCESSED_DIR, f"{nodule_id}_mask.npy"), mask)
    
    # Get cluster from scan to extract size information
    cluster = scan.cluster_annotations()[cluster_index]
    avg_diameter = np.mean([ann.diameter for ann in cluster])
    avg_volume = np.mean([ann.volume for ann in cluster])
    
    metadata = {
        'patient_id': scan.patient_id,
        'nodule_index': cluster_index,
        'avg_malignancy': float(malignancy),
        'avg_diameter_mm': float(avg_diameter),
        'avg_volume_mm3': float(avg_volume),
        'slice_thickness': float(scan.slice_thickness),
        'pixel_spacing': float(scan.pixel_spacing),
        'shape': list(vol.shape)
    }
    
    with open(os.path.join(cfg.PROCESSED_DIR, f"{nodule_id}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)


def process_scan_nodules(scan, current_total_saved, target_total):
    """
    Downloads DICOM (if needed), loads volume, and processes nodules.
    
    Args:
        scan: pylidc Scan object
        current_total_saved: Current count of saved nodules
        target_total: Target total number of nodules to save
        
    Returns:
        int: Number of nodules successfully saved from this scan
    """
    expected_dir = download_scan_if_needed(scan)
    if not expected_dir:
        return 0

    try:
        # Load entire scan volume once
        vol_full = scan.to_volume()
        clusters = scan.cluster_annotations()

        print(f"  Slice thickness: {scan.slice_thickness:.2f} mm")
        
        saved_count = 0

        for i, cluster in enumerate(clusters):
            # Stop if we hit the global target
            if current_total_saved + saved_count >= target_total:
                break
                
            # Filter: Average malignancy > threshold
            malignancy = np.mean([ann.malignancy for ann in cluster])
            if malignancy <= cfg.MALIGNANCY_THRESHOLD:
                continue

            nodule_id = f"{scan.patient_id}_nodule_{i}"
            
            try:
                vol, mask = get_padded_nodule(cluster, scan, vol_full)
                save_nodule_data(nodule_id, vol, mask, scan, i, malignancy)
                
                saved_count += 1
                print(f"  ✓ Saved {nodule_id}: shape={vol.shape}, malignancy={malignancy:.2f}")

            except Exception as e:
                print(f"  ✗ Error processing nodule {nodule_id}: {e}")
                continue
            
        return saved_count
        
    except Exception as e:
        print(f"  ✗ Processing failed for scan {scan.patient_id}: {e}")
        return 0
"""Geometric metrics for evaluating 3D segmentation masks.

This module provides functions for computing standard geometric evaluation metrics
between predicted and ground truth 3D binary masks. All metrics support physical
spacing to convert from voxel coordinates to real-world units (typically millimeters).

Implemented metrics:
    - Volume computation in physical units
    - Absolute Relative Volume Error (ARVE)
    - Dice Similarity Coefficient (DSC)
    - Symmetric Hausdorff Distance (HD)

These metrics are commonly used in medical image segmentation evaluation,
particularly for tumour and nodule delineation tasks.
"""
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist


def compute_volume(mask: np.ndarray, voxel_volume: float) -> float:
    """
    Compute physical volume from a binary mask.

    Args:
        mask: 3D numpy array (0/1)
        voxel_volume: physical volume per voxel (e.g., mm^3)

    Returns:
        float: volume in physical units
    """
    return float(np.sum(mask > 0) * voxel_volume)


def absolute_relative_volume_error(pred_mask: np.ndarray, gt_mask: np.ndarray, voxel_volume: float) -> float:
    """
    Absolute Relative Volume Error (ARVE).

    ARVE = |V_pred - V_gt| / V_gt
    """
    v_pred = compute_volume(pred_mask, voxel_volume)
    v_gt = compute_volume(gt_mask, voxel_volume)
    if v_gt == 0:
        return 0.0 if v_pred == 0 else float("inf")
    return abs(v_pred - v_gt) / v_gt


def dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Dice Similarity Coefficient (DSC).
    """
    pred = pred_mask > 0
    gt = gt_mask > 0
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom


def _surface_points(mask: np.ndarray) -> np.ndarray:
    """
    Extract surface voxels of a binary mask.
    """
    if np.count_nonzero(mask) == 0:
        return np.empty((0, 3), dtype=np.float32)
    eroded = binary_erosion(mask, iterations=1)
    surface = np.logical_and(mask > 0, np.logical_not(eroded))
    return np.column_stack(np.where(surface))


def hausdorff_distance(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """
    Compute symmetric Hausdorff Distance between two binary masks in physical units.

    Args:
        pred_mask: 3D numpy array
        gt_mask: 3D numpy array
        spacing: (z_spacing, y_spacing, x_spacing)

    Returns:
        float: Hausdorff Distance in physical units
    """
    pred_pts = _surface_points(pred_mask)
    gt_pts = _surface_points(gt_mask)

    if pred_pts.size == 0 and gt_pts.size == 0:
        return 0.0
    if pred_pts.size == 0 or gt_pts.size == 0:
        return float("inf")

    # Convert voxel indices to physical coordinates
    pred_pts = pred_pts * np.array(spacing, dtype=np.float32)
    gt_pts = gt_pts * np.array(spacing, dtype=np.float32)

    dists = cdist(pred_pts, gt_pts, metric="euclidean")
    hd_pred_to_gt = dists.min(axis=1).max()
    hd_gt_to_pred = dists.min(axis=0).max()

    return float(max(hd_pred_to_gt, hd_gt_to_pred))
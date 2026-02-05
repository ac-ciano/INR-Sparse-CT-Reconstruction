"""
Cubic interpolation baseline for lung nodule reconstruction.
Provides volume interpolation along the z-axis for comparison with SIREN.
"""
import os
import csv
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d

from .data_loader import get_nodule_loader
from .radiometric_metrics import calculate_all_metrics
from . import config as cfg


def _get_kept_slices_mask_3d(train_mask_flat, volume_shape):
    """
    Extract kept slice indices from 1D slice mask.
    
    Args:
        train_mask_flat: 1D boolean array indicating kept slices
        volume_shape: Shape of the volume (D, H, W)
        
    Returns:
        1D boolean mask of kept slices
    """
    return train_mask_flat


def cubic_z_interpolate_volume(volume, train_mask_flat, device="cpu"):
    """
    Reconstruct full volume by cubic interpolation along z-axis,
    keeping available slices intact.

    Args:
        volume: numpy array (D, H, W) - normalized volume
        train_mask_flat: 1D boolean mask (len == D) indicating kept slices
        device: Computation device ('cpu' or 'cuda')

    Returns:
        numpy array (D, H, W): Reconstructed volume with interpolated slices
        
    Raises:
        ValueError: If fewer than 2 slices are available for interpolation
    """
    d, h, w = volume.shape

    kept_z = np.where(train_mask_flat)[0]
    
    if len(kept_z) < 2:
        raise ValueError(
            f"Not enough kept slices to interpolate. Found {len(kept_z)}, need at least 2."
        )
    
    # Extract kept slices and flatten for interpolation
    vol_sparse = volume[kept_z]
    flat_sparse = vol_sparse.reshape(len(kept_z), -1)
    
    # Create cubic interpolation function
    f = interp1d(
        kept_z, 
        flat_sparse, 
        kind='cubic', 
        axis=0, 
        fill_value="extrapolate"
    )

    # Interpolate for all z indices
    all_z = np.arange(d)
    recon_flat = f(all_z)  # (D, H*W)

    # Reshape back to volume
    recon = recon_flat.reshape(d, h, w)

    # Enforce ground truth on known slices (avoids floating point drift)
    recon[kept_z] = volume[kept_z]

    return recon.astype(volume.dtype)


def run_cubic_z(
    nodule_id,
    data_dir=None,
    csv_dir=None,
    recon_dir=None,
    device=None,
):
    """
    Run cubic interpolation baseline along z, compute validation metrics,
    and save results.

    Args:
        nodule_id: Identifier for the nodule to process
        data_dir: Directory containing processed nodules (default: cfg.PROCESSED_DIR)
        csv_dir: Directory for CSV logs (default: cfg.BASELINE_LOGS_DIR)
        recon_dir: Directory for reconstructed volumes (default: cfg.RECONSTRUCTED_INTERPOLATION_DIR)
        device: Computation device (default: cfg.DEFAULT_DEVICE)

    Returns:
        dict: Dictionary containing:
            - val_loss: Validation MSE loss
            - mae: Mean Absolute Error
            - psnr: Peak Signal-to-Noise Ratio
            - rmse: Root Mean Squared Error
            - ssim: Structural Similarity Index
            - csv_path: Path to saved CSV log
            - recon_path: Path to saved reconstruction
    """
    # Use config defaults if not specified
    data_dir = data_dir or str(cfg.PROCESSED_DIR)
    csv_dir = csv_dir or str(cfg.BASELINE_LOGS_DIR)
    recon_dir = recon_dir or str(cfg.RECONSTRUCTED_INTERPOLATION_DIR)
    device = device or cfg.DEFAULT_DEVICE

    # Ensure directories exist
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    print(f"Loading dataset: {nodule_id}")
    dataset = get_nodule_loader(
        nodule_id=nodule_id,
        data_dir=data_dir,
        normalize_coords=True,
    )

    vol_norm = dataset.data_flat.reshape(dataset.shape)

    print("Performing cubic interpolation along z...")
    # Reconstruct volume using normalized input
    recon = cubic_z_interpolate_volume(
        volume=vol_norm, 
        train_mask_flat=dataset.train_mask,
        device=device,
    )

    print("Computing validation metrics...")
    # Prepare validation targets
    excluded_indices = dataset.get_excluded_indices()
    excluded_coords, excluded_densities = dataset.get_excluded_coords()

    # Convert flat indices to 3D coordinates
    D, H, W = dataset.shape
    z = excluded_indices // (H * W)
    y = (excluded_indices % (H * W)) // W
    x = excluded_indices % W
    preds = recon[z, y, x]

    # Move to specified device for metric calculation
    preds_t = torch.from_numpy(preds).float().to(device)
    targets_t = excluded_densities.to(device)

    val_loss = F.mse_loss(preds_t, targets_t).item()
    val_metrics = calculate_all_metrics(preds_t, targets_t)

    # Save CSV log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(csv_dir, f"{nodule_id}_cubic_z_{timestamp}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "val_loss",
            "val_mae",
            "val_psnr",
            "val_rmse",
            "val_ssim",
        ])
        writer.writerow([
            val_loss,
            val_metrics["mae"],
            val_metrics["psnr"],
            val_metrics["rmse"],
            val_metrics["ssim"],
        ])

    print(f"✓ CSV log saved: {csv_path}")

    # Denormalize and save reconstructed volume
    recon_denorm = recon * (dataset.MAX_HU - dataset.MIN_HU) + dataset.MIN_HU
    recon_denorm = np.clip(recon_denorm, dataset.MIN_HU, dataset.MAX_HU)

    recon_path = os.path.join(recon_dir, f"{nodule_id}_cubic_z_recon.npy")
    np.save(recon_path, recon_denorm.astype(np.float32))

    print(f"✓ Reconstruction saved: {recon_path}")
    print(f"\nValidation Metrics:")
    print(f"  MSE Loss: {val_loss:.6f}")
    print(f"  MAE: {val_metrics['mae']:.6f}")
    print(f"  PSNR: {val_metrics['psnr']:.2f} dB")
    print(f"  RMSE: {val_metrics['rmse']:.6f}")
    print(f"  SSIM: {val_metrics['ssim']:.6f}")

    return {
        "val_loss": val_loss,
        **val_metrics,
        "csv_path": csv_path,
        "recon_path": recon_path,
    }
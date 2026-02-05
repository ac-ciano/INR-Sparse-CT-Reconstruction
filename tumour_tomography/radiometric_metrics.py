"""Radiometric metrics for evaluating image reconstruction quality.

This module provides PyTorch-based functions for computing standard image quality
metrics between model predictions and ground truth values. All metrics operate
on tensors and are designed for evaluating continuous intensity reconstruction
from implicit neural representations.

Implemented metrics:
    - Peak Signal-to-Noise Ratio (PSNR) - logarithmic quality measure in dB
    - Mean Absolute Error (MAE) - L1 distance metric
    - Root Mean Squared Error (RMSE) - L2 distance metric in original units
    - Mean Squared Error (MSE) - standard regression loss
    - Structural Similarity Index (SSIM) - perceptual quality metric

These metrics are commonly used for evaluating CT reconstruction quality
and SIREN model performance on medical imaging tasks.
"""
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity

def calculate_psnr(predictions, targets, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        max_val: Maximum possible value in the data (default 1.0 for normalized [0, 1] range)
    
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(predictions, targets)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def calculate_mae(predictions, targets):
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
    
    Returns:
        MAE value
    """
    mae = F.l1_loss(predictions, targets)
    return mae.item()

def calculate_rmse(predictions, targets):
    """
    Calculate Root Mean Squared Error (RMSE).
    More interpretable than MSE, same units as input.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
    
    Returns:
        RMSE value
    """
    mse = F.mse_loss(predictions, targets)
    rmse = torch.sqrt(mse)
    return rmse.item()

def calculate_ssim(predictions, targets, data_range=1.0):
    """
    Calculate Structural Similarity Index (SSIM).
    
    SSIM measures perceptual similarity between two signals, considering
    luminance, contrast, and structure. For 1D voxel data (flattened from
    volumetric images), the data is reshaped to a 2D representation for
    meaningful SSIM computation.
    
    Args:
        predictions: Model predictions (torch.Tensor or numpy array)
        targets: Ground truth values (torch.Tensor or numpy array)
        data_range: The data range of the input (max - min). Default 1.0
                    for normalized [0, 1] data.
    
    Returns:
        SSIM value in range [-1, 1], where 1 indicates perfect similarity
    """
    # Convert to numpy if torch tensors
    if isinstance(predictions, torch.Tensor):
        pred_np = predictions.detach().cpu().numpy().flatten()
    else:
        pred_np = np.asarray(predictions).flatten()
    
    if isinstance(targets, torch.Tensor):
        tgt_np = targets.detach().cpu().numpy().flatten()
    else:
        tgt_np = np.asarray(targets).flatten()
    
    # Handle identical inputs (perfect similarity)
    if np.array_equal(pred_np, tgt_np):
        return 1.0
    
    n = len(pred_np)
    
    # For very small arrays, SSIM is not meaningful
    if n < 7:
        # Fall back to simple correlation-based similarity
        if np.std(pred_np) == 0 or np.std(tgt_np) == 0:
            return 1.0 if np.allclose(pred_np, tgt_np) else 0.0
        corr = np.corrcoef(pred_np, tgt_np)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    
    # Reshape 1D data to 2D for SSIM computation
    # Choose dimensions that are as square as possible
    height = int(np.sqrt(n))
    width = n // height
    usable_n = height * width
    
    # Truncate to fit rectangular shape
    pred_2d = pred_np[:usable_n].reshape(height, width)
    tgt_2d = tgt_np[:usable_n].reshape(height, width)
    
    # Determine appropriate window size (must be odd and <= min dimension)
    min_dim = min(height, width)
    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(3, win_size)  # Minimum window size of 3
    
    # Compute SSIM
    ssim_value = structural_similarity(
        pred_2d, 
        tgt_2d, 
        data_range=data_range,
        win_size=win_size
    )
    
    return float(ssim_value)

def calculate_all_metrics(predictions, targets, max_val=1.0):
    """
    Calculate all metrics at once.
    
    Computes MSE, MAE, PSNR, RMSE, and SSIM between predictions and targets.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        max_val: Maximum possible value for PSNR and SSIM data_range calculation
    
    Returns:
        Dictionary with all metrics: mse, mae, psnr, rmse, ssim
    """
    with torch.no_grad():
        metrics = {
            'mse': F.mse_loss(predictions, targets).item(),
            'mae': calculate_mae(predictions, targets),
            'psnr': calculate_psnr(predictions, targets, max_val),
            'rmse': calculate_rmse(predictions, targets),
            'ssim': calculate_ssim(predictions, targets, data_range=max_val)
        }
    return metrics

# Explicitly export functions
__all__ = ['calculate_psnr', 'calculate_mae', 'calculate_rmse', 'calculate_ssim', 'calculate_all_metrics']
"""Data loading utilities for SIREN-based tumour reconstruction.

This module provides PyTorch Dataset implementations for loading and preprocessing
nodule CT volumes for implicit neural representation training. It handles coordinate
normalization, train/validation splitting based on slice skipping, and volume
reconstruction from model predictions.

The primary class NoduleCoordinateDataset implements a coordinate-based data loading
scheme where 3D spatial coordinates are mapped to CT density values, enabling
training of SIREN networks for continuous volumetric representation.

Typical usage:
    dataset = get_nodule_loader('PATIENT_001_nodule_0', data_dir)
    train_loader = DataLoader(dataset, batch_size=2048, shuffle=True)
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NoduleCoordinateDataset(Dataset):
    """
    Dataset for training SIREN on a single nodule volume.
    Returns (x, y, z) coordinates and corresponding density values.
    """
    MIN_HU = -1000.0
    MAX_HU = 400.0
    
    def __init__(self, nodule_id, data_dir, normalize_coords=True):
        """Initialize the dataset by loading and preprocessing a nodule volume.
        
        Loads the nodule volume, normalizes CT Hounsfield Units to [0, 1] range,
        creates a 3D coordinate grid normalized to [-1, 1], and establishes
        train/validation splits using slice-skipping (every 5th slice for training).
        
        Args:
            nodule_id: Unique identifier for the nodule (e.g., 'PATIENT_001_nodule_0').
            data_dir: Path to directory containing processed nodule files.
            normalize_coords: Whether to normalize coordinates (legacy parameter,
                coordinates are always normalized via linspace).
        """
        self.nodule_id = nodule_id
        
        # Load volume
        vol_path = os.path.join(data_dir, f"{nodule_id}_vol.npy")
        self.volume = np.load(vol_path).astype(np.float32)
        
        # Normalize volume values to [0, 1] (optional but recommended for SIREN)
        self.volume = (self.volume - self.MIN_HU) / (self.MAX_HU - self.MIN_HU)
        self.volume = np.clip(self.volume, 0, 1)
        
        self.shape = self.volume.shape
        self.normalize_coords = normalize_coords
        
        # Create coordinate grid
        # CHANGED: Use linspace to ensure exact [-1, 1] range
        D, H, W = self.shape
        z_coords = np.linspace(-1, 1, D)
        y_coords = np.linspace(-1, 1, H)
        x_coords = np.linspace(-1, 1, W)
        
        # Meshgrid (indexing='ij' for D, H, W)
        self.coords_z, self.coords_y, self.coords_x = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        
        # Flatten
        self.coords = np.stack([self.coords_x.flatten(), self.coords_y.flatten(), self.coords_z.flatten()], axis=1).astype(np.float32)
        self.data_flat = self.volume.flatten() # ADDED: Store flat data for easy access/testing
        
        # Create train/val split (slice skipping)
        self.train_mask = np.zeros(D, dtype=bool)
        self.train_mask[::5] = True # Keep every 5th slice for training
        
        # Create full masks for flat arrays
        z_flat_indices = np.arange(len(self.coords)) // (H * W)
        self.train_indices = np.where(np.isin(z_flat_indices, np.where(self.train_mask)[0]))[0]
        self.val_indices = np.where(~np.isin(z_flat_indices, np.where(self.train_mask)[0]))[0]
        
        self.train_coords = self.coords[self.train_indices]
        self.train_values = self.data_flat[self.train_indices]
        
    def _normalize_coordinates(self, coords):
        """Normalize coordinates to [-1, 1] range (deprecated).
        
        This method is deprecated as coordinate normalization is now handled
        directly in __init__ using np.linspace for exact range guarantees.
        
        Args:
            coords: Input coordinates array.
            
        Returns:
            The input coordinates unchanged.
        """
        return coords
    
    def __len__(self):
        """Return the number of training samples (voxels from training slices).
        
        Returns:
            Number of coordinate-value pairs available for training.
        """
        return len(self.train_coords)
    
    def __getitem__(self, idx):
        """Get a single training sample by index.
        
        Args:
            idx: Index of the sample to retrieve.
            
        Returns:
            Tuple of (coordinates, density) where coordinates is a float32 array
            of shape (3,) containing (x, y, z) in [-1, 1] and density is a 
            float32 scalar in [0, 1].
        """
        return self.train_coords[idx], self.train_values[idx]
    
    def get_all_coords(self):
        """Get all voxel coordinates in the volume.
        
        Returns:
            Tensor of shape (D*H*W, 3) containing all (x, y, z) coordinates
            in the volume, ordered for reshaping back to (D, H, W).
        """
        return torch.from_numpy(self.coords)
        
    def get_excluded_coords(self):
        """Get validation set coordinates and ground truth values.
        
        The validation set consists of voxels from slices excluded from training
        (slices not divisible by 5), used for evaluating interpolation quality.
        
        Returns:
            Tuple of (coordinates, densities) as PyTorch tensors:
                - coordinates: Tensor of shape (num_val_voxels, 3)
                - densities: Tensor of shape (num_val_voxels,)
        """
        val_c = self.coords[self.val_indices]
        val_v = self.data_flat[self.val_indices]
        return torch.from_numpy(val_c), torch.from_numpy(val_v)
        
    def get_excluded_indices(self):
        """Get indices of validation voxels in the flattened volume array.
        
        Returns:
            NumPy array of integer indices corresponding to validation voxels.
        """
        return self.val_indices
    
    def reconstruct_volume(self, predictions):
        """Reshape flat predictions back to the original 3D volume shape.
        
        Args:
            predictions: Flat array or tensor of predicted density values,
                ordered to match self.coords (D*H*W elements).
                
        Returns:
            NumPy array of shape (D, H, W) containing the reconstructed volume.
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        return predictions.reshape(self.shape)


def get_nodule_loader(nodule_id, data_dir, normalize_coords=True):
    """
    Create a Dataset for a single nodule.
    
    Args:
        nodule_id: ID of the nodule
        data_dir: Directory containing processed data
        normalize_coords: Whether to normalize coordinates
        
    Returns:
        NoduleCoordinateDataset object
    """
    dataset = NoduleCoordinateDataset(
        nodule_id=nodule_id,
        data_dir=data_dir,
        normalize_coords=normalize_coords
    )
    
    return dataset
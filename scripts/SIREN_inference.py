"""
SIREN Inference and Reconstruction Module.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from tumour_tomography.models import TumorSIREN
from tumour_tomography.data_loader import get_nodule_loader
from tumour_tomography import config as cfg


def load_best_model(nodule_id, device):
    """
    Locates and loads the most recent model checkpoint for a specific nodule.
    
    Args:
        nodule_id: The unique identifier for the nodule.
        device: Computation device ('cpu' or 'cuda').
        
    Returns:
        TumorSIREN: The loaded PyTorch model in evaluation mode.
    """
    # Search in fixed_epochs directory first, then early_stopping
    for subdir in [cfg.MODELS_FIXED_EPOCHS_DIR, cfg.MODELS_EARLY_STOPPING_DIR]:
        candidates = list(subdir.glob(f"{nodule_id}_*.pth"))
        if candidates:
            # Pick most recent
            model_path = max(candidates, key=lambda p: p.stat().st_mtime)
            print(f"Loading checkpoint: {model_path}")
            
            checkpoint = torch.load(model_path, map_location=device)
            
            # Extract architecture from checkpoint if available
            if 'architecture' in checkpoint:
                arch = checkpoint['architecture']
                hidden_features = arch.get('hidden_features', cfg.DEFAULT_HIDDEN_FEATURES)
                omega_0 = arch.get('omega_0', cfg.DEFAULT_OMEGA_0)
            else:
                hidden_features = cfg.DEFAULT_HIDDEN_FEATURES
                omega_0 = cfg.DEFAULT_OMEGA_0
                print("Warning: Architecture info not in checkpoint, using defaults")
            
            model = TumorSIREN(
                hidden_features=hidden_features,
                hidden_layers=cfg.DEFAULT_HIDDEN_LAYERS,
                omega_0=omega_0
            )
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.to(device)
            model.eval()
            
            return model
    
    raise FileNotFoundError(f"No model found for nodule {nodule_id}")


def restore_scale(volume_norm, min_hu, max_hu):
    """Denormalizes volume to HU scale."""
    volume = volume_norm * (max_hu - min_hu) + min_hu
    return np.clip(volume, min_hu, max_hu)


def reconstruct_volume(model, dataset, batch_size, device):
    """Performs volumetric reconstruction via batched inference."""
    all_coords = dataset.get_all_coords()
    total_voxels = len(all_coords)
    preds = []
    
    with torch.no_grad():
        for i in tqdm(range(0, total_voxels, batch_size), desc="Inference"):
            batch_coords = all_coords[i : i + batch_size].to(device)
            batch_out = model(batch_coords)
            preds.append(batch_out.cpu().numpy())
            
    flat_preds = np.concatenate(preds, axis=0).flatten()
    vol_reconstructed_norm = flat_preds.reshape(dataset.shape)
    
    return vol_reconstructed_norm


def main():
    """Command-line entry point for SIREN volumetric reconstruction.
    
    Parses command-line arguments, loads the trained model checkpoint,
    performs batched inference over the full coordinate grid, and saves
    the reconstructed volume in Hounsfield Units.
    
    The reconstructed volume is saved to RECONSTRUCTED_SIREN_DIR with
    filename {nodule_id}_siren_recon.npy.
    """
    parser = argparse.ArgumentParser(description="SIREN Inference and Reconstruction")
    parser.add_argument('--nodule-id', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=cfg.DEFAULT_INFERENCE_BATCH_SIZE)
    parser.add_argument('--device', type=str, default=cfg.DEFAULT_DEVICE,
                       choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Get device with fallback
    device = cfg.get_device(args.device)
    
    # Load Dataset
    print(f"Loading data for {args.nodule_id}...")
    dataset = get_nodule_loader(
        nodule_id=args.nodule_id,
        data_dir=str(cfg.PROCESSED_DIR),
        normalize_coords=True
    )
    
    # Load Model
    model = load_best_model(args.nodule_id, device)
    
    # Inference
    print("Running volume reconstruction...")
    vol_norm = reconstruct_volume(model, dataset, args.batch_size, device)
    
    # Restore Original Scale
    print("Restoring original scale (HU)...")
    vol_hu = restore_scale(vol_norm, dataset.MIN_HU, dataset.MAX_HU)
    
    # Save Output
    save_path = cfg.RECONSTRUCTED_SIREN_DIR / f"{args.nodule_id}_siren_recon.npy"
    np.save(save_path, vol_hu.astype(np.float32))
    
    print("-" * 60)
    print(f"Reconstruction Complete.")
    print(f"Original Shape: {dataset.shape}")
    print(f"Output Shape:   {vol_hu.shape}")
    print(f"Value Range:    [{vol_hu.min():.2f}, {vol_hu.max():.2f}] HU")
    print(f"File Saved:     {save_path}")
    print("-" * 60)


if __name__ == "__main__":
    main()
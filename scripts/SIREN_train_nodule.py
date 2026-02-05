"""
Script to train a SIREN model on a single nodule.
Saves trained model and generates training visualizations.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from tumour_tomography.models import TumorSIREN
from tumour_tomography.data_loader import get_nodule_loader
from tumour_tomography.SIREN_training import train_single_nodule
from tumour_tomography import config as cfg


def get_random_nodule(data_dir, seed=42):
    """
    Select a random nodule from available processed data.
    
    Args:
        data_dir: Directory containing processed nodules
        seed: Random seed for reproducibility
        
    Returns:
        nodule_id: Selected nodule identifier (without _vol.npy suffix)
    """
    vol_files = glob.glob(os.path.join(data_dir, "*_vol.npy"))
    
    if not vol_files:
        raise FileNotFoundError(f"No processed nodule files found in {data_dir}")
    
    # Extract nodule IDs (remove _vol.npy suffix)
    nodule_ids = [os.path.basename(f).replace('_vol.npy', '') for f in vol_files]
    
    # Set seed and select
    np.random.seed(seed)
    selected = np.random.choice(nodule_ids)
    
    print(f"Available nodules: {len(nodule_ids)}")
    print(f"Selected nodule (seed={seed}): {selected}")
    
    return selected

def get_untrained_nodule(seed=None):
    """
    Select a random nodule that hasn't been trained yet.
    
    Args:
        seed: Random seed for reproducibility (None for random selection)
        
    Returns:
        nodule_id: Selected nodule identifier (without _vol.npy suffix)
    """
    vol_files = list(cfg.PROCESSED_DIR.glob("*_vol.npy"))
    
    if not vol_files:
        raise FileNotFoundError(f"No processed nodule files found in {cfg.PROCESSED_DIR}")
    
    # Extract all available nodule IDs
    all_nodule_ids = [f.stem.replace('_vol', '') for f in vol_files]
    
    # Get already trained nodule IDs from both subdirectories
    trained_nodule_ids = set()
    
    for subdir in [cfg.MODELS_EARLY_STOPPING_DIR, cfg.MODELS_FIXED_EPOCHS_DIR]:
        if subdir.exists():
            model_files = list(subdir.glob("*.pth"))
            for model_file in model_files:
                # Extract nodule_id from filename (format: nodule_id_timestamp.pth)
                filename = model_file.stem
                # Remove timestamp part (last 16 chars: _YYYYMMDD_HHMMSS)
                nodule_id = '_'.join(filename.split('_')[:-2])
                trained_nodule_ids.add(nodule_id)
    
    # Find untrained nodules
    untrained_nodule_ids = [nid for nid in all_nodule_ids if nid not in trained_nodule_ids]
    
    if not untrained_nodule_ids:
        raise ValueError(f"All {len(all_nodule_ids)} nodules have already been trained!")
    
    # Set seed if provided
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(None)
    
    selected = np.random.choice(untrained_nodule_ids)
    
    print(f"Total nodules available: {len(all_nodule_ids)}")
    print(f"Already trained: {len(trained_nodule_ids)}")
    print(f"Untrained nodules: {len(untrained_nodule_ids)}")
    print(f"Selected nodule: {selected}")
    
    return selected


def save_model(model, nodule_id, history, early_stopping_used):
    """
    Save trained model with metadata in appropriate subdirectory.
    
    Args:
        model: Trained TumorSIREN model
        nodule_id: Nodule identifier
        history: Training history dictionary
        early_stopping_used: Whether early stopping was used
    """
    save_dir = cfg.MODELS_EARLY_STOPPING_DIR if early_stopping_used else cfg.MODELS_FIXED_EPOCHS_DIR
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{nodule_id}_{timestamp}.pth"
    model_path = save_dir / model_filename
    
    # Save model state and training metadata
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'nodule_id': nodule_id,
        'timestamp': timestamp,
        'best_epoch': history['best_epoch'],
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
        'early_stopping_used': early_stopping_used,
        # Architecture info for reproducibility
        'architecture': {
            'hidden_features': model.net[0].linear.out_features,
            'omega_0': model.net[0].omega_0,
        }
    }
    
    torch.save(checkpoint, model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    return model_path


def main():
    """Command-line entry point for SIREN nodule training.
    
    Parses command-line arguments, initializes the model and dataset,
    runs training with specified hyperparameters, and saves the trained
    model checkpoint. Supports both early stopping and fixed-epoch training.
    
    Command-line arguments control nodule selection, training hyperparameters
    (epochs, batch size, learning rate, etc.), and model architecture
    (hidden layers, features, omega_0, dropout).
    """
    parser = argparse.ArgumentParser(description='Train SIREN on a lung nodule')
    parser.add_argument('--nodule-id', type=str, default=None,
                       help='Specific nodule ID. If not provided, selects untrained nodule.')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for nodule selection')
    parser.add_argument('--epochs', type=int, default=cfg.DEFAULT_MAX_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=cfg.DEFAULT_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=cfg.DEFAULT_LEARNING_RATE)
    parser.add_argument('--no-early-stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=cfg.DEFAULT_PATIENCE)
    parser.add_argument('--hidden-layers', type=int, default=cfg.DEFAULT_HIDDEN_LAYERS)
    parser.add_argument('--hidden-features', type=int, default=cfg.DEFAULT_HIDDEN_FEATURES)
    parser.add_argument('--omega-0', type=float, default=cfg.DEFAULT_OMEGA_0)
    parser.add_argument('--dropout', type=float, default=cfg.DEFAULT_DROPOUT)
    parser.add_argument('--coord-noise', type=float, default=cfg.DEFAULT_COORD_NOISE_STD)
    parser.add_argument('--device', type=str, default=cfg.DEFAULT_DEVICE,
                       choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Get device with fallback
    device = cfg.get_device(args.device)
    
    # Determine nodule to train on
    if args.nodule_id is None:
        nodule_id = get_untrained_nodule(seed=args.seed)
    else:
        nodule_id = args.nodule_id
    
    early_stopping_used = not args.no_early_stopping
    
    print("=" * 80)
    print("SIREN NODULE TRAINING")
    print("=" * 80)
    print(f"Nodule ID: {nodule_id}")
    print(f"Data directory: {cfg.PROCESSED_DIR}")
    print(f"Device: {device}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.epochs}")
    print(f"Early stopping: {early_stopping_used}")
    if early_stopping_used:
        print(f"Patience: {args.patience}")
    print("=" * 80)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = get_nodule_loader(
        nodule_id=nodule_id,
        data_dir=str(cfg.PROCESSED_DIR),
        normalize_coords=True
    )
    
    # Initialize model
    print("\nInitializing SIREN model...")
    model = TumorSIREN(
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        omega_0=args.omega_0,
        dropout=args.dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    print("\n" + "=" * 80)
    trained_model, history = train_single_nodule(
        model=model,
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stopping=early_stopping_used,
        patience=args.patience,
        device=str(device),
        save_csv=True,
        csv_dir=str(cfg.TRAINING_LOGS_DIR),
        coord_noise_std=args.coord_noise
    )
    
    # Save trained model
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    model_path = save_model(trained_model, nodule_id, history, early_stopping_used)
    
    # Training summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    if history['val_loss']:
        print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
    print(f"Training mode: {'Early Stopping' if early_stopping_used else 'Fixed Epochs'}")
    print(f"Model saved to: {model_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
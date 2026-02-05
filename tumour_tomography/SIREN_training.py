"""Training utilities for SIREN-based tumour reconstruction.

This module provides the training pipeline for TumorSIREN models, including
learning rate scheduling, early stopping, and comprehensive metric logging.
The training approach uses a warmup phase followed by cosine annealing with
warm restarts for stable convergence.

Key components:
    - WarmupCosineScheduler: Custom LR scheduler with warmup and cosine annealing
    - train_single_nodule: Main training function with early stopping support
    - Comprehensive CSV logging of training/validation metrics

The training pipeline supports coordinate noise injection for regularization,
gradient clipping for stability, and validation-based model selection using
held-out CT slices (slice-skipping strategy).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os
import csv
from datetime import datetime
from torch.utils.data import DataLoader
from .radiometric_metrics import calculate_all_metrics

def set_seed(seed=42):
    """Set random seeds for reproducibility across PyTorch and NumPy.
    
    Sets seeds for torch CPU, torch CUDA (all devices), and NumPy random
    number generators to ensure reproducible training results.
    
    Args:
        seed: Integer seed value for all random number generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine annealing restarts.
    
    Implements a two-phase learning rate schedule:
    1. Warmup phase: Linear increase from 0 to base_lr over warmup_epochs
    2. Cosine phase: Cosine annealing with warm restarts (SGDR-style)
    
    The restart period doubles after each cycle (controlled by T_mult).
    
    Attributes:
        optimizer: The optimizer whose learning rate will be adjusted.
        warmup_epochs: Number of epochs for the linear warmup phase.
        T_0: Initial period for cosine annealing cycle.
        T_mult: Multiplier for cycle length after each restart.
        eta_min: Minimum learning rate during cosine annealing.
        base_lrs: Initial learning rates from optimizer param groups.
        last_epoch: Current epoch counter.
        cycle_len: Current cosine cycle length.
        cycle_epoch: Epoch within the current cosine cycle.
    """
    
    def __init__(self, optimizer, warmup_epochs, T_0=100, T_mult=2, eta_min=1e-6):
        """Initialize the warmup cosine scheduler.
        
        Args:
            optimizer: PyTorch optimizer to schedule.
            warmup_epochs: Number of epochs for linear warmup phase.
            T_0: Initial period for the first cosine annealing cycle.
            T_mult: Factor by which T_0 is multiplied after each restart.
            eta_min: Minimum learning rate (lower bound during annealing).
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_epoch = 0
        self.cycle_len = T_0
        self.cycle_epoch = 0
        
        self.step() 

    def step(self):
        """Advance the scheduler by one epoch and update learning rates.
        
        During warmup, learning rate increases linearly from 0 to base_lr.
        After warmup, applies cosine annealing with warm restarts, where
        the cycle length doubles after each restart.
        """
        self.last_epoch += 1
        
        if self.last_epoch <= self.warmup_epochs:
            # Warmup Phase (Linear)
            alpha = self.last_epoch / self.warmup_epochs
            lrs = [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine Annealing Phase
            # Adjust epoch counter relative to end of warmup
            curr_epoch = self.last_epoch - self.warmup_epochs
            
            # Check for restart
            if self.cycle_epoch >= self.cycle_len:
                self.cycle_len *= self.T_mult
                self.cycle_epoch = 0
            
            # Cosine formula
            import math
            coeffs = 0.5 * (1 + math.cos(math.pi * self.cycle_epoch / self.cycle_len))
            lrs = [self.eta_min + (base_lr - self.eta_min) * coeffs for base_lr in self.base_lrs]
            
            self.cycle_epoch += 1
            
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
            
    def get_last_lr(self):
        """Get the current learning rates for all parameter groups.
        
        Returns:
            List of current learning rates, one per optimizer param group.
        """
        return [group['lr'] for group in self.optimizer.param_groups]


def train_single_nodule(model, dataset, 
                        epochs=2000, 
                        batch_size=2048, 
                        lr=5e-5, 
                        early_stopping=True,
                        patience=200, 
                        device='cuda',
                        save_csv=True,
                        csv_dir='logs/training',
                        seed=42,
                        use_scheduler=True,
                        warmup_epochs=50,
                        cosine_T0=100,
                        cosine_Tmult=2,
                        grad_clip=1.0,
                        coord_noise_std=0.0):
    """
    Trains the SIREN model on a single nodule with Early Stopping.

    Returns:
        tuple: (trained_model, history_dict)
    
    Args:
        model: The TumorSIREN instance
        dataset: The NoduleCoordinateDataset (initialized for a specific nodule)
        epochs: Maximum number of training epochs
        batch_size: Training batch size
        lr: Initial learning rate for Adam optimizer
        early_stopping: Whether to use early stopping based on validation loss
        patience: Epochs to wait for validation improvement before stopping
        save_csv: Whether to save training history as CSV
        csv_dir: Directory to save CSV logs
        seed: Random seed for reproducibility
        use_scheduler: Whether to use learning rate scheduler
        warmup_epochs: Number of epochs for linear warmup (default: 50)
        cosine_T0: Cosine scheduler first restart period (default: 100)
        cosine_Tmult: Cosine scheduler restart period multiplier (default: 2)
        grad_clip: Gradient clipping value (None to disable)
        coord_noise_std: Standard deviation of Gaussian noise added to coordinates (e.g. 1e-4)
    """

    # Set random seed for reproducibility
    set_seed(seed)

    # Setup CSV logging
    if save_csv:
        os.makedirs(csv_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(csv_dir, f"{dataset.nodule_id}_{timestamp}(lr{lr}--bs{batch_size}).csv")
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'train_loss', 'train_mae', 'train_psnr', 'train_rmse', 'train_ssim',
                            'val_loss', 'val_mae', 'val_psnr', 'val_rmse', 'val_ssim',
                            'learning_rate', 'best_epoch'])
    
    # Setup
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Setup scheduler: Warmup + Cosine Annealing
    scheduler = None
    if use_scheduler:
        scheduler = WarmupCosineScheduler(
            optimizer, 
            warmup_epochs=warmup_epochs,
            T_0=cosine_T0,
            T_mult=cosine_Tmult,
            eta_min=1e-6
        )
    
    criterion = nn.MSELoss()
    
    # Set worker_init_fn for reproducibility
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Prepare validation data
    val_coords, val_densities = dataset.get_excluded_coords()
    
    has_validation = False
    if val_coords is not None and len(val_coords) > 0:
        val_coords = val_coords.to(device)
        val_targets = val_densities.to(device)
        has_validation = True
    
    # Determine training mode
    if early_stopping and has_validation:
        training_mode = "early_stopping"
        print(f"Training mode: Early Stopping (patience={patience})")
    elif has_validation:
        training_mode = "fixed_with_validation"
        print(f"Training mode: Fixed epochs with validation tracking")
    else:
        training_mode = "fixed_no_validation"
        print("Warning: No excluded slices found. Training for fixed epochs without validation.")

    # Metrics Storage
    history = {
        'train_loss': [],
        'train_mae': [],
        'train_psnr': [],
        'train_rmse': [],
        'train_ssim': [],
        'val_loss': [],
        'val_mae': [],
        'val_psnr': [],
        'val_rmse': [],
        'val_ssim': [],
        'best_epoch': 0,
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    print(f"--- Starting Training on {dataset.nodule_id} ---")
    print(f"Device: {device} | LR: {lr} | Batch: {batch_size} | Epochs: {epochs}")
    print(f"Scheduler: Warmup({warmup_epochs}) + CosineRestart(T0={cosine_T0}, Tmult={cosine_Tmult})")
    print(f"Early Stopping: {early_stopping} | Grad Clip: {grad_clip}")
    if save_csv:
        print(f"Logging to: {csv_path}")
    
    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        epoch_train_preds = []
        epoch_train_targets = []
        
        for coords, density in train_loader:
            coords = coords.to(device)
            targets = density.to(device)

            # coordinate jittering
            if coord_noise_std > 0:
                noise = torch.randn_like(coords) * coord_noise_std
                coords = coords + noise
            
            optimizer.zero_grad()
            preds = model(coords)
            loss = criterion(preds, targets)
            loss.backward()
            
            # Apply gradient clipping for stability
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            batch_losses.append(loss.item())
            
            epoch_train_preds.append(preds.detach().cpu())
            epoch_train_targets.append(targets.detach().cpu())
        
        # Calculate training metrics
        avg_train_loss = np.mean(batch_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Compute additional training metrics
        all_train_preds = torch.cat(epoch_train_preds, dim=0)
        all_train_targets = torch.cat(epoch_train_targets, dim=0)
        train_metrics = calculate_all_metrics(all_train_preds, all_train_targets)
        
        history['train_mae'].append(train_metrics['mae'])
        history['train_psnr'].append(train_metrics['psnr'])
        history['train_rmse'].append(train_metrics['rmse'])
        history['train_ssim'].append(train_metrics['ssim'])
        
        # Track current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        # Validation Step
        if has_validation:
            model.eval()
            with torch.no_grad():
                val_preds = model(val_coords)
                val_loss = criterion(val_preds, val_targets).item()
                val_metrics = calculate_all_metrics(val_preds, val_targets)
                
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_metrics['mae'])
            history['val_psnr'].append(val_metrics['psnr'])
            history['val_rmse'].append(val_metrics['rmse'])
            history['val_ssim'].append(val_metrics['ssim'])
            
            # Update learning rate scheduler (step every epoch)
            if scheduler is not None:
                scheduler.step()
            
            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                history['best_epoch'] = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            # Progress Print
            if epoch % 50 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                      f"Val PSNR: {val_metrics['psnr']:.2f} | Val SSIM: {val_metrics['ssim']:.4f} | LR: {current_lr:.2e}")
            
            # CSV Logging
            if save_csv:
                csv_writer.writerow([epoch, avg_train_loss, train_metrics['mae'], 
                                    train_metrics['psnr'], train_metrics['rmse'], train_metrics['ssim'],
                                    val_loss, val_metrics['mae'], 
                                    val_metrics['psnr'], val_metrics['rmse'], val_metrics['ssim'],
                                    current_lr, history['best_epoch']])
                csv_file.flush()

            # Early Stopping Check
            if training_mode == "early_stopping" and epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}!")
                print(f"Best Validation Loss: {best_val_loss:.6f} at epoch {history['best_epoch']}")
                break
        else:
            # No validation available
            # No validation
            if scheduler is not None:
                scheduler.step()
                
            if epoch % 50 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.6f} | "
                      f"Train PSNR: {train_metrics['psnr']:.2f} | Train SSIM: {train_metrics['ssim']:.4f} | LR: {current_lr:.2e}")
            
            # CSV Logging (no validation)
            if save_csv:
                csv_writer.writerow([epoch, avg_train_loss, train_metrics['mae'],
                                    train_metrics['psnr'], train_metrics['rmse'], train_metrics['ssim'],
                                    None, None, None, None, None, current_lr, None])
                csv_file.flush()

        # Save final epoch model for fixed training
        if training_mode in ["fixed_with_validation", "fixed_no_validation"] and epoch == epochs:
            best_model_state = copy.deepcopy(model.state_dict())
            history['best_epoch'] = epoch

    # Close CSV file
    if save_csv:
        csv_file.close()
        print(f"\nTraining log saved to {csv_path}")

    # Load best model based on training mode
    if training_mode == "early_stopping":
        print(f"Loading best model from epoch {history['best_epoch']}")
    elif training_mode == "fixed_with_validation":
        print(f"Using final epoch model (epoch {history['best_epoch']})")
    else:
        print(f"Using final epoch model (epoch {epochs})")
    
    model.load_state_dict(best_model_state)
    
    return model, history
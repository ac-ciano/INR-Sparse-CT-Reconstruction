"""
Configuration module for Oncology SIREN project.
Centralizes all paths, hyperparameters, and settings.
"""
import os
import numpy as np
from pathlib import Path

# Compatibility fix for legacy libraries using deprecated NumPy attributes
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Reconstruction directories
RECONSTRUCTED_SIREN_DIR = DATA_DIR / "reconstructed-SIREN"
RECONSTRUCTED_INTERPOLATION_DIR = DATA_DIR / "reconstructed-interpolation"

# Mask directories for evaluation
PROPAGATED_MASKS_SIREN_DIR = DATA_DIR / "propagated_masks_siren"
PROPAGATED_MASKS_INTERPOLATION_DIR = DATA_DIR / "propagated_masks_interpolation"
PRED_MASKS_SIREN_DIR = DATA_DIR / "pred_masks_siren"
PRED_MASKS_INTERPOLATION_DIR = DATA_DIR / "pred_masks_interpolation"


# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_EARLY_STOPPING_DIR = MODELS_DIR / "early_stopping"
MODELS_FIXED_EPOCHS_DIR = MODELS_DIR / "fixed_epochs"

# Logging directories
LOGS_DIR = PROJECT_ROOT / "logs"
TRAINING_LOGS_DIR = LOGS_DIR / "training"
BASELINE_LOGS_DIR = LOGS_DIR / "baselines"
INFERENCE_LOGS_DIR = LOGS_DIR / "inference"
PROPAGATED_EVAL_LOGS_DIR = LOGS_DIR / "propagated_evaluation"

# Create all necessary directories
for directory in [
    RAW_DATA_DIR, PROCESSED_DIR, 
    RECONSTRUCTED_SIREN_DIR, RECONSTRUCTED_INTERPOLATION_DIR,
    PROPAGATED_MASKS_SIREN_DIR, PROPAGATED_MASKS_INTERPOLATION_DIR,
    PRED_MASKS_SIREN_DIR, PRED_MASKS_INTERPOLATION_DIR,
    MODELS_EARLY_STOPPING_DIR, MODELS_FIXED_EPOCHS_DIR,
    TRAINING_LOGS_DIR, BASELINE_LOGS_DIR, INFERENCE_LOGS_DIR,
    PROPAGATED_EVAL_LOGS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA EXTRACTION PARAMETERS
# ============================================================================
SLICE_THICKNESS_LIMIT = 1.0  # mm
PAD_MM = 10  # Padding around nodule in millimeters
CONSENSUS_THRESHOLD = 0.5  # Threshold for annotation consensus
MALIGNANCY_THRESHOLD = 3  # Minimum malignancy score
TARGET_NODULE_COUNT = 70  # Target number of nodules to extract

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
DEFAULT_HIDDEN_FEATURES = 256
DEFAULT_HIDDEN_LAYERS = 3
DEFAULT_OMEGA_0 = 30.0
DEFAULT_DROPOUT = 0.5

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_BATCH_SIZE = 2048
DEFAULT_MAX_EPOCHS = 2000
DEFAULT_PATIENCE = 200
DEFAULT_SEED = 42

# Scheduler settings
DEFAULT_USE_SCHEDULER = True
DEFAULT_WARMUP_EPOCHS = 50
DEFAULT_COSINE_T0 = 100
DEFAULT_COSINE_TMULT = 2

# Regularization
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_COORD_NOISE_STD = 0.0001

# ============================================================================
# BASELINE PARAMETERS
# ============================================================================
DEFAULT_INTERPOLATION_METHOD = 'cubic_z'
ROI_GENERATION_MARGIN_MM = 3.0
ROI_GENERATION_MODE = 'dilation' # 'box' for rectangular expansion, 'dilation' for isotropic 3D dilation

# ============================================================================
# MASKING & EVALUATION PARAMETERS
# ============================================================================
HU_DENSITY_THRESHOLD = -400.0
MASK_PROPAGATION_STD_MULTIPLIER = 4.0

# ============================================================================
# INFERENCE PARAMETERS
# ============================================================================
DEFAULT_INFERENCE_BATCH_SIZE = 32768

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
import torch

def get_device(requested_device='cuda'):
    """
    Get appropriate computation device with fallback.
    
    Args:
        requested_device: Requested device ('cuda' or 'cpu')
        
    Returns:
        torch.device: Available device
    """
    if requested_device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if requested_device == 'cuda':
        print("Warning: CUDA requested but not available. Using CPU.")
    return torch.device('cpu')

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# HOUNSFIELD UNIT RANGES
# ============================================================================
MIN_HU = -1000  # Air
MAX_HU = 400    # Bone/calcification
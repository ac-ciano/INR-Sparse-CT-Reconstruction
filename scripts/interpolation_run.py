"""
Script to run cubic interpolation along z baseline on lung nodules.
Provides a baseline comparison for SIREN reconstruction quality.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import numpy as np
from tumour_tomography import config as cfg
from tumour_tomography.interpolation import run_cubic_z


def get_random_nodule(data_dir, seed=42):
    """
    Select a random nodule from available processed data.
    
    Args:
        data_dir: Directory containing processed nodules
        seed: Random seed for reproducibility
        
    Returns:
        str: Selected nodule identifier (without _vol.npy suffix)
        
    Raises:
        FileNotFoundError: If no processed nodule files are found
    """
    vol_files = glob.glob(os.path.join(data_dir, "*_vol.npy"))
    
    if not vol_files:
        raise FileNotFoundError(f"No processed nodule files found in {data_dir}")
    
    nodule_ids = [os.path.basename(f).replace("_vol.npy", "") for f in vol_files]
    
    np.random.seed(seed)
    selected = np.random.choice(nodule_ids)
    
    print(f"Available nodules: {len(nodule_ids)}")
    print(f"Selected nodule (seed={seed}): {selected}")
    
    return selected


def main():
    """Main execution function for interpolation baseline."""
    parser = argparse.ArgumentParser(
        description="Cubic interpolation along z baseline for lung nodule reconstruction"
    )
    parser.add_argument(
        "--nodule-id", 
        type=str, 
        default=None,
        help="Specific nodule ID. If not provided, selects random nodule."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=cfg.DEFAULT_SEED,
        help=f"Random seed for nodule selection (default: {cfg.DEFAULT_SEED})"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=cfg.DEFAULT_DEVICE,
        choices=["cpu", "cuda"],
        help="Computation device"
    )
    
    args = parser.parse_args()

    # Get device with fallback
    device = cfg.get_device(args.device)

    # Determine nodule to process
    if args.nodule_id is None:
        nodule_id = get_random_nodule(str(cfg.PROCESSED_DIR), seed=args.seed)
    else:
        nodule_id = args.nodule_id

    print("=" * 80)
    print("CUBIC-Z INTERPOLATION BASELINE")
    print("=" * 80)
    print(f"Nodule ID: {nodule_id}")
    print(f"Data directory: {cfg.PROCESSED_DIR}")
    print(f"Device: {device}")
    print(f"Output directory: {cfg.RECONSTRUCTED_INTERPOLATION_DIR}")
    print("=" * 80)
    print()

    # Run interpolation
    results = run_cubic_z(
        nodule_id=nodule_id,
        data_dir=str(cfg.PROCESSED_DIR),
        csv_dir=str(cfg.BASELINE_LOGS_DIR),
        recon_dir=str(cfg.RECONSTRUCTED_INTERPOLATION_DIR),
        device=str(device),
    )

    print("\n" + "=" * 80)
    print("INTERPOLATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
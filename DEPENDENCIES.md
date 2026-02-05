# Dependency Versions

## Production Environment

This document records the exact versions used in the experiments reported in the paper. All versions were extracted from the `onc_lung_inr` conda environment used for running experiments.

### Core Environment

- **Python**: 3.10.19
- **CUDA**: 12.1
- **PyTorch CUDA**: 12.1

### Deep Learning Framework

| Package | Version | Source |
|---------|---------|--------|
| torch | 2.5.1 | conda (pytorch channel) |
| torchvision | 0.20.1 | PyPI |
| torchaudio | 2.5.1 | PyPI |
| pytorch-cuda | 12.1 | conda (nvidia channel) |

### Numerical & Scientific Computing

| Package | Version | Source |
|---------|---------|--------|
| numpy | 2.2.6 | conda-forge |
| scipy | 1.15.2 | conda-forge |
| pandas | 2.3.3 | conda-forge |

### Image Processing & Machine Learning

| Package | Version | Source |
|---------|---------|--------|
| scikit-image | 0.25.2 | conda-forge |
| scikit-learn | 1.7.2 | conda-forge |
| Pillow | 10.4.0 | conda-forge |

### Medical Imaging

| Package | Version | Source |
|---------|---------|--------|
| pydicom | 3.0.1 | conda-forge |
| pylidc | 0.2.3 | PyPI |
| SimpleITK | 2.5.3 | conda-forge |
| tcia-utils | 3.2.1 | PyPI |

### Visualization

| Package | Version | Source |
|---------|---------|--------|
| matplotlib | 3.10.1 | conda-forge |
| pyvista | 0.46.4 | conda-forge |
| plotly | 6.5.1 | PyPI |

### Development Tools

| Package | Version | Source |
|---------|---------|--------|
| jupyterlab | 4.5.2 | conda-forge |
| ipywidgets | 8.1.8 | conda-forge |
| tqdm | 4.67.1 | conda-forge |

### Testing

| Package | Version | Source |
|---------|---------|--------|
| pytest | 9.0.2 | PyPI |
| pytest-cov | 7.0.0 | PyPI |

## Installation Methods

### Method 1: Conda (Recommended for Exact Reproduction)

```bash
conda env create -f environment.yml
conda activate onc_lung_inr
```

This will install the exact versions listed above from the specified channels.

### Method 2: pip

```bash
pip install -r requirements.txt
```

For GPU support with CUDA 12.1:
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

## Hardware Specifications

The experiments were conducted on:
- **GPU**: NVIDIA RTX 3090
- **CUDA Version**: 12.1
- **RAM**: 16GB+ recommended
- **OS**: Windows 10

## Version History

### 2026-02-04: Initial Version Pinning
- Extracted exact versions from `onc_lung_inr` conda environment
- Updated `requirements.txt` with pinned versions
- Updated `environment.yml` with specific versions
- Updated README.md with accurate version badges and prerequisites
- Updated `pyproject.toml` with Python 3.10 constraint

## Notes

1. **PyTorch Installation**: PyTorch was installed from conda (pytorch channel), but torchvision and torchaudio were installed from PyPI for compatibility.

2. **PyLIDC**: The `pylidc` package is not available in conda, so it must be installed via pip.

3. **TCIA Utils**: The `tcia-utils` package provides utilities for accessing The Cancer Imaging Archive and is only available via PyPI.

4. **NumPy 2.x**: This project uses NumPy 2.2.6. Be aware of potential compatibility issues if using older code that assumes NumPy 1.x behavior.

5. **Python Version**: The experiments used Python 3.10.19 specifically. While other 3.10.x versions may work, 3.10.19 is recommended for exact reproduction.

## Reproducibility

To ensure reproducibility:
1. Use the `environment.yml` file for conda-based installation (preferred)
2. Verify GPU setup: `python -c "import torch; print(torch.cuda.is_available())"`
3. Run the test suite: `pytest tests/ -v`
4. Check that all tests pass before running experiments

## Future Updates

If updating dependencies:
1. Test thoroughly with the existing test suite
2. Document any breaking changes
3. Update this file with new versions and reasons for updates
4. Consider maintaining separate `requirements-dev.txt` for development dependencies

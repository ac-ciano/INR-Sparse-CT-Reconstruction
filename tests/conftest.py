import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tumour_tomography.models import SineLayer, TumorSIREN


@pytest.fixture
def small_volume():
    return np.random.randn(16, 16, 16).astype(np.float32)


@pytest.fixture
def sample_coords_batch():
    batch_size = 32
    return torch.randn(batch_size, 3) * 0.5


@pytest.fixture
def sine_layer_first():
    return SineLayer(in_features=3, out_features=64, is_first=True, omega_0=30)


@pytest.fixture
def sine_layer_hidden():
    return SineLayer(in_features=64, out_features=64, is_first=False, omega_0=30)


@pytest.fixture
def siren_model():
    return TumorSIREN(hidden_features=128, hidden_layers=3, omega_0=30, dropout=0.1)


@pytest.fixture
def siren_model_no_dropout():
    return TumorSIREN(hidden_features=64, hidden_layers=2, omega_0=30, dropout=0.0)


@pytest.fixture
def binary_mask_cube():
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[3:7, 3:7, 3:7] = 1
    return mask


@pytest.fixture
def binary_mask_sphere():
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    center = np.array([10, 10, 10])
    for z in range(20):
        for y in range(20):
            for x in range(20):
                if np.linalg.norm(np.array([z, y, x]) - center) <= 5:
                    mask[z, y, x] = 1
    return mask


@pytest.fixture
def perfect_predictions():
    values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
    return values, values.clone()


@pytest.fixture
def known_error_predictions():
    pred = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    target = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    return pred, target

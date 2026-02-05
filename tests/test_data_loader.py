import pytest
import torch
import numpy as np
import os
import tempfile


class TestCoordinateNormalization:

    @pytest.mark.parametrize("shape", [
        (10, 10, 10),
        (16, 16, 16),
        (20, 30, 40),
        (5, 10, 15),
    ])
    def test_linspace_coordinates_in_range(self, shape):
        D, H, W = shape
        z_coords = np.linspace(-1, 1, D)
        y_coords = np.linspace(-1, 1, H)
        x_coords = np.linspace(-1, 1, W)
        
        assert z_coords.min() == -1.0
        assert z_coords.max() == 1.0
        assert y_coords.min() == -1.0
        assert y_coords.max() == 1.0
        assert x_coords.min() == -1.0
        assert x_coords.max() == 1.0

    def test_meshgrid_shape_matches_volume(self):
        D, H, W = 8, 12, 16
        z_coords = np.linspace(-1, 1, D)
        y_coords = np.linspace(-1, 1, H)
        x_coords = np.linspace(-1, 1, W)
        
        coords_z, coords_y, coords_x = np.meshgrid(
            z_coords, y_coords, x_coords, indexing='ij'
        )
        
        assert coords_z.shape == (D, H, W)
        assert coords_y.shape == (D, H, W)
        assert coords_x.shape == (D, H, W)

    def test_meshgrid_ij_indexing_order(self):
        D, H, W = 4, 6, 8
        z_coords = np.linspace(-1, 1, D)
        y_coords = np.linspace(-1, 1, H)
        x_coords = np.linspace(-1, 1, W)
        
        coords_z, coords_y, coords_x = np.meshgrid(
            z_coords, y_coords, x_coords, indexing='ij'
        )
        
        assert np.allclose(coords_z[:, 0, 0], z_coords)
        assert np.allclose(coords_y[0, :, 0], y_coords)
        assert np.allclose(coords_x[0, 0, :], x_coords)

    def test_flattened_coords_total_count(self):
        D, H, W = 5, 7, 9
        z_coords = np.linspace(-1, 1, D)
        y_coords = np.linspace(-1, 1, H)
        x_coords = np.linspace(-1, 1, W)
        
        coords_z, coords_y, coords_x = np.meshgrid(
            z_coords, y_coords, x_coords, indexing='ij'
        )
        
        coords = np.stack([
            coords_x.flatten(),
            coords_y.flatten(),
            coords_z.flatten()
        ], axis=1)
        
        assert coords.shape == (D * H * W, 3)

    def test_all_flattened_coords_in_range(self):
        D, H, W = 10, 12, 14
        z_coords = np.linspace(-1, 1, D)
        y_coords = np.linspace(-1, 1, H)
        x_coords = np.linspace(-1, 1, W)
        
        coords_z, coords_y, coords_x = np.meshgrid(
            z_coords, y_coords, x_coords, indexing='ij'
        )
        
        coords = np.stack([
            coords_x.flatten(),
            coords_y.flatten(),
            coords_z.flatten()
        ], axis=1)
        
        assert coords.min() >= -1.0
        assert coords.max() <= 1.0


class TestHUNormalization:

    MIN_HU = -1000.0
    MAX_HU = 400.0

    @pytest.mark.parametrize("hu_value,expected_normalized", [
        (-1000.0, 0.0),
        (400.0, 1.0),
        (-300.0, 0.5),
        (-1500.0, 0.0),
        (1000.0, 1.0),
    ])
    def test_hu_normalization_values(self, hu_value, expected_normalized):
        volume = np.array([[[hu_value]]], dtype=np.float32)
        normalized = (volume - self.MIN_HU) / (self.MAX_HU - self.MIN_HU)
        normalized = np.clip(normalized, 0, 1)
        assert np.isclose(normalized[0, 0, 0], expected_normalized)

    def test_hu_normalization_clipping(self):
        volume = np.array([[[-2000.0, 1000.0]]], dtype=np.float32)
        normalized = (volume - self.MIN_HU) / (self.MAX_HU - self.MIN_HU)
        normalized = np.clip(normalized, 0, 1)
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0


class TestTrainValidationSplit:

    def test_train_mask_every_5th_slice(self):
        D = 25
        train_mask = np.zeros(D, dtype=bool)
        train_mask[::5] = True
        
        expected_train_indices = [0, 5, 10, 15, 20]
        actual_train_indices = np.where(train_mask)[0].tolist()
        assert actual_train_indices == expected_train_indices

    def test_train_val_indices_no_overlap(self):
        D, H, W = 20, 10, 10
        
        train_mask = np.zeros(D, dtype=bool)
        train_mask[::5] = True
        
        z_flat_indices = np.arange(D * H * W) // (H * W)
        train_indices = np.where(np.isin(z_flat_indices, np.where(train_mask)[0]))[0]
        val_indices = np.where(~np.isin(z_flat_indices, np.where(train_mask)[0]))[0]
        
        assert len(np.intersect1d(train_indices, val_indices)) == 0

    def test_train_val_cover_all_voxels(self):
        D, H, W = 15, 8, 8
        total_voxels = D * H * W
        
        train_mask = np.zeros(D, dtype=bool)
        train_mask[::5] = True
        
        z_flat_indices = np.arange(total_voxels) // (H * W)
        train_indices = np.where(np.isin(z_flat_indices, np.where(train_mask)[0]))[0]
        val_indices = np.where(~np.isin(z_flat_indices, np.where(train_mask)[0]))[0]
        
        assert len(train_indices) + len(val_indices) == total_voxels


class TestVolumeReconstruction:

    @pytest.mark.parametrize("shape", [
        (8, 8, 8),
        (10, 12, 14),
        (16, 16, 16),
    ])
    def test_reconstruct_volume_shape(self, shape):
        predictions_flat = np.random.randn(np.prod(shape))
        reconstructed = predictions_flat.reshape(shape)
        assert reconstructed.shape == shape

    def test_reconstruct_from_tensor(self):
        shape = (10, 10, 10)
        predictions = torch.randn(np.prod(shape))
        predictions_np = predictions.detach().cpu().numpy()
        reconstructed = predictions_np.reshape(shape)
        assert reconstructed.shape == shape

    def test_reconstruction_preserves_values(self):
        shape = (5, 6, 7)
        original = np.random.randn(*shape).astype(np.float32)
        flat = original.flatten()
        reconstructed = flat.reshape(shape)
        assert np.allclose(original, reconstructed)


class TestNoduleCoordinateDatasetIntegration:

    @pytest.fixture
    def mock_volume_file(self, tmp_path):
        nodule_id = "test_nodule"
        volume = np.random.randn(20, 20, 20).astype(np.float32) * 500 - 500
        vol_path = tmp_path / f"{nodule_id}_vol.npy"
        np.save(vol_path, volume)
        return tmp_path, nodule_id

    def test_dataset_creation_with_mock_data(self, mock_volume_file):
        data_dir, nodule_id = mock_volume_file
        
        from tumour_tomography.data_loader import NoduleCoordinateDataset
        dataset = NoduleCoordinateDataset(nodule_id, str(data_dir))
        
        assert len(dataset) > 0
        assert dataset.shape == (20, 20, 20)

    def test_dataset_coords_in_range(self, mock_volume_file):
        data_dir, nodule_id = mock_volume_file
        
        from tumour_tomography.data_loader import NoduleCoordinateDataset
        dataset = NoduleCoordinateDataset(nodule_id, str(data_dir))
        
        coords = dataset.coords
        assert coords.min() >= -1.0
        assert coords.max() <= 1.0

    def test_dataset_getitem_returns_tuple(self, mock_volume_file):
        data_dir, nodule_id = mock_volume_file
        
        from tumour_tomography.data_loader import NoduleCoordinateDataset
        dataset = NoduleCoordinateDataset(nodule_id, str(data_dir))
        
        coord, value = dataset[0]
        assert coord.shape == (3,)
        assert isinstance(value, (float, np.floating))

    def test_dataset_get_all_coords(self, mock_volume_file):
        data_dir, nodule_id = mock_volume_file
        
        from tumour_tomography.data_loader import NoduleCoordinateDataset
        dataset = NoduleCoordinateDataset(nodule_id, str(data_dir))
        
        all_coords = dataset.get_all_coords()
        assert isinstance(all_coords, torch.Tensor)
        assert all_coords.shape == (20 * 20 * 20, 3)

    def test_dataset_get_excluded_coords(self, mock_volume_file):
        data_dir, nodule_id = mock_volume_file
        
        from tumour_tomography.data_loader import NoduleCoordinateDataset
        dataset = NoduleCoordinateDataset(nodule_id, str(data_dir))
        
        val_coords, val_values = dataset.get_excluded_coords()
        assert isinstance(val_coords, torch.Tensor)
        assert isinstance(val_values, torch.Tensor)
        assert len(val_coords) == len(val_values)

    def test_dataset_reconstruct_volume(self, mock_volume_file):
        data_dir, nodule_id = mock_volume_file
        
        from tumour_tomography.data_loader import NoduleCoordinateDataset
        dataset = NoduleCoordinateDataset(nodule_id, str(data_dir))
        
        fake_predictions = np.random.randn(20 * 20 * 20)
        reconstructed = dataset.reconstruct_volume(fake_predictions)
        assert reconstructed.shape == (20, 20, 20)

    def test_volume_values_normalized(self, mock_volume_file):
        data_dir, nodule_id = mock_volume_file
        
        from tumour_tomography.data_loader import NoduleCoordinateDataset
        dataset = NoduleCoordinateDataset(nodule_id, str(data_dir))
        
        assert dataset.volume.min() >= 0.0
        assert dataset.volume.max() <= 1.0

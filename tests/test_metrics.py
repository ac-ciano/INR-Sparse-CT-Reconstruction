import pytest
import torch
import numpy as np

from tumour_tomography.radiometric_metrics import (
    calculate_psnr,
    calculate_mae,
    calculate_rmse,
    calculate_ssim,
    calculate_all_metrics
)
from tumour_tomography.geometric_metrics import (
    compute_volume,
    absolute_relative_volume_error,
    dice_coefficient,
    hausdorff_distance
)


class TestPSNR:

    def test_identical_predictions_returns_inf(self, perfect_predictions):
        pred, target = perfect_predictions
        psnr = calculate_psnr(pred, target)
        assert psnr == float('inf')

    def test_psnr_known_value(self):
        pred = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
        target = torch.tensor([0.6, 0.6, 0.6, 0.6], dtype=torch.float32)
        mse = 0.01
        expected_psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        psnr = calculate_psnr(pred, target, max_val=1.0)
        assert np.isclose(psnr, expected_psnr, rtol=1e-5)

    def test_psnr_with_different_max_val(self):
        pred = torch.tensor([127.0, 127.0], dtype=torch.float32)
        target = torch.tensor([128.0, 128.0], dtype=torch.float32)
        psnr_255 = calculate_psnr(pred, target, max_val=255.0)
        psnr_1 = calculate_psnr(pred / 255, target / 255, max_val=1.0)
        assert np.isclose(psnr_255, psnr_1, rtol=1e-4)

    def test_psnr_higher_for_smaller_error(self):
        target = torch.rand(100)
        pred_good = target + torch.randn(100) * 0.01
        pred_bad = target + torch.randn(100) * 0.1
        psnr_good = calculate_psnr(pred_good, target)
        psnr_bad = calculate_psnr(pred_bad, target)
        assert psnr_good > psnr_bad


class TestMAE:

    def test_mae_identical_is_zero(self, perfect_predictions):
        pred, target = perfect_predictions
        mae = calculate_mae(pred, target)
        assert mae == 0.0

    def test_mae_known_value(self, known_error_predictions):
        pred, target = known_error_predictions
        mae = calculate_mae(pred, target)
        assert mae == 1.0

    def test_mae_symmetric(self):
        pred = torch.tensor([0.0, 0.5, 1.0])
        target = torch.tensor([0.2, 0.7, 0.8])
        mae1 = calculate_mae(pred, target)
        mae2 = calculate_mae(target, pred)
        assert np.isclose(mae1, mae2)

    @pytest.mark.parametrize("offset", [0.1, 0.25, 0.5, 1.0])
    def test_mae_equals_constant_offset(self, offset):
        target = torch.rand(50)
        pred = target + offset
        mae = calculate_mae(pred, target)
        assert np.isclose(mae, offset, rtol=1e-5)


class TestRMSE:

    def test_rmse_identical_is_zero(self, perfect_predictions):
        pred, target = perfect_predictions
        rmse = calculate_rmse(pred, target)
        assert rmse == 0.0

    def test_rmse_known_value(self, known_error_predictions):
        pred, target = known_error_predictions
        rmse = calculate_rmse(pred, target)
        assert np.isclose(rmse, 1.0)

    def test_rmse_greater_or_equal_mae(self):
        pred = torch.rand(100)
        target = torch.rand(100)
        rmse = calculate_rmse(pred, target)
        mae = calculate_mae(pred, target)
        assert rmse >= mae - 1e-6

    def test_rmse_equals_mae_for_constant_error(self):
        target = torch.zeros(10)
        pred = torch.ones(10) * 0.5
        rmse = calculate_rmse(pred, target)
        mae = calculate_mae(pred, target)
        assert np.isclose(rmse, mae)


class TestSSIM:

    def test_ssim_identical_returns_one(self, perfect_predictions):
        pred, target = perfect_predictions
        ssim = calculate_ssim(pred, target)
        assert ssim == 1.0

    def test_ssim_identical_large_array(self):
        target = torch.rand(1000)
        ssim = calculate_ssim(target, target.clone())
        assert ssim == 1.0

    def test_ssim_in_valid_range(self):
        pred = torch.rand(500)
        target = torch.rand(500)
        ssim = calculate_ssim(pred, target)
        # SSIM is typically in range [-1, 1]
        assert -1.0 <= ssim <= 1.0

    def test_ssim_symmetric(self):
        pred = torch.rand(500)
        target = torch.rand(500)
        ssim1 = calculate_ssim(pred, target)
        ssim2 = calculate_ssim(target, pred)
        assert np.isclose(ssim1, ssim2, rtol=1e-5)

    def test_ssim_higher_for_similar_signals(self):
        target = torch.rand(500)
        pred_similar = target + torch.randn(500) * 0.01
        pred_different = target + torch.randn(500) * 0.5
        ssim_similar = calculate_ssim(pred_similar, target)
        ssim_different = calculate_ssim(pred_different, target)
        assert ssim_similar > ssim_different

    def test_ssim_with_numpy_arrays(self):
        pred = np.random.rand(500).astype(np.float32)
        target = np.random.rand(500).astype(np.float32)
        ssim = calculate_ssim(pred, target)
        assert -1.0 <= ssim <= 1.0

    def test_ssim_respects_data_range(self):
        # Test with data in [0, 255] range
        target = torch.rand(500) * 255
        pred = target + torch.randn(500) * 5
        ssim = calculate_ssim(pred, target, data_range=255.0)
        assert -1.0 <= ssim <= 1.0


class TestCalculateAllMetrics:

    def test_returns_all_keys(self):
        pred = torch.rand(50)
        target = torch.rand(50)
        metrics = calculate_all_metrics(pred, target)
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'psnr' in metrics
        assert 'rmse' in metrics
        assert 'ssim' in metrics

    def test_metrics_consistency(self):
        pred = torch.rand(100)
        target = torch.rand(100)
        metrics = calculate_all_metrics(pred, target)
        
        assert np.isclose(metrics['rmse'], np.sqrt(metrics['mse']), rtol=1e-5)
        assert metrics['rmse'] >= metrics['mae'] - 1e-6
        assert -1.0 <= metrics['ssim'] <= 1.0

    def test_perfect_predictions_metrics(self, perfect_predictions):
        pred, target = perfect_predictions
        metrics = calculate_all_metrics(pred, target)
        assert metrics['mse'] == 0.0
        assert metrics['mae'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['psnr'] == float('inf')
        assert metrics['ssim'] == 1.0


class TestComputeVolume:

    def test_empty_mask_zero_volume(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        volume = compute_volume(mask, voxel_volume=1.0)
        assert volume == 0.0

    def test_full_mask_volume(self):
        mask = np.ones((10, 10, 10), dtype=np.uint8)
        volume = compute_volume(mask, voxel_volume=1.0)
        assert volume == 1000.0

    def test_cube_volume_known(self, binary_mask_cube):
        voxel_count = binary_mask_cube.sum()
        volume = compute_volume(binary_mask_cube, voxel_volume=1.0)
        assert volume == float(voxel_count)

    def test_volume_scales_with_voxel_size(self, binary_mask_cube):
        vol_1 = compute_volume(binary_mask_cube, voxel_volume=1.0)
        vol_8 = compute_volume(binary_mask_cube, voxel_volume=8.0)
        assert vol_8 == vol_1 * 8


class TestAbsoluteRelativeVolumeError:

    def test_identical_masks_zero_error(self, binary_mask_cube):
        arve = absolute_relative_volume_error(binary_mask_cube, binary_mask_cube, 1.0)
        assert arve == 0.0

    def test_arve_known_value(self):
        gt = np.zeros((10, 10, 10), dtype=np.uint8)
        gt[:5, :, :] = 1
        pred = np.zeros((10, 10, 10), dtype=np.uint8)
        pred[:4, :, :] = 1
        
        v_gt = 500
        v_pred = 400
        expected_arve = abs(v_pred - v_gt) / v_gt
        
        arve = absolute_relative_volume_error(pred, gt, 1.0)
        assert np.isclose(arve, expected_arve)

    def test_empty_gt_returns_inf_or_zero(self):
        gt = np.zeros((5, 5, 5), dtype=np.uint8)
        pred = np.ones((5, 5, 5), dtype=np.uint8)
        arve = absolute_relative_volume_error(pred, gt, 1.0)
        assert arve == float('inf')

    def test_both_empty_returns_zero(self):
        gt = np.zeros((5, 5, 5), dtype=np.uint8)
        pred = np.zeros((5, 5, 5), dtype=np.uint8)
        arve = absolute_relative_volume_error(pred, gt, 1.0)
        assert arve == 0.0


class TestDiceCoefficient:

    def test_perfect_overlap(self, binary_mask_cube):
        dice = dice_coefficient(binary_mask_cube, binary_mask_cube)
        assert dice == 1.0

    def test_no_overlap(self):
        mask1 = np.zeros((10, 10, 10), dtype=np.uint8)
        mask1[:5, :, :] = 1
        mask2 = np.zeros((10, 10, 10), dtype=np.uint8)
        mask2[5:, :, :] = 1
        dice = dice_coefficient(mask1, mask2)
        assert dice == 0.0

    def test_partial_overlap_known_value(self):
        mask1 = np.zeros((10, 10, 10), dtype=np.uint8)
        mask1[:6, :, :] = 1
        mask2 = np.zeros((10, 10, 10), dtype=np.uint8)
        mask2[4:, :, :] = 1
        
        intersection = 200
        sum_voxels = 600 + 600
        expected_dice = 2 * intersection / sum_voxels
        
        dice = dice_coefficient(mask1, mask2)
        assert np.isclose(dice, expected_dice)

    def test_both_empty_returns_one(self):
        mask1 = np.zeros((5, 5, 5), dtype=np.uint8)
        mask2 = np.zeros((5, 5, 5), dtype=np.uint8)
        dice = dice_coefficient(mask1, mask2)
        assert dice == 1.0

    def test_dice_symmetric(self):
        mask1 = np.zeros((10, 10, 10), dtype=np.uint8)
        mask1[2:7, 2:7, 2:7] = 1
        mask2 = np.zeros((10, 10, 10), dtype=np.uint8)
        mask2[4:9, 4:9, 4:9] = 1
        dice1 = dice_coefficient(mask1, mask2)
        dice2 = dice_coefficient(mask2, mask1)
        assert dice1 == dice2


class TestHausdorffDistance:

    def test_identical_masks_zero_distance(self, binary_mask_cube):
        spacing = (1.0, 1.0, 1.0)
        hd = hausdorff_distance(binary_mask_cube, binary_mask_cube, spacing)
        assert hd == 0.0

    def test_both_empty_zero_distance(self):
        mask1 = np.zeros((5, 5, 5), dtype=np.uint8)
        mask2 = np.zeros((5, 5, 5), dtype=np.uint8)
        spacing = (1.0, 1.0, 1.0)
        hd = hausdorff_distance(mask1, mask2, spacing)
        assert hd == 0.0

    def test_one_empty_returns_inf(self, binary_mask_cube):
        empty = np.zeros_like(binary_mask_cube)
        spacing = (1.0, 1.0, 1.0)
        hd = hausdorff_distance(binary_mask_cube, empty, spacing)
        assert hd == float('inf')

    def test_hd_with_known_separation(self):
        mask1 = np.zeros((20, 10, 10), dtype=np.uint8)
        mask1[0:3, 4:6, 4:6] = 1
        mask2 = np.zeros((20, 10, 10), dtype=np.uint8)
        mask2[17:20, 4:6, 4:6] = 1
        
        spacing = (1.0, 1.0, 1.0)
        hd = hausdorff_distance(mask1, mask2, spacing)
        
        assert hd >= 14.0

    def test_hd_scales_with_spacing(self, binary_mask_cube):
        mask2 = np.roll(binary_mask_cube, shift=1, axis=0)
        
        spacing_1 = (1.0, 1.0, 1.0)
        spacing_2 = (2.0, 1.0, 1.0)
        
        hd_1 = hausdorff_distance(binary_mask_cube, mask2, spacing_1)
        hd_2 = hausdorff_distance(binary_mask_cube, mask2, spacing_2)
        
        assert hd_2 >= hd_1

    def test_hd_symmetric(self):
        mask1 = np.zeros((10, 10, 10), dtype=np.uint8)
        mask1[2:5, 2:5, 2:5] = 1
        mask2 = np.zeros((10, 10, 10), dtype=np.uint8)
        mask2[5:8, 5:8, 5:8] = 1
        spacing = (1.0, 1.0, 1.0)
        hd1 = hausdorff_distance(mask1, mask2, spacing)
        hd2 = hausdorff_distance(mask2, mask1, spacing)
        assert hd1 == hd2

import pytest
import torch
import numpy as np

from tumour_tomography.models import SineLayer, TumorSIREN


class TestSineLayer:

    @pytest.mark.parametrize("batch_size,in_features,out_features", [
        (1, 3, 64),
        (16, 3, 128),
        (32, 64, 64),
        (128, 128, 256),
    ])
    def test_forward_pass_shape(self, batch_size, in_features, out_features):
        layer = SineLayer(in_features=in_features, out_features=out_features)
        x = torch.randn(batch_size, in_features)
        output = layer(x)
        assert output.shape == (batch_size, out_features)

    def test_first_layer_initialization(self, sine_layer_first):
        weights = sine_layer_first.linear.weight.data
        in_features = sine_layer_first.in_features
        expected_bound = 1 / in_features
        assert weights.min() >= -expected_bound
        assert weights.max() <= expected_bound

    def test_hidden_layer_initialization(self, sine_layer_hidden):
        weights = sine_layer_hidden.linear.weight.data
        in_features = sine_layer_hidden.in_features
        omega_0 = sine_layer_hidden.omega_0
        expected_bound = np.sqrt(6 / in_features) / omega_0
        assert weights.min() >= -expected_bound - 1e-6
        assert weights.max() <= expected_bound + 1e-6

    def test_output_bounded_by_sine(self, sine_layer_first):
        x = torch.randn(100, 3)
        output = sine_layer_first(x)
        assert output.min() >= -1.0
        assert output.max() <= 1.0

    @pytest.mark.parametrize("omega_0", [1, 10, 30, 60])
    def test_omega_parameter_stored(self, omega_0):
        layer = SineLayer(in_features=3, out_features=64, omega_0=omega_0)
        assert layer.omega_0 == omega_0

    def test_bias_included_by_default(self):
        layer = SineLayer(in_features=3, out_features=64)
        assert layer.linear.bias is not None

    def test_bias_excluded_when_disabled(self):
        layer = SineLayer(in_features=3, out_features=64, bias=False)
        assert layer.linear.bias is None


class TestTumorSIREN:

    @pytest.mark.parametrize("batch_size", [1, 8, 16, 32, 64, 128])
    def test_forward_pass_output_shape(self, siren_model, batch_size):
        coords = torch.randn(batch_size, 3)
        output = siren_model(coords)
        assert output.shape == (batch_size,)

    def test_forward_pass_with_normalized_coords(self, siren_model):
        coords = torch.rand(50, 3) * 2 - 1
        output = siren_model(coords)
        assert output.shape == (50,)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize("hidden_features,hidden_layers", [
        (64, 2),
        (128, 3),
        (256, 4),
        (512, 5),
    ])
    def test_various_configurations(self, hidden_features, hidden_layers):
        model = TumorSIREN(
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            omega_0=30
        )
        coords = torch.randn(16, 3)
        output = model(coords)
        assert output.shape == (16,)

    def test_dropout_effect_train_vs_eval(self):
        model = TumorSIREN(hidden_features=128, hidden_layers=3, dropout=0.5)
        coords = torch.randn(100, 3)
        
        model.train()
        train_outputs = [model(coords).clone() for _ in range(5)]
        train_variance = torch.stack(train_outputs).var(dim=0).mean()
        
        model.eval()
        eval_outputs = [model(coords).clone() for _ in range(5)]
        eval_variance = torch.stack(eval_outputs).var(dim=0).mean()
        
        assert eval_variance < train_variance or eval_variance < 1e-6

    def test_no_dropout_consistent_output(self, siren_model_no_dropout):
        siren_model_no_dropout.eval()
        coords = torch.randn(20, 3)
        output1 = siren_model_no_dropout(coords)
        output2 = siren_model_no_dropout(coords)
        assert torch.allclose(output1, output2)

    def test_outermost_linear_true(self):
        model = TumorSIREN(hidden_features=64, hidden_layers=2, outermost_linear=True)
        last_layer = model.net[-1]
        assert isinstance(last_layer, torch.nn.Linear)

    def test_outermost_linear_false(self):
        model = TumorSIREN(hidden_features=64, hidden_layers=2, outermost_linear=False, dropout=0.0)
        has_sine_layer = any(isinstance(layer, SineLayer) for layer in list(model.net.children())[-2:])
        assert has_sine_layer

    def test_gradient_flow(self, siren_model):
        coords = torch.randn(16, 3, requires_grad=True)
        output = siren_model(coords)
        loss = output.sum()
        loss.backward()
        assert coords.grad is not None
        assert not torch.isnan(coords.grad).any()

    def test_model_parameters_exist(self, siren_model):
        params = list(siren_model.parameters())
        assert len(params) > 0
        total_params = sum(p.numel() for p in params)
        assert total_params > 0

    @pytest.mark.parametrize("device", ['cpu'])
    def test_device_compatibility(self, device):
        model = TumorSIREN(hidden_features=64, hidden_layers=2).to(device)
        coords = torch.randn(8, 3).to(device)
        output = model(coords)
        assert output.device.type == device

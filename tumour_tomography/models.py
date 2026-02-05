"""SIREN model implementation with sinusoidal activation functions.

This module implements Sinusoidal Representation Networks (SIREN) for implicit
neural representation of volumetric medical imaging data. SIREN networks use
periodic sinusoidal activation functions that enable learning of high-frequency
details in continuous signals.

Key components:
    - SineLayer: Building block with sinusoidal activation and SIREN initialization
    - TumorSIREN: Complete network architecture for 3D coordinate-to-density mapping

The implementation follows the original SIREN paper (Sitzmann et al., 2020) with
specialized weight initialization to maintain signal statistics through deep networks.

Reference:
    Sitzmann, V., et al. "Implicit Neural Representations with Periodic Activation
    Functions." NeurIPS 2020.
"""
import torch
import torch.nn as nn
import numpy as np


class SineLayer(nn.Module):
    """
    Single layer with sinusoidal activation function.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        bias: Whether to include bias term
        is_first: Whether this is the first layer (affects initialization)
        omega_0: Frequency parameter for sinusoidal activation
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        """Initialize a SineLayer with SIREN-specific weight initialization.
        
        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Whether to include a bias term in the linear transformation.
            is_first: Whether this is the first layer in the network.
                First layers use uniform initialization in [-1/in_features, 1/in_features].
            omega_0: Frequency scaling factor for the sinusoidal activation.
                Higher values increase the frequency of the sine activations.
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize layer weights according to SIREN initialization scheme.
        
        For the first layer, weights are uniformly sampled from [-1/in_features, 1/in_features].
        For subsequent layers, weights are sampled from [-sqrt(6/in_features)/omega_0, 
        sqrt(6/in_features)/omega_0] to preserve signal variance through the network.
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        """Apply linear transformation followed by scaled sinusoidal activation.
        
        Args:
            input: Input tensor of shape (batch_size, in_features).
            
        Returns:
            Output tensor of shape (batch_size, out_features) with values in [-1, 1].
        """
        return torch.sin(self.omega_0 * self.linear(input))

class TumorSIREN(nn.Module):
    """SIREN network for implicit representation of 3D tumour volumes.
    
    This model takes 3D spatial coordinates as input and outputs predicted
    CT density values, enabling continuous representation of volumetric
    medical imaging data. The architecture uses sinusoidal activation
    functions throughout for capturing high-frequency spatial details.
    
    Attributes:
        net: Sequential container holding all network layers.
    """
    
    def __init__(self, hidden_features=256, hidden_layers=3, omega_0=30, outermost_linear=True, dropout=0.5):
        """Initialize the TumorSIREN network.
        
        Args:
            hidden_features: Number of features in hidden layers.
            hidden_layers: Number of hidden layers (excluding input/output).
            omega_0: Frequency scaling factor for sine activations.
            outermost_linear: If True, use linear activation for output layer.
                If False, use sinusoidal activation.
            dropout: Dropout probability applied after each hidden layer.
        """
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(SineLayer(3, hidden_features, is_first=True, omega_0=omega_0))
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(SineLayer(hidden_features, hidden_features, omega_0=omega_0))
            layers.append(nn.Dropout(dropout))
        # Final layer
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, 1)
            
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / omega_0,
                    np.sqrt(6 / hidden_features) / omega_0
                )
            
            layers.append(final_linear)
        else:
            layers.append(SineLayer(hidden_features, 1, omega_0=omega_0))
            layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, coords):
        """Predict CT density values for given 3D coordinates.
        
        Args:
            coords: Input coordinates tensor of shape (batch_size, 3) with
                values normalized to range [-1, 1] for x, y, z dimensions.
                
        Returns:
            Predicted density values tensor of shape (batch_size,) with
            values typically in range [0, 1] after training.
        """
        output = self.net(coords)
        return output.squeeze(-1)
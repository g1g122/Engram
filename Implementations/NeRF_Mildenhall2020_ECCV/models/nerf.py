"""
NeRF MLP (Multi-Layer Perceptron) Network.

Implements the core neural radiance field as described in Mildenhall et al. 2020.
The network maps positional-encoded 3D coordinates and view directions to
volume density (sigma) and view-dependent RGB color.

Architecture:
    - 8 fully-connected layers (width 256) with ReLU activations for the
      density (geometry) branch, with a skip connection at layer 5.
    - Density sigma is predicted from the geometry feature alone.
    - A separate 1-layer branch (width 128) takes the geometry feature
      concatenated with the encoded view direction to predict RGB color.

Reference: Mildenhall et al. 2020, Section 3 & Supplementary Figure 7.
"""

import torch
import torch.nn as nn


class NeRF(nn.Module):
    """Neural Radiance Field MLP.

    Args:
        pos_dim: Dimensionality of positional-encoded input (default 63).
        dir_dim: Dimensionality of direction-encoded input (default 27).
        net_width: Width of hidden layers in the density branch (default 256).
        color_width: Width of the hidden layer in the color branch (default 128).
        net_depth: Number of layers in the density branch (default 8).
        skip_layer: Index of the layer that receives a skip connection (default 4).
    """

    def __init__(
        self,
        pos_dim: int = 63,
        dir_dim: int = 27,
        net_width: int = 256,
        color_width: int = 128,
        net_depth: int = 8,
        skip_layer: int = 4,
    ):
        super().__init__()
        self.skip_layer = skip_layer

        # --- Density (geometry) branch ---
        # Build net_depth layers; the skip_layer receives concatenated input.
        self.density_layers = nn.ModuleList()
        for i in range(net_depth):
            if i == 0:
                in_features = pos_dim
            elif i == skip_layer:
                in_features = net_width + pos_dim
            else:
                in_features = net_width
            self.density_layers.append(nn.Linear(in_features, net_width))

        # Raw density output (single scalar, no activation here;
        # ReLU is applied in forward() to guarantee non-negative sigma)
        self.sigma_head = nn.Linear(net_width, 1)

        # --- Color (appearance) branch ---
        # Bottleneck: project geometry feature before combining with direction
        self.feature_proj = nn.Linear(net_width, net_width)

        # Direction-conditioned color prediction
        self.color_hidden = nn.Linear(net_width + dir_dim, color_width)
        self.rgb_head = nn.Linear(color_width, 3)

    def forward(self, pos_enc: torch.Tensor, dir_enc: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict (RGB, sigma) for a batch of points.

        Args:
            pos_enc: Positional-encoded 3D coordinates, shape [B, pos_dim].
            dir_enc: Directional-encoded view directions, shape [B, dir_dim].

        Returns:
            Tensor of shape [B, 4] — (R, G, B, sigma) per sample point.
        """
        # --- Density branch with skip connection ---
        h = pos_enc
        for i, layer in enumerate(self.density_layers):
            if i == self.skip_layer:
                h = torch.cat([h, pos_enc], dim=-1)
            h = torch.relu(layer(h))

        # Density: apply ReLU to ensure non-negative values
        sigma = torch.relu(self.sigma_head(h))

        # --- Color branch ---
        # Project geometry feature (no activation, raw linear projection)
        feature = self.feature_proj(h)

        # Concatenate with view direction encoding
        feature = torch.cat([feature, dir_enc], dim=-1)

        # Predict RGB via one hidden layer
        feature = torch.relu(self.color_hidden(feature))
        rgb = torch.sigmoid(self.rgb_head(feature))

        return torch.cat([rgb, sigma], dim=-1)

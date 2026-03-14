"""
Positional Encoding (Embedder) for NeRF.

Maps low-dimensional input coordinates into a higher-dimensional space using
sinusoidal functions at exponentially increasing frequencies. This overcomes
the spectral bias of MLPs, enabling them to represent high-frequency detail
in geometry and appearance.

Reference: Mildenhall et al. 2020, Section 5.1, Equation (4).
"""

import torch
import torch.nn as nn


class Embedder(nn.Module):
    """Positional encoding via sinusoidal frequency bands.

    For a scalar input p, produces:
        γ(p) = (sin(2^0 π p), cos(2^0 π p), ..., sin(2^{L-1} π p), cos(2^{L-1} π p))

    Applied independently to each component of the input vector,
    so a 3D input with L frequency bands yields a 3 × 2L = 6L dimensional output.

    Args:
        in_dims: Dimensionality of the raw input (e.g. 3 for xyz or direction).
        num_freqs: Number of frequency bands L (10 for position, 4 for direction).
        include_input: If True, prepend the raw input to the encoding,
            giving output dim = in_dims + in_dims * 2 * num_freqs.
    """

    def __init__(self, in_dims: int, num_freqs: int, include_input: bool = True):
        super().__init__()
        self.in_dims = in_dims
        self.num_freqs = num_freqs
        self.include_input = include_input

        self.out_dims = in_dims * num_freqs * 2
        if include_input:
            self.out_dims += in_dims

        # freq_bands: [2^0, 2^1, ..., 2^{L-1}]  (shape [L])
        freq_bands = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input coordinates.

        Args:
            x: Input tensor of shape [..., in_dims].

        Returns:
            Encoded tensor of shape [..., out_dims].
        """
        # x[..., None] : [..., in_dims, 1]
        # freq_bands   : [L]  → broadcast to [..., in_dims, L]
        scaled = x[..., None] * self.freq_bands * torch.pi

        # Interleave sin and cos: [..., in_dims, L] → [..., in_dims, L, 2]
        sin_cos = torch.stack([scaled.sin(), scaled.cos()], dim=-1)

        # Flatten last two dims: [..., in_dims, L, 2] → [..., in_dims * 2L]
        encoded = sin_cos.reshape(*x.shape[:-1], -1)

        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)

        return encoded


def get_embedder(num_freqs: int, in_dims: int = 3, include_input: bool = True):
    """Create an Embedder and return it along with its output dimensionality.

    Convenience factory matching the interface expected by the NeRF model.

    Args:
        num_freqs: Number of frequency bands L.
        in_dims: Input dimensionality (default 3).
        include_input: Whether to include raw input in output.

    Returns:
        (embedder, out_dims): The Embedder module and its output dimensionality.

    Examples:
        >>> embed_pos, pos_dim = get_embedder(10)  # L=10 for position → 63
        >>> embed_dir, dir_dim = get_embedder(4)   # L=4 for direction  → 27
    """
    embedder = Embedder(in_dims, num_freqs, include_input)
    return embedder, embedder.out_dims

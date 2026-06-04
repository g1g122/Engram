"""
Cortical Sheet Mapping for TDANN.

Responsibility:
    - CORE COMPONENT: Manages mapping from feature map indices (c, h, w) -> physical coords (x, y).
    - Define physical geometry for each layer based on config (surface area, neighborhood size).
    - Stage 1 Init: Assign positions on 2D grid preserving retinotopy.
    - Store channel permutation indices for Stage 2 shuffling results.
    - Provide neighborhood sampling for Spatial Loss computation.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn


@dataclass
class LayerGeometry:
    """Physical geometry parameters for a single layer's cortical sheet."""
    layer_id: str                    # e.g., "L2", "L4", "L9"
    region: str                      # e.g., "Retina", "V1", "VTC"
    surface_area_mm2: float          # Total surface area in mm^2
    neighborhood_sigma_mm: float     # Neighborhood radius for spatial loss
    num_channels: int                # Number of channels (C dimension)
    spatial_size: int                # Spatial dimension (H = W for square feature maps)

    @property
    def sheet_side_mm(self) -> float:
        """Side length of the square cortical sheet in mm."""
        return math.sqrt(self.surface_area_mm2)

    @property
    def num_units(self) -> int:
        """Total number of units in this layer."""
        return self.num_channels * self.spatial_size * self.spatial_size


class CorticalSheet(nn.Module):
    """
    Manages the 2D cortical sheet embedding for all layers (L2-L9).

    Each unit in the feature map tensor (c, h, w) is assigned a fixed physical
    position (x, y) on a 2D cortical sheet. Positions are initialized to preserve
    retinotopy: units with similar receptive fields are placed nearby.

    Key concepts:
        - Retinotopic initialization: (h, w) determines base position on sheet
        - Channel spreading: channels at same (h, w) are spread locally
        - Neighborhood: defined by neighborhood_sigma_mm for spatial loss
    """

    def __init__(self, layer_configs: Dict[str, dict], feature_map_sizes: Dict[str, Tuple[int, int, int]]):
        """
        Args:
            layer_configs: Config dict from YAML, keyed by layer_id ("L2" to "L9").
                           Each entry has: surface_area_mm2, neighborhood_sigma_mm, region.
            feature_map_sizes: Dict mapping layer_id to (C, H, W) tuple.
                               E.g., {"L2": (64, 56, 56), "L4": (128, 28, 28), ...}
        """
        super().__init__()
        self.geometries: Dict[str, LayerGeometry] = {}
        # positions[layer_id] -> Tensor of shape (num_units, 2) for (x, y) coords
        self.positions: Dict[str, torch.Tensor] = {}
        # Permutation indices: used after Stage 2 shuffling
        # perm_indices[layer_id] -> LongTensor of shape (num_units,)
        self.perm_indices: Dict[str, torch.Tensor] = {}

        for layer_id, config in layer_configs.items():
            if layer_id not in feature_map_sizes:
                continue
            c, h, w = feature_map_sizes[layer_id]
            assert h == w, f"Expected square feature maps, got H={h}, W={w}"

            geom = LayerGeometry(
                layer_id=layer_id,
                region=config['region'],
                surface_area_mm2=config['surface_area_mm2'],
                neighborhood_sigma_mm=config['neighborhood_sigma_mm'],
                num_channels=c,
                spatial_size=h
            )
            self.geometries[layer_id] = geom

            # Initialize positions with retinotopic layout
            positions = self._init_retinotopic_positions(geom)
            # Register as buffer (non-trainable, but saved with model)
            self.register_buffer(f'pos_{layer_id}', positions)
            self.positions[layer_id] = positions

            # Initialize identity permutation (no shuffling yet)
            perm = torch.arange(geom.num_units, dtype=torch.long)
            self.register_buffer(f'perm_{layer_id}', perm)
            self.perm_indices[layer_id] = perm

    def _init_retinotopic_positions(self, geom: LayerGeometry) -> torch.Tensor:
        """
        Initialize unit positions preserving retinotopy with local channel spreading.

        For a feature map of shape (C, H, W), we assign positions such that:
        1. Units at same (h, w) but different channels are spread locally
        2. Overall layout preserves the spatial grid structure (retinotopy)

        The sheet is divided into H×W grid cells. Within each cell, C channels
        are arranged in a local grid pattern.

        Returns:
            Tensor of shape (C*H*W, 2) containing (x, y) positions in mm.
        """
        c, h = geom.num_channels, geom.spatial_size
        sheet_side = geom.sheet_side_mm

        # Grid cell size (each (h, w) location gets a cell)
        cell_size = sheet_side / h

        # Arrange channels within each cell in a sqrt(C) x sqrt(C) sub-grid
        # This is the "local spreading" for the short-reach problem
        channels_per_side = int(math.ceil(math.sqrt(c)))
        channel_spacing = cell_size / (channels_per_side + 1)

        positions = torch.zeros(c * h * h, 2)

        # PyTorch flatten order: C -> H -> W (C slowest, W fastest)
        # index = ci * (H * W) + hi * W + wi
        for ci in range(c):
            # Local offset within each cell for this channel
            local_row = ci // channels_per_side
            local_col = ci % channels_per_side
            offset_x = (local_col - (channels_per_side - 1) / 2) * channel_spacing
            offset_y = (local_row - (channels_per_side - 1) / 2) * channel_spacing

            for hi in range(h):
                for wi in range(h):  # W = H for square maps
                    # Base position for this grid cell (center)
                    base_x = (wi + 0.5) * cell_size
                    base_y = (hi + 0.5) * cell_size

                    unit_idx = ci * (h * h) + hi * h + wi
                    positions[unit_idx, 0] = base_x + offset_x
                    positions[unit_idx, 1] = base_y + offset_y

        return positions

    def get_positions(self, layer_id: str) -> torch.Tensor:
        """
        Get current positions for a layer, applying any shuffling permutation.

        Args:
            layer_id: Layer identifier ("L2" to "L9")

        Returns:
            Tensor of shape (num_units, 2) with (x, y) positions in mm.
            result[i] is the physical (x, y) coordinate of unit i.
        """
        base_pos = getattr(self, f'pos_{layer_id}')
        perm = getattr(self, f'perm_{layer_id}')
        # perm[i] = j means: unit i is assigned to base_pos[j]
        # So result[i] = base_pos[perm[i]] = position of unit i
        return base_pos[perm]

    def set_permutation(self, layer_id: str, perm_indices: torch.Tensor):
        """
        Set the channel permutation after Stage 2 shuffling.

        Args:
            layer_id: Layer identifier
            perm_indices: LongTensor of shape (num_units,).
                          perm[i] = j means unit i should be placed at base_pos[j].
        """
        setattr(self, f'perm_{layer_id}', perm_indices)
        self.perm_indices[layer_id] = perm_indices

    def unit_index_to_chw(self, layer_id: str, unit_idx: int) -> Tuple[int, int, int]:
        """
        Convert flat unit index to (c, h, w) tuple.

        PyTorch flatten order: C -> H -> W
        index = c * (H * W) + h * W + w
        """
        geom = self.geometries[layer_id]
        num_c, size_h = geom.num_channels, geom.spatial_size
        hw = size_h * size_h

        ci = unit_idx // hw
        remainder = unit_idx % hw
        hi = remainder // size_h
        wi = remainder % size_h

        return (ci, hi, wi)

    def chw_to_unit_index(self, layer_id: str, c: int, h: int, w: int) -> int:
        """Convert (c, h, w) tuple to flat unit index. PyTorch order: C -> H -> W."""
        geom = self.geometries[layer_id]
        size_h = geom.spatial_size
        return c * (size_h * size_h) + h * size_h + w

    def sample_neighborhood(
        self,
        layer_id: str,
        center_idx: Optional[int] = None,
        max_units: int = 500
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample units within a neighborhood for spatial loss computation.

        Args:
            layer_id: Layer identifier
            center_idx: Index of center unit. If None, randomly sample a center.
            max_units: Maximum number of units to return (for computational efficiency).

        Returns:
            Tuple of:
                - unit_indices: LongTensor of shape (N,) with sampled unit indices
                - positions: Tensor of shape (N, 2) with positions of sampled units
        """
        geom = self.geometries[layer_id]
        positions = self.get_positions(layer_id)
        num_units = geom.num_units
        radius = geom.neighborhood_sigma_mm

        if center_idx is None:
            center_idx = torch.randint(0, num_units, (1,)).item()

        center_pos = positions[center_idx]

        # Compute distances from center
        distances = torch.norm(positions - center_pos, dim=1)

        # Find units within neighborhood radius
        within_radius = distances <= radius
        neighbor_indices = torch.where(within_radius)[0]

        # Subsample if too many units
        if len(neighbor_indices) > max_units:
            perm = torch.randperm(len(neighbor_indices))[:max_units]
            neighbor_indices = neighbor_indices[perm]

        neighbor_positions = positions[neighbor_indices]

        return neighbor_indices, neighbor_positions

    def get_pairwise_distances(self, layer_id: str, unit_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Euclidean distances between units.

        Args:
            layer_id: Layer identifier
            unit_indices: LongTensor of shape (N,) with unit indices

        Returns:
            Tensor of shape (N, N) with pairwise distances in mm.
        """
        positions = self.get_positions(layer_id)
        pos_subset = positions[unit_indices]  # (N, 2)
        # Pairwise distance computation
        diff = pos_subset.unsqueeze(0) - pos_subset.unsqueeze(1)  # (N, N, 2)
        distances = torch.norm(diff, dim=2)  # (N, N)
        return distances

    def save_positions(self, path: str):
        """Save all positions and permutations to file."""
        state = {
            'geometries': {k: vars(v) for k, v in self.geometries.items()},
            'positions': {k: v.cpu() for k, v in self.positions.items()},
            'perm_indices': {k: v.cpu() for k, v in self.perm_indices.items()}
        }
        torch.save(state, path)

    def load_positions(self, path: str):
        """Load positions and permutations from file."""
        state = torch.load(path)
        for layer_id, perm in state['perm_indices'].items():
            self.set_permutation(layer_id, perm)


def get_feature_map_sizes() -> Dict[str, Tuple[int, int, int]]:
    """
    Return the expected feature map sizes for ResNet-18 layers.

    These are fixed based on ResNet-18 architecture with 224x224 input.
    """
    return {
        'L2': (64, 56, 56),    # After layer1[0]
        'L3': (64, 56, 56),    # After layer1[1]
        'L4': (128, 28, 28),   # After layer2[0]
        'L5': (128, 28, 28),   # After layer2[1]
        'L6': (256, 14, 14),   # After layer3[0]
        'L7': (256, 14, 14),   # After layer3[1]
        'L8': (512, 7, 7),     # After layer4[0]
        'L9': (512, 7, 7),     # After layer4[1]
    }

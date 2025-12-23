"""
Relative Spatial Loss (SL_Rel) for TDANN.

Responsibility:
    - CORE OBJECTIVE: Encourage nearby units to have correlated responses.
    - Formula: SL_Rel = 1 - Corr(response_similarities, inverse_distances)  (Equation 3)
    - Inverse distance: D_i = 1 / (d_i + 1)  (Equation 4)
    - Compute within local neighborhoods for computational feasibility.
    - Sum across all layers with weight α (Equation 5).
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from models.cortical_sheet import CorticalSheet


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Pearson correlation between two 1D tensors.

    Args:
        x, y: 1D tensors of the same length.

    Returns:
        Scalar tensor with correlation coefficient.
    """
    x_centered = x - x.mean()
    y_centered = y - y.mean()

    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())

    # Avoid division by zero
    return numerator / (denominator + 1e-8)


def compute_pairwise_response_correlations(
    responses: torch.Tensor,
    unit_indices: torch.Tensor
) -> torch.Tensor:
    """
    Compute pairwise Pearson correlations between unit responses.

    Args:
        responses: Tensor of shape (B, num_units) with responses across batch.
        unit_indices: LongTensor of shape (N,) specifying which units to use.

    Returns:
        Tensor of shape (N*(N-1)/2,) with upper-triangle pairwise correlations.
    """
    # Extract responses for selected units: (B, N)
    selected = responses[:, unit_indices]
    n = selected.shape[1]

    # Compute correlation matrix: (N, N)
    # Center each unit's responses across batch
    centered = selected - selected.mean(dim=0, keepdim=True)
    # Normalize
    norms = torch.sqrt((centered ** 2).sum(dim=0, keepdim=True) + 1e-8)
    normalized = centered / norms
    # Correlation matrix
    corr_matrix = normalized.T @ normalized / selected.shape[0]

    # Extract upper triangle (excluding diagonal)
    triu_indices = torch.triu_indices(n, n, offset=1, device=responses.device)
    pairwise_corrs = corr_matrix[triu_indices[0], triu_indices[1]]

    return pairwise_corrs


def compute_pairwise_inverse_distances(
    positions: torch.Tensor,
    unit_indices: torch.Tensor
) -> torch.Tensor:
    """
    Compute pairwise inverse distances between units.

    Formula: D_i = 1 / (d_i + 1)  (Equation 4)

    Args:
        positions: Tensor of shape (total_units, 2) with (x, y) coords.
        unit_indices: LongTensor of shape (N,) specifying which units.

    Returns:
        Tensor of shape (N*(N-1)/2,) with upper-triangle inverse distances.
    """
    # Extract positions for selected units: (N, 2)
    selected = positions[unit_indices]
    n = selected.shape[0]

    # Pairwise Euclidean distances: (N, N)
    diff = selected.unsqueeze(0) - selected.unsqueeze(1)  # (N, N, 2)
    distances = torch.norm(diff, dim=2)  # (N, N)

    # Inverse distance with +1 to avoid division by zero
    inv_distances = 1.0 / (distances + 1.0)

    # Extract upper triangle
    triu_indices = torch.triu_indices(n, n, offset=1, device=positions.device)
    pairwise_inv_dist = inv_distances[triu_indices[0], triu_indices[1]]

    return pairwise_inv_dist


class SpatialLoss(nn.Module):
    """
    Computes the Relative Spatial Loss (SL_Rel) across all layers.

    SL_Rel = 1 - Corr(response_correlations, inverse_distances)

    The loss encourages units that are physically close on the cortical
    sheet to have correlated responses.
    """

    def __init__(
        self,
        cortical_sheet: CorticalSheet,
        alpha: float = 0.25,
        num_neighborhoods: int = 10,
        max_units_per_neighborhood: int = 500
    ):
        """
        Args:
            cortical_sheet: CorticalSheet module with position mappings.
            alpha: Weight of spatial loss (α in Equation 5). Default 0.25.
            num_neighborhoods: Number of random neighborhoods to sample per layer.
            max_units_per_neighborhood: Max units per neighborhood for efficiency.
        """
        super().__init__()
        self.cortical_sheet = cortical_sheet
        self.alpha = alpha
        self.num_neighborhoods = num_neighborhoods
        self.max_units = max_units_per_neighborhood

    def compute_layer_loss(
        self,
        layer_id: str,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spatial loss for a single layer.

        Args:
            layer_id: Layer identifier ("L2" to "L9").
            features: Feature map tensor of shape (B, C, H, W).

        Returns:
            Scalar loss tensor for this layer.
        """
        b, c, h, w = features.shape
        device = features.device

        # Flatten features: (B, C*H*W)
        responses = features.view(b, -1)

        # Get positions for this layer
        positions = self.cortical_sheet.get_positions(layer_id).to(device)

        # Sample multiple neighborhoods and average their losses
        losses = []
        for _ in range(self.num_neighborhoods):
            # Sample a neighborhood
            unit_indices, _ = self.cortical_sheet.sample_neighborhood(
                layer_id,
                center_idx=None,  # Random center
                max_units=self.max_units
            )
            unit_indices = unit_indices.to(device)

            if len(unit_indices) < 3:
                # Need at least 3 units for meaningful correlation
                continue

            # Compute pairwise response correlations
            response_corrs = compute_pairwise_response_correlations(
                responses, unit_indices
            )

            # Compute pairwise inverse distances
            inv_distances = compute_pairwise_inverse_distances(
                positions, unit_indices
            )

            # SL_Rel = 1 - Corr(r, D)
            corr = pearson_correlation(response_corrs, inv_distances)
            loss = 1.0 - corr
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses).mean()

    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total spatial loss across all layers.

        Args:
            features: Dict mapping layer_id to feature tensors.
                      E.g., {"L2": (B,64,56,56), "L4": (B,128,28,28), ...}

        Returns:
            Tuple of:
                - total_loss: Weighted sum of layer losses (α * Σ SL_k)
                - layer_losses: Dict mapping layer_id to individual losses
        """
        layer_losses = {}
        total_loss = torch.tensor(0.0, device=next(iter(features.values())).device)

        for layer_id, feat in features.items():
            # Skip non-layer outputs (pooled, projections, logits)
            if layer_id not in ['L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9']:
                continue

            loss = self.compute_layer_loss(layer_id, feat)
            layer_losses[layer_id] = loss
            total_loss = total_loss + loss

        # Apply weight α
        total_loss = self.alpha * total_loss

        return total_loss, layer_losses

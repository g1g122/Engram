"""
SimCLR Loss (NT-Xent) for TDANN.

Responsibility:
    - Implement NT-Xent (Normalized Temperature-scaled Cross Entropy Loss).
    - Standard contrastive self-supervised objective.
    - Used in Stage 1 (Pretrain) and Stage 3 (Joint Train).
    - Encourages same-image augmented views to be similar, different images to be dissimilar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRLoss(nn.Module):
    """
    NT-Xent loss for SimCLR contrastive learning.

    Given a batch of N images, each image is augmented twice to produce 2N views.
    For each anchor view, the other augmented view of the same image is the positive,
    and all other 2(N-1) views are negatives.

    Loss = -log( exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) )

    where (i, j) are positive pairs and k iterates over all other samples.
    """

    def __init__(self, temperature: float = 0.5):
        """
        Args:
            temperature: Temperature parameter τ for scaling similarities.
                         Lower temperature makes the distribution sharper.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss for two sets of augmented views.

        Args:
            z1: Embeddings from first augmentation, shape (N, D).
            z2: Embeddings from second augmentation, shape (N, D).
                Both should be L2-normalized.

        Returns:
            Scalar loss tensor.
        """
        batch_size = z1.shape[0]
        device = z1.device

        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate: [z1_0, z1_1, ..., z1_{N-1}, z2_0, z2_1, ..., z2_{N-1}]
        # Shape: (2N, D)
        z = torch.cat([z1, z2], dim=0)

        # Compute similarity matrix: (2N, 2N)
        # sim[i, j] = z[i] · z[j] / τ
        sim_matrix = torch.mm(z, z.T) / self.temperature

        # Create mask for positive pairs
        # For z1[i], positive is z2[i] (at index i + N)
        # For z2[i], positive is z1[i] (at index i)
        # Positive pairs: (i, i+N) and (i+N, i) for i in [0, N)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),  # z1 -> z2
            torch.arange(batch_size, device=device)                    # z2 -> z1
        ])

        # Mask out self-similarities (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        # Cross entropy loss: each row should predict its positive pair
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class SimCLRProjector(nn.Module):
    """
    Projection head for SimCLR.

    Maps backbone features to a lower-dimensional space where
    the contrastive loss is applied.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 2048, output_dim: int = 128):
        """
        Args:
            input_dim: Dimension of backbone output (512 for ResNet-18).
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension for contrastive space.
        """
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Backbone output, shape (B, input_dim).

        Returns:
            Projected embeddings, shape (B, output_dim).
        """
        return self.projector(x)

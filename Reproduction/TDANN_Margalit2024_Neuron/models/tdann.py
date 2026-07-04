"""
TDANN (Topographic Deep Artificial Neural Network) Main Model.

Responsibility:
    - Combine ResNet18Backbone + CorticalSheet + SimCLRProjector.
    - Forward pass returns features from all layers for spatial loss.
    - Support both self-supervised (SimCLR) and supervised training.
    - Enable weight re-initialization (Stage 5) while preserving cortical positions.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from models.backbone import ResNet18Backbone
from models.cortical_sheet import CorticalSheet, get_feature_map_sizes
from objectives.simclr_loss import SimCLRProjector
from objectives.spatial_loss import SpatialLoss


class TDANN(nn.Module):
    """
    Topographic Deep Artificial Neural Network.

    Integrates:
        - ResNet-18 backbone for feature extraction
        - Cortical sheet for spatial organization
        - SimCLR projector for self-supervised learning (optional)
        - Spatial loss computation across layers
    """

    def __init__(
        self,
        layer_configs: Dict[str, dict],
        use_simclr: bool = True,
        spatial_loss_alpha: float = 0.25,
        pretrained_backbone: bool = False
    ):
        """
        Args:
            layer_configs: Config dict from YAML for cortical sheet.
            use_simclr: If True, use SimCLR (self-supervised). If False, use supervised.
            spatial_loss_alpha: Weight α for spatial loss (default 0.25 from paper).
            pretrained_backbone: If True, initialize backbone with ImageNet weights.
        """
        super().__init__()

        # Backbone: ResNet-18 with intermediate layer outputs
        self.backbone = ResNet18Backbone(pretrained=pretrained_backbone)

        # Cortical Sheet: manages (c,h,w) -> (x,y) mappings
        feature_map_sizes = get_feature_map_sizes()
        self.cortical_sheet = CorticalSheet(layer_configs, feature_map_sizes)

        # SimCLR projector (only used if use_simclr=True)
        self.use_simclr = use_simclr
        if use_simclr:
            self.projector = SimCLRProjector(
                input_dim=512,      # ResNet-18 final layer output
                hidden_dim=2048,
                output_dim=128
            )

        # Spatial Loss module
        self.spatial_loss_module = SpatialLoss(
            cortical_sheet=self.cortical_sheet,
            alpha=spatial_loss_alpha,
            num_neighborhoods=10,
            max_units_per_neighborhood=500
        )

        # Classification head (only used if use_simclr=False)
        if not use_simclr:
            self.classifier = self.backbone.get_classifier()

    def forward(
        self,
        x: torch.Tensor,
        return_projections: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through TDANN.

        Args:
            x: Input images, shape (B, 3, 224, 224).
            return_projections: If True and using SimCLR, return projected embeddings.

        Returns:
            Dictionary with keys:
                - 'L2' to 'L9': Feature maps from each layer
                - 'pooled': Global average pooled features (B, 512)
                - 'projections': (optional) SimCLR projections (B, 128)
                - 'logits': (optional) Classification logits if not using SimCLR
        """
        # Extract features from all layers
        features = self.backbone(x)

        # Add projections or logits depending on mode
        if self.use_simclr and return_projections:
            pooled = features['pooled']
            projections = self.projector(pooled)
            features['projections'] = projections
        elif not self.use_simclr:
            pooled = features['pooled']
            logits = self.classifier(pooled)
            features['logits'] = logits

        return features

    def compute_spatial_loss(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute spatial loss across all layers.

        Args:
            features: Output from forward(), containing 'L2' to 'L9'.

        Returns:
            Tuple of (total_spatial_loss, layer_losses_dict)
        """
        return self.spatial_loss_module(features)

    def reset_backbone_weights(self):
        """
        Re-initialize all backbone weights randomly.

        Called at Step 5 of training: "All network weights are randomly re-initialized."
        The cortical sheet positions are NOT reset (they are frozen after Stage 2).
        """
        self.backbone.reset_parameters()

        # Also reset projector/classifier if present
        if self.use_simclr and hasattr(self, 'projector'):
            for module in self.projector.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0, 0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        elif not self.use_simclr and hasattr(self, 'classifier'):
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            nn.init.zeros_(self.classifier.bias)

    def load_cortical_positions(self, path: str):
        """Load cortical sheet positions from file (after Stage 2 shuffling)."""
        self.cortical_sheet.load_positions(path)

    def save_cortical_positions(self, path: str):
        """Save cortical sheet positions to file."""
        self.cortical_sheet.save_positions(path)

    def set_layer_permutation(self, layer_id: str, perm_indices: torch.Tensor):
        """Set permutation for a specific layer (used in Stage 2 shuffling)."""
        self.cortical_sheet.set_permutation(layer_id, perm_indices)

    def get_layer_positions(self, layer_id: str) -> torch.Tensor:
        """Get current positions for a layer."""
        return self.cortical_sheet.get_positions(layer_id)


def create_tdann(
    layer_configs: Dict[str, dict],
    use_simclr: bool = True,
    spatial_loss_alpha: float = 0.25,
    pretrained_backbone: bool = False
) -> TDANN:
    """
    Factory function to create TDANN model.

    Args:
        layer_configs: Layer configuration from YAML.
        use_simclr: Use SimCLR (True) or supervised classification (False).
        spatial_loss_alpha: Weight for spatial loss.
        pretrained_backbone: Initialize with ImageNet weights.

    Returns:
        Initialized TDANN model.
    """
    return TDANN(
        layer_configs=layer_configs,
        use_simclr=use_simclr,
        spatial_loss_alpha=spatial_loss_alpha,
        pretrained_backbone=pretrained_backbone
    )

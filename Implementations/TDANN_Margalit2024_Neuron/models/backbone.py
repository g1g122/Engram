"""
ResNet-18 Backbone with Intermediate Layer Outputs for TDANN.

Responsibility:
    - Wrap torchvision's ResNet-18 to expose outputs from all 8 BasicBlocks.
    - Return feature maps for Layer 2-9 (TDANN terminology), needed for Spatial Loss.
    - Layer 1 (initial 7x7 conv) is NOT embedded per paper design.
    - Provide reset_parameters() for Step 5 of training (weight re-initialization).
"""

from typing import Dict
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Backbone(nn.Module):
    """
    ResNet-18 backbone that returns intermediate feature maps from all 8 BasicBlocks.

    Layer mapping (per paper Table in STAR Methods):
        L2: layer1[0] -> 64 × 56 × 56  = 200,704 units (Retina)
        L3: layer1[1] -> 64 × 56 × 56  = 200,704 units (Retina)
        L4: layer2[0] -> 128 × 28 × 28 = 100,352 units (V1)
        L5: layer2[1] -> 128 × 28 × 28 = 100,352 units (V1)
        L6: layer3[0] -> 256 × 14 × 14 = 50,176 units  (V2)
        L7: layer3[1] -> 256 × 14 × 14 = 50,176 units  (V4)
        L8: layer4[0] -> 512 × 7 × 7   = 25,088 units  (VTC)
        L9: layer4[1] -> 512 × 7 × 7   = 25,088 units  (VTC)
    """

    def __init__(self, pretrained: bool = False):
        """
        Args:
            pretrained: If True, load ImageNet-pretrained weights.
                        Used in Stage 1 (task-only pretraining) and Stage 2 (shuffling).
        """
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = resnet18(weights=weights)

        # Layer 1: Initial conv block (not embedded in cortical sheet)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        # Layers 2-9: 4 stages × 2 blocks = 8 BasicBlocks
        self.layer1 = base_model.layer1  # Blocks 0, 1 -> L2, L3
        self.layer2 = base_model.layer2  # Blocks 2, 3 -> L4, L5
        self.layer3 = base_model.layer3  # Blocks 4, 5 -> L6, L7
        self.layer4 = base_model.layer4  # Blocks 6, 7 -> L8, L9

        # For task loss computation (SimCLR or classification)
        self.avgpool = base_model.avgpool
        self.fc = base_model.fc

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning all intermediate layer outputs.

        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Dictionary with keys 'L2' through 'L9' containing feature maps,
            plus 'output' for the final pooled representation (used for task loss).
        """
        features = {}

        # Layer 1: Initial conv (not embedded)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 2-3: Stage 1 (Retina)
        x = self.layer1[0](x)
        features['L2'] = x
        x = self.layer1[1](x)
        features['L3'] = x

        # Layer 4-5: Stage 2 (V1)
        x = self.layer2[0](x)
        features['L4'] = x
        x = self.layer2[1](x)
        features['L5'] = x

        # Layer 6-7: Stage 3 (V2, V4)
        x = self.layer3[0](x)
        features['L6'] = x
        x = self.layer3[1](x)
        features['L7'] = x

        # Layer 8-9: Stage 4 (VTC)
        x = self.layer4[0](x)
        features['L8'] = x
        x = self.layer4[1](x)
        features['L9'] = x

        # Final pooled output for task loss
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features['pooled'] = x

        return features

    def get_classifier(self) -> nn.Linear:
        """Return the classification head for supervised training."""
        return self.fc

    def reset_parameters(self):
        """
        Re-initialize all network weights randomly.
        Called at Step 5 of training: "All network weights are randomly re-initialized."
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)


def create_backbone(pretrained: bool = False) -> ResNet18Backbone:
    """Factory function to create the backbone."""
    return ResNet18Backbone(pretrained=pretrained)

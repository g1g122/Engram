import torch.nn as nn

from .backbones import get_backbone


class ImageClassifier(nn.Module):
    """Image classifier built from a backbone and a linear classification head."""

    def __init__(self, backbone, n_outputs, num_classes, dropout=0.0):
        super().__init__()

        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}.")

        self.backbone = backbone
        self.n_outputs = n_outputs
        self.num_classes = num_classes

        if dropout > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(n_outputs, num_classes),
            )
        else:
            self.classifier = nn.Linear(n_outputs, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def build_image_classifier(
    num_classes,
    backbone_name="moa_clip_vit_b16",
    model_name=None,
    pretrained=True,
    classifier_dropout=0.0,
    **backbone_kwargs,
):
    backbone, n_outputs, backbone_trainable_params, injected_names = get_backbone(
        name=backbone_name,
        model_name=model_name,
        pretrained=pretrained,
        **backbone_kwargs,
    )

    model = ImageClassifier(
        backbone=backbone,
        n_outputs=n_outputs,
        num_classes=num_classes,
        dropout=classifier_dropout,
    )

    trainable_params = list(backbone_trainable_params)
    trainable_params.extend(
        p for p in model.classifier.parameters() if p.requires_grad
    )

    metadata = {
        "backbone_name": backbone_name,
        "model_name": model_name,
        "n_outputs": n_outputs,
        "num_classes": num_classes,
        "injected_names": injected_names,
    }

    return model, trainable_params, metadata

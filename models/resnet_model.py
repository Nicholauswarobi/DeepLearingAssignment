"""Advanced model: ResNet18 pretrained on ImageNet, fine-tuned for 4-class
corn leaf disease classification via transfer learning.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_resnet18(
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.4,
) -> nn.Module:
    """Load torchvision's ResNet18 and replace the final FC layer for our 4 classes.

    freeze_backbone=True trains only the new classifier head (feature extraction
    mode); False fine-tunes the whole network (default, generally best accuracy).
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model

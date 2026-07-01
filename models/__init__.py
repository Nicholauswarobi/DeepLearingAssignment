"""Model factory: builds the baseline CNN or the ResNet18 transfer-learning model by name."""
from __future__ import annotations

import torch.nn as nn

from models.baseline_cnn import CustomCNN
from models.resnet_model import build_resnet18

MODEL_NAMES = ("baseline", "resnet18")


def get_model(
    name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    dropout: float = 0.4,
) -> nn.Module:
    name = name.lower()
    if name == "baseline":
        return CustomCNN(num_classes=num_classes, dropout=dropout)
    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    raise ValueError(f"Unknown model name '{name}'. Choose from {MODEL_NAMES}.")

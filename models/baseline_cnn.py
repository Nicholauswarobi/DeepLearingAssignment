"""Baseline: a custom CNN trained from scratch, used as the performance floor
that the transfer-learning model (ResNet18) is compared against.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2D -> BatchNorm -> ReLU -> MaxPool, the repeating unit of the baseline CNN."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class CustomCNN(nn.Module):
    """4-block CNN: 3 -> 32 -> 64 -> 128 -> 256 channels, GAP, dropout, FC classifier.

    Input: (B, 3, 224, 224) -> Output: (B, num_classes) logits.
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),      # 224 -> 112
            ConvBlock(32, 64),     # 112 -> 56
            ConvBlock(64, 128),    # 56 -> 28
            ConvBlock(128, 256),   # 28 -> 14
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)

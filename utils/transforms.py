"""torchvision transform pipelines for training, evaluation, and inference."""
from __future__ import annotations

import torch
from torchvision import transforms

import config


class AddGaussianNoise(torch.nn.Module):
    """Injects zero-mean Gaussian noise into a tensor image (applied after ToTensor)."""

    def __init__(self, std: float = 0.05):
        super().__init__()
        self.std = std

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)


def get_train_transforms(image_size: tuple[int, int] = config.IMAGE_SIZE) -> transforms.Compose:
    """Augmentation-heavy pipeline used for the training split only."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.RandomApply([AddGaussianNoise(std=0.05)], p=0.3),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: tuple[int, int] = config.IMAGE_SIZE) -> transforms.Compose:
    """Deterministic resize + normalize pipeline for validation/test/inference."""
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


def denormalize(tensor):
    """Invert ImageNet normalization for visualization (returns HWC numpy in [0, 1])."""
    import numpy as np

    mean = np.array(config.IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(config.IMAGENET_STD).reshape(3, 1, 1)
    img = tensor.cpu().numpy() * std + mean
    img = np.clip(img, 0.0, 1.0)
    return np.transpose(img, (1, 2, 0))

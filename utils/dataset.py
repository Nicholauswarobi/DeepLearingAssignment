"""PyTorch Dataset/DataLoader construction for the Corn Leaf Disease dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config
from utils.transforms import get_eval_transforms, get_train_transforms


class CornLeafDataset(Dataset):
    """Reads a manifest CSV (filepath,label,label_idx) and serves (image, label) pairs."""

    def __init__(self, manifest_path: Path, transform: Optional[Callable] = None):
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. Run `python -m utils.split_data` first."
            )
        self.df = pd.read_csv(manifest_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label_idx"])

    def class_counts(self) -> pd.Series:
        return self.df["label"].value_counts().reindex(config.CLASS_NAMES, fill_value=0)


def compute_class_weights(dataset: CornLeafDataset) -> torch.Tensor:
    """Inverse-frequency class weights, used to counter class imbalance."""
    counts = dataset.class_counts().values.astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: tuple[int, int] = config.IMAGE_SIZE,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from the pre-computed manifests."""
    train_ds = CornLeafDataset(config.TRAIN_MANIFEST, transform=get_train_transforms(image_size))
    val_ds = CornLeafDataset(config.VAL_MANIFEST, transform=get_eval_transforms(image_size))
    test_ds = CornLeafDataset(config.TEST_MANIFEST, transform=get_eval_transforms(image_size))

    sampler = None
    shuffle = True
    if use_weighted_sampler:
        counts = train_ds.class_counts().values.astype(float)
        counts[counts == 0] = 1.0
        class_weights = 1.0 / counts
        sample_weights = train_ds.df["label_idx"].map(
            lambda i: class_weights[i]
        ).values
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    pin_memory = config.DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, val_loader, test_loader

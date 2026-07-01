"""Builds reproducible, stratified train/validation/test manifests.

The raw dataset ships as PlantVillage/train/<class>/*.jpg and
PlantVillage/val/<class>/*.jpg (no test split). This script pools every
image per class from both raw folders, shuffles deterministically (seeded),
and re-splits into train/val/test according to config.SPLIT_RATIOS. Results
are written as CSV manifests (filepath,label,label_idx) under
dataset/splits/, which every Dataset/DataLoader in this project reads from.

Run directly to (re)generate the splits:
    python -m utils.split_data
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_images_per_class(raw_dirs: List[Path], class_names: List[str]) -> Dict[str, List[Path]]:
    """Pool every image belonging to each class across all raw directories."""
    images: Dict[str, List[Path]] = defaultdict(list)
    for raw_dir in raw_dirs:
        if not raw_dir.exists():
            logger.warning("Raw data directory not found, skipping: %s", raw_dir)
            continue
        for class_name in class_names:
            class_dir = raw_dir / class_name
            if not class_dir.exists():
                logger.warning("Class folder not found, skipping: %s", class_dir)
                continue
            for path in class_dir.iterdir():
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    images[class_name].append(path)
    return images


def _stratified_split(
    images_per_class: Dict[str, List[Path]],
    ratios: Dict[str, float],
    seed: int,
) -> Dict[str, List[Tuple[Path, str]]]:
    """Shuffle each class's images deterministically and cut into splits."""
    assert abs(sum(ratios.values()) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    rng = random.Random(seed)
    splits: Dict[str, List[Tuple[Path, str]]] = {name: [] for name in ratios}

    for class_name, paths in images_per_class.items():
        shuffled = paths.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(n * ratios["train"]))
        n_val = int(round(n * ratios["val"]))

        train_paths = shuffled[:n_train]
        val_paths = shuffled[n_train : n_train + n_val]
        test_paths = shuffled[n_train + n_val :]

        splits["train"].extend((p, class_name) for p in train_paths)
        splits["val"].extend((p, class_name) for p in val_paths)
        splits["test"].extend((p, class_name) for p in test_paths)

    return splits


def build_splits(force: bool = False) -> None:
    """Create train/val/test manifest CSVs plus a dataset-stats JSON report."""
    if config.TRAIN_MANIFEST.exists() and not force:
        logger.info("Manifests already exist, skipping split generation (use force=True to rebuild).")
        return

    logger.info("Scanning raw dataset directories: %s", config.RAW_DATA_DIRS)
    images_per_class = _collect_images_per_class(config.RAW_DATA_DIRS, config.CLASS_NAMES)

    total = sum(len(v) for v in images_per_class.values())
    if total == 0:
        raise FileNotFoundError(
            f"No images found under {config.RAW_DATA_DIRS}. Check that the PlantVillage "
            "dataset folder is present and class subfolders match config.CLASS_NAMES."
        )
    logger.info("Found %d images across %d classes.", total, len(images_per_class))

    splits = _stratified_split(images_per_class, config.SPLIT_RATIOS, seed=config.SEED)
    class_to_idx = {name: i for i, name in enumerate(config.CLASS_NAMES)}

    stats = {"total_images": total, "per_class": {}, "per_split": {}}
    for class_name, paths in images_per_class.items():
        stats["per_class"][class_name] = len(paths)

    for split_name, records in splits.items():
        rows = [
            {
                "filepath": str(path.resolve()),
                "label": label,
                "label_idx": class_to_idx[label],
            }
            for path, label in records
        ]
        df = pd.DataFrame(rows).sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)
        out_path = config.SPLITS_DIR / f"{split_name}.csv"
        df.to_csv(out_path, index=False)
        logger.info("Wrote %d rows to %s", len(df), out_path)
        stats["per_split"][split_name] = {
            "total": len(df),
            "per_class": df["label"].value_counts().to_dict(),
        }

    stats_path = config.RESULTS_DIR / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Dataset statistics saved to %s", stats_path)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    build_splits(force=True)

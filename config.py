"""Central configuration for the Corn Leaf Disease Classification project.

All paths, hyperparameters, and runtime settings live here so every script
(train.py, evaluate.py, predict.py, tune.py, app.py, notebook) shares one
source of truth.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Raw dataset as provided (PlantVillage/train, PlantVillage/val).
RAW_DATA_DIRS: List[Path] = [
    PROJECT_ROOT / "PlantVillage" / "train",
    PROJECT_ROOT / "PlantVillage" / "val",
]

# Manifest-based train/validation/test split (CSV files: filepath,label).
# A manifest approach is used instead of physically copying tens of
# thousands of images into dataset/train|validation|test folders: it is
# just as reproducible, avoids duplicating ~350MB of images on disk, and is
# regenerated deterministically (seeded) by utils/split_data.py.
DATASET_DIR: Path = PROJECT_ROOT / "dataset"
SPLITS_DIR: Path = DATASET_DIR / "splits"
TRAIN_MANIFEST: Path = SPLITS_DIR / "train.csv"
VAL_MANIFEST: Path = SPLITS_DIR / "val.csv"
TEST_MANIFEST: Path = SPLITS_DIR / "test.csv"

CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
PLOTS_DIR: Path = PROJECT_ROOT / "plots"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

for _dir in (CHECKPOINT_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR, SPLITS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
CLASS_NAMES: List[str] = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
]
NUM_CLASSES: int = len(CLASS_NAMES)

# Short, human-friendly labels used in plots/reports.
CLASS_SHORT_NAMES: List[str] = [
    "Gray Leaf Spot",
    "Common Rust",
    "Healthy",
    "Northern Leaf Blight",
]

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

IMAGE_SIZE: tuple[int, int] = (224, 224)
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and torch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Training hyperparameters (defaults; overridable via CLI args)
# --------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    model_name: str = "resnet18"  # "baseline" | "resnet18"
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # "adam" | "sgd"
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 1e-4
    num_workers: int = 4
    mixed_precision: bool = True
    seed: int = SEED
    pretrained: bool = True
    dropout: float = 0.4
    label_smoothing: float = 0.0
    use_class_weights: bool = True


DEFAULT_TRAINING_CONFIG = TrainingConfig()

# Hyperparameter search space (used by tune.py)
HP_SEARCH_SPACE = {
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "batch_size": [16, 32, 64],
    "optimizer": ["adam", "sgd"],
}

# --------------------------------------------------------------------------
# Output file locations
# --------------------------------------------------------------------------
def best_ckpt_path(model_name: str) -> Path:
    return CHECKPOINT_DIR / f"{model_name}_best.pth"


def last_ckpt_path(model_name: str) -> Path:
    return CHECKPOINT_DIR / f"{model_name}_last.pth"


def model_results_dir(model_name: str) -> Path:
    d = RESULTS_DIR / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_plots_dir(model_name: str) -> Path:
    d = PLOTS_DIR / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d

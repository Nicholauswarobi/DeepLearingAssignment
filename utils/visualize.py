"""Plotting helpers: sample grids, training curves, confusion matrix, ROC/PR curves,
and baseline-vs-advanced model comparison charts. Every function saves a PNG and
returns the matplotlib Figure so notebooks can also display it inline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "ggplot")


def _save(fig: plt.Figure, save_path: Path | None) -> None:
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_sample_images(dataset, class_names: List[str], save_path: Path | None = None, n: int = 12):
    import random

    from utils.transforms import denormalize

    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    cols = 4
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, indices):
        image, label_idx = dataset[idx]
        img = denormalize(image) if hasattr(image, "cpu") else image
        ax.imshow(img)
        ax.set_title(class_names[label_idx], fontsize=9)
        ax.axis("off")
    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.suptitle("Sample Training Images (after augmentation)", fontsize=14)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_class_distribution(counts: Dict[str, int], save_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(counts.keys())
    values = list(counts.values())
    ax.bar(range(len(names)), values, color=sns.color_palette("viridis", len(names)))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Number of images")
    ax.set_title("Class Distribution")
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_training_curves(history: Dict[str, List[float]], save_path: Path | None = None, model_name: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} — Loss Curve".strip(" —"))
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Accuracy")
    axes[1].plot(history["val_acc"], label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name} — Accuracy Curve".strip(" —"))
    axes[1].legend()

    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names: List[str], save_path: Path | None = None, model_name: str = ""):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}".strip(" —"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_roc_curve(y_true, y_probs, class_names: List[str], save_path: Path | None = None, model_name: str = ""):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (One-vs-Rest) — {model_name}".strip(" —"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_precision_recall_curve(y_true, y_probs, class_names: List[str], save_path: Path | None = None, model_name: str = ""):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}".strip(" —"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_model_comparison(results: Dict[str, Dict[str, float]], metrics_to_plot: Sequence[str], save_path: Path | None = None):
    """results: {model_name: {metric_name: value, ...}, ...}"""
    model_names = list(results.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / max(len(model_names), 1)

    for i, model_name in enumerate(model_names):
        values = [results[model_name].get(m, 0.0) for m in metrics_to_plot]
        ax.bar(x + i * width, values, width, label=model_name)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metrics_to_plot, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs. Advanced Model Comparison")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig

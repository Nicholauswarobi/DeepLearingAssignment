"""Metric computation helpers shared by evaluate.py, train.py and the notebook."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
) -> Dict:
    """Accuracy, macro/weighted precision-recall-F1, and one-vs-rest ROC-AUC."""
    n_classes = len(class_names)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    try:
        # roc_auc_score(multi_class="ovr") expects raw integer labels (n_samples,), not a
        # binarized matrix -- it binarizes internally per class.
        metrics["roc_auc_macro"] = float(
            roc_auc_score(y_true, y_probs, average="macro", multi_class="ovr", labels=list(range(n_classes)))
        )
        metrics["roc_auc_weighted"] = float(
            roc_auc_score(y_true, y_probs, average="weighted", multi_class="ovr", labels=list(range(n_classes)))
        )
    except ValueError:
        # Can happen if a class is entirely absent from y_true in a tiny sample.
        metrics["roc_auc_macro"] = float("nan")
        metrics["roc_auc_weighted"] = float("nan")

    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    return metrics


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(checkpoint_path: Path) -> float:
    return round(Path(checkpoint_path).stat().st_size / (1024 * 1024), 2)

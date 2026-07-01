"""Evaluate a trained checkpoint on the held-out test split.

Generates: classification report, confusion matrix, ROC curve,
precision-recall curve, and (when results exist for more than one model)
a baseline-vs-advanced comparison table/plot.

Examples
--------
    python evaluate.py --model resnet18
    python evaluate.py --model baseline --checkpoint checkpoints/baseline_last.pth
    python evaluate.py --model both --compare
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models import get_model
from utils.dataset import CornLeafDataset
from utils.logger import setup_logger
from utils.metrics import compute_metrics, count_parameters, model_size_mb
from utils.transforms import get_eval_transforms
from utils.visualize import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_precision_recall_curve,
    plot_roc_curve,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    p.add_argument("--model", choices=["baseline", "resnet18", "both"], default="resnet18")
    p.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path (single model only).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--compare", action="store_true", help="Also emit a model comparison table/plot.")
    return p.parse_args()


def load_model(model_name: str, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def run_inference(model, loader: DataLoader, device: torch.device):
    all_probs, all_preds, all_labels = [], [], []
    n_images = 0
    start = time.time()
    for images, labels in tqdm(loader, desc="test", leave=False):
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        n_images += images.size(0)
    elapsed = time.time() - start

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
        elapsed / max(n_images, 1),
    )


def evaluate_model(model_name: str, checkpoint_path: Path, args: argparse.Namespace, logger) -> Dict:
    device = config.DEVICE
    logger.info("Evaluating '%s' from %s on %s", model_name, checkpoint_path, device)

    model, checkpoint = load_model(model_name, checkpoint_path, device)

    test_ds = CornLeafDataset(config.TEST_MANIFEST, transform=get_eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    y_true, y_pred, y_probs, avg_inference_time = run_inference(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred, y_probs, config.CLASS_NAMES)
    metrics["avg_inference_time_ms"] = avg_inference_time * 1000
    metrics["num_parameters"] = count_parameters(model)
    metrics["model_size_mb"] = model_size_mb(checkpoint_path)
    metrics["checkpoint_epoch"] = checkpoint.get("epoch")
    metrics["val_acc_at_checkpoint"] = checkpoint.get("val_acc")

    results_dir = config.model_results_dir(model_name)
    plots_dir = config.model_plots_dir(model_name)

    report_text = metrics.pop("classification_report")
    (results_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_confusion_matrix(y_true, y_pred, config.CLASS_NAMES, plots_dir / "confusion_matrix.png", model_name)
    plot_roc_curve(y_true, y_probs, config.CLASS_NAMES, plots_dir / "roc_curve.png", model_name)
    plot_precision_recall_curve(y_true, y_probs, config.CLASS_NAMES, plots_dir / "precision_recall_curve.png", model_name)

    logger.info("Test accuracy: %.4f | F1 (macro): %.4f | ROC-AUC (macro): %.4f",
                metrics["accuracy"], metrics["f1_macro"], metrics["roc_auc_macro"])
    logger.info("\n%s", report_text)

    return metrics


def build_comparison(model_names: List[str], logger) -> None:
    results = {}
    for name in model_names:
        metrics_path = config.model_results_dir(name) / "metrics.json"
        meta_path = config.model_results_dir(name) / "training_meta.json"
        if not metrics_path.exists():
            logger.warning("No results found for '%s', skipping in comparison.", name)
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if meta_path.exists():
            metrics.update(json.loads(meta_path.read_text(encoding="utf-8")))
        results[name] = metrics

    if len(results) < 2:
        logger.warning("Comparison requires results for at least 2 models; found %d.", len(results))

    comparison_metrics = [
        "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro",
        "training_time_seconds", "avg_inference_time_ms", "model_size_mb",
    ]
    rows = []
    for name, m in results.items():
        rows.append({"model": name, **{k: m.get(k) for k in comparison_metrics}})

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "model_comparison.csv", index=False)
    logger.info("Model comparison table:\n%s", df.to_string(index=False))

    plot_model_comparison(
        {r["model"]: r for r in rows},
        metrics_to_plot=["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro"],
        save_path=config.PLOTS_DIR / "model_comparison.png",
    )


def main() -> None:
    args = parse_args()
    logger = setup_logger("evaluate", config.LOGS_DIR / "evaluate.log")

    model_names = ["baseline", "resnet18"] if args.model == "both" else [args.model]

    for name in model_names:
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else config.best_ckpt_path(name)
        if not checkpoint_path.exists():
            logger.error("Checkpoint not found: %s. Train the model first with train.py.", checkpoint_path)
            continue
        evaluate_model(name, checkpoint_path, args, logger)

    if args.compare or args.model == "both":
        build_comparison(["baseline", "resnet18"], logger)


if __name__ == "__main__":
    main()

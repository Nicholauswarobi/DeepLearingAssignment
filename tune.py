"""Lightweight hyperparameter search over learning rate, batch size, and optimizer.

Trains short runs (few epochs, default model=baseline for speed) across the grid
defined in config.HP_SEARCH_SPACE, ranks combinations by best validation accuracy,
and writes results/hyperparam_search.csv. Re-run the winning combination with
train.py at full epoch budget for the final model.

Example
-------
    python tune.py --model baseline --epochs 5 --trials 6
"""
from __future__ import annotations

import argparse
import itertools
import random
import time

import pandas as pd
import torch.nn as nn
from torch import amp

import config
from models import get_model
from train import build_optimizer, run_epoch
from utils.dataset import get_dataloaders
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperparameter search for corn leaf disease classifiers.")
    p.add_argument("--model", choices=["baseline", "resnet18"], default="baseline")
    p.add_argument("--epochs", type=int, default=5, help="Short epoch budget per trial.")
    p.add_argument("--trials", type=int, default=6, help="Number of random combinations to try (0 = full grid).")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("tune", config.LOGS_DIR / "tune.log")
    config.set_seed(args.seed)
    device = config.DEVICE

    grid = list(itertools.product(
        config.HP_SEARCH_SPACE["learning_rate"],
        config.HP_SEARCH_SPACE["batch_size"],
        config.HP_SEARCH_SPACE["optimizer"],
    ))
    rng = random.Random(args.seed)
    rng.shuffle(grid)
    combos = grid if args.trials <= 0 else grid[: args.trials]

    logger.info("Running %d hyperparameter trial(s) for model=%s, %d epochs each.",
                len(combos), args.model, args.epochs)

    results = []
    for i, (lr, batch_size, optimizer_name) in enumerate(combos, start=1):
        logger.info("[Trial %d/%d] lr=%s batch_size=%s optimizer=%s", i, len(combos), lr, batch_size, optimizer_name)

        train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size, num_workers=2)
        model = get_model(args.model, num_classes=config.NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model, optimizer_name, lr, config.DEFAULT_TRAINING_CONFIG.weight_decay)
        scaler = amp.GradScaler(device=device.type, enabled=(device.type == "cuda"))

        best_val_acc = 0.0
        start = time.time()
        for epoch in range(args.epochs):
            _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True, device.type == "cuda")
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, device, False, device.type == "cuda")
            best_val_acc = max(best_val_acc, val_acc)
        elapsed = time.time() - start

        logger.info("  -> best_val_acc=%.4f (%.1fs)", best_val_acc, elapsed)
        results.append({
            "learning_rate": lr, "batch_size": batch_size, "optimizer": optimizer_name,
            "best_val_acc": best_val_acc, "trial_time_seconds": round(elapsed, 1),
        })

    df = pd.DataFrame(results).sort_values("best_val_acc", ascending=False).reset_index(drop=True)
    out_path = config.RESULTS_DIR / "hyperparam_search.csv"
    df.to_csv(out_path, index=False)
    logger.info("Hyperparameter search results saved to %s", out_path)
    logger.info("Best combination:\n%s", df.iloc[0].to_string())


if __name__ == "__main__":
    main()

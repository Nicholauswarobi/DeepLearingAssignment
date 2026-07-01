"""Train the baseline CNN and/or the ResNet18 transfer-learning model on the
Corn Leaf Disease dataset.

Examples
--------
    python train.py --model resnet18
    python train.py --model baseline --epochs 20 --batch-size 64
    python train.py --model both --optimizer sgd --lr 0.0005
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch import amp
from tqdm import tqdm

import config
from models import get_model
from utils.dataset import compute_class_weights, get_dataloaders, CornLeafDataset
from utils.early_stopping import EarlyStopping
from utils.logger import setup_logger
from utils.metrics import count_parameters
from utils.split_data import build_splits
from utils.transforms import get_train_transforms
from utils.visualize import plot_class_distribution, plot_sample_images, plot_training_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train corn leaf disease classifiers.")
    p.add_argument("--model", choices=["baseline", "resnet18", "both"], default="resnet18")
    p.add_argument("--epochs", type=int, default=config.DEFAULT_TRAINING_CONFIG.epochs)
    p.add_argument("--batch-size", type=int, default=config.DEFAULT_TRAINING_CONFIG.batch_size)
    p.add_argument("--lr", type=float, default=config.DEFAULT_TRAINING_CONFIG.learning_rate)
    p.add_argument("--weight-decay", type=float, default=config.DEFAULT_TRAINING_CONFIG.weight_decay)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default=config.DEFAULT_TRAINING_CONFIG.optimizer)
    p.add_argument("--patience", type=int, default=config.DEFAULT_TRAINING_CONFIG.early_stopping_patience)
    p.add_argument("--num-workers", type=int, default=config.DEFAULT_TRAINING_CONFIG.num_workers)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision even on CUDA.")
    p.add_argument("--no-pretrained", action="store_true", help="Train ResNet18 from random init.")
    p.add_argument("--no-class-weights", action="store_true", help="Disable class-imbalance weighting.")
    p.add_argument("--dropout", type=float, default=config.DEFAULT_TRAINING_CONFIG.dropout)
    return p.parse_args()


def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)


def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    scaler,
    device: torch.device,
    train: bool,
    use_amp: bool,
) -> tuple[float, float]:
    model.train(mode=train)
    running_loss, correct, total = 0.0, 0, 0

    desc = "train" if train else "val  "
    with torch.set_grad_enabled(train):
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def train_one_model(model_name: str, args: argparse.Namespace, logger) -> Dict:
    logger.info("=" * 70)
    logger.info("Training model: %s", model_name)
    logger.info("=" * 70)

    config.set_seed(args.seed)
    device = config.DEVICE
    use_amp = (not args.no_amp) and device.type == "cuda"

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = get_model(
        model_name,
        num_classes=config.NUM_CLASSES,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)
    logger.info("Trainable parameters: %s", f"{count_parameters(model):,}")

    class_weights = None
    if not args.no_class_weights:
        train_ds = CornLeafDataset(config.TRAIN_MANIFEST, transform=get_train_transforms())
        class_weights = compute_class_weights(train_ds).to(device)
        logger.info("Class weights: %s", class_weights.cpu().numpy().round(3).tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.DEFAULT_TRAINING_CONFIG.scheduler_factor,
        patience=config.DEFAULT_TRAINING_CONFIG.scheduler_patience,
    )
    scaler = amp.GradScaler(device=device.type, enabled=use_amp)
    early_stopping = EarlyStopping(patience=args.patience)

    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True, use_amp)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, device, False, use_amp)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %02d/%02d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f | lr=%.6f",
            epoch, args.epochs, train_loss, train_acc, val_loss, val_acc, current_lr,
        )

        torch.save(
            {"model_name": model_name, "epoch": epoch, "state_dict": model.state_dict(),
             "val_acc": val_acc, "config": vars(args)},
            config.last_ckpt_path(model_name),
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_name": model_name, "epoch": epoch, "state_dict": model.state_dict(),
                 "val_acc": val_acc, "config": vars(args)},
                config.best_ckpt_path(model_name),
            )
            logger.info("New best model saved (val_acc=%.4f).", val_acc)

        if early_stopping.step(val_loss):
            logger.info("Early stopping triggered after epoch %d.", epoch)
            break

    elapsed = time.time() - start_time
    logger.info("Training finished in %.1fs. Best val_acc=%.4f", elapsed, best_val_acc)

    history_path = config.LOGS_DIR / f"{model_name}_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    plot_training_curves(
        history, save_path=config.model_plots_dir(model_name) / "training_curves.png", model_name=model_name
    )

    meta = {
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "training_time_seconds": elapsed,
        "epochs_run": len(history["train_loss"]),
        "num_parameters": count_parameters(model),
    }
    (config.model_results_dir(model_name) / "training_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def main() -> None:
    args = parse_args()
    logger = setup_logger("train", config.LOGS_DIR / "train.log")

    config.set_seed(args.seed)
    logger.info("Device: %s", config.DEVICE)

    build_splits(force=False)

    # One-off EDA artifacts (dataset overview required by the assignment brief).
    dataset_stats_path = config.RESULTS_DIR / "dataset_stats.json"
    if dataset_stats_path.exists():
        stats = json.loads(dataset_stats_path.read_text(encoding="utf-8"))
        plot_class_distribution(stats["per_class"], save_path=config.PLOTS_DIR / "class_distribution.png")

    preview_ds = CornLeafDataset(config.TRAIN_MANIFEST, transform=get_train_transforms())
    plot_sample_images(
        preview_ds, config.CLASS_NAMES, save_path=config.PLOTS_DIR / "sample_augmented_images.png"
    )

    model_names = ["baseline", "resnet18"] if args.model == "both" else [args.model]
    summary = {}
    for name in model_names:
        summary[name] = train_one_model(name, args, logger)

    logger.info("Training summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

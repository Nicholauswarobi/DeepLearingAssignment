"""Run inference with a trained checkpoint on a single image or a folder of images.

Examples
--------
    python predict.py --model resnet18 --image path/to/leaf.jpg
    python predict.py --model resnet18 --dir path/to/folder --save-csv results/predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

import config
from models import get_model
from utils.logger import setup_logger
from utils.transforms import get_eval_transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict corn leaf disease class for one or more images.")
    p.add_argument("--model", choices=["baseline", "resnet18"], default="resnet18")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--image", type=str, default=None, help="Path to a single image.")
    p.add_argument("--dir", type=str, default=None, help="Path to a folder of images for batch prediction.")
    p.add_argument("--save-csv", type=str, default=str(config.RESULTS_DIR / "predictions.csv"))
    return p.parse_args()


def load_predictor(model_name: str, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_image(model, image_path: Path, device: torch.device) -> Tuple[str, float, List[float], float]:
    """Returns (predicted_class, confidence, all_class_probabilities, inference_time_ms)."""
    transform = get_eval_transforms()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    start = time.time()
    logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    elapsed_ms = (time.time() - start) * 1000

    pred_idx = int(probs.argmax())
    return config.CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs.tolist(), elapsed_ms


def collect_images(dir_path: Path) -> List[Path]:
    return sorted(p for p in dir_path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    args = parse_args()
    logger = setup_logger("predict", config.LOGS_DIR / "predict.log")

    if not args.image and not args.dir:
        raise SystemExit("Provide either --image <path> or --dir <path>.")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else config.best_ckpt_path(args.model)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}. Train the model first with train.py.")

    device = config.DEVICE
    model = load_predictor(args.model, checkpoint_path, device)
    logger.info("Loaded '%s' from %s on %s", args.model, checkpoint_path, device)

    image_paths = [Path(args.image)] if args.image else collect_images(Path(args.dir))
    if not image_paths:
        raise SystemExit("No images found to predict on.")

    rows = []
    for path in image_paths:
        pred_class, confidence, probs, elapsed_ms = predict_image(model, path, device)
        logger.info("%s -> %s (confidence=%.4f, %.1fms)", path.name, pred_class, confidence, elapsed_ms)
        rows.append(
            {
                "filename": str(path),
                "predicted_class": pred_class,
                "confidence": round(confidence, 4),
                "inference_time_ms": round(elapsed_ms, 2),
                **{f"prob_{cls}": round(p, 4) for cls, p in zip(config.CLASS_NAMES, probs)},
            }
        )

    save_path = Path(args.save_csv)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %d prediction(s) to %s", len(rows), save_path)


if __name__ == "__main__":
    main()

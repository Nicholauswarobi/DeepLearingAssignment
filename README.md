# Corn Leaf Disease Classification

A complete, production-quality PyTorch pipeline for classifying maize (corn) leaf images into
four categories using a custom CNN baseline and a ResNet18 transfer-learning model. Built for
the **Deep Learning for Image Classification & Segmentation** assignment (Image Analysis
Groups 9).

Runs entirely locally in VS Code — no Google Colab required. The device (CPU/CUDA) is detected
automatically at runtime.

## Contents

- [Dataset](#dataset)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Interactive UI](#interactive-ui-streamlit)
- [Jupyter notebook](#jupyter-notebook)
- [Model architectures](#model-architectures)
- [Training details](#training-details)
- [Evaluation outputs](#evaluation-outputs)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)

## Dataset

**Corn (maize) Leaf Disease Dataset** (PlantVillage), 4 classes, RGB JPEGs (~256x256), ~60MB total:

| Class | Short name | Train | Val | Total |
|---|---|---:|---:|---:|
| `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | Gray Leaf Spot | 410 | 103 | 513 |
| `Corn_(maize)___Common_rust_` | Common Rust | 953 | 239 | 1,192 |
| `Corn_(maize)___healthy` | Healthy | 929 | 233 | 1,162 |
| `Corn_(maize)___Northern_Leaf_Blight` | Northern Leaf Blight | 788 | 197 | 985 |

The raw data ships pre-split into `PlantVillage/train/<class>/` and `PlantVillage/val/<class>/`
folders, with no test split. `utils/split_data.py` pools every image per class from both raw
folders and deterministically (seed=42) re-splits into **train/validation/test (70/15/15)**,
writing manifest CSVs (`filepath,label,label_idx`) to `dataset/splits/`. Every `Dataset` in this
project reads from these manifests instead of physically copying the ~3,850 source images —
this is fully reproducible and avoids duplicating the dataset on disk. Re-run
`python -m utils.split_data` to regenerate the splits at any time.

### Dataset challenges

- **Class imbalance** — Gray Leaf Spot has ~half the samples of Common Rust. Addressed with a
  `WeightedRandomSampler` for training batches and inverse-frequency class weights in
  `CrossEntropyLoss` (see `utils/dataset.py`), rather than duplicating files.
- **Visual similarity** — Gray Leaf Spot and Northern Leaf Blight both present as elongated
  necrotic lesions and are the most commonly confused pair (see confusion matrices).
- **Resolution/lighting variance** — handled by resizing to 224x224 and normalizing with
  ImageNet statistics, applied consistently to both models for a fair comparison.
- **Background noise** (soil, adjacent leaves, hands) — mitigated with random crop / affine /
  perspective augmentation during training.

### Preprocessing & augmentation

All splits are resized to 224x224 and normalized with ImageNet mean/std (required for the
pretrained ResNet18 backbone; applied identically to the baseline CNN for a fair comparison).
Training images additionally go through (`utils/transforms.py::get_train_transforms`):

- `RandomResizedCrop`, `RandomHorizontalFlip` — standard flip/crop augmentation
- `RandomRotation` — rotation augmentation
- `ColorJitter` (brightness/contrast/saturation/hue) — lighting robustness
- `RandomAffine` + `RandomPerspective` — geometric/viewpoint robustness
- `AddGaussianNoise` (custom, `p=0.3`, applied post-`ToTensor`) — noise injection, simulating
  sensor noise / compression artifacts in field-captured leaf photos

Validation/test/inference use a deterministic `Resize` + `Normalize` pipeline only (no
augmentation), so evaluation numbers reflect real-world inference conditions.

## Project structure

```
DeepLearingAssignment/
├── config.py                 # Paths, hyperparameters, device detection, reproducibility
├── train.py                  # Train baseline CNN and/or ResNet18
├── evaluate.py                # Evaluate a checkpoint on the test set + model comparison
├── predict.py                 # CLI single-image / batch inference
├── tune.py                    # Hyperparameter grid/random search
├── app.py                     # Streamlit UI for interactive testing
├── requirements.txt
├── README.md
├── PlantVillage/               # Raw dataset (train/, val/), as provided
├── dataset/splits/             # Generated train.csv / val.csv / test.csv manifests
├── models/
│   ├── baseline_cnn.py         # CustomCNN (Conv-BN-ReLU-MaxPool blocks)
│   ├── resnet_model.py         # ResNet18 transfer-learning builder
│   └── __init__.py             # get_model(name) factory
├── utils/
│   ├── split_data.py           # Stratified train/val/test manifest generation
│   ├── dataset.py               # CornLeafDataset + get_dataloaders (weighted sampling)
│   ├── transforms.py            # Train/eval torchvision transform pipelines
│   ├── metrics.py               # Accuracy/precision/recall/F1/ROC-AUC computation
│   ├── visualize.py             # All plotting (samples, curves, confusion matrix, ROC, PR)
│   ├── early_stopping.py        # EarlyStopping (patience on val loss)
│   └── logger.py                # Console + file logging setup
├── notebook/
│   └── Corn_Leaf_Disease_Classification.ipynb
├── checkpoints/                 # {model}_best.pth / {model}_last.pth (generated)
├── results/                     # metrics.json, classification_report.txt, comparison.csv (generated)
├── plots/                       # training curves, confusion matrix, ROC/PR curves (generated)
└── logs/                        # train.log, evaluate.log, predict.log, *_history.json (generated)
```

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
```

CUDA users: if `torch.cuda.is_available()` returns `False` after installing, install the
CUDA-enabled wheel for your GPU from https://pytorch.org/get-started/locally/ — the default
`pip install torch` above installs a CPU-only build unless a CUDA toolkit is detected.
Everything in this project (`config.DEVICE`) works unchanged on CPU or CUDA.

## Quickstart

```bash
# 1. Build the train/val/test manifests (auto-run by train.py if missing)
python -m utils.split_data

# 2. Train both models (defaults: 30 epochs, batch 32, Adam, lr 1e-3)
python train.py --model both

# 3. Evaluate both on the held-out test set + generate comparison table/plot
python evaluate.py --model both --compare

# 4. Predict on a new image
python predict.py --model resnet18 --image path/to/leaf.jpg

# 5. Launch the interactive testing UI
streamlit run app.py
```

## Usage

### Training — `train.py`

```bash
python train.py --model resnet18 --epochs 30 --batch-size 32 --lr 0.001 --optimizer adam
python train.py --model baseline --epochs 20 --batch-size 64 --optimizer sgd --lr 0.0005
python train.py --model both --no-amp                 # disable mixed precision
python train.py --model resnet18 --no-class-weights     # disable imbalance weighting
```

Key flags: `--model {baseline,resnet18,both}`, `--epochs`, `--batch-size`, `--lr`,
`--weight-decay`, `--optimizer {adam,sgd}`, `--patience` (early stopping), `--dropout`,
`--seed`, `--no-amp`, `--no-pretrained`, `--no-class-weights`.

Each run: seeds everything (`config.set_seed`), builds/reuses the dataset manifests, saves
`checkpoints/{model}_best.pth` (highest val accuracy) and `checkpoints/{model}_last.pth` (final
epoch) every epoch, applies `ReduceLROnPlateau` + early stopping on validation loss, and writes
`logs/{model}_history.json` plus `plots/{model}/training_curves.png`.

### Evaluation — `evaluate.py`

```bash
python evaluate.py --model resnet18
python evaluate.py --model baseline --checkpoint checkpoints/baseline_last.pth
python evaluate.py --model both --compare
```

Writes, per model, to `results/{model}/`: `metrics.json`, `classification_report.txt`, and to
`plots/{model}/`: `confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`. With
`--compare` (or `--model both`), also writes `results/model_comparison.csv` and
`plots/model_comparison.png`.

### Prediction — `predict.py`

```bash
python predict.py --model resnet18 --image samples/leaf1.jpg
python predict.py --model resnet18 --dir samples/ --save-csv results/predictions.csv
```

Prints the predicted class + confidence per image and writes a CSV with per-class
probabilities and inference time.

### Hyperparameter search — `tune.py`

```bash
python tune.py --model baseline --epochs 5 --trials 6
```

Searches `config.HP_SEARCH_SPACE` (learning rates `[1e-3, 5e-4, 1e-4]`, batch sizes
`[16, 32, 64]`, optimizers `[adam, sgd]`) with a reduced epoch budget, ranks by best validation
accuracy, and writes `results/hyperparam_search.csv`. Re-run `train.py` with the winning
combination at the full epoch budget for the final model.

## Interactive UI (Streamlit)

```bash
streamlit run app.py
```

`app.py` provides four tabs for testing trained checkpoints without touching the command line:

1. **Single Image** — upload a leaf photo, see the predicted class, confidence, per-class
   probability bar chart, and inference time.
2. **Batch Prediction** — upload multiple images, get a results table + downloadable CSV.
3. **Model Performance** — view saved confusion matrix, ROC curve, PR curve, classification
   report, and the baseline-vs-ResNet18 comparison table/plot (from `evaluate.py` output).
4. **Training Curves** — view saved loss/accuracy curves and the dataset class distribution.

The sidebar lets you switch between the `baseline` and `resnet18` models and pick any available
checkpoint (`{model}_best.pth` / `{model}_last.pth`) from `checkpoints/`. Train at least one
model first (`python train.py --model resnet18`) so a checkpoint exists to load.

## Jupyter notebook

`notebook/Corn_Leaf_Disease_Classification.ipynb` walks through the entire pipeline end-to-end —
dataset overview, augmentation preview, model architecture summary, training both models,
evaluation, model comparison, a hyperparameter-search cell, and a single-image prediction demo.
It imports and calls the exact same functions used by `train.py` / `evaluate.py` / `predict.py`
(no duplicated logic), so notebook results land in the same `checkpoints/`, `results/`, and
`plots/` folders the CLI scripts and Streamlit app read from. A `NOTEBOOK_EPOCHS` variable lets
you run a quick demo (e.g. 5 epochs) before committing to the full 30-epoch training run.

## Model architectures

**Baseline — `CustomCNN`** (`models/baseline_cnn.py`): 4 blocks of
`Conv2D → BatchNorm → ReLU → MaxPool` (channels 3→32→64→128→256), global average pooling,
dropout, and a `Linear(256→128) → ReLU → Dropout → Linear(128→4)` classifier head. Trained from
scratch — the performance floor the advanced model is measured against.

**Advanced — `ResNet18`** (`models/resnet_model.py`): `torchvision.models.resnet18` pretrained
on ImageNet, final FC layer replaced with `Dropout → Linear(512→4)`, fine-tuned end-to-end
(`--no-pretrained` trains from random init instead, `freeze_backbone=True` in
`build_resnet18` switches to feature-extraction-only mode).

Both models output raw logits (softmax is applied at inference time via
`torch.nn.functional.softmax`, and internally by `nn.CrossEntropyLoss` during training) —
standard PyTorch practice for numerical stability.

## Training details

| Setting | Default |
|---|---|
| Epochs | 30 (early stopping patience 7 on val loss) |
| Batch size | 32 |
| Optimizer | Adam (SGD with momentum 0.9 also supported) |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| LR scheduler | `ReduceLROnPlateau` (factor 0.5, patience 3) |
| Loss | `CrossEntropyLoss` with inverse-frequency class weights |
| Regularization | Dropout (0.4), BatchNorm, weight decay, early stopping |
| Mixed precision | Enabled automatically on CUDA (`torch.amp`), no-op on CPU |
| Seed | 42 (Python `random`, NumPy, PyTorch CPU+CUDA, cuDNN deterministic) |

## Evaluation outputs

For each model: **Accuracy, Precision, Recall, F1-score (macro & weighted), ROC-AUC
(one-vs-rest, macro & weighted)**, plus a full `classification_report`, confusion matrix, ROC
curve, and precision-recall curve — one per class. `evaluate.py --compare` additionally reports
**training time, average inference time per image, parameter count, and checkpoint file size**
side-by-side for the baseline vs. advanced model in `results/model_comparison.csv` and
`plots/model_comparison.png`.

## Reproducibility

`config.set_seed(42)` seeds Python's `random`, NumPy, and PyTorch (CPU + all CUDA devices) and
sets `cudnn.deterministic = True` / `cudnn.benchmark = False`. Dataset splits are generated with
the same fixed seed, so re-running `utils.split_data.build_splits(force=True)` reproduces
identical train/val/test manifests.

## Troubleshooting

- **`FileNotFoundError` for a manifest CSV** — run `python -m utils.split_data` (or just run
  `train.py`, which calls it automatically).
- **`Checkpoint not found`** when running `evaluate.py`, `predict.py`, or the Streamlit app —
  train the corresponding model first with `train.py`.
- **Slow CPU training** — reduce `--epochs`, use `--model baseline` (much smaller/faster than
  ResNet18), or reduce `--batch-size`; mixed precision (`torch.amp`) only accelerates CUDA runs.
- **Streamlit shows "No checkpoints found"** — the sidebar model/checkpoint dropdowns only list
  files matching `checkpoints/{model}_*.pth`; train that model or pick the other one.

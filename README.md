# Operation Cyber-Histology

Post-incident reconstruction of a PyTorch image-classification pipeline for the MAI/IDL SS26 final assignment.

The project restores training and inference for three convolutional architectures across four medical image datasets:

- `AlexNet`
- `VGG16`
- `ResNet18`

Supported datasets:

- `cells`
- `chest`
- `lesions`
- `orgs`

## Repository Layout

```text
.
+-- Code/
|   +-- config.json      # Central experiment configuration
|   +-- data.py          # Dataset loading, train/validation split, normalization
|   +-- evaluate.py      # Test-set inference and classification metrics
|   +-- models.py        # AlexNet, VGG16, and ResNet18 definitions
|   +-- train.py         # Main training entry point
|   +-- trainer.py       # Training and validation loop
|   +-- utils.py         # CSV/reporting helpers
+-- AUDIT_LOG.md         # Technical bug and correction log
+-- REPORT.md            # Benchmark summary and model recommendation notes
+-- README.md
+-- assignment_final.pdf
```

The `data/` and `results/` folders are intentionally ignored by Git because the dataset files and generated metrics are local artifacts.

## Prerequisites

- Python 3.10 or newer
- PyTorch
- NumPy
- scikit-learn
- Matplotlib

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

Install dependencies:

```bash
python3 -m pip install torch numpy scikit-learn matplotlib
```

## Data Setup

Download the assignment data from the official course link and place the files in a local `data/` folder at the repository root.

Expected files:

```text
data/
+-- cells.pt
+-- chest.pt
+-- lesions.pt
+-- orgs.pt
+-- organs.pt
```

Each `.pt` file must contain:

- `train_images`
- `train_labels`
- `test_images`
- `test_labels`

## Configuration

The training pipeline is controlled through `Code/config.json`.

Important fields:

| Field | Purpose |
|---|---|
| `DATA` | Default dataset for a single run |
| `MODEL` | Default model for a single run |
| `DATA_PATH` | Relative or absolute path to the dataset folder |
| `BATCH_SIZE` | Mini-batch size |
| `EPOCHS` | Number of training epochs |
| `LEARNING_RATE` | Adam optimizer learning rate |
| `DATASETS` | Dataset-specific channel and class counts |
| `MODELS` | Supported model names |
| `VAL_SPLIT` | Fraction of training data reserved for validation |

## Training

Run Task 1 using the configuration in `Code/config.json`:

```bash
python3 Code/train.py --task task1
```

Run the Task 2 green benchmark matrix:

```bash
python3 Code/train.py --task task2
```

Run Task 3 transfer-learning experiments:

```bash
python3 Code/train.py --task task3
```

To run only one dataset/model pair, set `RUN_ALL` to `false` and edit `DATA`, `MODEL`, and `EPOCHS` in `Code/config.json`.

## Outputs

Training prints epoch-level training and validation loss/accuracy.

After inference, the pipeline reports:

- accuracy
- macro precision
- macro recall
- macro F1-score

Task-specific metrics are appended to:

```text
results/task1_test_metrics.csv
results/task2_test_metrics.csv
results/task3_test_metrics.csv
```

## Verification Commands

Check that the Python files compile:

```bash
python3 -m py_compile Code/train.py Code/trainer.py Code/data.py Code/models.py Code/evaluate.py Code/utils.py Code/runner.py Code/transfer.py
```

Run a short smoke training job by temporarily setting `RUN_ALL` to `false`, choosing one `DATA`/`MODEL` pair in `Code/config.json`, and then running:

```bash
python3 Code/train.py --task task1
```

Check model output shapes for all dataset/model combinations:

```bash
python3 - <<'PY'
import sys, torch
sys.path.insert(0, "Code")
import models

datasets = {
    "cells": (3, 8),
    "chest": (1, 2),
    "lesions": (3, 7),
    "orgs": (1, 11),
}

for dataset, (channels, classes) in datasets.items():
    x = torch.randn(2, channels, 64, 64)
    for model_name in ["AlexNet", "VGG16", "ResNet18"]:
        model = getattr(models, model_name)(in_channels=channels, num_classes=classes)
        model.eval()
        with torch.no_grad():
            y = model(x)
        print(dataset, model_name, tuple(y.shape))
PY
```

## Current Benchmark Status

The completed Task 2 benchmark is stored in `results/task2_test_metrics.csv`. It uses 20 epochs and includes baseline/lightweight results, accuracy metrics, runtime, peak memory, and inference latency.

All four datasets have at least one model above the assignment target. See `REPORT.md` for the final Task 2 benchmark table and model recommendations.

## Notes for Final Submission

Before submitting, confirm that:

- `README.md`, `AUDIT_LOG.md`, and `REPORT.md` are committed.
- `results/task1_test_metrics.csv`, `results/task2_test_metrics.csv`, and `results/task3_test_metrics.csv` are present as final result artifacts.
- `REPORT.md` contains final Task 2 and Task 3 metrics.
- The final branch required by the course contains the production-ready code and documentation.

## References

- GitHub Docs: About READMEs - https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes
- PyTorch Docs: Reproducibility - https://docs.pytorch.org/docs/stable/notes/randomness.html
- scikit-learn Docs: Classification metrics - https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

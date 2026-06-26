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
|   +-- inference.py     # Test-set inference and classification metrics
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

Run the default configuration:

```bash
python3 Code/train.py
```

Run one specific dataset/model pair:

```bash
python3 Code/train.py --data cells --model AlexNet
python3 Code/train.py --data chest --model ResNet18
python3 Code/train.py --data lesions --model VGG16
```

Run every configured dataset/model pair:

```bash
python3 Code/train.py --all
```

Use a separate configuration file:

```bash
python3 Code/train.py --config path/to/config.json
```

## Outputs

Training prints epoch-level training and validation loss/accuracy.

After inference, the pipeline reports:

- accuracy
- macro precision
- macro recall
- macro F1-score

Metrics are appended to:

```text
results/test_metrics.csv
```

## Verification Commands

Check that the Python files compile:

```bash
python3 -m py_compile Code/train.py Code/trainer.py Code/data.py Code/models.py Code/inference.py Code/utils.py
```

Run a short smoke training job:

```bash
python3 Code/train.py --data cells --model AlexNet
```

Check model output shapes for all dataset/model combinations:

```bash
python3 - <<'PY'
import sys, torch
sys.path.insert(0, "Code")
import models

datasets = {
    "cells": (3, 8),
    "chest": (1, 3),
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

The current local smoke benchmark for `cells` + `AlexNet` reached 89.62% test accuracy after 3 epochs. The assignment target for `cells` is 90%, so this result is close but should be rerun with more epochs before final submission.

The full benchmark matrix across all four datasets and all three models must be completed before final grading. See `REPORT.md` for the current benchmark table and remaining work.

## Notes for Final Submission

Before submitting, confirm that:

- `README.md`, `AUDIT_LOG.md`, and `REPORT.md` are committed.
- The full dataset/model benchmark matrix has been run.
- `REPORT.md` contains final metrics for all required permutations.
- The final branch required by the course contains the production-ready code and documentation.

## References

- GitHub Docs: About READMEs - https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes
- PyTorch Docs: Reproducibility - https://docs.pytorch.org/docs/stable/notes/randomness.html
- scikit-learn Docs: Classification metrics - https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

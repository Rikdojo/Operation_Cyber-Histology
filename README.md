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
- `organs` for the Task 3 scarce-data experiment

## Repository Layout

```text
.
+-- Code/
|   +-- config.json      # Central experiment configuration
|   +-- data.py          # Dataset loading, train/validation split, normalization
|   +-- inference.py     # Test-set inference and classification metrics
|   +-- models.py        # AlexNet, VGG16, and ResNet18 definitions
|   +-- run_green.py     # Task 2 green benchmark runner
|   +-- run_task3.py     # Task 3 organs transfer benchmark runner
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
| `PATIENCE` | Early-stopping patience based on validation loss |
| `CHECKPOINT_DIR` | Folder for best validation-loss model files |
| `HISTORY_DIR` | Folder for train/validation loss plots |

Data loading uses a seeded random train/validation split and training-only per-channel normalization.

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

Best model checkpoints are saved under:

```text
results/models/
```

Loss-curve PNG files are saved under:

```text
results/history/
```

## Green Initiative Benchmark

Task 2 adds lightweight model variants and a green benchmark runner.

Lightweight models:

- `LightAlexNet`
- `LightVGG16`
- `LightResNet18`

Check the planned green experiments without training:

```bash
python3 Code/run_green.py --dry-run
```

Run one quick green comparison:

```bash
python3 Code/run_green.py --datasets cells --models AlexNet --variants Baseline,Lightweight --epochs 1
```

Run the full green benchmark matrix:

```bash
python3 Code/run_green.py
```

Green metrics are written to:

```text
results/green_metrics.csv
```

The green runner logs accuracy, precision, recall, macro F1, parameter count, training runtime, inference latency per sample, and CUDA peak memory when running on a CUDA GPU such as Colab T4.

## Task 3 Organs Transfer Benchmark

Task 3 keeps the Task 2 lightweight model work and adds a small transfer-learning experiment for the scarce `organs` data.

The default setup in `Code/config.json` uses:

- source dataset: `chest`
- target dataset: `organs`
- model: `LightVGG16`
- modes: `scratch`, `pretrained`, `finetune`

The assignment's new scarce dataset is `organs.pt`. The older `orgs.pt` file is still part of the main restored benchmark matrix, but Task 3 should use `organs.pt`.

Run the Task 3 benchmark:

```bash
python3 Code/run_task3.py
```

For a quick local check, reduce only the target training epochs:

```bash
python3 Code/run_task3.py --epochs 1
```

If the source checkpoint should be trained again:

```bash
python3 Code/run_task3.py --force-source
```

Task 3 metrics are written to:

```text
results/task3_metrics.csv
```

Current local Task 3 result: `LightVGG16` fine-tuning from `chest` to `organs` reached 57.50% accuracy after 20 target epochs.

The source checkpoint is saved to:

```text
results/task3_chest_lightvgg16.pt
```

## Verification Commands

Check that the Python files compile:

```bash
python3 -m py_compile Code/train.py Code/trainer.py Code/data.py Code/models.py Code/inference.py Code/utils.py Code/run_green.py Code/run_task3.py
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
    "chest": (1, 2),
    "lesions": (3, 7),
    "orgs": (1, 11),
    "organs": (1, 11),
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

The current local smoke benchmark from the earlier Task 1/2 work reached 89.62% test accuracy for `cells` + `AlexNet` after 3 epochs. The assignment target for `cells` is 90%, so this result is close but should be rerun with more epochs before final submission.

The full benchmark matrix across all four datasets and all three models must be completed before final grading. See `REPORT.md` for the current benchmark table and remaining work.

## Notes for Final Submission

Before submitting, confirm that:

- `README.md`, `AUDIT_LOG.md`, and `REPORT.md` are committed.
- The full dataset/model benchmark matrix has been run.
- `REPORT.md` contains final metrics for all required permutations.
- `results/task3_metrics.csv` has been generated for the organs transfer comparison.
- The final branch required by the course contains the production-ready code and documentation.

## References

- GitHub Docs: About READMEs - https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes
- PyTorch Docs: Reproducibility - https://docs.pytorch.org/docs/stable/notes/randomness.html
- scikit-learn Docs: Classification metrics - https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

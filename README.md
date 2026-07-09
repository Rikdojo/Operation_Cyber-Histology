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
|   +-- data.py          # Dataset loading, train/validation split, normalization, data augmentation
|   +-- evaluate.py      # Test-set inference and classification metrics
|   +-- models.py        # AlexNet, VGG16, ResNet18, and lightweight variants
|   +-- runner.py        # Coordinates training, evaluation, checkpointing, and result collection
|   +-- train.py         # Main entry point for Task 1, Task 2, and Task 3 runs
|   +-- trainer.py       # Training, validation, early stopping, and best-state restoration
|   +-- transfer.py      # Scratch, feature-extraction, and fine-tuning model setup
|   +-- utils.py         # CSV writing and training-history plot helpers
+-- tests/
|   +-- test_pipeline.py # Automated unit tests for pipeline components
+-- AUDIT_LOG.md         # Technical bug and correction log
+-- REPORT.md            # Benchmark summary and model recommendation notes
+-- README.md
+-- assignment_final.pdf
```

The `data/` folder is ignored because the dataset files are local artifacts. The `results/` folder is ignored for newly generated files, while the final required CSV files and Task 3 source checkpoint are tracked as submission artifacts.

## Prerequisites

- Python 3.10 or newer
- PyTorch
- NumPy
- scikit-learn
- Matplotlib

## Environment Setup

Choose one of the following environment options.

### Option 1: Python venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

### Option 2: Conda

```bash
conda create -n cyber_histology python=3.10
conda activate cyber_histology
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

| Field           | Purpose                                                                        |
| --------------- | ------------------------------------------------------------------------------ |
| `RUN_ALL`       | Whether to run all configured dataset-model pairs instead of one selected pair |
| `DATA`          | Default dataset for a single run                                               |
| `DATA_PATH`     | Relative or absolute path to the dataset folder                                |
| `BATCH_SIZE`    | Mini-batch size                                                                |
| `MODELS`        | Baseline model architectures available for Task 1 and Task 2                   |
| `MODEL`         | Default model for a single run                                                 |
| `DATASETS`      | Dataset-specific channel and class counts                                      |
| `DROP_RATE`     | Dropout probability                                                            |
| `ACTIVATION`    | Activation function used by ResNet-style blocks                                |
| `LEARNING_RATE` | Adam optimizer learning rate                                                   |
| `EPOCHS`        | Maximum number of training epochs                                              |
| `VAL_SPLIT`     | Fraction of training data reserved for validation                              |
| `SEED`          | Random seed for experiment setup and the train/validation split                |
| `OUTPUT_DIR`    | Directory for saved metrics, checkpoints, and training-history plots           |
| `PATIENCE`      | Early-stopping patience based on validation loss                               |
| `task2`         | Task 2 benchmark configuration for baseline/lightweight model comparisons      |
| `task3`         | Task 3 transfer-learning configuration for the scarce `organs` experiment      |

Data loading uses a seeded random train/validation split and training-only per-channel normalization.

## Running the Pipeline

Run commands from the repository root.

### Task 1: Corrected Baseline Pipeline

```bash
python3 Code/train.py --task task1
```

### Task 2: Green Initiative Benchmark

```bash
python3 Code/train.py --task task2
```

### Task 3: Scarce-Data Transfer Learning Benchmark

```bash
python3 Code/train.py --task task3
```

To run only one configured debug experiment, set `RUN_ALL` to `false` and edit `DATA`, `MODEL`, and `EPOCHS` in `Code/config.json`.

## Outputs

Training prints epoch-level training and validation loss/accuracy.

After inference, the pipeline reports:

- accuracy
- macro precision
- macro recall
- macro F1-score

Metrics are saved to task-specific CSV files:

```text
results/task1_test_metrics.csv
results/task2_test_metrics.csv
results/task3_test_metrics.csv
```

Model checkpoints are saved under:

```text
results/model/
```

Loss-curve PNG files are saved under:

```text
results/history/
```

## Green Initiative Benchmark

Task 2 compares baseline and lightweight model configurations to evaluate the trade-off between classification performance and computational efficiency.

The `task2` block in `Code/config.json` defines:

- `MODELS`: `AlexNet`, `VGG16`, and `ResNet18`
- `MODES`: `Baseline` and `Light`

Task 2 metrics are written to:

```text
results/task2_test_metrics.csv
```

The Task 2 pipeline records classification performance and efficiency metrics:

- accuracy
- macro precision
- macro recall
- macro F1-score
- trainable and total parameter count
- training runtime
- inference latency per sample
- peak CUDA or MPS memory when available

Task 2 also creates the source checkpoint required by Task 3:

```text
results/model/Light_ResNet18_orgs.pt
```

## Task 3 Organs Transfer Benchmark

Task 3 keeps the Task 2 lightweight model work and adds a transfer-learning experiment for the scarce `organs` data.

The `task3` block in `Code/config.json` defines:

- source dataset: `orgs`
- target dataset: `organs`
- model: `Light_ResNet18`
- transfer modes: `scratch`, `feature_extraction`, and `fine_tune`
- data augmentation: disabled by default
- source checkpoint: `results/model/Light_ResNet18_orgs.pt`

The `orgs.pt` dataset is used as the source dataset for learning reusable image features. The smaller `organs.pt` dataset is used as the target dataset for the scarce-data transfer-learning experiment.

Run Task 2 before Task 3 so that the source checkpoint exists.

Task 3 metrics are written to:

```text
results/task3_test_metrics.csv
```

Task 3 saves one target-model checkpoint for each transfer-learning mode:

```text
results/model/Light_ResNet18_scratch_organs.pt
results/model/Light_ResNet18_feature_extraction_organs.pt
results/model/Light_ResNet18_fine_tune_organs.pt
```

## Testing Framework

The repository includes automated unit tests in the `tests/` directory.

Check that all Python modules compile successfully:

```bash
python3 -m compileall Code
```

Run the automated test suite:

```bash
python3 -m unittest discover -s tests
```

The tests verify:

- required configuration sections
- Task 1 and Task 2 experiment-list generation
- model output shapes
- dataset loading and data splitting
- Task 2 result metrics and resource-measurement fields
- device selection and CPU memory handling


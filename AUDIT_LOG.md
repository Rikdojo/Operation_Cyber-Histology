# Incident Audit Log

This log documents the main defects discovered in the recovered Operation Cyber-Histology codebase and the commits that introduced the corresponding fixes in the repository history.

## Scope

Audited files:

- `Code/data.py`
- `Code/models.py`
- `Code/train.py`
- `Code/trainer.py`
- `Code/inference.py`
- `Code/config.json`

## Technical Audit Table

| File name | Problem manifestation | Root cause | Correction implemented | Git commit hash |
|---|---|---|---|---|
| `Code/config.json` | Pipeline could not be controlled externally and required code edits for different datasets/models. | Dataset name, model name, channel count, class count, learning rate, batch size, and epoch settings were not centralized. | Added a JSON configuration file and routed training through configuration values. | `6307acb` |
| `Code/config.json` | Only one dataset shape could be represented safely. | Dataset-specific metadata was stored as flat global fields instead of a dataset registry. | Added a `DATASETS` mapping for `cells`, `chest`, `lesions`, and `orgs`, each with channels and class counts. | `79c80f6` |
| `Code/data.py` | Data loading failed when dataset files used the restored assignment naming convention. | Loader expected a hardcoded alternate file suffix instead of `{dataset}.pt`. | Standardized file lookup to `{dataset}.pt`. | `24fbfee` |
| `Code/data.py` | Validation metrics were unreliable because validation samples could also be included in training. | The training split used the full recovered training tensor before slicing out validation data. | Split `train_images` and `train_labels` so validation samples are excluded from the training dataset. | `6307acb` |
| `Code/data.py` | Validation and test performance could be inflated by preprocessing leakage. | Normalization statistics were not restricted to the training split. | Computed mean and standard deviation from training data only, then reused them for train, validation, and test tensors. | `6307acb` |
| `Code/trainer.py` | `CrossEntropyLoss` could fail or train on incorrectly shaped targets. | Labels were stored as `(N, 1)` tensors and not consistently converted to one-dimensional integer class targets. | Converted labels with `.view(-1).long()` during training and evaluation. | `7535784` |
| `Code/trainer.py` | Gradients could accumulate across mini-batches and destabilize learning. | Optimizer gradients were not cleared before each backward pass. | Called `optimizer.zero_grad()` before computing gradients for each batch. | `6307acb` |
| `Code/trainer.py` | Accuracy calculation was harder to reason about and shadowed a Python built-in. | The variable name `sum` was used for sample counting. | Renamed the counter to `total` and used it consistently in loss/accuracy calculations. | `7535784` |
| `Code/models.py` | ResNet18 produced no logits, causing training/inference to fail. | `forward()` called the classifier but did not return its output. | Returned `self.classifier(out)` from the ResNet18 forward pass. | `6307acb` |
| `Code/models.py` | ResNet18 gradients could be weakened or neutralized by a non-learning activation placeholder. | Default activation was set to identity-like behavior instead of a nonlinear activation. | Set the default activation to `ReLU` and made activation configurable. | `6307acb` |
| `Code/models.py` | VGG blocks could receive incorrect channel dimensions after the first convolution in a block. | `current_in_channels` was not advanced after each convolution. | Updated `current_in_channels` inside the VGG block construction loop. | `6307acb` |
| `Code/models.py` | VGG configuration C tail convolutions had incorrect spatial behavior. | The 1x1 convolution tail used the same padding as 3x3 convolutions. | Used zero padding for 1x1 convolutions and normal padding for 3x3 convolutions. | `6307acb` |
| `Code/models.py` | AlexNet failed for grayscale datasets and datasets with class counts other than 11. | Input channels and output classes were hardcoded. | Changed AlexNet to accept `in_channels` and `num_classes` from the configuration. | `6307acb` |
| `Code/models.py` | AlexNet classifier shape was incompatible with the feature tensor for 64x64 inputs. | The first linear layer expected 2048 features while the adapted feature extractor produced 3072. | Set the first AlexNet classifier layer to `nn.Linear(3072, 1024)`. | `6307acb` |
| `Code/train.py` | Only a single hardcoded experiment could be run cleanly. | Training entry point did not expose dataset/model selection. | Added command-line arguments for `--data`, `--model`, `--config`, and `--all`. | `b3baeb2` |
| `Code/train.py` | Multi-dataset and multi-model execution was not available through one entry point. | No run list existed for the Cartesian product of configured datasets and models. | Added run-list generation from the `DATASETS` and `MODELS` config entries. | `b3baeb2` |
| `Code/inference.py` | GPU inference failed when the model was on GPU but test images stayed on CPU. | Inference moved neither input images nor predictions consistently across devices. | Moved images to the selected device before forward pass and moved labels/predictions back to CPU for metric calculation. | `4ac2bf0` |
| `Code/inference.py` | Macro metrics could break on classes with no predicted samples. | Classification metrics did not guard against undefined precision/recall cases. | Used `zero_division=0` for macro precision, recall, and F1 calculation. | `4ac2bf0` |
| `Code/utils.py` | Results output could fail if the target folder did not exist. | CSV writing assumed the parent output directory was already present. | Created the output directory before writing metrics. | `2de48be` |

## Task 2 Green Initiative Audit Additions

| File name | Problem manifestation | Root cause | Correction implemented | Git commit hash |
|---|---|---|---|---|
| `Code/models.py` | Task 2 lightweight variants could not run because `LightAlexNet`, `LightVGG16`, and `LightResNet18` were invalid empty stubs. | The classes contained placeholder `return` statements instead of real model definitions. | Implemented three valid lightweight architectures with reduced channel counts, adaptive pooling, and the same input/output contract as the baseline models. | `e7a5bde` |
| `Code/runner.py` | Green Initiative experiments did not produce a clean efficiency matrix for comparison. | The experiment runner mixed baseline and lightweight naming conventions and returned inconsistent metric keys. | Standardized baseline/lightweight model resolution and returned one row per experiment with model variant, parameter count, runtime, latency, memory, and classification metrics. | `0a1647f` |
| `Code/run_green.py` | There was no direct command to run the full Task 2 green benchmark matrix. | Task 2 orchestration logic existed only as helper code and was not exposed as a clear reproducible entry point. | Added `Code/run_green.py` with dry-run support, dataset/model/variant filters, epoch override, and CSV output to `results/green_metrics.csv`. | `0a1647f` |
| `Code/inference.py` | Task 2 could not quantify inference efficiency. | Inference only needed accuracy metrics for Task 1 and did not expose latency or peak memory values. | Added inference timing, latency per sample, and CUDA peak inference memory reporting while preserving accuracy, precision, recall, and macro F1. | `e7a5bde` |
| `README.md` / `REPORT.md` | Task 2 workflow and green-analysis expectations were not documented. | Documentation still focused mainly on Task 1 reconstruction and benchmark reporting. | Added Green Initiative run commands, output file description, parameter reduction table, and instructions for completing the final green analysis after GPU benchmarking. | `0a1647f` |

## Current Open Risks

| Area | Risk | Recommendation |
|---|---|---|
| Automated tests | The current `integration` branch does not contain a committed `tests/` folder, even though smoke tests existed earlier in commit `02ba9db`. | Restore or recreate tests before final submission. Minimum tests should cover config parsing, model output shapes, data loading, and a tiny training loop. |
| Full benchmark evidence | `results/test_metrics.csv` currently contains local smoke results for `cells` + `AlexNet`, not the full 12-run benchmark matrix. | Run all dataset/model combinations and update `REPORT.md` with final metrics. |
| Dependency reproducibility | Dependencies are documented in `README.md`, but there is no committed `requirements.txt`. | Add a locked or minimal dependency file before final packaging. |
| Device selection | The code selects CUDA or CPU, but Apple Silicon MPS is available on this machine and is not used. | Consider adding MPS support for faster local experimentation on Mac. |
| Output configuration | `OUTPUT_DIR` exists in `config.json`, but `train.py` currently writes to `results/test_metrics.csv` directly. | Route result output through `OUTPUT_DIR` for full configuration consistency. |

## Verification Evidence

Local checks performed on the current `integration` worktree:

```bash
python3 -m py_compile Code/train.py Code/trainer.py Code/data.py Code/models.py Code/inference.py Code/utils.py
```

All listed files compiled successfully.

Model forward-pass shape checks succeeded for all combinations of:

- 4 datasets
- 3 model architectures

The current local smoke run completed:

```bash
python3 Code/train.py --data cells --model AlexNet
```

The 3-epoch local result recorded in `results/test_metrics.csv` reached 89.62% test accuracy and 88.30% macro F1 on `cells` + `AlexNet`.

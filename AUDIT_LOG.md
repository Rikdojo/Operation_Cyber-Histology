# Incident Audit Log

This log documents the main defects discovered in the recovered Operation Cyber-Histology codebase and the commits that introduced the corresponding fixes in the repository history.

## Scope

Audited files:

- `Code/data.py`
- `Code/models.py`
- `Code/train.py`
- `Code/trainer.py`
- `Code/evaluate.py`
- `Code/config.json`

## Technical Audit Table

| File name           | Problem manifestation                                                                              | Root cause                                                                                                                | Correction implemented                                                                                                   | Git commit hash |
| ------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | --------------- |
| `Code/config.json`  | Pipeline could not be controlled externally and required code edits for different datasets/models. | Dataset name, model name, channel count, class count, learning rate, batch size, and epoch settings were not centralized. | Added a JSON configuration file and routed training through configuration values.                                        | `6307acb`       |
| `Code/config.json`  | Only one dataset shape could be represented safely.                                                | Dataset-specific metadata was stored as flat global fields instead of a dataset registry.                                 | Added a `DATASETS` mapping for `cells`, `chest`, `lesions`, and `orgs`, each with channels and class counts.             | `79c80f6`       |
| `Code/data.py`      | Data loading failed when dataset files used the restored assignment naming convention.             | Loader expected a hardcoded alternate file suffix instead of `{dataset}.pt`.                                              | Standardized file lookup to `{dataset}.pt`.                                                                              | `24fbfee`       |
| `Code/data.py`      | Validation metrics were unreliable because validation samples could also be included in training.  | The training split used the full recovered training tensor before slicing out validation data.                            | Split `train_images` and `train_labels` so validation samples are excluded from the training dataset.                    | `6307acb`       |
| `Code/data.py`      | Validation and test performance could be inflated by preprocessing leakage.                        | Normalization statistics were not restricted to the training split.                                                       | Computed mean and standard deviation from training data only, then reused them for train, validation, and test tensors.  | `6307acb`       |
| `Code/trainer.py`   | `CrossEntropyLoss` could fail or train on incorrectly shaped targets.                              | Labels were stored as `(N, 1)` tensors and not consistently converted to one-dimensional integer class targets.           | Converted labels with `.view(-1).long()` during training and evaluation.                                                 | `7535784`       |
| `Code/trainer.py`   | Gradients could accumulate across mini-batches and destabilize learning.                           | Optimizer gradients were not cleared before each backward pass.                                                           | Called `optimizer.zero_grad()` before computing gradients for each batch.                                                | `6307acb`       |
| `Code/trainer.py`   | Accuracy calculation was harder to reason about and shadowed a Python built-in.                    | The variable name `sum` was used for sample counting.                                                                     | Renamed the counter to `total` and used it consistently in loss/accuracy calculations.                                   | `7535784`       |
| `Code/models.py`    | ResNet18 produced no logits, causing training/inference to fail.                                   | `forward()` called the classifier but did not return its output.                                                          | Returned `self.classifier(out)` from the ResNet18 forward pass.                                                          | `6307acb`       |
| `Code/models.py`    | ResNet18 gradients could be weakened or neutralized by a non-learning activation placeholder.      | Default activation was set to identity-like behavior instead of a nonlinear activation.                                   | Set the default activation to `ReLU` and made activation configurable.                                                   | `6307acb`       |
| `Code/models.py`    | VGG blocks could receive incorrect channel dimensions after the first convolution in a block.      | `current_in_channels` was not advanced after each convolution.                                                            | Updated `current_in_channels` inside the VGG block construction loop.                                                    | `6307acb`       |
| `Code/models.py`    | VGG configuration C tail convolutions had incorrect spatial behavior.                              | The 1x1 convolution tail used the same padding as 3x3 convolutions.                                                       | Used zero padding for 1x1 convolutions and normal padding for 3x3 convolutions.                                          | `6307acb`       |
| `Code/models.py`    | AlexNet failed for grayscale datasets and datasets with class counts other than 11.                | Input channels and output classes were hardcoded.                                                                         | Changed AlexNet to accept `in_channels` and `num_classes` from the configuration.                                        | `6307acb`       |
| `Code/models.py`    | AlexNet classifier shape was incompatible with the feature tensor for 64x64 inputs.                | The first linear layer expected 2048 features while the adapted feature extractor produced 3072.                          | Set the first AlexNet classifier layer to `nn.Linear(3072, 1024)`.                                                       | `6307acb`       |
| `Code/train.py`     | Only a single hardcoded experiment could be run cleanly.                                           | Training entry point did not expose dataset/model selection.                                                              | Added run-list support for configured dataset/model combinations.                                                        | `b3baeb2`       |
| `Code/train.py`     | Running `python3 Code/train.py --task ...` could crash before training started.                    | The parser only defined `--task`, but `main()` still tried to read `args.config`.                                         | Loaded the default `Code/config.json` directly and kept `--task` as the only command-line switch.                        | `180f2b1`       |
| `Code/train.py`     | `RUN_ALL=false` still behaved like a multi-model run.                                             | The single-run branch returned every configured model instead of the selected `DATA` and `MODEL`.                         | Made task 1 return one configured dataset/model pair and task 2 return one baseline debug run when `RUN_ALL` is false.   | `180f2b1`       |
| `Code/runner.py`    | Relative data paths pointed at `Code/data` instead of the repository-level `data` folder.          | `get_data_path()` resolved paths from the `Code/` directory.                                                             | Resolved relative data paths from the project root with `Path(__file__).resolve().parent.parent`.                        | `4582999`       |
| `Code/config.json`  | JSON silently overwrote one `task2` block.                                                        | The config file contained duplicate `task2` keys.                                                                         | Removed the duplicate block so task 2 has one clear model/mode registry.                                                 | `1fb8472`       |
| `Code/evaluate.py`  | GPU inference failed when the model was on GPU but test images stayed on CPU.                     | Inference moved neither input images nor predictions consistently across devices.                                         | Moved images to the selected device before forward pass and moved labels/predictions back to CPU for metric calculation. | `4ac2bf0`       |
| `Code/evaluate.py`  | Macro metrics could break on classes with no predicted samples.                                   | Classification metrics did not guard against undefined precision/recall cases.                                            | Used `zero_division=0` for macro precision, recall, and F1 calculation.                                                  | `4ac2bf0`       |
| `Code/utils.py`     | Results output could fail if the target folder did not exist.                                      | CSV writing assumed the parent output directory was already present.                                                      | Created the output directory before writing metrics.                                                                     | `2de48be`       |
| `Code/runner.py`    | Green benchmarking could not report useful local device behavior on Apple Silicon.                 | Device selection only considered CUDA and CPU, and memory reporting only used CUDA peak-memory APIs.                      | Added CUDA/MPS/CPU device selection, MPS memory reporting, and an explicit CPU fallback value.                            | `1fb8472`       |
| `Code/train.py`     | Task 2 benchmark CSV did not clearly separate baseline and light runs.                             | Model names were changed during execution, but the output row did not preserve an explicit run mode.                      | Added `mode`, total/trainable parameter counts, runtime, latency, and memory fields to the CSV row.                      | `1fb8472`       |
| `Code/data.py`      | Tests failed in a fresh environment without `torchvision`.                                        | Data augmentation depended on `torchvision.transforms` for a small optional flip/affine step.                             | Replaced the dependency with a simple PyTorch horizontal flip for training augmentation.                                  | `1fb8472`       |
| `tests/test_pipeline.py` | The integration branch had no committed testing framework.                                      | The original recovered project did not include tests for config, model shapes, loaders, or runner helpers.                | Added a simple `unittest` pipeline test file covering config, run lists, model shapes, data loading, green metrics, and device fallback. | `6163c34` |

## Current Open Risks

| Area                       | Risk                                                                                                                                        | Recommendation                                                                                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Full benchmark evidence    | The final 12-run task 1 benchmark matrix and task 2 green matrix still need to be generated from the current code.                         | Run all dataset/model combinations and update `REPORT.md` with final metrics.                                                                              |
| Dependency reproducibility | Dependencies are documented in `README.md`, but there is no committed `requirements.txt`.                                                   | Add a small dependency file before final packaging.                                                                                                        |
| README/code mismatch       | Some README commands and file names may still describe older script names or output names.                                                   | Update README after the final code path is frozen.                                                                                                         |
| Task 3 checkpoint          | `feature_extraction` and `fine_tune` modes require the source checkpoint named in `task3.CHECKPOINT`.                                       | Train the source model first or add a clear source-checkpoint preparation step before running those modes.                                                  |

## Verification Evidence

Local checks performed on the current `integration` worktree:

```bash
python3 -m py_compile Code/train.py Code/trainer.py Code/data.py Code/models.py Code/evaluate.py Code/runner.py Code/transfer.py Code/utils.py
```

All listed files compiled successfully.

The current testing framework runs with:

```bash
python3 -m unittest discover -s tests
```

The tests cover config loading, task 1/task 2 run lists, model output shapes, synthetic `.pt` data loading, green metric output fields, and device/memory fallback behavior.

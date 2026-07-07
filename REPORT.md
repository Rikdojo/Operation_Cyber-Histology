# Consolidated Benchmark Report

## Executive Summary

The reconstructed pipeline trains and evaluates the recovered convolutional classifiers through a shared configuration file. The final artifacts cover the corrected Task 1 baseline matrix, the Task 2 Green Initiative matrix, and the Task 3 scarce-data transfer experiment.

The final result artifacts are stored in:

```text
results/task1_test_metrics.csv
results/task2_test_metrics.csv
results/task3_test_metrics.csv
```

All final runs use 20 epochs with early stopping enabled. The CSV files include classification metrics plus green-efficiency measurements: parameter count, training runtime, peak memory, inference runtime, and inference latency per sample.

## Assignment Accuracy Targets

| Dataset | Minimal accuracy |
|---|---:|
| `cells` | 90% |
| `chest` | 87% |
| `lesions` | 67% |
| `orgs` | 83% |

## Current Validated Benchmark Results

These results are taken from the completed 20-epoch Task 2 CSV in `results/task2_test_metrics.csv`. Task 2 contains both the baseline models and lightweight variants, so this table selects the best final model for each assignment dataset from the complete benchmark matrix.

| Dataset | Best model | Mode | Accuracy | Precision | Recall | Macro F1 | Target | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| `cells` | `Light_ResNet18` | Light | 97.25% | 97.18% | 96.93% | 97.02% | 90.00% | Pass |
| `chest` | `ResNet18` | Baseline | 89.90% | 92.44% | 86.79% | 88.57% | 87.00% | Pass |
| `lesions` | `Light_ResNet18` | Light | 76.16% | 54.22% | 44.12% | 46.98% | 67.00% | Pass |
| `orgs` | `Light_ResNet18` | Light | 92.15% | 91.51% | 91.34% | 91.31% | 83.00% | Pass |

## Task 1 Baseline Matrix

Task 1 evaluates the corrected baseline `AlexNet`, `VGG16`, and `ResNet18` implementations across the four required datasets.

| Dataset | Model | Epochs | Parameters | Train time | Train memory | Accuracy | Precision | Recall | Macro F1 | Latency/sample |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cells` | `AlexNet` | 20 | 5,693,544 | 60.3s | 186.6 MB | 94.18% | 94.93% | 92.67% | 93.59% | 0.116 ms |
| `cells` | `VGG16` | 20 | 12,631,624 | 380.2s | 607.0 MB | 97.16% | 97.21% | 96.66% | 96.90% | 0.491 ms |
| `cells` | `ResNet18` | 20 | 11,172,936 | 635.5s | 837.4 MB | 96.73% | 96.39% | 96.64% | 96.46% | 1.004 ms |
| `chest` | `AlexNet` | 20 | 5,682,690 | 28.5s | 183.5 MB | 86.70% | 91.23% | 82.26% | 84.41% | 0.105 ms |
| `chest` | `VGG16` | 20 | 12,627,394 | 103.2s | 608.8 MB | 89.42% | 91.58% | 86.41% | 88.07% | 0.490 ms |
| `chest` | `ResNet18` | 20 | 11,168,706 | 327.0s | 833.9 MB | 89.90% | 92.44% | 86.79% | 88.57% | 1.020 ms |
| `lesions` | `AlexNet` | 20 | 5,692,519 | 48.6s | 185.9 MB | 75.46% | 53.26% | 47.11% | 48.07% | 0.114 ms |
| `lesions` | `VGG16` | 20 | 12,631,111 | 225.9s | 609.6 MB | 72.92% | 43.98% | 38.49% | 38.13% | 0.488 ms |
| `lesions` | `ResNet18` | 20 | 11,172,423 | 496.8s | 835.1 MB | 76.11% | 60.98% | 48.32% | 50.84% | 1.016 ms |
| `orgs` | `AlexNet` | 20 | 5,691,915 | 87.5s | 185.7 MB | 89.51% | 88.58% | 88.60% | 88.34% | 0.094 ms |
| `orgs` | `VGG16` | 20 | 12,632,011 | 433.3s | 610.4 MB | 90.62% | 89.70% | 89.89% | 89.57% | 0.483 ms |
| `orgs` | `ResNet18` | 20 | 11,173,323 | 958.1s | 838.0 MB | 91.98% | 90.68% | 90.92% | 90.68% | 0.999 ms |

## Full Task 2 Benchmark Matrix

Task 2 compares each baseline architecture with its lightweight counterpart. Baseline rows are included here as the reference point for the Green Initiative comparison.

| Dataset | Model | Mode | Epochs | Parameters | Train time | Train memory | Accuracy | Precision | Recall | Macro F1 | Latency/sample |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cells` | `AlexNet` | Baseline | 20 | 5,693,544 | 60.3s | 186.6 MB | 94.18% | 94.93% | 92.67% | 93.59% | 0.116 ms |
| `cells` | `Light_AlexNet` | Light | 20 | 1,522,472 | 67.1s | 115.2 MB | 96.17% | 95.65% | 96.40% | 95.93% | 0.129 ms |
| `cells` | `VGG16` | Baseline | 20 | 12,631,624 | 380.2s | 607.0 MB | 97.16% | 97.21% | 96.66% | 96.90% | 0.491 ms |
| `cells` | `Light_VGG16` | Light | 20 | 4,447,752 | 287.9s | 409.8 MB | 95.73% | 96.48% | 94.94% | 95.62% | 0.375 ms |
| `cells` | `ResNet18` | Baseline | 20 | 11,172,936 | 635.5s | 837.4 MB | 96.73% | 96.39% | 96.64% | 96.46% | 1.004 ms |
| `cells` | `Light_ResNet18` | Light | 20 | 4,184,008 | 685.7s | 659.7 MB | 97.25% | 97.18% | 96.93% | 97.02% | 0.826 ms |
| `chest` | `AlexNet` | Baseline | 20 | 5,682,690 | 28.5s | 183.5 MB | 86.70% | 91.23% | 82.26% | 84.41% | 0.105 ms |
| `chest` | `Light_AlexNet` | Light | 20 | 1,517,378 | 20.9s | 112.7 MB | 84.29% | 89.37% | 79.23% | 81.29% | 0.123 ms |
| `chest` | `VGG16` | Baseline | 20 | 12,627,394 | 103.2s | 608.8 MB | 89.42% | 91.58% | 86.41% | 88.07% | 0.490 ms |
| `chest` | `Light_VGG16` | Light | 20 | 4,445,826 | 103.9s | 407.0 MB | 86.38% | 90.29% | 82.09% | 84.13% | 0.373 ms |
| `chest` | `ResNet18` | Baseline | 20 | 11,168,706 | 327.0s | 833.9 MB | 89.90% | 92.44% | 86.79% | 88.57% | 1.020 ms |
| `chest` | `Light_ResNet18` | Light | 20 | 4,181,314 | 251.1s | 653.8 MB | 83.65% | 89.32% | 78.29% | 80.35% | 0.819 ms |
| `lesions` | `AlexNet` | Baseline | 20 | 5,692,519 | 48.6s | 185.9 MB | 75.46% | 53.26% | 47.11% | 48.07% | 0.114 ms |
| `lesions` | `Light_AlexNet` | Light | 20 | 1,522,407 | 41.1s | 115.2 MB | 75.21% | 49.51% | 42.57% | 45.23% | 0.113 ms |
| `lesions` | `VGG16` | Baseline | 20 | 12,631,111 | 225.9s | 609.6 MB | 72.92% | 43.98% | 38.49% | 38.13% | 0.488 ms |
| `lesions` | `Light_VGG16` | Light | 20 | 4,447,623 | 169.2s | 409.2 MB | 71.92% | 37.78% | 30.37% | 29.99% | 0.382 ms |
| `lesions` | `ResNet18` | Baseline | 20 | 11,172,423 | 496.8s | 835.1 MB | 76.11% | 60.98% | 48.32% | 50.84% | 1.016 ms |
| `lesions` | `Light_ResNet18` | Light | 20 | 4,183,751 | 402.4s | 658.0 MB | 76.16% | 54.22% | 44.12% | 46.98% | 0.824 ms |
| `orgs` | `AlexNet` | Baseline | 20 | 5,691,915 | 87.5s | 185.7 MB | 89.51% | 88.58% | 88.60% | 88.34% | 0.094 ms |
| `orgs` | `Light_AlexNet` | Light | 20 | 1,517,963 | 72.3s | 114.2 MB | 90.01% | 88.66% | 89.03% | 88.72% | 0.088 ms |
| `orgs` | `VGG16` | Baseline | 20 | 12,632,011 | 433.3s | 610.4 MB | 90.62% | 89.70% | 89.89% | 89.57% | 0.483 ms |
| `orgs` | `Light_VGG16` | Light | 20 | 4,446,987 | 321.8s | 406.5 MB | 91.35% | 90.63% | 90.18% | 90.23% | 0.368 ms |
| `orgs` | `ResNet18` | Baseline | 20 | 11,173,323 | 958.1s | 838.0 MB | 91.98% | 90.68% | 90.92% | 90.68% | 0.999 ms |
| `orgs` | `Light_ResNet18` | Light | 20 | 4,183,627 | 772.8s | 660.0 MB | 92.15% | 91.51% | 91.34% | 91.31% | 0.826 ms |

## Methodology

The current pipeline sets Python/PyTorch random seeds, uses a seeded random train/validation split, computes z-score normalization statistics from the training split only, trains with `CrossEntropyLoss` and Adam, restores the best validation-loss model after early stopping, and logs macro-averaged precision/recall/F1 plus runtime, memory, and latency.

Macro-averaged metrics are appropriate because the datasets are multi-class and may have class imbalance. Macro averaging gives each class equal weight instead of allowing large classes to dominate the score.

## Architecture Recommendations

| Dataset | Recommendation | Reason |
|---|---|---|
| `cells` | `Light_ResNet18` | Highest accuracy and macro F1 while using fewer parameters than baseline `ResNet18`. |
| `chest` | `ResNet18` | Best accuracy and macro F1; the lightweight version loses too much accuracy on this dataset. |
| `lesions` | `Light_ResNet18` | Slightly highest accuracy and lower memory/latency than baseline `ResNet18`; macro F1 remains weak, so class imbalance should be discussed. |
| `orgs` | `Light_ResNet18` | Highest accuracy and macro F1 with lower memory and latency than baseline `ResNet18`. |

## Green Initiative Analysis

Task 2 adds lightweight versions of the restored model families:

- `Light_AlexNet`
- `Light_VGG16`
- `Light_ResNet18`

| Model family | Baseline parameters | Lightweight parameters | Parameter reduction |
|---|---:|---:|---:|
| AlexNet | 5,690,167 | 1,520,055 | 73.29% |
| VGG16 | 12,630,535 | 4,447,047 | 64.79% |
| ResNet18 | 11,171,847 | 4,183,175 | 62.56% |

Parameter counts vary slightly by dataset because the classifier output size changes with the number of classes. The table reports the mean parameter count across the four datasets.

The next table summarizes average cost changes from each baseline family to its lightweight counterpart across the four assignment datasets.

| Model family | Parameter reduction | Train-memory reduction | Latency reduction | Training-time change | Accuracy change |
|---|---:|---:|---:|---:|---:|
| AlexNet | 73.29% | 38.34% | -5.78% | -10.49% | -0.04 pp |
| VGG16 | 64.79% | 32.98% | 23.23% | -22.74% | -1.19 pp |
| ResNet18 | 62.56% | 21.32% | 18.44% | -12.63% | -1.38 pp |

The lightweight models substantially reduce parameter count and memory. `Light_VGG16` and `Light_ResNet18` also reduce inference latency on average. `Light_AlexNet` reduces parameters and memory but has slightly worse average latency in these runs, so its main benefit is model-size reduction rather than faster inference.

The completed benchmark supports `Light_ResNet18` as the best overall green recommendation for `cells`, `lesions`, and `orgs`. It keeps the residual architecture's accuracy advantage while reducing parameter count, memory footprint, and inference latency relative to baseline `ResNet18`. For `chest`, baseline `ResNet18` should be kept because it is the strongest result above the 87% target.

## Task 3: Organs Scarce-Data Transfer

The new `organs` dataset is smaller than the original `orgs` dataset and has 11 classes. Task 3 compares training from scratch against transfer from the larger `orgs` profile. The source checkpoint is:

```text
results/model/Light_ResNet18_orgs.pt
```

Task 3 results are written to `results/task3_test_metrics.csv`.

| Dataset | Model | Mode | Epochs | Trainable params | Train time | Memory | Accuracy | Precision | Recall | Macro F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `organs` | `Light_ResNet18_scratch` | `scratch` | 20 | 4,183,627 | 59.8s | 121.4 MB | 64.00% | 60.86% | 56.62% | 56.95% |
| `organs` | `Light_ResNet18_feature_extraction` | `feature_extraction` | 20 | 2,827 | 18.9s | 29.8 MB | 62.50% | 55.83% | 54.64% | 52.79% |
| `organs` | `Light_ResNet18_fine_tune` | `fine_tune` | 20 | 2,266,379 | 15.5s | 316.3 MB | 66.50% | 64.36% | 58.85% | 58.39% |

All three Task 3 modes exceed the requested 40% test accuracy. The best result is `fine_tune`, with 66.50% accuracy and 58.39% macro F1. This suggests that the `orgs` source checkpoint provides useful features, but the target task still benefits from adapting the deeper residual stage to the scarce `organs` images.

The `organs` target set has only 450 training samples after the validation split and 50 validation samples. Because the validation set is small, individual percentage points are noisy. The practical recommendation is to use `fine_tune` for the current Task 3 result, then rerun with multiple random seeds when more compute is available.

## Current Limitations

- The results come from one run per configuration, so random-seed variance is not measured.
- The `lesions` dataset has weak macro F1 despite passing the accuracy target.
- The `organs` validation split is small, so Task 3 validation percentages are noisy.
- Memory values are runtime-dependent and may differ slightly on another GPU.
- The train/validation split is seeded, but the training-loader shuffle order does not currently pass an explicit seeded generator.

## Reproducibility Checklist

- Install dependencies documented in `README.md`.
- Confirm all five `.pt` files exist in `data/`: `cells.pt`, `chest.pt`, `lesions.pt`, `orgs.pt`, and `organs.pt`.
- Run syntax checks.
- Run the automated unit tests.
- Confirm `results/task1_test_metrics.csv`, `results/task2_test_metrics.csv`, and `results/task3_test_metrics.csv` are present.
- Confirm `results/model/Light_ResNet18_orgs.pt` is present before rerunning Task 3 transfer modes.
- Keep `EPOCHS` set to 20 in `Code/config.json` for reproducibility with the final CSV files.

## References

- PyTorch reproducibility guidance: https://docs.pytorch.org/docs/stable/notes/randomness.html
- scikit-learn classification metric definitions: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
- Assignment dossier: `assignment_final.pdf`

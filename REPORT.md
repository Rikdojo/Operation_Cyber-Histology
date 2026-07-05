# Consolidated Benchmark Report

## Executive Summary

The reconstructed pipeline can train and evaluate the recovered convolutional classifiers through a shared configuration file. The Task 2 benchmark has been completed for all four datasets, all three baseline model families, and their lightweight variants.

The final result artifacts are stored in `results/task1_test_metrics.csv`, `results/task2_test_metrics.csv`, and `results/task3_test_metrics.csv`. These results use 20 epochs and include accuracy metrics plus green-efficiency measurements: parameter count, training runtime, peak memory, inference runtime, and inference latency per sample.

## Assignment Accuracy Targets

| Dataset | Minimal accuracy |
|---|---:|
| `cells` | 90% |
| `chest` | 87% |
| `lesions` | 67% |
| `orgs` | 83% |

## Current Validated Benchmark Results

These results are taken from the completed 20-epoch Task 2 CSV in `results/task2_test_metrics.csv`.

| Dataset | Best model | Mode | Accuracy | Macro F1 | Target | Status |
|---|---|---|---:|---:|---:|---|
| `cells` | `Light_ResNet18` | Light | 97.25% | 97.02% | 90.00% | Pass |
| `chest` | `ResNet18` | Baseline | 89.90% | 88.57% | 87.00% | Pass |
| `lesions` | `Light_ResNet18` | Light | 76.16% | 46.98% | 67.00% | Pass |
| `orgs` | `Light_ResNet18` | Light | 92.15% | 91.31% | 83.00% | Pass |

## Full Task 2 Benchmark Matrix

| Dataset | Model | Mode | Epochs | Parameters | Train time | Train memory | Accuracy | Macro F1 | Latency/sample |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `cells` | `AlexNet` | Baseline | 20 | 5,693,544 | 60.3s | 186.6 MB | 94.18% | 93.59% | 0.116 ms |
| `cells` | `Light_AlexNet` | Light | 20 | 1,522,472 | 67.1s | 115.2 MB | 96.17% | 95.93% | 0.129 ms |
| `cells` | `VGG16` | Baseline | 20 | 12,631,624 | 380.2s | 607.0 MB | 97.16% | 96.90% | 0.491 ms |
| `cells` | `Light_VGG16` | Light | 20 | 4,447,752 | 287.9s | 409.8 MB | 95.73% | 95.62% | 0.375 ms |
| `cells` | `ResNet18` | Baseline | 20 | 11,172,936 | 635.5s | 837.4 MB | 96.73% | 96.46% | 1.004 ms |
| `cells` | `Light_ResNet18` | Light | 20 | 4,184,008 | 685.7s | 659.7 MB | 97.25% | 97.02% | 0.826 ms |
| `chest` | `AlexNet` | Baseline | 20 | 5,682,690 | 28.5s | 183.5 MB | 86.70% | 84.41% | 0.105 ms |
| `chest` | `Light_AlexNet` | Light | 20 | 1,517,378 | 20.9s | 112.7 MB | 84.29% | 81.29% | 0.123 ms |
| `chest` | `VGG16` | Baseline | 20 | 12,627,394 | 103.2s | 608.8 MB | 89.42% | 88.07% | 0.490 ms |
| `chest` | `Light_VGG16` | Light | 20 | 4,445,826 | 103.9s | 407.0 MB | 86.38% | 84.13% | 0.373 ms |
| `chest` | `ResNet18` | Baseline | 20 | 11,168,706 | 327.0s | 833.9 MB | 89.90% | 88.57% | 1.020 ms |
| `chest` | `Light_ResNet18` | Light | 20 | 4,181,314 | 251.1s | 653.8 MB | 83.65% | 80.35% | 0.819 ms |
| `lesions` | `AlexNet` | Baseline | 20 | 5,692,519 | 48.6s | 185.9 MB | 75.46% | 48.07% | 0.114 ms |
| `lesions` | `Light_AlexNet` | Light | 20 | 1,522,407 | 41.1s | 115.2 MB | 75.21% | 45.23% | 0.113 ms |
| `lesions` | `VGG16` | Baseline | 20 | 12,631,111 | 225.9s | 609.6 MB | 72.92% | 38.13% | 0.488 ms |
| `lesions` | `Light_VGG16` | Light | 20 | 4,447,623 | 169.2s | 409.2 MB | 71.92% | 29.99% | 0.382 ms |
| `lesions` | `ResNet18` | Baseline | 20 | 11,172,423 | 496.8s | 835.1 MB | 76.11% | 50.84% | 1.016 ms |
| `lesions` | `Light_ResNet18` | Light | 20 | 4,183,751 | 402.4s | 658.0 MB | 76.16% | 46.98% | 0.824 ms |
| `orgs` | `AlexNet` | Baseline | 20 | 5,691,915 | 87.5s | 185.7 MB | 89.51% | 88.34% | 0.094 ms |
| `orgs` | `Light_AlexNet` | Light | 20 | 1,517,963 | 72.3s | 114.2 MB | 90.01% | 88.72% | 0.088 ms |
| `orgs` | `VGG16` | Baseline | 20 | 12,632,011 | 433.3s | 610.4 MB | 90.62% | 89.57% | 0.483 ms |
| `orgs` | `Light_VGG16` | Light | 20 | 4,446,987 | 321.8s | 406.5 MB | 91.35% | 90.23% | 0.368 ms |
| `orgs` | `ResNet18` | Baseline | 20 | 11,173,323 | 958.1s | 838.0 MB | 91.98% | 90.68% | 0.999 ms |
| `orgs` | `Light_ResNet18` | Light | 20 | 4,183,627 | 772.8s | 660.0 MB | 92.15% | 91.31% | 0.826 ms |

## Methodology

The current pipeline uses deterministic PyTorch seeding, a seeded random train/validation split, training-split-only z-score normalization, `CrossEntropyLoss`, Adam optimization, early stopping with patience, macro-averaged precision/recall/F1, and runtime/memory/latency logging.

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

| Model family | Baseline parameters | Lightweight parameters | Reduction |
|---|---:|---:|---:|
| AlexNet | 5,690,167 | 1,520,055 | 73.29% |
| VGG16 | 12,630,535 | 4,447,047 | 64.79% |
| ResNet18 | 11,171,847 | 4,183,175 | 62.56% |

Parameter counts vary slightly by dataset because the classifier output size changes with the number of classes. The table reports the mean parameter count across the four datasets.

The completed benchmark supports `Light_ResNet18` as the best overall green recommendation for `cells`, `lesions`, and `orgs`. It keeps the residual architecture's accuracy advantage while reducing parameter count, memory footprint, and inference latency relative to baseline `ResNet18`. For `chest`, baseline `ResNet18` should be kept because it is the strongest result above the 87% target.

## Task 3: Organs Scarce-Data Transfer

The new `organs` dataset is smaller than the original `orgs` dataset and has 11 classes. Task 3 compares training from scratch against transfer from the larger `orgs` profile. The source checkpoint is:

```text
results/model/Light_ResNet18_orgs.pt
```

Task 3 results are written to `results/task3_test_metrics.csv`.

| Dataset | Model | Mode | Epochs | Trainable params | Train time | Memory | Accuracy | Macro F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `organs` | `Light_ResNet18` | `scratch` | 20 | 4,183,627 | 59.8s | 121.4 MB | 64.00% | 56.95% |
| `organs` | `Light_ResNet18` | `feature_extraction` | 20 | 2,827 | 18.9s | 29.8 MB | 62.50% | 52.79% |
| `organs` | `Light_ResNet18` | `fine_tune` | 20 | 2,266,379 | 15.5s | 316.3 MB | 66.50% | 58.39% |

All three Task 3 modes exceed the requested 40% test accuracy. The best result is `fine_tune`, with 66.50% accuracy and 58.39% macro F1. This suggests that the `orgs` source checkpoint provides useful features, but the target task still benefits from adapting the deeper residual stage to the scarce `organs` images.

The `organs` target set has only 450 training samples after the validation split and 50 validation samples. Because the validation set is small, individual percentage points are noisy. The practical recommendation is to use `fine_tune` for the current Task 3 result, then rerun with multiple random seeds when more compute is available.

## Current Limitations

- The results come from one run per configuration, so random-seed variance is not measured.
- The `lesions` dataset has weak macro F1 despite passing the accuracy target.
- The `organs` validation split is small, so Task 3 validation percentages are noisy.
- Memory values are runtime-dependent and may differ slightly on another GPU.

## Reproducibility Checklist

- Install dependencies documented in `README.md`.
- Confirm all five `.pt` files exist in `data/`: `cells.pt`, `chest.pt`, `lesions.pt`, `orgs.pt`, and `organs.pt`.
- Run syntax checks.
- Confirm `results/task1_test_metrics.csv`, `results/task2_test_metrics.csv`, and `results/task3_test_metrics.csv` are present.
- Confirm `results/model/Light_ResNet18_orgs.pt` is present before rerunning Task 3 transfer modes.
- Keep `EPOCHS` set to 20 in `Code/config.json` for reproducibility with the final CSV files.

## References

- PyTorch reproducibility guidance: https://docs.pytorch.org/docs/stable/notes/randomness.html
- scikit-learn classification metric definitions: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
- Assignment dossier: `assignment_final.pdf`

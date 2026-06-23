"""
Run all configured dataset/model experiments and save the final test metrics.
"""
import argparse
import csv
from pathlib import Path
import time

import torch

from train import get_run_list, load_config, run_training, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full benchmark matrix.")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--output", default=None, help="CSV output path")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for this run")
    parser.add_argument("--dry-run", action="store_true", help="Only print the planned runs")
    return parser.parse_args()


def write_results(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "model",
        "epochs",
        "test_loss",
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "runtime_seconds"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["EPOCHS"] = args.epochs

    runs = get_run_list(config, run_all=True)
    if args.dry_run:
        for data_name, model_name in runs:
            print(f"{data_name},{model_name}")
        return

    set_seed(config.get("SEED", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmark executing on device: {device}")

    rows = []
    for data_name, model_name in runs:
        started = time.perf_counter()
        _, metrics = run_training(data_name, model_name, config, device)
        runtime = time.perf_counter() - started

        rows.append({
            "dataset": data_name,
            "model": model_name,
            "epochs": config["EPOCHS"],
            "test_loss": round(metrics["loss"], 4),
            "accuracy": round(metrics["accuracy"], 2),
            "precision": round(metrics["precision"], 2),
            "recall": round(metrics["recall"], 2),
            "macro_f1": round(metrics["macro_f1"], 2),
            "runtime_seconds": round(runtime, 2)
        })

    output_path = Path(args.output) if args.output else Path(config["OUTPUT_DIR"]) / "benchmark_results.csv"
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent.parent / output_path

    write_results(rows, output_path)
    print(f"\nSaved benchmark results to {output_path}")


if __name__ == "__main__":
    main()

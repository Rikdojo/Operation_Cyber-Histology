"""
Run the Task 2 Green Initiative benchmark matrix.
"""
import argparse
import json
import random
from pathlib import Path

import torch

from runner import run_experiment
from utils import write_csv


def load_config(config_path=None):
    config_path = Path(config_path) if config_path else Path(__file__).resolve().parent / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv_list(value):
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def build_experiments(config, datasets=None, models=None, variants=None):
    datasets = datasets or list(config["DATASETS"].keys())
    models = models or config["MODELS"]
    variants = variants or config.get("VARIANTS", ["Baseline", "Lightweight"])

    for dataset_name in datasets:
        dataset_config = config["DATASETS"][dataset_name]
        for model_name in models:
            for variant_name in variants:
                yield {
                    **config,
                    **dataset_config,
                    "DATA": dataset_name,
                    "MODEL": model_name,
                    "VARIANT": variant_name,
                }


def output_path_for(config, output_arg):
    output_path = Path(output_arg) if output_arg else Path(config.get("OUTPUT_DIR", "results")) / "green_metrics.csv"
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent.parent / output_path
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline and lightweight green benchmarks.")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--output", default=None, help="CSV output path")
    parser.add_argument("--epochs", type=int, default=None, help="Override EPOCHS for this run")
    parser.add_argument("--datasets", default=None, help="Comma-separated datasets, e.g. cells,chest")
    parser.add_argument("--models", default=None, help="Comma-separated models, e.g. AlexNet,ResNet18")
    parser.add_argument("--variants", default=None, help="Comma-separated variants, e.g. Baseline,Lightweight")
    parser.add_argument("--dry-run", action="store_true", help="Print planned experiments without training")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["EPOCHS"] = args.epochs

    set_seed(config.get("SEED", 42))

    experiments = list(build_experiments(
        config,
        datasets=parse_csv_list(args.datasets),
        models=parse_csv_list(args.models),
        variants=parse_csv_list(args.variants),
    ))

    if args.dry_run:
        for experiment in experiments:
            print(f"{experiment['DATA']},{experiment['MODEL']},{experiment['VARIANT']},{experiment['EPOCHS']} epochs")
        return

    output_path = output_path_for(config, args.output)
    for experiment in experiments:
        print("\n" + "=" * 60)
        print(f"{experiment['VARIANT']} | {experiment['MODEL']} | {experiment['DATA']}")
        print("=" * 60)
        result = run_experiment(experiment)
        write_csv(result, output_path=output_path)

    print(f"\nSaved green benchmark metrics to {output_path}")


if __name__ == "__main__":
    main()

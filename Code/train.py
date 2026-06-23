import json
from pathlib import Path

from utils import write_csv
from runner import run_experiment


def main():
    base_dir = Path(__file__).resolve().parent

    with open(base_dir / "config.json", "r") as f:
        config = json.load(f)

    for dataset_name, dataset_info in config["DATASETS"].items():
        for model_name in config["MODELS"]:
            for variant_name in config["VARIANTS"]:
                experiment_config = {
                    **config["TRAINING"],
                    **dataset_info,
                    "DATA": dataset_name,
                    "MODEL": model_name,
                    "VARIANT": variant_name,
                }

                print("\n" + "=" * 60)
                print(f"{variant_name} | {model_name} | {dataset_name}")
                print("=" * 60)

                result = run_experiment(experiment_config)

                write_csv(
                    result,
                    base_dir / "results.csv"
                )


if __name__ == "__main__":
    main()
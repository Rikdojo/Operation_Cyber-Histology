"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from trainer import Trainer


def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.json"
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_path(config):
    data_path = Path(config["DATA_PATH"])
    if data_path.is_absolute():
        return data_path
    return Path(__file__).resolve().parent.parent / data_path


def get_run_list(config, run_all=False):
    if run_all:
        return [(data_name, model_name)
                for data_name in config["DATASETS"]
                for model_name in config["MODELS"]]
    return [(config["DATA"], config["MODEL"])]


def build_model(model_name, dataset_name, config):
    if dataset_name not in config["DATASETS"]:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if model_name not in config["MODELS"]:
        raise ValueError(f"Unknown model: {model_name}")

    dataset_config = config["DATASETS"][dataset_name]
    model_class = getattr(models, model_name)
    model = model_class(
        in_channels=dataset_config["channels"],
        num_classes=dataset_config["num_classes"],
        drop_rate=config.get("DROP_RATE", 0.5),
        activation_str=config.get("ACTIVATION", None)
    )
    return model


def prepare_experiment(data_name, model_name, config, device):
    data_path = get_data_path(config)
    train_loader, val_loader, test_loader = get_loaders(
        data=data_name,
        data_path=data_path,
        batch_size=config["BATCH_SIZE"],
        val_split=config.get("VAL_SPLIT", 0.1)
    )

    model = build_model(model_name, data_name, config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    trainer = Trainer(model, criterion, optimizer, device)

    return trainer, train_loader, val_loader, test_loader


def run_training(data_name, model_name, config, device):
    print(f"\nRunning {model_name} on {data_name}")
    trainer, train_loader, val_loader, test_loader = prepare_experiment(
        data_name, model_name, config, device
    )

    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])
    dataset_config = config["DATASETS"][data_name]
    test_metrics = trainer.evaluate_metrics(test_loader, dataset_config["num_classes"])

    print("Test Metrics | "
          f"Loss: {test_metrics['loss']:.4f} - "
          f"Accuracy: {test_metrics['accuracy']:.2f}% - "
          f"Precision: {test_metrics['precision']:.2f}% - "
          f"Recall: {test_metrics['recall']:.2f}% - "
          f"Macro F1: {test_metrics['macro_f1']:.2f}%")
    return trainer, test_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train histology classification models.")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--data", default=None, help="Dataset name from config")
    parser.add_argument("--model", default=None, help="Model name from config")
    parser.add_argument("--all", action="store_true", help="Run every dataset/model pair")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.data:
        config["DATA"] = args.data
    if args.model:
        config["MODEL"] = args.model

    set_seed(config.get("SEED", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training executing on device: {device}")

    run_all = args.all or config.get("RUN_ALL", False)
    for data_name, model_name in get_run_list(config, run_all=run_all):
        run_training(data_name, model_name, config, device)

if __name__ == "__main__":
    main()

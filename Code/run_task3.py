"""
Task 3: scarce organs data with simple transfer learning.

The idea is:
1. learn useful low-level features from chest,
2. reuse them for the smaller organs dataset,
3. compare scratch, frozen transfer, and full fine-tuning.
"""
import argparse
import json
import random
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim

import models
from data import get_loaders
from inference import run_inference
from trainer import Trainer
from utils import plot_losses, write_csv


def load_config(config_path=None):
    config_path = Path(config_path) if config_path else Path(__file__).resolve().parent / "config.json"
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


def project_path(config, key):
    path = Path(config[key])
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent.parent / path


def build_model(config, dataset_name, model_name, device):
    dataset_config = config["DATASETS"][dataset_name]
    model_class = getattr(models, model_name)
    return model_class(
        in_channels=dataset_config["channels"],
        num_classes=dataset_config["num_classes"],
        drop_rate=config.get("DROP_RATE", 0.5),
        activation_str=config.get("ACTIVATION", "ReLU"),
    ).to(device)


def replace_classifier(model, num_classes, device):
    if isinstance(model.classifier, nn.Sequential):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes).to(device)
    else:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes).to(device)


def set_trainable_parts(model, mode):
    for parameter in model.parameters():
        parameter.requires_grad = mode != "frozen"

    if mode == "frozen":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True


def train_and_test(model, config, dataset_name, epochs, device, run_name, save_best_path):
    train_loader, val_loader, test_loader = get_loaders(
        data=dataset_name,
        data_path=get_data_path(config),
        batch_size=config["BATCH_SIZE"],
        val_split=config.get("VAL_SPLIT", 0.1),
        seed=config.get("SEED", 42),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["LEARNING_RATE"],
    )
    trainer = Trainer(model, criterion, optimizer, device)

    started = time.perf_counter()
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        patience=config.get("PATIENCE"),
        save_best_path=save_best_path,
    )
    training_time = time.perf_counter() - started

    history_path = project_path(config, "HISTORY_DIR") / f"{run_name}_loss.png"
    plot_losses(
        {run_name: (history["train_loss"], history["val_loss"])},
        title=run_name,
        output_path=history_path,
    )

    metrics = run_inference(trainer.model, test_loader, device)
    return metrics, training_time


def source_checkpoint_path(config):
    path = Path(config["TASK3"]["CHECKPOINT"])
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent.parent / path


def prepare_source_model(config, device, force_train=False):
    task_config = config["TASK3"]
    checkpoint = source_checkpoint_path(config)

    if checkpoint.exists() and not force_train:
        print(f"Using existing source checkpoint: {checkpoint}")
        return checkpoint

    print(f"\nTraining source model on {task_config['SOURCE_DATA']}")
    model = build_model(config, task_config["SOURCE_DATA"], task_config["MODEL"], device)
    train_and_test(
        model,
        config,
        task_config["SOURCE_DATA"],
        task_config["SOURCE_EPOCHS"],
        device,
        run_name=f"task3_source_{task_config['SOURCE_DATA']}",
        save_best_path=checkpoint,
    )
    return checkpoint


def build_target_model(config, mode, checkpoint, device):
    task_config = config["TASK3"]
    target_data = task_config["TARGET_DATA"]
    model_name = task_config["MODEL"]

    if mode == "scratch":
        return build_model(config, target_data, model_name, device)

    model = build_model(config, task_config["SOURCE_DATA"], model_name, device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    replace_classifier(model, config["DATASETS"][target_data]["num_classes"], device)
    set_trainable_parts(model, mode)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Run Task 3 organs transfer benchmark.")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--epochs", type=int, default=None, help="Override target epochs")
    parser.add_argument("--source-epochs", type=int, default=None, help="Override chest source epochs")
    parser.add_argument("--force-source", action="store_true", help="Retrain the chest source model")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["TASK3"]["TARGET_EPOCHS"] = args.epochs
    if args.source_epochs is not None:
        config["TASK3"]["SOURCE_EPOCHS"] = args.source_epochs

    set_seed(config.get("SEED", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Task 3 executing on device: {device}")

    task_config = config["TASK3"]
    checkpoint = prepare_source_model(config, device, force_train=args.force_source)
    rows = []

    for mode in task_config["MODES"]:
        print("\n" + "=" * 60)
        print(f"Task 3 mode: {mode}")
        print("=" * 60)

        model = build_target_model(config, mode, checkpoint, device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        metrics, training_time = train_and_test(
            model,
            config,
            task_config["TARGET_DATA"],
            task_config["TARGET_EPOCHS"],
            device,
            run_name=f"task3_{mode}_{task_config['TARGET_DATA']}",
            save_best_path=project_path(config, "CHECKPOINT_DIR") / f"task3_{mode}_{task_config['TARGET_DATA']}.pt",
        )

        rows.append({
            "dataset": task_config["TARGET_DATA"],
            "model": task_config["MODEL"],
            "mode": mode,
            "epochs": task_config["TARGET_EPOCHS"],
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "training_time_seconds": training_time,
            "accuracy": metrics["accuracy"] * 100,
            "precision": metrics["precision"] * 100,
            "recall": metrics["recall"] * 100,
            "macro_f1": metrics["macro_f1"] * 100,
            "inference_time_seconds": metrics["inference_time"],
            "latency_per_sample_seconds": metrics["latency_per_sample"],
        })

    output_path = Path(config.get("OUTPUT_DIR", "results")) / "task3_metrics.csv"
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent.parent / output_path
    write_csv(rows, output_path=output_path)
    print(f"\nSaved Task 3 metrics to {output_path}")


if __name__ == "__main__":
    main()

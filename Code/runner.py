import torch
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from data import get_loaders
import models
from trainer import Trainer
import time
from inference import run_inference
from utils import plot_losses


def resolve_model_name(model_name, variant):
    if variant == "Lightweight":
        return f"Light{model_name}"
    if variant == "Baseline":
        return model_name
    raise ValueError(f"Unknown model variant: {variant}")


def project_path(config, key):
    path = Path(config[key])
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent.parent / path


def run_experiment(experiment_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training executing on device: {device}")
    peak_train_memory = None

    data_path = Path(experiment_config["DATA_PATH"])
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent.parent / data_path

    train_loader, val_loader, test_loader = get_loaders(
        data=experiment_config["DATA"],
        data_path=data_path,
        batch_size=experiment_config["BATCH_SIZE"],
        val_split=experiment_config.get("VAL_SPLIT", 0.1),
        seed=experiment_config.get("SEED", 42),
    )

    class_name = resolve_model_name(experiment_config["MODEL"], experiment_config["VARIANT"])
    model_class = getattr(models, class_name)
    model = model_class(
        in_channels=experiment_config["channels"],
        num_classes=experiment_config["num_classes"],
        drop_rate=experiment_config.get("DROP_RATE", 0.5),
        activation_str=experiment_config["ACTIVATION"],
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=experiment_config["LEARNING_RATE"])
    trainer = Trainer(model, criterion, optimizer, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    checkpoint_path = project_path(experiment_config, "CHECKPOINT_DIR") / f"{class_name}_{experiment_config['DATA']}.pt"
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=experiment_config["EPOCHS"],
        patience=experiment_config.get("PATIENCE"),
        save_best_path=checkpoint_path,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_train_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Peak training memory: {peak_train_memory:.2f} MB")
    training_time = time.perf_counter() - start_time
    print(f"Total training time: {training_time:.2f} seconds")

    history_path = project_path(experiment_config, "HISTORY_DIR") / f"{class_name}_{experiment_config['DATA']}_loss.png"
    plot_losses(
        {f"{class_name} {experiment_config['DATA']}": (history["train_loss"], history["val_loss"])},
        title=f"{class_name} on {experiment_config['DATA']}",
        output_path=history_path,
    )

    inference_metrics = run_inference(trainer.model, test_loader, device)

    return {
        "dataset": experiment_config["DATA"],
        "model": experiment_config["MODEL"],
        "variant": experiment_config["VARIANT"],
        "model_class": class_name,
        "epochs": experiment_config["EPOCHS"],
        "parameters": num_params,
        "training_runtime_seconds": training_time,
        "peak_train_memory_mb": peak_train_memory,
        "accuracy": inference_metrics["accuracy"] * 100,
        "precision": inference_metrics["precision"] * 100,
        "recall": inference_metrics["recall"] * 100,
        "macro_f1": inference_metrics["macro_f1"] * 100,
        "inference_time_seconds": inference_metrics["inference_time"],
        "latency_per_sample_seconds": inference_metrics["latency_per_sample"],
        "peak_inference_memory_mb": inference_metrics["peak_inference_memory"],
    }

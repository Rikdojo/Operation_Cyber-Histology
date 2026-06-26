
import torch
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from data import get_loaders
import models
from trainer import Trainer
import time
from inference import run_inference

def run_experiment(experiment_config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training executing on device: {device}")

    data_path = Path(experiment_config["DATA_PATH"])
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent.parent / data_path

    train_loader, val_loader, test_loader = get_loaders(
        data=experiment_config["DATA"],
        data_path=data_path,
        batch_size=experiment_config["BATCH_SIZE"],
        val_split=experiment_config.get("VAL_SPLIT", 0.1),
    )

    if experiment_config["VARIANT"] == "Lightweight":
        class_name = f"Light{experiment_config['MODEL']}"
    elif experiment_config["VARIANT"] == "Baseline":
        class_name = experiment_config['MODEL']

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
    trainer.fit(train_loader, val_loader, epochs=experiment_config["EPOCHS"])
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_train_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Peak training memory: {peak_train_memory:.2f} MB")
    training_time = time.perf_counter() - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    inference_metrics = run_inference(trainer.model, test_loader, device)

    performance_list = {
    "Model": (f'{experiment_config["VARIANT"]}_'f'{experiment_config["MODEL"]}'),
    "dataset": experiment_config["DATA"],
    "parameters": num_params,
    "training_time": training_time,
    "peak_train_memory": peak_train_memory if device.type == "cuda" else None,
    "accuracy": inference_metrics["accuracy"],
    "precision": inference_metrics["precision"],
    "recall": inference_metrics["recall"],
    "macro_f1": inference_metrics["macro_f1"],
    "inference_time": inference_metrics["inference_time"],
    "latency_per_sample": inference_metrics["latency_per_sample"],
    "peak_inference_memory": inference_metrics["peak_inference_memory"],
}

    return performance_list

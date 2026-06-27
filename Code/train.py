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
from inference import run_inference
from trainer import Trainer
import models
from utils import write_csv
import time 



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
    return Path(__file__).resolve().parent / data_path


def get_run_list(config, run_all=False):
    if run_all:
        return [(data_name, model_name)
                for data_name in config["DATASETS"]
                for model_name in config["MODELS"]]
    return [(config["DATA"], config["MODEL"])]


def parse_args():
    parser = argparse.ArgumentParser(description="Train histology classification models.")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--data", default=None, help="Dataset name from config")
    parser.add_argument("--model", default=None, help="Model name from config")
    parser.add_argument("--all", action="store_true", help="Run every dataset/model pair")
    return parser.parse_args()

def start_gpu_memory_measurement(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()


def get_peak_gpu_memory_mb(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return None


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
    rows = []

    for data_name, model_name in get_run_list(config, run_all=run_all):
        print(f"\nRunning {model_name} on {data_name}")
        
        train_loader, val_loader, test_loader = get_loaders(data_name, data_path=get_data_path(config), batch_size=config["BATCH_SIZE"], val_split=config.get("VAL_SPLIT"))
        model_class = getattr(models, model_name)
        model = model_class(in_channels=config["DATASETS"][data_name]["channels"],num_classes=config["DATASETS"][data_name]["num_classes"],drop_rate=config.get("DROP_RATE", 0.5), activation_str=config.get("ACTIVATION", None)).to(device)
        
        num_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        print(f"Trainable parameters: {num_params:,}")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
        trainer = Trainer(model, criterion, optimizer, device)
        

        start_gpu_memory_measurement(device)
        start_time = time.perf_counter()
        trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])        
        peak_train_memory = get_peak_gpu_memory_mb(device)
           
        training_time = time.perf_counter() - start_time
        print(f"Total training time: {training_time:.2f} seconds")
        if peak_train_memory is not None:
            print(f"Peak training memory: {peak_train_memory:.2f} MB")

        start_gpu_memory_measurement(device)
        inference_start_time = time.perf_counter()

        test_metrics = run_inference(model, test_loader, device)

        peak_inference_memory = get_peak_gpu_memory_mb(device)
        inference_time = time.perf_counter() - inference_start_time

        rows.append({
            "dataset": data_name,
            "model": model_name,
            "epochs": config["EPOCHS"],
            "trainable_parameters": num_params,
            "training_time": training_time,
            "peak_training_memory_mb": peak_train_memory,
            "accuracy": test_metrics["accuracy"] * 100,
            "precision": test_metrics["precision"] * 100,
            "recall": test_metrics["recall"] * 100,
            "macro_f1": test_metrics["macro_f1"] * 100,
            "inference_time": inference_time,
            "peak_inference_memory_mb": peak_inference_memory,
            "inference_latency_per_sample": inference_time / len(test_loader.dataset),
        })

    output_dir = Path(config.get("OUTPUT_DIR", "results"))
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent.parent / output_dir
    write_csv(rows, output_path=output_dir / "test_metrics.csv")


if __name__ == "__main__":
    main()


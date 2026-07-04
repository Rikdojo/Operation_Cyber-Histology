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
    return Path(__file__).resolve().parent/ data_path


def get_run_list(config, task, run_all=False):
    if run_all:
        if task == "task1":
            return [(data_name, model_name, None)
                    for data_name in config["DATASETS"]
                    for model_name in config["MODELS"]]
        elif task == "task2":
            return [(data_name, model_name, mode)
                    for data_name in config["DATASETS"]
                    for model_name in config["task2"]["MODELS"]
                    for mode in config["task2"]["MODES"]]
    return [
        (config["DATA"], model_name, None)
        for model_name in config["MODELS"]
    ]

def parse_args():
    parser = argparse.ArgumentParser(description="Train histology classification models.")
    parser.add_argument("--task",choices=["task1", "task2", "task3"],default="task1",)
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
    task = args.task
    rows = []
    experiment_list = get_run_list(config,task, run_all=run_all)

    for data_name, model_name,mode in experiment_list:
        
        train_loader, val_loader, test_loader = get_loaders(data_name, data_path=get_data_path(config), batch_size=config["BATCH_SIZE"], val_split=config.get("VAL_SPLIT"), seed=config.get("SEED", 42))
        
        if task == "task1":
            model_class = getattr(models, model_name)
            model = model_class(in_channels=config["DATASETS"][data_name]["channels"],num_classes=config["DATASETS"][data_name]["num_classes"],drop_rate=config.get("DROP_RATE", 0.5), activation_str=config.get("ACTIVATION", None)).to(device)
            
        if task == "task2":
            if mode == "Light":
                model_name = f"{mode}_{model_name}"
            model_class = getattr(models, model_name)
            model = model_class(in_channels=config["DATASETS"][data_name]["channels"],num_classes=config["DATASETS"][data_name]["num_classes"],drop_rate=config.get("DROP_RATE", 0.5), activation_str=config.get("ACTIVATION", None), mode=mode).to(device)
            
        print(f"\nRunning {model_name} on {data_name}")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
        trainer = Trainer(model, criterion, optimizer, device)
        trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])

        test_metrics = run_inference(model, test_loader, device)

        print( 
        "Test Metrics | "
        f"Accuracy: {test_metrics['accuracy'] * 100:.2f}% - "
        f"Precision: {test_metrics['precision'] * 100:.2f}% - "
        f"Recall: {test_metrics['recall'] * 100:.2f}% - "
        f"Macro F1: {test_metrics['macro_f1'] * 100:.2f}%")


        rows.append({"dataset": data_name, 
            "model": model_name, 
            "epochs": config["EPOCHS"],
            "accuracy": test_metrics["accuracy"] * 100,
            "precision": test_metrics["precision"] * 100,
            "recall": test_metrics["recall"] * 100,
            "macro_f1": test_metrics["macro_f1"] * 100,
        })
    output_dir = Path(config.get("OUTPUT_DIR", "results"))
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent.parent / output_dir
    write_csv(rows,output_path=output_dir / f"{task}_test_metrics.csv")


if __name__ == "__main__":
    main()


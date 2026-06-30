
import torch
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from data import get_loaders
from trainer import Trainer
import time
from evaluate import evaluate_model


def start_gpu_memory_measurement(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

def get_peak_gpu_memory_mb(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return None

def get_data_path(config):
    data_path = Path(config["DATA_PATH"])
    if data_path.is_absolute():
        return data_path
    return Path(__file__).resolve().parent / data_path

def run_experiment(model,config, model_name,data_name, device):

    train_loader, val_loader, test_loader = get_loaders(data_name, data_path=get_data_path(config), batch_size=config["BATCH_SIZE"], val_split=config.get("VAL_SPLIT"), seed=config.get("SEED", 42)) 
        
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

    accuracy, precision, recall, macro_f1 = evaluate_model(model, test_loader, device)

    peak_inference_memory = get_peak_gpu_memory_mb(device)
    inference_time = time.perf_counter() - inference_start_time
    print(f"Total inference time: {inference_time:.2f} seconds")

    checkpoint_dir = Path(config["OUTPUT_DIR"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
   
    
    checkpoint_path = checkpoint_dir / f"{model_name}_{data_name}.pt"
    
    torch.save(model.state_dict(), checkpoint_path)

    test_metrics = {
        "num_params": num_params,
        "training_time": training_time,
        "peak_train_memory": peak_train_memory,
        "inference_time": inference_time,
        "inference_latency_per_sample": inference_time/len(test_loader.dataset),
        "peak_inference_memory": peak_inference_memory, 
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1 
    }


    return test_metrics

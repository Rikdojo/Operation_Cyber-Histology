"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer

def main():   
    # Load config from the same folder as train.py, so it works when running python Code/train.py.
    config_path = Path(__file__).resolve().parent / "config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    # Use GPU if it is available; otherwise use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training executing on device: {device}")

    # Make the data path stable, instead of depending on the terminal folder.
    data_path = Path(config["DATA_PATH"])
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent.parent / data_path

    # Load training and validation data using values from the config file.
    train_loader, val_loader, _ = get_loaders(data=config["DATA"], data_path=data_path, batch_size=config["BATCH_SIZE"])

    # Select the model class from models.py using the model name in config.json.
    model_class = getattr(models, config["MODEL"])

    # Pass channel count and class count from config, so models work for different datasets.
    # Use a normal default dropout instead of hardcoding 0.99, which is too high for training.
    model = model_class(
        in_channels=config["CHANNELS"],
        num_classes=config["NUM_CLASSES"],
        drop_rate=config.get("DROP_RATE", 0.5),
        activation_str=config.get("ACTIVATION", None)
    ).to(device)

    # CrossEntropyLoss is used for multi-class classification.
    criterion = nn.CrossEntropyLoss()

    # Adam updates model weights using the learning rate from config.json.
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    # Trainer contains the training and validation loops.
    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])

if __name__ == "__main__":
    main()

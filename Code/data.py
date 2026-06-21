"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(data, data_path, batch_size, val_split=0.1):
    d_path = Path(data_path) / f"{data}.pt"
    if not d_path.exists():
        d_path = Path(data_path) / f"{data}_data.pt"

    data_dict = torch.load(d_path)

    total_samples = data_dict['train_images'].shape[0]
    val_size = int(total_samples * val_split)
    val_start = total_samples - val_size

    # Keep validation samples out of the training set.
    train_data = data_dict['train_images'][:val_start]
    train_labels = data_dict['train_labels'][:val_start]
    val_data = data_dict['train_images'][val_start:]
    val_labels = data_dict['train_labels'][val_start:]

    # Normalize all splits with training statistics only.
    mean = train_data.mean()
    std = train_data.std()
    train_data = (train_data - mean) / std
    val_data = (val_data - mean) / std
    test_data = (data_dict['test_images'] - mean) / std
    
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, data_dict['test_labels'])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

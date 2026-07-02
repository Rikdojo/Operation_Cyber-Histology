"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def normalize_with_train_stats(train_data, val_data, test_data):
    mean = train_data.mean(dim=(0, 2, 3), keepdim=True)
    std = train_data.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    return (
        (train_data - mean) / std,
        (val_data - mean) / std,
        (test_data - mean) / std,
    )


def get_loaders(data, data_path, batch_size, val_split=0.1, seed=42):

    d_path = Path(data_path) / f"{data}.pt"
    data_dict = torch.load(d_path)

    total_samples = data_dict['train_images'].shape[0]
    val_size = max(1, int(total_samples * val_split))
    val_size = min(val_size, total_samples - 1)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_samples, generator=generator)

    train_idx = indices[:-val_size]
    val_idx = indices[-val_size:]

    train_data = data_dict['train_images'][train_idx]
    train_labels = data_dict['train_labels'][train_idx]
    val_data = data_dict['train_images'][val_idx]
    val_labels = data_dict['train_labels'][val_idx]

    train_data, val_data, test_data = normalize_with_train_stats(
        train_data,
        val_data,
        data_dict['test_images'],
    )
    
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, data_dict['test_labels']) 

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(data, data_path, batch_size, val_split=0.1):

    d_path = Path(data_path) / f"{data}.pt"
    data_dict = torch.load(d_path)

    total_samples = data_dict['train_images'].shape[0]
    val_size = int(total_samples * val_split)
    val_start = total_samples - val_size

    train_data = data_dict['train_images'][:val_start] # add this line to exclude validation data in training data 
    train_labels = data_dict['train_labels'][:val_start]
    val_data = data_dict['train_images'][val_start:]
    val_labels = data_dict['train_labels'][val_start:]

    mean = train_data.mean() # normalize the image data with z-score normalization using training statistics for all splits to avoid validation/test leakage.
    std = train_data.std()
    train_data = (train_data - mean) / std
    val_data = (val_data - mean) / std
    test_data = (data_dict['test_images'] - mean) / std # normalize test data with the same mean and std as the training data.
    
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, data_dict['test_labels']) 

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

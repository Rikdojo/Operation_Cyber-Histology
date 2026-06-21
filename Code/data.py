"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(data, data_path, batch_size, val_split=0.1):
    
    #I made the loader support both possible dataset file names so the training script does not fail because of a small naming difference.
    d_path = Path(data_path) / f"{data}.pt"
    if not d_path.exists():
        d_path = Path(data_path) / f"{data}_data.pt"

    data_dict = torch.load(d_path)

    total_samples = data_dict['train_images'].shape[0]
    val_size = int(total_samples * val_split)
    val_start = total_samples - val_size

    # Keep validation samples out of the training set.
    #I fixed the split so validation samples are not included in training. This makes validation accuracy more reliable.
    train_data = data_dict['train_images'][:val_start]
    train_labels = data_dict['train_labels'][:val_start]
    val_data = data_dict['train_images'][val_start:]
    val_labels = data_dict['train_labels'][val_start:]

    #I normalized train, validation, and test images using only training statistics to avoid data leakage.
    #This ensures that the model does not gain information about the validation and test sets during training, leading to more reliable evaluation results.

    mean = train_data.mean()
    std = train_data.std()
    train_data = (train_data - mean) / std
    val_data = (val_data - mean) / std
    test_data = (data_dict['test_images'] - mean) / std
    
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    #After adding normalization, the test dataset should use the normalized test_data.
# If training data is normalized but test data is raw, evaluation is inconsistent.
    test_dataset = TensorDataset(test_data, data_dict['test_labels'])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

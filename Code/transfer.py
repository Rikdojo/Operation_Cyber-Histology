from pathlib import Path
import torch
import torch.nn as nn
import models


def build_pretrained_model(config, mode, device):
    model_name = config["task3"]["MODEL"]
   
    model_class = getattr(models, model_name)
  
    if mode == "scratch":
        return model_class(
            in_channels=config["task3"]["channels"],
            num_classes=config["task3"]["num_classes"],
        ).to(device)
    
    # pretrained / finetune
    model = model_class(
        in_channels=config["task3"]["FROM_DATA"]["channels"],
        num_classes=config["task3"]["FROM_DATA"]["num_classes"],
    ).to(device)

    checkpoint = torch.load(config["task3"]["CHECKPOINT"], map_location=device)
    model.load_state_dict(checkpoint)

    # freeze parameters 

    #adjust classifier for new number of classes
    if isinstance(model.classifier, nn.Sequential):# VGG / AlexNet
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, config["task3"]["num_classes"]).to(device)
    else:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, config["task3"]["num_classes"]).to(device)

    # freeze parameters 
    for p in model.parameters():
        p.requires_grad = False

    #unfreeze classifier parameters
    for p in model.classifier.parameters():
        p.requires_grad = True

    if mode == "finetune":
        if "ResNet" in model_name:
            for p in model.stage4.parameters():
                p.requires_grad = True
        else:
            for p in model.features[-1].parameters():
                p.requires_grad = True
    return model
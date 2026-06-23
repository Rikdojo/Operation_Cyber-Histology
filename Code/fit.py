"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
from networkx import config
import torch
import copy
from utils import plot_losses

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct, sum = 0, 0
        
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device).squeeze(1).long() # Convert labels from shape [batch_size, 1] to [batch_size],because CrossEntropyLoss expects class indices as a 1D tensor.
            self.optimizer.zero_grad() # add zero the parameter gradients to prevent accumulation of gradients across batches.
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            sum += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return running_loss / sum, (correct / sum) * 100

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct, total = 0, 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device).squeeze(1).long() # Convert labels from shape [batch_size, 1] to [batch_size],because CrossEntropyLoss expects class indices as a 1D tensor.               
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return running_loss / total, (correct / total) * 100

    def fit(self, train_loader, val_loader, config, patience=20):
        print("\n Starting Training Routine...")
        print("-" * 50)
        best_val = float("inf")
        best_state = None
        best_epoch = -1
        current_patience = 0
        train_losses, val_losses =  [], []
        epochs = config["EPOCHS"]
        data_type = config["DATA"]
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        
            if val_loss < best_val: # to save the best model on validation data. 
                best_val = val_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.model.state_dict())
                current_patience = 0
            else:
                current_patience += 1

            if current_patience >= patience: # early stopping
                print("early stopping")
                break
        histories = { "Loss": (train_losses, val_losses)}
        model_name = self.model.__class__.__name__
        plot_losses(histories,title=(f'{config["VARIANT"]}_'f'{config["MODEL"]}_'f'{config["DATA"]} training history'))
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print("Best model loaded with validation loss: {:.4f}".format(best_val))

        torch.save(self.model.state_dict(),f'Model/{config["VARIANT"]}_'f'{config["MODEL"]}_'f'{config["DATA"]}_model.pt')
        print("-" * 50)
        print(f"Best validation loss: {best_val:.4f} at epoch {best_epoch}")
        print("Training Complete!")

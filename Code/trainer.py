"""
MAI/IDL SS26 - Final assignment.

MG 6/6/2026
"""
from copy import deepcopy
from pathlib import Path

import torch


class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device).view(-1).long()

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return running_loss / total, (correct / total) * 100

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device).view(-1).long()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return running_loss / total, (correct / total) * 100

    def fit(self, train_loader, val_loader, epochs, patience=None, save_best_path=None):
        print("\n Starting Training Routine...")
        print("-" * 50)
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(f"Epoch [{epoch + 1:02d}/{epochs:02d}] | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience and epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if save_best_path is not None and best_state is not None:
            save_best_path = Path(save_best_path)
            save_best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, save_best_path)
            print(f"Best model saved to {save_best_path}")

        print("-" * 50)
        print("Training Complete!")
        return history

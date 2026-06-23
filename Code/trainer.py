"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch


def macro_classification_scores(labels, predictions, num_classes):
    labels = labels.view(-1).cpu()
    predictions = predictions.view(-1).cpu()
    accuracy = predictions.eq(labels).float().mean().item() * 100

    precisions, recalls, f1_scores = [], [], []
    for class_id in range(num_classes):
        predicted_class = predictions.eq(class_id)
        true_class = labels.eq(class_id)

        true_positive = (predicted_class & true_class).sum().item()
        false_positive = (predicted_class & ~true_class).sum().item()
        false_negative = (~predicted_class & true_class).sum().item()

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": accuracy,
        "precision": sum(precisions) / num_classes * 100,
        "recall": sum(recalls) / num_classes * 100,
        "macro_f1": sum(f1_scores) / num_classes * 100
    }


class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        # I renamed sum to total to avoid shadowing a Python built-in and make the accuracy calculation easier to read.
        # Now total= sum
        correct, total = 0, 0
        
        for images, labels in dataloader:
 # reshaped labels into a one-dimensional integer tensor because that is what CrossEntropyLoss expects.

            images = images.to(self.device)
            labels = labels.to(self.device).view(-1).long()
            
            # Clear old gradients before calculating this batch.
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
             # Validation uses the same loss function as training.
            # otherwise validation can fail even if training works.
                images = images.to(self.device)
                labels = labels.to(self.device).view(-1).long()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return running_loss / total, (correct / total) * 100

    def evaluate_metrics(self, dataloader, num_classes):
        self.model.eval()
        running_loss = 0.0
        total = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device).view(-1).long()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                predictions = outputs.argmax(dim=1)

                running_loss += loss.item() * images.size(0)
                total += labels.size(0)
                all_labels.append(labels.cpu())
                all_predictions.append(predictions.cpu())

        labels = torch.cat(all_labels)
        predictions = torch.cat(all_predictions)
        scores = macro_classification_scores(labels, predictions, num_classes)
        scores["loss"] = running_loss / total
        return scores

    def fit(self, train_loader, val_loader, epochs):
        print("\n Starting Training Routine...")
        print("-" * 50)
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
        print("-" * 50)
        print("Training Complete!")

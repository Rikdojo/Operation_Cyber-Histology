import torch
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def run_inference(model, test_loader, device):
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    label_list = []
    prediction_list = []
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).squeeze(1).long()

            outputs = model(images)
            _, predicted = outputs.max(1)

            total_samples += labels.size(0)
            label_list.extend(labels.cpu().tolist())
            prediction_list.extend(predicted.cpu().tolist())

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_inference_memory = (torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        print(f"Peak inference memory: {peak_inference_memory:.2f} MB")

    inference_time = time.perf_counter() - start_time
    latency_per_sample = inference_time / total_samples

    accuracy = accuracy_score(label_list, prediction_list)
    precision = precision_score(label_list, prediction_list, average="macro", zero_division=0)
    recall = recall_score(label_list, prediction_list, average="macro", zero_division=0)
    macro_f1 = f1_score(label_list, prediction_list, average="macro", zero_division=0)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print(f"Total inference time: {inference_time:.2f} seconds")
    print(f"Inference latency: {latency_per_sample * 1000:.4f} ms/sample")

    performance_list = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,      
        "inference_time": inference_time,     
        "latency_per_sample": latency_per_sample,
        "peak_inference_memory": peak_inference_memory if device.type == "cuda" else None   
    }   
    return performance_list


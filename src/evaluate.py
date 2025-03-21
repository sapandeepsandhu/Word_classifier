import torch

def evaluate_model(model, dataloader, device, classes, dataset_name="Validation"):
    """
    Evaluates the model on the given dataloader, printing overall and per-class accuracy.
    
    Args:
        model: The PyTorch model to evaluate.
        dataloader: DataLoader for the dataset (validation or test).
        device: The device (CPU, CUDA, or MPS) used for evaluation.
        classes: List of class names.
        dataset_name (str): A name for the dataset (e.g., "Validation" or "Test").
    """
    model.eval()
    correct = 0
    total = 0
    # For per-class accuracy
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for label, pred in zip(labels, predicted):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
    
    overall_acc = 100 * correct / total
    print(f"\n{dataset_name} Accuracy: {overall_acc:.2f}%")
    for i, cls in enumerate(classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"  {cls}: {acc:.2f}%")
        else:
            print(f"  {cls}: No samples in {dataset_name} set")
    
    model.train()

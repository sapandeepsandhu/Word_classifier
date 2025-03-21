import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_metrics(model, dataloader, device, classes):
    """
    Evaluates the model on the given dataloader and prints overall accuracy,
    precision, recall, F1 score, and the confusion matrix.
    
    Args:
        model: The trained PyTorch model.
        dataloader: DataLoader for the dataset (validation or test).
        device: The device on which the model runs (CPU, CUDA, or MPS).
        classes: List of class names.
    """
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("Final Metrics:")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    
    model.train()

if __name__ == "__main__":
    # Import the data pipeline and model using relative imports
    from .data import get_dataloaders
    from .model import WordCNN
    
    # Set the dataset directory (relative to the project root)
    data_dir = "dataset"  
    batch_size = 32
    # Obtain train and validation loaders; for demonstration, we use the validation set as our test set.
    train_loader, val_loader, classes = get_dataloaders(data_dir, batch_size=batch_size, train_split=0.8)
    test_loader = val_loader  # For demonstration purposes
    
    # Set the device
    device = torch.device("mps" if torch.backends.mps.is_available() 
                          else "cuda" if torch.cuda.is_available() 
                          else "cpu")
    
    # Initialize the model and load saved weights
    num_classes = len(classes)
    model = WordCNN(num_classes=num_classes).to(device)
    model_path = "outputs/word_classifier.pth"  # Updated relative path from project root
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Compute and print final metrics on the test set
    compute_metrics(model, test_loader, device, classes)

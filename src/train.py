import os
import torch
import torch.nn as nn
import torch.optim as optim
from .data import get_dataloaders
from .model import WordCNN
from .evaluate import evaluate_model

def print_model_stats(model):
    """Print total number of parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")

def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001, image_size=(64, 64), train_split=0.8):
    # Load the data
    train_loader, val_loader, classes = get_dataloaders(data_dir, image_size, batch_size, train_split)
    num_classes = len(classes)
    
    # Set up the device
    device = torch.device("mps" if torch.backends.mps.is_available() 
                          else "cuda" if torch.cuda.is_available() 
                          else "cpu")
    print(f"Training on device: {device}")
    
    # Initialize the model, loss function, and optimizer
    model = WordCNN(num_classes=num_classes).to(device)
    print_model_stats(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            batch_count += 1
            
            # Print details for each batch
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"\nEpoch [{epoch+1}/{num_epochs}] completed. Average Loss: {epoch_loss:.4f}")
        
        # Evaluate on validation set at the end of each epoch
        evaluate_model(model, val_loader, device, classes, dataset_name="Validation")
    
    # Save the trained model
    os.makedirs("outputs", exist_ok=True)
    model_path = "outputs/word_classifier.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    data_dir = "/Users/sapandeepsinghsandhu/Desktop/Word_classifier/dataset"  # Adjust path as needed
    train_model(data_dir)

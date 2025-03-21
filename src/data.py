import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Optional: Set an interactive backend if needed.
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on your system

def get_dataloaders(data_dir, image_size=(64, 64), batch_size=32, train_split=0.8):
    """
    Loads the dataset from data_dir using ImageFolder,
    applies transforms, splits the dataset into training and validation sets,
    prints relevant statistics, and returns DataLoaders along with class names.
    
    Args:
        data_dir (str): Path to the dataset directory.
        image_size (tuple): Target size to resize images (width, height).
        batch_size (int): Batch size for the DataLoaders.
        train_split (float): Proportion of dataset to use for training.
    
    Returns:
        train_loader, val_loader, classes: DataLoaders for train and validation sets, and list of class names.
    """
    # Define transformations: resizing, random rotation, horizontal flip, and conversion to tensor.
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomRotation(10),         # Rotate ±10° degrees
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally
        transforms.ToTensor(),
        # Optionally, add normalization:
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load dataset using ImageFolder
    dataset = ImageFolder(root=data_dir, transform=transform)
    classes = dataset.classes
    total_size = len(dataset)
    print("Data Loading:")
    print(f"  - Data directory: {data_dir}")
    print(f"  - Total images: {total_size}")
    print(f"  - Classes: {classes}")

    # Split the dataset into train and validation sets.
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print("Data Splitting:")
    print(f"  - Training size: {train_size}")
    print(f"  - Validation size: {val_size}")

    # Create DataLoaders for training and validation sets.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("DataLoaders:")
    print(f"  - Batch size: {batch_size}")
    
    return train_loader, val_loader, classes

def show_batch(batch, classes):
    """
    Displays a preview of a batch.
    
    Args:
        batch: Tuple (images, labels) from DataLoader.
        classes: List of class names.
    """
    images, labels = batch
    print(f"Batch shape: {images.shape}")
    print("Labels:", labels)
    
    # Plot a grid of 9 images from the batch.
    plt.figure(figsize=(8, 8))
    for i in range(min(9, images.shape[0])):
        plt.subplot(3, 3, i+1)
        # Convert tensor to numpy array and transpose dimensions for visualization
        npimg = images[i].cpu().numpy().transpose((1, 2, 0))
        plt.imshow(npimg)
        plt.title(classes[labels[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # For quick debugging, run this module directly.
    data_dir = "../dataset"  # Adjust path as needed
    train_loader, val_loader, classes = get_dataloaders(data_dir)
    
    # Get one batch from the train_loader and show it.
    batch = next(iter(train_loader))
    print("\nOne batch preview:")
    show_batch(batch, classes)

import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from .model import WordCNN

def load_model(model_path, num_classes=2, device=None):
    """
    Loads the trained model from the given path.
    
    Args:
        model_path (str): Path to the saved model (.pth file).
        num_classes (int): Number of classes (default: 2).
        device: Torch device (CPU, CUDA, or MPS). If None, auto-detect.
    
    Returns:
        model: Loaded and evaluated model.
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() 
                              else "cuda" if torch.cuda.is_available() 
                              else "cpu")
    model = WordCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def infer_image(image_path, model, classes, image_size=(64, 64)):
    """
    Processes a single image and returns the predicted class along with the image.
    
    Args:
        image_path (str): Path to the input image.
        model: The trained PyTorch model.
        classes (list): List of class names.
        image_size (tuple): Size to which the image will be resized.
    
    Returns:
        predicted_class (str): The predicted class label.
        image (PIL.Image): The original image loaded for visualization.
    """
    # Define the same transformation as during training
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # Optionally, add normalization if it was used during training
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # add batch dimension
    
    # Ensure the input tensor is on the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    predicted_class = classes[predicted.item()]
    return predicted_class, image

if __name__ == "__main__":
    # Set paths (adjust these as necessary)
    model_path = "outputs/word_classifier.pth"
    image_path = "/Users/sapandeepsinghsandhu/Desktop/Word_classifier/dataset/english/eng__1014.jpeg"  # replace with an actual image file path
    
    # Define class names. These should match your training dataset order.
    classes = ["english", "punjabi"]
    
    # Set device and load the model
    device = torch.device("mps" if torch.backends.mps.is_available() 
                          else "cuda" if torch.cuda.is_available() 
                          else "cpu")
    model = load_model(model_path, num_classes=2, device=device)
    
    # Perform inference on the specified image
    predicted_class, image = infer_image(image_path, model, classes)
    print("Predicted class:", predicted_class)
    
    # Display the image with the predicted label
    plt.figure(figsize=(6,6))
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis("off")
    plt.show()

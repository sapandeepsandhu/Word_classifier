import torch.nn as nn
import torch.nn.functional as F

class WordCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(WordCNN, self).__init__()
        # First convolutional block: Conv + ReLU + Pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # Third convolutional block (optional, can add if desired)
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Calculate the size of the flattened features.
        # With input size 64x64, after two poolings (each reduces dimensions by 2):
        # 64 -> 32 -> 16 (if using 2 conv blocks)
        # So the feature map from conv2 is 32 channels, 16x16 in size.
        self.fc1 = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        # Convolution Block 1
        x = self.pool(F.relu(self.conv1(x)))
        # Convolution Block 2
        x = self.pool(F.relu(self.conv2(x)))
        # Uncomment below if you add a third block
        # x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the feature maps into a vector
        x = x.view(x.size(0), -1)
        # Final fully-connected layer for classification
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    # Quick test for model output shape
    import torch
    model = WordCNN(num_classes=2)
    sample_input = torch.randn(1, 3, 64, 64)  # one random image, 3 channels, 64x64
    output = model(sample_input)
    print("Output shape:", output.shape)  # Expected: torch.Size([1, 2])

"""
CNN Model for MNIST Digit Classification
Trains a simple convolutional neural network to classify handwritten digits (0-9)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision
import torchvision.transforms.v2 as transforms


# Load MNIST dataset
train_set = torchvision.datasets.MNIST("./data/", train=True, download=True)
valid_set = torchvision.datasets.MNIST("./data/", train=False, download=True)

# Data augmentation for training (rotation, affine transforms)
train_trans = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
])

# No augmentation for validation (keep it deterministic)
test_trans = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
])

train_set.transform = train_trans
valid_set.transform = test_trans

# Create data loaders
batch_size = 128
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)


class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST
    Architecture: 2 conv blocks + 2 fully connected layers
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            
            # Flatten and fully connected layers
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 output classes (digits 0-9)
        )
    
    def forward(self, x):
        return self.features(x)


# Initialize model and training components
model = SimpleCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 5


def train_model(model, train_loader, loss_fn, optimizer, epochs):
    """Train the model for the specified number of epochs"""
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        
        # Iterate through batches
        for images, labels in train_loader:
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")


def evaluate_model(model, test_loader):
    """Evaluate the model on test data"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Train the model
print("Training CNN model...")
train_model(model, train_loader, loss_fn, optimizer, epochs)

# Evaluate on test set
print("\nEvaluating on test set...")
evaluate_model(model, test_loader)

# Save the trained model
torch.save(model.state_dict(), 'mnist_cnn_model.pth')
print("\n✓ Model saved to mnist_cnn_model.pth")










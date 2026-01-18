"""
Transfer Learning CNN for MNIST
Uses a pretrained SimpleCNN as a feature extractor and trains a new classification head.
This helps the model converge faster and achieve better accuracy with less data.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision
import torchvision.transforms.v2 as transforms


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for MNIST
    Used as the base model for transfer learning
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.features(x)


class TransferCNN(nn.Module):
    """
    Transfer Learning CNN for MNIST
    Separates convolutional features (frozen) from classification head (trainable)
    This allows us to reuse learned features from pretrained SimpleCNN
    """
    def __init__(self, num_classes=10):
        super(TransferCNN, self).__init__()
        
        # Feature extractor: convolutional layers (will be loaded from pretrained model)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Classifier head: fully connected layers (will be trained from scratch)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_pretrained_conv_weights(pretrained_path, transfer_model, map_location=None):
    """
    Load convolutional weights from pretrained SimpleCNN into TransferCNN
    Only copies conv layer weights, not the fully connected layers
    """
    sd = torch.load(pretrained_path, map_location=map_location)
    new_sd = transfer_model.state_dict()
    
    # Map pretrained conv layer keys to transfer model
    # Conv layers are at indices 0 and 3 in SimpleCNN.features
    mapping = {
        'features.0.weight': 'features.0.weight',
        'features.0.bias': 'features.0.bias',
        'features.3.weight': 'features.3.weight',
        'features.3.bias': 'features.3.bias',
    }
    
    for src_key, dst_key in mapping.items():
        if src_key in sd and dst_key in new_sd:
            new_sd[dst_key] = sd[src_key]
    
    transfer_model.load_state_dict(new_sd)


def get_dataloaders(batch_size=128):
    """Load MNIST training and test data with augmentation"""
    train_trans = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    test_trans = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    train_set = torchvision.datasets.MNIST("./data/", train=True, download=True)
    test_set = torchvision.datasets.MNIST("./data/", train=False, download=True)
    train_set.transform = train_trans
    test_set.transform = test_trans

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_epoch(model, loader, loss_fn, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate(model, loader, device):
    """Test the model and return accuracy"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Train the transfer learning model"""
    parser = argparse.ArgumentParser(description="MNIST Transfer Learning with pretrained CNN")
    parser.add_argument('--pretrained', type=str, default='mnist_cnn_model.pth', help='path to pretrained weights')
    parser.add_argument('--freeze-features', action='store_true', help='freeze convolutional layers')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = TransferCNN(num_classes=10).to(device)

    # Load pretrained conv weights
    try:
        load_pretrained_conv_weights(args.pretrained, model, map_location=device)
        print(f"✓ Loaded pretrained weights from {args.pretrained}")
    except Exception as e:
        print(f"✗ Could not load pretrained weights: {e}")

    # Optionally freeze conv layers
    if args.freeze_features:
        for p in model.features.parameters():
            p.requires_grad = False
        print("✓ Froze convolutional layers")

    # Load data
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    # Setup training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Quick validation: forward pass
    model.eval()
    sample_imgs, _ = next(iter(train_loader))
    with torch.no_grad():
        out = model(sample_imgs.to(device))
    print(f"✓ Model check: output shape {out.shape}, trainable params: {count_parameters(model)}")

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f} | Test Acc: {acc:.2f}%")

    # Save model
    torch.save(model.state_dict(), 'mnist_cnn_transfer.pth')
    print("✓ Saved model to mnist_cnn_transfer.pth")


if __name__ == '__main__':
    main()

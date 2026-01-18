"""
MNIST Ensemble Classifier - Web Interface
Combines predictions from MLP, CNN, and Transfer Learning CNN models
using weighted voting to classify handwritten digits
"""

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import numpy as np


# Set up the page
st.set_page_config(page_title="MNIST Ensemble Classifier", layout="centered")
st.title("MNIST Ensemble Classifier")
st.write("Draw a digit (0-9) or upload an image to classify using an ensemble of three neural network models.")


# Drawing canvas for user input
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Option to upload an image instead of drawing
uploaded = st.file_uploader("Or upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])


def preprocess_pil(img: Image.Image):
    """Convert PIL image to tensor (28x28 grayscale normalized)"""
    img = img.convert('L')  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors (white digit on black background)
    img = img.resize((28, 28))  # Resize to MNIST size
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])
    tensor = transform(img).unsqueeze(0)
    return tensor

@st.cache_resource
def load_models(device='cpu'):
    """Load all three trained models"""
    
    # Model 1: Simple MLP
    class SimpleMLP(nn.Module):
        def __init__(self):
            super(SimpleMLP, self).__init__()
            self.flatten = nn.Flatten()
            self.features = nn.Sequential(
                nn.Linear(28 * 28, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            return self.features(x)

    mlp = SimpleMLP()
    try:
        mlp.load_state_dict(torch.load('mnist_mlp_model.pth', map_location=device))
    except Exception as e:
        st.error(f'Error loading MLP model: {e}')

    # Model 2: CNN (Convolutional Neural Network)
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )
        def forward(self, x):
            return self.features(x)

    cnn = SimpleCNN()
    try:
        cnn.load_state_dict(torch.load('mnist_cnn_model.pth', map_location=device))
    except Exception:
        st.warning('Could not load `mnist_cnn_model.pth`')

    # Model 3: Transfer Learning CNN
    class TransferCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )
        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    transfer = TransferCNN()
    try:
        transfer.load_state_dict(torch.load('mnist_cnn_transfer.pth', map_location=device))
    except Exception:
        st.warning('Could not load `mnist_cnn_transfer.pth`')

    # Set all models to evaluation mode (no training)
    for m in (mlp, cnn, transfer):
        m.eval()

    return mlp, cnn, transfer

mlp_model, cnn_model, transfer_model = load_models()


def predict_ensemble(tensor):
    """
    Make predictions using all three models and combine with weighted voting
    Weights: MLP 10%, CNN 30%, Transfer 60%
    """
    with torch.no_grad():
        # MLP needs flattened input
        mlp_in = tensor.view(1, -1)
        out_mlp = mlp_model(mlp_in)
        
        # CNN and Transfer need (1, 1, 28, 28) format
        out_cnn = cnn_model(tensor)
        out_transfer = transfer_model(tensor)

        # Convert outputs to probabilities
        p_mlp = torch.softmax(out_mlp, dim=1).squeeze(0).numpy()
        p_cnn = torch.softmax(out_cnn, dim=1).squeeze(0).numpy()
        p_transfer = torch.softmax(out_transfer, dim=1).squeeze(0).numpy()

        # Weighted ensemble (transfer model is most reliable, so gets 60%)
        final = 0.1 * p_mlp + 0.3 * p_cnn + 0.6 * p_transfer
        pred = int(np.argmax(final))
    
    return pred, final

input_tensor = None

# Get image from canvas or upload
if uploaded is not None:
    # User uploaded an image
    img = Image.open(uploaded)
    input_tensor = preprocess_pil(img)
elif canvas_result.image_data is not None:
    # User drew something on the canvas
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L')
    input_tensor = preprocess_pil(img)


# Make prediction and display results
if input_tensor is not None:
    pred, final_probs = predict_ensemble(input_tensor)
    
    # Display the prediction
    st.success(f"**Prediction: {pred}**")
    
    # Show probability chart
    st.subheader("Confidence by Digit")
    st.bar_chart(final_probs)

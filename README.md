# MNIST Ensemble Classifier

A machine learning project that combines three neural networks (MLP, CNN, and Transfer Learning CNN) to classify handwritten digits (0-9) with high accuracy.

## Project Overview

This project demonstrates:
- **Data Augmentation**: Rotating and transforming images during training
- **Convolutional Neural Networks (CNN)**: Learning visual features automatically
- **Transfer Learning**: Reusing pretrained weights to speed up training
- **Model Ensembling**: Combining predictions from multiple models for better accuracy

## Models Included

1. **SimpleMLP** - Fully connected neural network (10% weight in ensemble)
2. **SimpleCNN** - Convolutional neural network (30% weight in ensemble)
3. **TransferCNN** - CNN with pretrained weights from SimpleCNN (60% weight in ensemble)

## Getting Started

### Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training Models

Train the MLP model:
```bash
./venv/bin/python mnist_mlp_model.py
```

Train the CNN model (with data augmentation):
```bash
./venv/bin/python mnist_cnn_model.py
```

Train the Transfer Learning CNN:
```bash
./venv/bin/python mnist_cnn_transfer.py --epochs 5
```

### Running the Web Interface

Start the Streamlit app:
```bash
./venv/bin/streamlit run app.py
```

Then open your browser to `http://localhost:8501`

You can either:
- Draw a digit on the canvas
- Upload an image of a digit

## How Ensemble Voting Works

The three models make independent predictions, then a weighted average combines them:

```
Final Prediction = (0.10 × MLP_prob) + (0.30 × CNN_prob) + (0.60 × TransferCNN_prob)
```

## Performance

- **MLP**: ~97% accuracy
- **CNN**: ~99% accuracy  
- **Transfer CNN**: ~98% accuracy
- **Ensemble**: ~99%+ accuracy

## Data Augmentation

Training uses random transformations to improve model robustness:
- Random rotation (±15 degrees)
- Random affine transformation (translation + scale)

## Technologies Used

- **PyTorch**: Deep learning framework
- **Streamlit**: Web interface
- **NumPy**: Numerical computing
- **Torchvision**: Computer vision utilities

---

*Student Project - Machine Learning Fundamentals*


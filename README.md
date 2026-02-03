# MNIST Sign Language Detection Using CNNs

## Overview
This project implements a deep learning-based system designed to classify American Sign Language (ASL) hand gestures using the Sign Language MNIST dataset. The system utilizes a custom Convolutional Neural Network (CNN) to process grayscale image data, train a robust classifier, and evaluate performance on unseen test data.

The framework is built to be modular and easy to execute, providing visualization of training progress and test predictions. The repository includes pre-generated outputs and results for reference.

## Key Objectives
- **Accurate Classification**: Detect and classify 24 distinct hand signs (excluding J and Z) from the MNIST-style dataset.
- **Model Optimization**: Utilize a custom CNN architecture with batch normalization and dropout to prevent overfitting.
- **Visual Validation**: Generate interpretable visualizations of training history, sample images, and prediction results.
- **Production-Ready Structure**: distinct separation of training, evaluation, and visualization modules.

## Data
- **Dataset**: Sign Language MNIST (28x28 grayscale images)
- **Classes**: 24 active classes (Letters A-Z, excluding J and Z which require motion)
- **Format**: CSV files (`sign_mnist_train.csv`, `sign_mnist_test.csv`) containing pixel values and labels.
- **Source**: Kaggle (assumed source for `archive (9)` dataset)

## Feature Engineering
At each step, raw pixel data is processed:
- **Normalization**: Pixel values (0-255) are scaled to (0-1) for stable training.
- **Reshaping**: Flat vectors are reshaped into `(1, 28, 28)` tensors to preserve spatial structure.
- **Batching**: Data is loaded in batches (default: 64) for efficient gradient updates.

## Models
### 1. Custom CNN Architecture
A specialized convolutional network designed for image classification:
- **Feature Estimator**: 3 Blocks of [Conv2d -> BatchNorm -> ReLU -> MaxPool] to extract hierarchical features.
- **Classifier**: Fully connected layers with Dropout (0.5) to map features to class probabilities.
- **Optimization**: Trained using Adam optimizer with Learning Rate Scheduler (`ReduceLROnPlateau`).

### 2. DenseNet (Alternative)
- The codebase also includes `densenet.py`, suggesting experimentation with DenseNet architectures for potentially higher accuracy.

## Alerting & Monitoring (Visualization)
The system automatically generates outputs to monitor performance:
- **Training Samples**: `training_samples.png` shows batch inputs.
- **Training History**: `training_history.png` plots accuracy and loss curves over epochs.
- **Test Predictions**: `test_predictions.png` and `test_result.png` display model predictions vs ground truth with color-coded validity (Green=Correct, Red=Incorrect).

## Evaluation Methodology
Performance is evaluated using standard classification metrics:
- **Accuracy**: Percentage of correctly classified signs on the test set.
- **Loss**: Cross-Entropy Loss to measure confidence calibration.
- **Visual Inspection**: Direct comparison of predicted labels against ground truth images.

## Key Findings
- **High Accuracy**: The custom CNN achieves competitive accuracy on the test set.
- **Robustness**: The model generalizes well to unseen data, as evidenced by the validation and test accuracy alignment.
- **Efficiency**: The architecture balances depth and speed, suitable for real-time inference on standard hardware.

## Project Structure
```
CL-AI/
├── custom_cnn.py       # Main training script
├── test_model.py       # Evaluation and testing script
├── densenet.py         # Alternative model architecture
├── visualize.py        # plotting utilities
├── CNN.pth             # Trained model weights
├── .gitignore          # Git configuration
└── README.md           # Project documentation
```
*(Note: Dataset `archive (9)` is excluded from version control. Generated outputs and results are included)*

## How to Run
### Prerequisites
- Python 3.x
- Dependencies: `torch`, `pandas`, `numpy`, `matplotlib`

### Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install torch pandas numpy matplotlib
```

### Training
To train the model and save weights (`CNN.pth`):
```bash
python custom_cnn.py
```

### Testing
To evaluate the trained model on the test set:
```bash
python test_model.py
```

**Note**: Ensure the dataset is located at the path specified in the scripts (currently configured for `archive (9)`), or update the paths in `custom_cnn.py` and `test_model.py`.

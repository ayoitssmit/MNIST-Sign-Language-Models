# MNIST Sign Language Detection

This project implements a deep learning-based system to classify American Sign Language (ASL) hand gestures using the Sign Language MNIST dataset. It features two distinct model architectures: a custom Convolutional Neural Network (CNN) and a fine-tuned MobileNetV2, allowing for performance comparison and deployment flexibility.

## Overview
The system processes grayscale image data to train robust classifiers for 24 distinct hand signs (Letters A-Z, excluding J and Z). It includes a complete pipeline for training, evaluation, validation, and model visualization.

## Key Features
- **Dual Architecture**:
  - **Custom CNN**: A lightweight, 3-block architecture optimized for quick training and efficiency.
  - **MobileNetV2**: A fine-tuned, pre-trained model utilizing transfer learning for enhanced accuracy and feature extraction.
- **Robustness**: Implements data augmentation (rotation, flipping), batch normalization, and dropout to prevent overfitting.
- **Visualization Suite**:
  - Training accuracy and loss curves.
  - Confusion matrices for detailed error analysis.
  - Computation graph visualization (`torchviz`).
  - ONNX model export for interoperability.

## Data
- **Dataset**: Sign Language MNIST (28x28 grayscale images).
- **Classes**: 24 active classes (0-25, excluding 9=J and 25=Z).
- **Input Processing**: Reshaped to `(1, 28, 28)` for Custom CNN and resized/repeated to `(3, 64, 64)` for MobileNetV2.

## Project Structure
```
CL-AI/
├── custom_cnn.py       # Training script for Custom CNN
├── mobilenetv2.py      # Training script for MobileNetV2 (Fine-tuning)
├── test_model.py       # Testing & evaluation script (Targeting MobileNetV2)
├── visualize.py        # Model architecture visualization and ONNX export
├── custom_cnn.pth      # Saved weights for Custom CNN
├── mv2slfinal.pth      # Saved weights for MobileNetV2
├── outputs/            # Generated artifacts (ONNX model, architecture diagrams)
├── .venv/              # Virtual environment
└── README.md           # Project documentation
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ayoitssmit/MNIST-Sign-Language-Models.git
   cd MNIST-Sign-Language-Models
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn torchviz torchsummary
   ```
   *(Note: `torchviz` requires GraphViz to be installed on your system)*

## Usage

### 1. Training Custom CNN
Trains the custom 3-block CNN from scratch.
```bash
python custom_cnn.py
```
- **Output**: Saves `custom_cnn.pth`, plots training history and confusion matrix.

### 2. Training MobileNetV2
Fine-tunes a pre-trained MobileNetV2 model.
```bash
python mobilenetv2.py
```
- **Output**: Saves `mobilenetv2_mnist_sl.pth`, logs detailed metrics.

### 3. Evaluation & Testing
Evaluates the trained MobileNetV2 model (`mv2slfinal.pth`) against the test set.
```bash
python test_model.py
```
- **Output**: Classification report, accuracy metrics, and prediction visualizations (`test_result_mv2.png`).

### 4. Visualization & Export
Generates a visual representation of the model architecture and exports it to ONNX format.
```bash
python visualize.py
```
- **Output**: `outputs/mobilenetv2_architecture.png`, `outputs/mobilenetv2_model.onnx`.

## Results
- **Custom CNN**: Balanced performance, lightweight.
- **MobileNetV2**: Higher accuracy potential due to transfer learning, robust against variations.

## Evaluation
Performance is measured using:
- **Accuracy**: Overall correct classification percentage.
- **Cross-Entropy Loss**: Model confidence measure.
- **Confusion Matrix**: To identify specific class misclassifications.

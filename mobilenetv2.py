import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2

from sklearn.metrics import classification_report, confusion_matrix

# =========================
# GLOBAL CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters optimized for performance on lower-end hardware
BATCH_SIZE = 32          # Kept low for memory efficiency
LEARNING_RATE = 1e-4     # Lower LR for fine-tuning
EPOCHS = 15              # Sufficient for convergence with fine-tuning

NUM_CLASSES = 24        # J and Z excluded
RESIZE_TO = 64          # 64x64 is a sweet spot for MobileNetV2 speed/accuracy on CPU

TRAIN_CSV_PATH = r"c:\Users\smits\Desktop\CL-AI\archive (9)\sign_mnist_train.csv"
TEST_CSV_PATH  = r"c:\Users\smits\Desktop\CL-AI\archive (9)\sign_mnist_test.csv"

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =========================
# DATASET
# =========================
class SignLanguageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        data = pd.read_csv(csv_file)
        self.images = data.iloc[:, 1:].values
        self.labels = data["label"].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Reshape to 28x28 (original size)
        image = self.images[idx].reshape(28, 28).astype(np.float32) / 255.0
        
        # Convert to tensor and add channel dim: (1, 28, 28)
        image = torch.tensor(image).unsqueeze(0)

        # Convert 1-channel to 3-channel for MobileNet
        image = image.repeat(3, 1, 1)

        # Resize to target size (e.g., 64x64)
        image = F.interpolate(
            image.unsqueeze(0),
            size=(RESIZE_TO, RESIZE_TO),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # Apply augmentations if any
        if self.transform:
            image = self.transform(image)

        original_label = int(self.labels[idx])

        # Map labels to 0-23 range
        # (J=9 and Z=25 are missing in the dataset)
        LABEL_MAP = {
            0: 0,   1: 1,   2: 2,   3: 3,   4: 4,
            5: 5,   6: 6,   7: 7,   8: 8,
            10: 9,  11: 10, 12: 11, 13: 12, 14: 13,
            15: 14, 16: 15, 17: 16, 18: 17, 19: 18,
            20: 19, 21: 20, 22: 21, 23: 22, 24: 23
        }

        # Safety check
        if original_label not in LABEL_MAP:
             # Fallback or error - usually shouldn't happen with valid dataset
            label_val = 0 
        else:
            label_val = LABEL_MAP[original_label]
            
        label = torch.tensor(label_val, dtype=torch.long)
        return image, label

# =========================
# MODEL
# =========================
class MobileNetV2_MNIST(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()

        # Load pre-trained MobileNetV2
        self.backbone = mobilenet_v2(weights="IMAGENET1K_V1")
        
        # UNFREEZE backbone for fine-tuning --> Better accuracy
        # Since we are resizing to 64x64 which is different from 224x224, 
        # adaptation is crucial.
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Replace classifier
        # MobileNetV2 uses a dropout layer before the final linear layer
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# =========================
# TRAIN / EVAL FUNCTION
# =========================
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

# =========================
# MAIN
# =========================
def main():
    print(f"Using device: {DEVICE}")

    # transforms for mild augmentation to prevent overfitting
    # without being too heavy on CPU
    train_transforms = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.1), # Sign language is handedness-sensitive, use carefully
    ])
    
    # Initialize Datasets
    # Note: We apply transforms inside the Dataset manually or here?
    # The current Dataset class design applies resize manually. 
    # Let's pass transforms to the dataset if we want them, but for strict consistency
    # with the custom_cnn file structure, we'll keep it simple.
    # However, to improve accuracy as requested, we allow the dataset to apply extra transforms.
    
    train_full = SignLanguageDataset(TRAIN_CSV_PATH, transform=train_transforms)
    test_set = SignLanguageDataset(TEST_CSV_PATH)

    # Split Train/Val
    train_size = int(0.8 * len(train_full))
    val_size = len(train_full) - train_size
    train_set, val_set = random_split(train_full, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set, BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_set, BATCH_SIZE, shuffle=False)

    # Initialize Model
    model = MobileNetV2_MNIST(NUM_CLASSES).to(DEVICE)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler to reduce LR when validation accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, factor=0.5
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    print("\nTraining MobileNetV2 (Fine-Tuning)...\n")
    
    start_train_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        # Step User LR Scheduler
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

    total_train_time = time.time() - start_train_time
    print(f"\nTotal Training Time: {total_train_time/60:.2f} minutes")

    # Save Model
    save_path = "mobilenetv2_mnist_sl.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved as {save_path}")

    # =========================
    # VISUALIZATION
    # =========================
    # 1. Accuracy & Loss Curves
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("MobileNetV2 Training Accuracy")
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("MobileNetV2 Training Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("mobilenetv2_training_history.png")
    plt.close()
    print("Graph saved: mobilenetv2_training_history.png")

    # =========================
    # TEST EVALUATION
    # =========================
    print("\nEvaluating on Test Set...")
    model.eval()
    y_true, y_pred = [], []

    start_time = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    inference_time = time.time() - start_time

    print("\nClassification Report (MobileNetV2):")
    print(classification_report(y_true, y_pred, digits=4))

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("MobileNetV2 Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("mobilenetv2_confusion_matrix.png")
    plt.close()
    print("Graph saved: mobilenetv2_confusion_matrix.png")

    print(f"Inference time (test set): {inference_time:.2f}s")
    print("Done!")

if __name__ == "__main__":
    main()

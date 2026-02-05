import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# =========================
# GLOBAL CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST Sign Language label mapping
# Original labels: 0–25 (A–Z)
# Missing: J (9), Z (25)

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20

IMG_HEIGHT = 28
IMG_WIDTH = 28
NUM_CLASSES = 24   # IMPORTANT: J and Z excluded

TRAIN_CSV_PATH = r"c:\Users\smits\Desktop\CL-AI\archive (9)\sign_mnist_train.csv"
TEST_CSV_PATH  = r"c:\Users\smits\Desktop\CL-AI\archive (9)\sign_mnist_test.csv"

torch.manual_seed(42)
np.random.seed(42)

# =========================
# DATASET
# =========================
class SignLanguageDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)

        self.images = data.iloc[:, 1:].values
        self.labels = data["label"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28).astype(np.float32)
        image = image / 255.0
        image = torch.tensor(image).unsqueeze(0)

        original_label = int(self.labels[idx])

        # Exclude J (9) and Z (25)
        LABEL_MAP = {
            0: 0,   1: 1,   2: 2,   3: 3,   4: 4,
            5: 5,   6: 6,   7: 7,   8: 8,
            10: 9,  11: 10, 12: 11, 13: 12, 14: 13,
            15: 14, 16: 15, 17: 16, 18: 17, 19: 18,
            20: 19, 21: 20, 22: 21, 23: 22, 24: 23
        }

        if original_label not in LABEL_MAP:
            raise ValueError(f"Invalid label found: {original_label}")

        label = torch.tensor(LABEL_MAP[original_label], dtype=torch.long)

        return image, label


# =========================
# MODEL
# =========================
class CustomCNN(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),  # lighter FC (ablation-ready)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# =========================
# TRAIN / EVAL FUNCTIONS
# =========================
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0, 0, 0

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

    train_dataset_full = SignLanguageDataset(TRAIN_CSV_PATH)
    test_dataset = SignLanguageDataset(TEST_CSV_PATH)

    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size

    train_dataset, val_dataset = random_split(
        train_dataset_full, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    model = CustomCNN(NUM_CLASSES).to(DEVICE)

    # =========================
    # MODEL STATS (PAPER)
    # =========================
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2
    )

    history = {"train_acc": [], "val_acc": []}

    print("\nTraining started...\n")
    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion
        )

        scheduler.step(val_acc)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

    torch.save(model.state_dict(), "custom_cnn.pth")

    # =========================
    # TEST EVALUATION
    # =========================
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

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    print(f"Inference time (test set): {inference_time:.2f}s")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()

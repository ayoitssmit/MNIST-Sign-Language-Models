import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# =========================
# GLOBAL CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

TEST_CSV_PATH = r"c:\Users\smits\Desktop\CL-AI\archive (9)\sign_mnist_test.csv"
MODEL_FILENAME = "mv2slfinal.pth"   
BATCH_SIZE = 64
NUM_CLASSES = 24
RESIZE_TO = 64

# =========================
# DATASET
# =========================
class SignLanguageDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.images = data.iloc[:, 1:].values
        self.labels = data["label"].values

        self.LABEL_MAP = {
            0:0,1:1,2:2,3:3,4:4,
            5:5,6:6,7:7,8:8,
            10:9,11:10,12:11,13:12,14:13,
            15:14,16:15,17:16,18:17,19:18,
            20:19,21:20,22:21,23:22,24:23
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].reshape(28, 28).astype(np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0)
        img = img.repeat(3, 1, 1)

        img = F.interpolate(
            img.unsqueeze(0),
            size=(RESIZE_TO, RESIZE_TO),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        label = self.LABEL_MAP[int(self.labels[idx])]
        return img, torch.tensor(label, dtype=torch.long)

# =========================
# MODEL (FIXED ARCHITECTURE)
# =========================
class MobileNetV2_SL(nn.Module):
    def __init__(self, num_classes=24, hidden_dim=256):
        super().__init__()
        # FIX 1: Rename 'model' to 'backbone' to match the saved file keys
        self.backbone = mobilenet_v2(weights=None)
        
        # FIX 2: Update classifier to match saved indices (1 and 4)
        # Structure: Dropout(0) -> Linear(1) -> ReLU(2) -> Dropout(3) -> Linear(4)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, hidden_dim), # Corresponds to classifier.1
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes) # Corresponds to classifier.4
        )

    def forward(self, x):
        return self.backbone(x)

# =========================
# VISUALIZATION
# =========================
def visualize(model, loader):
    print("Generating visualization...")
    images, labels = next(iter(loader))
    images = images.to(DEVICE)

    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(1)

    images = images.cpu()
    preds = preds.cpu()
    labels = labels.cpu()

    acc = preds.eq(labels).float().mean() * 100

    plt.figure(figsize=(10, 10))
    plt.suptitle(f"MobileNetV2 Predictions (Batch Acc: {acc:.1f}%)")

    for i in range(min(9, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        img = images[i].permute(1, 2, 0)
        plt.imshow(img, cmap="gray")

        color = "green" if preds[i] == labels[i] else "red"
        plt.title(f"P:{preds[i].item()} | T:{labels[i].item()}", color=color)
        plt.axis("off")

    plt.savefig("test_result_mv2.png")
    plt.close()
    print("Visualization saved as test_result_mv2.png")

# =========================
# TEST PIPELINE
# =========================
def test_model():
    print(f"\n--- Testing MobileNetV2 Model: {MODEL_FILENAME} ---")

    if not os.path.exists(TEST_CSV_PATH):
        print("❌ Test CSV not found")
        return

    if not os.path.exists(MODEL_FILENAME):
        print(f"❌ Model file '{MODEL_FILENAME}' not found")
        return

    # Load data
    test_dataset = SignLanguageDataset(TEST_CSV_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")

    # Load State Dict First to detect structure
    print("Loading model weights...")
    state_dict = torch.load(MODEL_FILENAME, map_location=DEVICE)
    
    # FIX 3: Detect hidden dimension size from the saved file automatically
    if 'backbone.classifier.1.weight' in state_dict:
        hidden_dim = state_dict['backbone.classifier.1.weight'].shape[0]
        print(f"✅ Detected hidden layer dimension: {hidden_dim}")
    else:
        hidden_dim = 256 # Default fallback
        print(f"⚠️ Could not detect hidden dim, using default: {hidden_dim}")

    # Initialize model with correct dim
    model = MobileNetV2_SL(NUM_CLASSES, hidden_dim=hidden_dim).to(DEVICE)
    
    # Load weights
    try:
        model.load_state_dict(state_dict)
        print("✅ Model weights loaded successfully")
    except RuntimeError as e:
        print(f"❌ Failed to load weights: {e}")
        return

    # Evaluation
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    print("Running evaluation...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    print("\n" + "="*35)
    print("FINAL TEST RESULTS (MobileNetV2)")
    print(f"Loss     : {total_loss / total:.4f}")
    print(f"Accuracy : {correct / total * 100:.2f}%")
    print("="*35)

    visualize(model, test_loader)

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    test_model()
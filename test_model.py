# test_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# --- IMPORT MODULES ---
# Now this import is safe because we added the if __name__ == "__main__" block
from custom_cnn import SignLanguageDataset, CustomCNN, TEST_CSV_PATH, DEVICE

# ---------------------------------------------------------
#  USER CONFIGURATION
# ---------------------------------------------------------
MODEL_FILENAME = 'CNN.pth'

MODEL_ARCHITECTURES = {
    'CNN.pth': CustomCNN,
}

# -----------------
#  TESTING SCRIPT
# -----------------
def visualize(model, loader):
    try:
        print("Generating visualization...")
        images, labels = next(iter(loader))
        images = images.to(DEVICE)
        
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        
        images = images.cpu()
        predictions = predictions.cpu()
        labels = labels.cpu()
        
        plt.figure(figsize=(10, 10))
        acc = predictions.eq(labels).float().mean() * 100
        plt.suptitle(f"Predictions for {MODEL_FILENAME} (Batch Acc: {acc:.1f}%)")
        
        for i in range(min(9, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].squeeze(), cmap='gray')
            
            pred_lab = predictions[i].item()
            true_lab = labels[i].item()
            
            color = 'green' if pred_lab == true_lab else 'red'
            plt.title(f"Pred: {pred_lab} | True: {true_lab}", color=color)
            plt.axis("off")
            
        # CHANGE: Save to file instead of show
        plt.savefig('test_result.png')
        plt.close()
        print("Visualization saved as 'test_result.png'.")
        
    except Exception as e:
        print(f"Visualization failed: {e}")

def test_model():
    print(f"--- Testing Model: {MODEL_FILENAME} ---")
    
    # 1. Determine Architecture
    if MODEL_FILENAME not in MODEL_ARCHITECTURES:
        print(f"Error: No architecture defined for '{MODEL_FILENAME}'")
        return
    
    model_class = MODEL_ARCHITECTURES[MODEL_FILENAME]
    print(f"Architecture: {model_class.__name__}")

    # 2. Load Data
    if not os.path.exists(TEST_CSV_PATH):
        print(f"Error: Test data not found at {TEST_CSV_PATH}")
        return
        
    print("Loading test dataset...")
    test_dataset = SignLanguageDataset(TEST_CSV_PATH)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")

    # 3. Load Model Weights
    if not os.path.exists(MODEL_FILENAME):
        print(f"Error: Model file '{MODEL_FILENAME}' not found. Please run custom_cnn.py first.")
        return

    model = model_class().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_FILENAME, map_location=DEVICE))
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    # 4. Evaluate
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    print("Running evaluation...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    final_loss = running_loss / total
    final_acc = correct / total * 100
    
    print("\n" + "="*30)
    print(f"FINAL TEST RESULTS")
    print(f"Loss: {final_loss:.4f}")
    print(f"Accuracy: {final_acc:.2f}%")
    print("="*30 + "\n")

    # 5. Visualize
    visualize(model, test_loader)

if __name__ == "__main__":
    test_model()
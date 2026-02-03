import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# -----------------
# GLOBAL CONSTANTS
# -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
IMG_HEIGHT = 28
IMG_WIDTH = 28
NUM_CLASSES = 26 

# Paths
TRAIN_CSV_PATH = r"c:\Users\smits\Desktop\CL-AI\archive (9)\sign_mnist_train.csv"
TEST_CSV_PATH = r"c:\Users\smits\Desktop\CL-AI\archive (9)\sign_mnist_test.csv"

# -----------------
# DATASET CLASS
# -----------------
class SignLanguageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        raw_data = pd.read_csv(csv_file)
        self.labels = raw_data['label'].values
        self.pixels = raw_data.iloc[:, 1:].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.pixels[idx].reshape(IMG_HEIGHT, IMG_WIDTH).astype(np.float32)
        image = image / 255.0  # Normalize
        image = torch.tensor(image).unsqueeze(0) # Add channel dimension (1, 28, 28)
        return image, torch.tensor(label, dtype=torch.long)

# -----------------
# MODEL ARCHITECTURE
# -----------------
class CustomCNN(nn.Module):
    def __init__(self, num_classes=26):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------
# MAIN EXECUTION
# -----------------
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. CHECK FILES
    if not os.path.exists(TRAIN_CSV_PATH) or not os.path.exists(TEST_CSV_PATH):
        print("Error: Dataset files not found.")
        exit()

    # 2. LOAD DATA
    full_train_dataset = SignLanguageDataset(TRAIN_CSV_PATH)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    test_dataset = SignLanguageDataset(TEST_CSV_PATH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # 3. VISUALIZE DATA (SAVE TO FILE)
    try:
        images, labels = next(iter(train_loader))
        plt.figure(figsize=(10, 10))
        plt.suptitle("Sample Images from Training Set")
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].squeeze(), cmap='gray')
            plt.title(f"Label: {labels[i].item()}")
            plt.axis("off")
        
        # CHANGED: Save instead of Show
        plt.savefig('training_samples.png')
        plt.close() 
        print("Graph saved: training_samples.png")
    except Exception as e:
        print(f"Visualization skipped: {e}")

    # 4. INITIALIZE MODEL
    model = CustomCNN(NUM_CLASSES).to(DEVICE)
    print("\nModel Architecture Created.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    # 5. TRAINING LOOP
    print("\nStarting model training...")
    
    def run_epoch(loader, is_training=True):
        if is_training:
            model.train()
        else:
            model.eval()
            
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.set_grad_enabled(is_training):
            for images, labels in loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                if is_training:
                    optimizer.zero_grad()
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                if is_training:
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(train_loader, is_training=True)
        val_loss, val_acc = run_epoch(val_loader, is_training=False)
        
        scheduler.step(val_acc)
        
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    print("Model training complete.")

    # 6. SAVE MODEL
    print("\nSaving the trained model...")
    torch.save(model.state_dict(), 'CNN.pth')
    print("Model saved successfully as 'CNN.pth'")

    # 7. EVALUATE ON TEST SET
    print("\nEvaluating on Test Set...")
    test_loss, test_acc = run_epoch(test_loader, is_training=False)
    print(f'Test accuracy: {test_acc * 100:.2f}%')

    # 8. PLOT HISTORY (SAVE TO FILE)
    try:
        epochs_range = range(EPOCHS)
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['accuracy'], label='Training Accuracy')
        plt.plot(epochs_range, history['val_accuracy'], label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history['loss'], label='Training Loss')
        plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.grid(True)
        
        # CHANGED: Save instead of Show
        plt.savefig('training_history.png')
        plt.close()
        print("Graph saved: training_history.png")
    except Exception as e:
        print(f"Plotting history skipped: {e}")

    # 9. VISUALIZE PREDICTIONS (SAVE TO FILE)
    try:
        plt.figure(figsize=(10, 10))
        plt.suptitle("Predictions on Test Images")
        
        images, labels = next(iter(test_loader))
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
        
        images = images.cpu()
        labels = labels.cpu()
        predictions = predictions.cpu()
        
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].squeeze(), cmap='gray')
            
            predicted_label = predictions[i].item()
            true_label = labels[i].item()
            
            title_color = 'green' if predicted_label == true_label else 'red'
            plt.title(f"Pred: {predicted_label}, True: {true_label}", color=title_color)
            plt.axis("off")
            
        # CHANGED: Save instead of Show
        plt.savefig('test_predictions.png')
        plt.close()
        print("Graph saved: test_predictions.png")
    except Exception as e:
        print(f"Prediction visualization skipped: {e}")
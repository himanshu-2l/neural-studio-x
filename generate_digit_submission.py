import os
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using PyTorch device: {device}")

def generate_synthetic_digits(n_train=2000, n_test=1000):
    y_train = np.random.randint(0, 10, size=n_train)
    X_train = np.random.randint(0, 256, size=(n_train, 784))
    X_test = np.random.randint(0, 256, size=(n_test, 784))
    train_df = pd.DataFrame(X_train, columns=[f'pixel{i}' for i in range(784)])
    train_df.insert(0, 'label', y_train)
    test_df = pd.DataFrame(X_test, columns=[f'pixel{i}' for i in range(784)])
    test_df.insert(0, 'ImageId', np.arange(1, n_test + 1))
    return train_df, test_df

# Search dataset
train_matches = glob.glob('/kaggle/input/**/train.csv', recursive=True) + glob.glob('../input/**/train.csv', recursive=True) + glob.glob('./**/train.csv', recursive=True)
test_matches = glob.glob('/kaggle/input/**/test.csv', recursive=True) + glob.glob('../input/**/test.csv', recursive=True) + glob.glob('./**/test.csv', recursive=True)

digit_train = [f for f in train_matches if 'digit' in f.lower() or 'mnist' in f.lower()]
digit_test = [f for f in test_matches if 'digit' in f.lower() or 'mnist' in f.lower()]

if len(digit_train) > 0 and len(digit_test) > 0:
    train_df = pd.read_csv(digit_train[0])
    test_df = pd.read_csv(digit_test[0])
    print(f"Loaded dataset: {digit_train[0]}")
else:
    print("Generating synthetic 28x28 digit data...")
    train_df, test_df = generate_synthetic_digits(2000, 1000)

labels = train_df['label'].values
pixel_cols = [c for c in train_df.columns if c != 'label']
X_data = (train_df[pixel_cols].values / 255.0).reshape(-1, 1, 28, 28).astype(np.float32)
y_data = labels.astype(np.int64)

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = DigitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training PyTorch CNN Model...")
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == lbls).sum().item()
    val_acc = (correct / len(val_loader.dataset)) * 100.0
    print(f"Epoch [{epoch}/{epochs}] -> Loss: {epoch_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

test_pixel_cols = [c for c in test_df.columns if c.lower() != 'imageid']
X_test_arr = (test_df[test_pixel_cols].values / 255.0).reshape(-1, 1, 28, 28).astype(np.float32)
test_tensor = torch.tensor(X_test_arr).to(device)

model.eval()
with torch.no_grad():
    test_preds = model(test_tensor).argmax(dim=1).cpu().numpy()

image_ids = test_df['ImageId'].values if 'ImageId' in test_df.columns else np.arange(1, len(test_preds) + 1)
sub_df = pd.DataFrame({'ImageId': image_ids, 'Label': test_preds})
sub_df.to_csv('submission_digit.csv', index=False)
print("SUCCESS: Saved 'submission_digit.csv' with shape", sub_df.shape)

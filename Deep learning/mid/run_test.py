import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Download data
print("=" * 50)
print("Q2.1: Downloading Fashion-MNIST...")
full_train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
test_dataset_raw = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True)

train_images_all = full_train_dataset.data.numpy()
train_labels_all = full_train_dataset.targets.numpy()
test_images_all = test_dataset_raw.data.numpy()
test_labels_all = test_dataset_raw.targets.numpy()

print(f"Training set size: {len(full_train_dataset)}")
print(f"Test set size: {len(test_dataset_raw)}")
print(f"Image shape: {full_train_dataset[0][0].size}")

# Q2.4: Prepare tensors
print("\n" + "=" * 50)
print("Q2.4: Preparing Tensors...")
X_all = train_images_all / 255.0
y_all = train_labels_all
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(X_all, y_all, test_size=10000, random_state=42)

X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.long)
X_val = torch.tensor(X_val_np, dtype=torch.float32)
y_val = torch.tensor(y_val_np, dtype=torch.long)
X_test = torch.tensor(test_images_all / 255.0, dtype=torch.float32)
y_test = torch.tensor(test_labels_all, dtype=torch.long)

# Test Q2.4
assert X_train.shape == (50000, 28, 28)
assert X_val.shape == (10000, 28, 28)
assert X_train.dtype == torch.float32
assert y_train.dtype == torch.long
assert X_train.max() <= 1.0 and X_train.min() >= 0.0
print("✅ Q2.4 Passed!")
print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Q3: DataLoaders
print("\n" + "=" * 50)
print("Q3: Building DataLoaders...")
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

batch_X, batch_y = next(iter(train_loader))
assert batch_X.shape == (64, 28, 28)
assert batch_y.dtype == torch.long
print("✅ Q3 Passed!")
print(f"   Batch images shape: {batch_X.shape}")
print(f"   Batch labels shape: {batch_y.shape}")

# Q4: Model
print("\n" + "=" * 50)
print("Q4: Building Model...")

class FashionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.model(x)

model = FashionClassifier()
test_input = torch.randn(5, 28, 28)
test_output = model(test_input)
assert test_output.shape == (5, 10)
total_params = sum(p.numel() for p in model.parameters())
print("✅ Q4 Passed!")
print(f"   Total parameters: {total_params}")
print(model)

# Q5.1: Setup
print("\n" + "=" * 50)
print("Q5.1: Setting up training...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epochs = 15

# Q5.2: Training Loop
print("\n" + "=" * 50)
print("Q5.2: Training for 15 epochs...")
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_losses.append(running_loss / total)
    train_accuracies.append(100 * correct / total)

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_losses.append(val_loss / total)
    val_accuracies.append(100 * correct / total)
    print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Train Acc: {train_accuracies[-1]:.2f}% | Val Acc: {val_accuracies[-1]:.2f}%')

assert len(train_losses) == 15
assert train_losses[-1] < train_losses[0]
print("✅ Q5.2 Passed!")
print(f"   Final Train Acc: {train_accuracies[-1]:.2f}%, Final Val Acc: {val_accuracies[-1]:.2f}%")

# Q5.3: Test Evaluation
print("\n" + "=" * 50)
print("Q5.3: Evaluating on test set...")
model.eval()
all_preds, all_labels = [], []
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
assert test_accuracy > 40.0
assert len(all_preds) == len(X_test)
print(f"✅ Q5.3 Passed! Test Accuracy: {test_accuracy:.2f}%")

# Q6.4: Classification Report
print("\n" + "=" * 50)
print("Q6.4: Classification Report")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Summary
print("=" * 50)
print("🎉 ALL TESTS PASSED!")
print(f"   Final Test Accuracy: {test_accuracy:.2f}%")
print("=" * 50)

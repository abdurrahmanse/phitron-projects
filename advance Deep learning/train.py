"""
Simple AgeCNN training script — mirrors the notebook cells exactly.
Run:  python3 train.py
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ─── Paths ────────────────────────────────────────────────────────────────────
CSV_PATH = "faces/train.csv"
IMG_DIR  = "faces/Train"

# ─── Label map & image size ───────────────────────────────────────────────────
LABEL_MAP = {"YOUNG": 0, "MIDDLE": 1, "OLD": 2}
IMG_SIZE  = 128

# ─── Transforms ───────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─── Dataset class ────────────────────────────────────────────────────────────
class FaceAgeDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform):
        self.df        = dataframe.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(os.path.join(self.img_dir, row["ID"])).convert("RGB")
        label = LABEL_MAP[row["Class"]]
        return self.transform(img), label

# ─── Simple CNN model ─────────────────────────────────────────────────────────
class AgeCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2)   # 128→64
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2)   # 64→32
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)  # 32→16
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2)  # 16→8
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        return self.classifier(x)

# ─── Train / eval helpers ─────────────────────────────────────────────────────
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += len(images)
    return total_loss / total, correct / total

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = loss_fn(outputs, labels)
            total_loss += loss.item() * len(images)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += len(images)
    return total_loss / total, correct / total

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    import sys

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    print(f"Total images: {len(df)}")
    print(df["Class"].value_counts().to_string())

    # ── Split ─────────────────────────────────────────────────────────────────
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Class"]
    )
    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)}")

    train_loader = DataLoader(
        FaceAgeDataset(train_df, IMG_DIR, train_transform),
        batch_size=32, shuffle=True,  num_workers=0
    )
    val_loader = DataLoader(
        FaceAgeDataset(val_df, IMG_DIR, val_transform),
        batch_size=32, shuffle=False, num_workers=0
    )

    imgs, lbls = next(iter(train_loader))
    print("Batch shape:", imgs.shape)

    # ── Device ────────────────────────────────────────────────────────────────
    torch.manual_seed(42)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model  = AgeCNN(num_classes=3).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params:,}")

    # ── Loss & optimiser ──────────────────────────────────────────────────────
    counts  = Counter(df["Class"].map(LABEL_MAP))
    weights = torch.tensor(
        [len(df) / (3 * counts[i]) for i in range(3)], dtype=torch.float
    ).to(device)
    print("Class weights:", [round(w.item(), 3) for w in weights])

    loss_fn   = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    EPOCHS       = 25
    best_val_acc = 0.0
    history      = {"train_loss": [], "val_loss": [],
                    "train_acc":  [], "val_acc":  []}

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), "best_model.pth")
            note = " <- best"
        else:
            note = ""

        print(f"{epoch:>5} | {tr_loss:>10.4f} | {tr_acc*100:>8.1f}% | "
              f"{vl_loss:>8.4f} | {vl_acc*100:>6.1f}%{note}", flush=True)

    # ── Load best & report ────────────────────────────────────────────────────
    model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
    _, final_acc = evaluate(model, val_loader, loss_fn, device)
    print(f"\nFinal Validation Accuracy: {final_acc*100:.2f}%")

    # ── Plots ─────────────────────────────────────────────────────────────────
    ep = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(ep, history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(ep, history["val_loss"],   label="Val Loss",   linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss over Epochs"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ep, [a*100 for a in history["train_acc"]], label="Train Acc", linewidth=2)
    ax2.plot(ep, [a*100 for a in history["val_acc"]],   label="Val Acc",   linewidth=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy over Epochs"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("AgeCNN — Training Results", fontsize=13)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120, bbox_inches="tight")
    print("Saved: training_curves.png")

    # ── Classification report ─────────────────────────────────────────────────
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images  = images.to(device)
            preds   = model(images).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    target_names = ["YOUNG", "MIDDLE", "OLD"]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig2, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1,2]); ax.set_xticklabels(target_names)
    ax.set_yticks([0,1,2]); ax.set_yticklabels(target_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(3):
        for j in range(3):
            color = "white" if cm[i,j] > cm.max()/2 else "black"
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color=color, fontsize=12)
    plt.colorbar(im); plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
    print("Saved: confusion_matrix.png")
    print(f"\nOverall Accuracy: {np.trace(cm)/cm.sum()*100:.2f}%")


if __name__ == "__main__":
    main()

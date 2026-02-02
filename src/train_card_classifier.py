import os
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# ================= CONFIG =================
DATA_DIR = r"C:\ElixerTracker\data\icons\dataset"
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-3
MODEL_OUT = r"C:\ElixerTracker\models\card_classifier.pth"
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --------- Data transforms (VERY IMPORTANT) ---------
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------- Load datasets ---------
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tfms)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

num_classes = len(train_ds.classes)
print("Detected classes:", train_ds.classes)

# --------- Model: Pretrained MobileNetV2 ---------
model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
)

# Freeze all backbone layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)

# --------- Training loop ---------
best_val_acc = 0.0
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # --------- Validation ---------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = correct / total if total > 0 else 0
    print(f"Epoch {epoch+1}: loss={epoch_loss:.3f}, val_acc={val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state": model.state_dict(),
            "classes": train_ds.classes
        }, MODEL_OUT)
        print("✅ Best model saved")

print("\nTraining finished.")
print("Best validation accuracy:", best_val_acc)
print("Model saved at:", MODEL_OUT)

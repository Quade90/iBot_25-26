import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# TRANSFORMS (5+ augmentations)
train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.RandomAffine(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# DATA
train_data = datasets.ImageFolder("data/train", transform=train_tf)
val_data   = datasets.ImageFolder("data/val", transform=val_tf)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32)

# MODEL
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

for p in model.parameters():
    p.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# LOSS & OPT
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5
)

# TRACKING
train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0

# TRAIN
epochs = 10

for epoch in range(epochs):
    model.train()
    correct, total = 0, 0
    run_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        preds = model(imgs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        run_loss += loss.item()
        _, p = torch.max(preds, 1)
        total += labels.size(0)
        correct += (p == labels).sum().item()

    train_loss = run_loss / len(train_loader)
    train_acc = 100 * correct / total

    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # VALIDATION
    model.eval()
    v_loss = 0
    v_correct, v_total = 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)

            v_loss += loss.item()
            _, p = torch.max(preds, 1)
            v_total += labels.size(0)
            v_correct += (p == labels).sum().item()

    val_loss = v_loss / len(val_loader)
    val_acc = 100 * v_correct / v_total

    val_losses.append(val_loss)
    val_accs.append(val_acc)

    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model")

    print(f"""
Epoch {epoch+1}/{epochs}
Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%
Val   Loss: {val_loss:.3f} | Val   Acc: {val_acc:.2f}%
""")

# PLOTS
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(train_losses,label="Train")
plt.plot(val_losses,label="Val")
plt.title("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs,label="Train")
plt.plot(val_accs,label="Val")
plt.title("Accuracy")
plt.legend()

plt.savefig("training_curves.png")
plt.show()

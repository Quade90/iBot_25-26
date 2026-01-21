import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data Augmentation & Transforms
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

# Loading data
train_data = datasets.ImageFolder("data/train", transform=train_tf)
val_data   = datasets.ImageFolder("data/val", transform=val_tf)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32)

print("Train classes:", train_data.class_to_idx)
print("Val classes:", val_data.class_to_idx)

num_classes = len(train_data.classes)
print(num_classes, train_data.classes)

# Model
model_choice = input("Choose model (resnet18/mobilenet): ").strip().lower()
if model_choice == "mobilenet":
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

elif model_choice == "resnet18":
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
else:
    raise ValueError("Invalid model choice. Choose 'resnet18' or 'mobilenet'.")

# Freeze everything
for p in model.parameters():
    p.requires_grad = False

# Unfreeze classifier
if model_choice == "resnet18":
    for p in model.fc.parameters():
        p.requires_grad = True
else:
    for p in model.classifier.parameters():
        p.requires_grad = True


model = model.to(device)

# Loss and Optimizer for intial training
criterion = nn.CrossEntropyLoss()

if model_choice == "resnet18":
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
else:  # mobilenet
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5
)

train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0

# Training loop
epochs = 10

print("Training the model...")

for epoch in range(epochs):
    # Training
    model.train()
    correct, total = 0, 0
    run_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        preds = model(imgs) # Forward pass
        labels = labels.long()
        loss = criterion(preds, labels)

        optimizer.zero_grad() # Backward pass
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

    # Validation
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

    # Identify best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = copy.deepcopy(model)

    print(f"""
Epoch {epoch+1}/{epochs}
Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%
Val   Loss: {val_loss:.3f} | Val   Acc: {val_acc:.2f}%
LR: {optimizer.param_groups[0]['lr']}
""")
    
def plot_grid(imgs, classes):
    """Plots a grid of images with true and predicted labels.
    
    Parameters:
        imgs: List of tuples (image_tensor, true_label, predicted_label)
        classes: List of class names corresponding to label indices.
    
    Returns:
        None
    """
    plt.figure(figsize=(15,8))
    for i, (img, true, pred) in enumerate(imgs):
        plt.subplot(4,5,i+1)

        # unnormalize
        img = img.cpu().permute(1,2,0)
        img = img * torch.tensor([0.229,0.224,0.225]) + \
              torch.tensor([0.485,0.456,0.406])
        img = img.clip(0,1)

        plt.imshow(img)
        plt.axis("off")
        color = "green" if true == pred else "red"
        plt.title(
            f"T:{classes[true]}\nP:{classes[pred]}",
            color=color, fontsize=9
        )

    plt.suptitle(
        "Correct (Top) vs Incorrect (Bottom) Predictions",
        fontsize=14
    )
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    
def show_predictions(model, loader):
    """Shows correct and incorrect predictions from the model.
    
    Parameters:
        model: Trained PyTorch model.
        loader: DataLoader for the dataset to evaluate.
    
    Returns:
        correct: Dictionary with lists of correct predictions per class.
        wrong: Dictionary with lists of incorrect predictions per class.
    """
    model.eval()

    correct = {0: [], 1: []}
    wrong = {0: [], 1: []}

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            _, predicted = torch.max(preds, 1)

            for i in range(len(imgs)):
                t = labels[i].item()
                p = predicted[i].item()

                if t == p and len(correct[t]) < 5:
                    correct[t].append((imgs[i], t, p))

                elif t != p and len(wrong[t]) < 5:
                    wrong[t].append((imgs[i], t, p))

                if all(len(correct[c])==5 for c in [0,1]) and \
                   all(len(wrong[c])==5 for c in [0,1]):
                    return correct, wrong

    return correct, wrong

    
# Plots for initial training and validation
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

# Fine-tuning
print("Fine-tuning best model...")
if model_choice == "resnet18":
    for name, param in best_model.layer4.named_parameters():
        if "conv2" in name:
            param.requires_grad = True

elif model_choice == "mobilenet":
    for name, param in best_model.features[-1].named_parameters():
        param.requires_grad = True

if model_choice == "resnet18":
    optimizer = optim.Adam([
        {'params': best_model.layer4.parameters(), 'lr': 1e-5},
        {'params': best_model.fc.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)

else:  # mobilenet
    optimizer = optim.Adam([
        {'params': best_model.features[-1].parameters(), 'lr': 1e-5},
        {'params': best_model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)

fine_tune_epochs = 10

scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.3
)

# Fine-tuning loop
ft_train_losses, ft_val_losses = [], []
ft_train_accs, ft_val_accs = [], []

for epoch in range(fine_tune_epochs):
    # Training
    best_model.train()
    correct, total = 0, 0
    run_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        preds = best_model(imgs)
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
    
    # Validation
    best_model.eval()
    v_loss = 0
    v_correct, v_total = 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            preds = best_model(imgs)
            loss = criterion(preds, labels)

            v_loss += loss.item()
            _, p = torch.max(preds, 1)
            v_total += labels.size(0)
            v_correct += (p == labels).sum().item()

    val_loss = v_loss / len(val_loader)
    val_acc = 100 * v_correct / v_total

    ft_train_losses.append(train_loss)
    ft_train_accs.append(train_acc)
    ft_val_losses.append(val_loss)
    ft_val_accs.append(val_acc)

    scheduler_ft.step(val_loss)

    # Save best fine-tuned model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(best_model.state_dict(), "best_finetuned.pth")

    print(f"""
[Fine-tune] Epoch {epoch+1}/{fine_tune_epochs}
Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%
Val   Loss: {val_loss:.3f} | Val   Acc: {val_acc:.2f}%
LR: {optimizer.param_groups[0]['lr']}
""")
    
# Plots for fine-tuning
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(ft_train_losses,label="FT Train")
plt.plot(ft_val_losses,label="FT Val")
plt.title("Fine-tuning Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(ft_train_accs,label="FT Train")
plt.plot(ft_val_accs,label="FT Val")
plt.title("Fine-tuning Accuracy")
plt.legend()

plt.savefig("finetune_curves.png")
plt.show()

correct_imgs, wrong_imgs = show_predictions(
    best_model, val_loader
)

plot_grid(correct_imgs[0] + correct_imgs[1] + wrong_imgs[0] + wrong_imgs[1], train_data.classes)



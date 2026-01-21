import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Transforms and DataLoader
tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_data = datasets.ImageFolder("data/test", transform=tf)
loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Trained and finetuned model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features,2)
model.load_state_dict(torch.load("best_finetuned.pth"))
model.to(device)
model.eval()

correct,total = 0,0
preds_all, labels_all = [], []

# Testing loop
print("Testing the model...")

with torch.no_grad():
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        _, p = torch.max(out,1)

        total += labels.size(0)
        correct += (p==labels).sum().item()

        preds_all += p.cpu().tolist()
        labels_all += labels.cpu().tolist()

acc = 100*correct/total
print("TEST ACCURACY:",acc)

# Confusion Matrix
cm = confusion_matrix(labels_all,preds_all)
class_names = test_data.classes
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()


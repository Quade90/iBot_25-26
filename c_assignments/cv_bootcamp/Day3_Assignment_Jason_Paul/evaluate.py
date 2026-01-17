import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"

# TRANSFORMS
tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_data = datasets.ImageFolder("data/test", transform=tf)
loader = DataLoader(test_data, batch_size=32, shuffle=True)

# MODEL
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features,2)
model.load_state_dict(torch.load("best_model.pth"))
model.to(device)
model.eval()

correct,total = 0,0
preds_all, labels_all = [], []

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

# CONFUSION MATRIX
cm = confusion_matrix(labels_all,preds_all)
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()

# Example predictions
import random

class_names = test_data.classes  # ['cats','dogs']

correct_imgs = []
wrong_imgs = []

cat_correct = 0
dog_correct = 0
cat_wrong = 0
dog_wrong = 0

with torch.no_grad():
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs,1)

        for i in range(len(imgs)):
            img = imgs[i].cpu()
            true = labels[i].item()
            pred = preds[i].item()

            # CORRECT
            if pred == true:
                if true == 0 and cat_correct < 2:
                    correct_imgs.append((img,pred,true))
                    cat_correct += 1

                if true == 1 and dog_correct < 3:
                    correct_imgs.append((img,pred,true))
                    dog_correct += 1

            # WRONG
            else:
                if true == 0 and cat_wrong < 3:
                    wrong_imgs.append((img,pred,true))
                    cat_wrong += 1

                if true == 1 and dog_wrong < 2:
                    wrong_imgs.append((img,pred,true))
                    dog_wrong += 1

        if len(correct_imgs) == 5 and len(wrong_imgs) == 5:
            break


def show_correct_wrong(correct_imgs, wrong_imgs, class_names):
    plt.figure(figsize=(18,8))

    # ---- TOP ROW: CORRECT ----
    for i,(img,pred,true) in enumerate(correct_imgs):
        plt.subplot(2,5,i+1)
        img = img.permute(1,2,0)
        img = img*0.229 + 0.485  # unnormalize approx
        plt.imshow(img)
        plt.axis("off")

        plt.title(
            f"Pred: {class_names[pred]}\n"
            f"Actual: {class_names[true]}",
            fontsize=10,
            color="green"
        )

    # ---- BOTTOM ROW: WRONG ----
    for i,(img,pred,true) in enumerate(wrong_imgs):
        plt.subplot(2,5,i+6)
        img = img.permute(1,2,0)
        img = img*0.229 + 0.485
        plt.imshow(img)
        plt.axis("off")

        plt.title(
            f"Pred: {class_names[pred]}\n"
            f"Actual: {class_names[true]}",
            fontsize=10,
            color="red"
        )

    plt.suptitle("Model Predictions\nTop: Correct | Bottom: Incorrect",
                 fontsize=18)

    plt.subplots_adjust(hspace=0.2)
    plt.savefig("prediction_examples.png")
    plt.show()

show_correct_wrong(correct_imgs, wrong_imgs, class_names)


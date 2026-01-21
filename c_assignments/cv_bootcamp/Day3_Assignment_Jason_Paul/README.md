# Cat or Dog Detector (Convolutional Neural Network (CNN))

## Overview
This project processes an **image** of a cat or a dog and identifies the animal in the picture using a neural network. The neural network is trained on multiple images of dogs and cats, the best model is selected and used for detection. The neural network chosen can be either **`ResNet-18`** or **`MobileNetV2`**. Statistics and model are saved for further use.

---

## How to Run

### 1. Start the **`train.py`** program
Run the train.py python script.

### 2. Choose model
When prompted, choose the model version user would prefer to use.
- Type **resnet18** for **`ResNet-18`**
- Type **mobilenet** for **`MobileNetV2`**
Upon choosing model, the program will train the model on dog and cat images from the training dataset, further fine-tune best model using training dataset and validate using new images from the validation dataset. Statistics are produced in the form of **training curves** and **fine-tuning curves**. All statistics and fine-tuned model will be saved.

### 3. Start the **`evaluate.py`** program
Run the evaluate.py python script.
The program evaluates the model saved from training using new images from the testing dataset. After evaluating a **confusion matrix** is created based on performance of model and saved.

### Files created:

- `best_model.pth` – Best model saved after initial training  
- `best_finetuned.pth` – Best model saved after fine-tuning  
- `training_curves.png` – Training and validation loss/accuracy graph  
- `finetune_curves.png` – Fine-tuning loss and accuracy graph  
- `confusion_matrix.png` – Confusion matrix on test dataset  
- `train.py` – Training and fine-tuning script  
- `evaluate.py` – Model evaluation script  
- `README.md` – Project documentation

---

## Final Test Accuracy
- Test Accuracy: **96.00%**

---

## Data Augmentation Methods:
- `RandomResizedCrop(224)`
    - Random zoom + random crop
- `RandomHorizontalFlip()`
    - Flips image left↔right
- `RandomRotation(15)`
    - Rotates image between `-15°` to `+15°`
- `ColorJitter(...)`
    - Random changes to brightness, contrast, saturation and hue.
- `RandomAffine(10)`
    - Random shifts, scaling, shearing

---

## Learning Rate Schedule
### Initial training phase
I used **ReduceLROnPlateau** as the learning rate scheduler.

- Initial learning rate: `0.001`
- Scheduler monitors: **validation loss**
- Patience: `3` epochs  
- Reduction factor: `0.5`

### Fine-tuning phase
During fine-tuning:
- Backbone layers: `5e-6`
- Classifier layers: `1e-4`
- Weight decay: `1e-4`

A separate scheduler was used with:
- Patience: `2`
- Reduction factor: `0.3`

---

## Observations
- There is a sweet spot when it comes to learning rate. Slightly too much or too less can make a lot of difference.
- **`ResNet-18`** seems to perform slightly better than **`MobileNetV2`**.
- Training accuracy quite heavily depends on the data augmentations present.

---

## Challenges Faced
- Installed the CPU version of PyTorch initially and had to reinstall the correct CUDA-enabled version.
- Overfitting issues where training accuracy increased but validation accuracy plateaued.
- Difficulty selecting an optimal learning rate.
- Scheduler confusion – ReduceLROnPlateau did not trigger initially because validation loss kept improving slightly.

---

## Bonus challenges
### Bonus 1: Fine-Tuning
After initial training, selected deeper layers of the pretrained network were unfrozen and trained with a lower learning rate. This allowed the model to adapt high-level features to the dataset while preserving useful pretrained representations, resulting in improved generalization.

### Bonus 2: **`MobileNetV2`** Comparison
Gave the user ability to use the `MobileNetV2` model along with `ResNet-18`.

### Bonus 3: Visualizing Predictions
Created grid of `10` correct predictions and `10` incorrect predictions with labels to give a visual idea of the models accuracy.

---

## Dependencies
- Python 3.8+
- torch
- torchvision
- numpy
- matplotlib
- scikit-learn
- seaborn
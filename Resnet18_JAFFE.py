from __future__ import print_function, division
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor
import torch.nn as nn
from PIL import Image
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd

cudnn.benchmark = True
plt.ion()
import torch.nn.functional as F


##obtaining data
# os.chdir('/Users/sofia/Downloads/')

files_jaffe = [
    f for f in os.listdir("jaffedbase") if os.path.isfile(os.path.join("jaffedbase", f))
]
files_jaffe.remove(".DS_Store")
emotion_raw = [files_jaffe[i].split(".")[1][:2] for i in range(len(files_jaffe))]
dict_emotions = {"AN": 0, "DI": 1, "FE": 2, "HA": 3, "SA": 4, "SU": 5, "NE": 6}
emotions = [*map(dict_emotions.get, emotion_raw)]

##functions for tensors
def create_image(pixels):
    img = Image.fromarray(pixels)
    img = img.convert("RGB")
    return img


def data_transforms():
    transf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transf


## processing data

array_images = [
    create_image(plt.imread("jaffedbase/" + files_jaffe[i]))
    for i in range(len(files_jaffe))
]
trans = data_transforms()
transforms_images = [trans(array_images[i]) for i in range(len(array_images))]

list_samples = []
for i in range(len(transforms_images)):
    sample = transforms_images[i], emotions[i]
    list_samples.append(sample)

## creating loaders
training_data = list_samples[:170]
val_data = list_samples[170:]
nb_samples_train = len(training_data)
nb_samples_val = len(val_data)
batch_size_train = 10
batch_size_val = 10
train_loader = DataLoader(training_data, batch_size=batch_size_train, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size_val, shuffle=True)
dataloaders = {"train": train_loader, "val": val_loader}

## model train


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_sizes = {"train": len(training_data), "val": len(val_data)}


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


##FINETUNING
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 7, bias=False)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.1)

##training
model_ft = train_model(
    model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25
)


torch.save(model_ft, "Resnet18_model_JAFFE_two.pt")

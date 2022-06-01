# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jupytext//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# #### In this notebook, we train a neural network on AMASS output to get an acceptable accuracy in movement classification problem

import pandas as pd
import sys
import os
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly
from sklearn.decomposition import PCA
import seaborn as sns
from os.path import dirname, join as pjoin
import scipy.io as sio
sys.path.insert(0, '/home/arefeh/Motion-Project/My Project/my_project/utils')
from DLC_functions import *
from sklearn.model_selection import train_test_split
# get some pytorch:
import torch
import torch.nn as nn
from torch.nn import MaxPool1d
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# confusion matrix from sklearn
from sklearn.metrics import confusion_matrix
# to get some idea of how long stuff will take to complete:
import time
# to see how unbalanced the data is:
from collections import Counter

# ### Load Data

Input_array = np.load("../data/03_processed/AMASS_velocity/Input_model.npy")
Labels = np.load("../data/03_processed/AMASS_velocity/labels.npy")
Subjects = np.load("../data/03_processed/AMASS_velocity/subjects.npy")
Labels_name = np.load("../data/03_processed/AMASS_velocity/labels_name.npy")
Labels_name.shape

Input_array = np.load("../data/03_processed/AMASS_resamplingvelocity/Input_model.npy")
Labels = np.load("../data/03_processed/AMASS_resamplingvelocity/labels.npy")
Subjects = np.load("../data/03_processed/AMASS_resamplingvelocity/subjects.npy")
Labels_name = np.load("../data/03_processed/AMASS_resamplingvelocity/labels_name.npy")
Labels_name.shape

plt.figure(figsize=(8,6))
plt.hist(Labels_name)
plt.xticks(rotation=90)
plt.show()


# h = np.histogram(Labels_name, np.unique(Labels_name))
plt.figure(figsize=(8,6))
plt.hist(Labels_name)
plt.xticks(rotation=90)
plt.show()


# +

Motions_train, Motions_test, labels_train, labels_test = train_test_split(Input_array, Labels,
                                                                          test_size=0.33, random_state=42)

class MotionDataset(Dataset):
    def __init__(self, train=True):
#         self.subjects = Subjects
        
        if train:
            self.labels = labels_train
            self.input_array = Motions_train
        else:
            self.labels = labels_test
            self.input_array = Motions_test
            

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (np.float32(np.squeeze(self.input_array[idx,:,:])), self.labels[idx])

        return sample


# +
motion_train = MotionDataset(train=True)
motion_test = MotionDataset(train=False)

print('TRAINING:')
for idx in range(len(motion_train)):
    pass
print(idx)
print(motion_train[idx][0].shape, Labels_name[motion_train[idx][1]])
print('\nTESTING:')
for idx in range(len(motion_test)):
    pass
print(idx)
print(motion_test[idx][0].shape, Labels_name[motion_test[idx][1]])

# -

# ### “1D” CNN in pytorch expects a 3D tensor as input: BxCxT

# +
# Hyperparameters
num_epochs = 200
num_classes = np.unique(Labels_name).shape[0]
batch_size = 100
learning_rate = 0.001

# Create training and test datasets
motion_train = MotionDataset(train=True)
motion_test = MotionDataset(train=False)

# Data loader
train_loader = DataLoader(dataset=motion_train, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=motion_test, batch_size=batch_size, shuffle=False)

# -

# ### Specific Model

class Mov1DCNN(nn.Module):
    def __init__(self):
        
        super(Mov1DCNN, self).__init__()

        self.layer1 = nn.Sequential(
          nn.Conv1d(in_channels=208, out_channels=124, kernel_size=6, stride=2),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
          nn.Conv1d(in_channels=124, out_channels=31, kernel_size=1),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2))

        self.dropout1 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(496, 2000)  # fix dimensions
        self.nl = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(2000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
#         print(out.size())
        out = self.layer2(out)
#         print(out.size(0))
        out = out.reshape(out.size(0), -1)
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.nl(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        # pick the most likely class:
        out = nn.functional.log_softmax(out, dim=1)
    
        return out


# +
### ADDING GPU ###
device = "cpu"

# create the model object:
model = Mov1DCNN()


# loss and optimizer:
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# -

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (motions, labels) in enumerate(train_loader):
        motions, labels = motions.to(device), labels.to(device)

    # Run the forward pass
        outputs = model(motions)
        loss = criterion(outputs, labels)
        # add l2 regularization
#         l2_lambda = 0.001
#         l2_norm = sum(p.pow(2.0).sum()for p in model.parameters())
#         loss = loss + l2_lambda * l2_norm
        # add l1 regularization
        l1_lambda = 0.001
        l1_norm = sum(abs(p).sum()for p in model.parameters())
        loss = loss + l1_lambda * l1_norm
        loss_list.append(loss.item())

    # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], "
                  f"Loss: {loss.item():.4f}, "
                  f"Accuracy: {((correct / total) * 100):.2f}%")

plt.plot(loss_list)
plt.xlabel("steps")
plt.ylabel("loss")
plt.show()
len(loss_list)

# Test the model
model.eval()
real_labels, predicted_labels = [], []
with torch.no_grad():
    correct = 0
    total = 0
    for motions, labels in test_loader:
        motions, labels = motions.to(device), labels.to(device)
        real_labels += list(labels)
        outputs = model(motions)
        _, predicted = torch.max(outputs.data, 1)
        predicted_labels += list(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test Accuracy of the model on the test moves: {(correct / total)*100:.3f}%")

# +
# chance level: 3.5%
# -



def plotConfusionMatrix(real_labels, predicted_labels, label_names):
    real_labels = [int(x) for x in real_labels]
    predicted_labels = [int(x) for x in predicted_labels]
    tick_names = [a.replace("_", " ") for a in label_names]
    cm = confusion_matrix(real_labels, predicted_labels, normalize='true')
    fig = plt.figure(figsize=(8,10))
    plt.imshow(cm)
    plt.xticks(range(len(tick_names)),tick_names, rotation=90)
    plt.yticks(range(len(tick_names)),tick_names)
    plt.xlabel('predicted move')
    plt.ylabel('real move')
    plt.show()
    return(cm)


# # padding with velocity 77% acc

#
name_labels = np.unique(Labels_name)
cm = plotConfusionMatrix(real_labels, predicted_labels, name_labels)

# # resampling with velocity 74% acc

name_labels = np.unique(Labels_name)
cm = plotConfusionMatrix(real_labels, predicted_labels, name_labels)

# # Truncation acc 30 %

# # Padding acc 75 %

# +

name_labels = np.unique(Labels_name)
cm = plotConfusionMatrix(real_labels, predicted_labels, name_labels)
# -

# ## Delete random movements in Resampling 71% acc

# +

name_labels = np.unique(Labels_name)
cm = plotConfusionMatrix(real_labels, predicted_labels, name_labels)
# -

len(name_labels)

# +
# regularization  69% acc

name_labels = np.unique(Labels_name)
cm = plotConfusionMatrix(real_labels, predicted_labels, name_labels)
# -

# ## Result of Truncation  45 % acc

# +

name_labels = np.unique(Labels_name)
cm = plotConfusionMatrix(real_labels, predicted_labels, name_labels)
cm.shape
# -

# ## Result of padding 51% acc
# 598 data samples

name_labels = np.unique(Labels_name)
cm = plotConfusionMatrix(real_labels, predicted_labels, name_labels)

# # Delete random movements in Padding 69% acc

name_labels = np.unique(Labels_name)
cm = plotConfusionMatrix(real_labels, predicted_labels, name_labels)

## regualriztion 66 % acc
name_labels = np.unique(Labels_name)
cm = plotConfusionMatrix(real_labels, predicted_labels, name_labels)

name_labels

name_labels

np.unique(predicted_labels)

Labels_name[motion_test[idx][1]]

motion_test



# ### try different kernel sizes

# +
class Mov1DCNN(nn.Module):
    def __init__(self,kernelsize,in_fc1_size):
        
        super(Mov1DCNN, self).__init__()

        self.layer1 = nn.Sequential(
          nn.Conv1d(in_channels=28, out_channels=124, kernel_size=kernelsize, stride=2),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
          nn.Conv1d(in_channels=124, out_channels=31, kernel_size=1),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2))

        self.dropout1 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(in_fc1_size, 2000)  # fix dimensions
        self.nl = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(2000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
#         print(out.size())
        out = self.layer2(out)
#         print(out.size(0))
        out = out.reshape(out.size(0), -1)
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.nl(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        # pick the most likely class:
        out = nn.functional.log_softmax(out, dim=1)
    
        return out
    
device = "cpu"
criterion = nn.CrossEntropyLoss()

kernels = [2,3,4,5,6,7]
in_fc1 = [124,124,124,93,93,93]
acc_kernels = []
for i,k in enumerate(kernels):
    print("kernel size:  ", k)
    model = Mov1DCNN(kernelsize = k,in_fc1_size = in_fc1[i])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (motions, labels) in enumerate(train_loader):
            motions, labels = motions.to(device), labels.to(device)

    # Run the forward pass
            outputs = model(motions)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

    # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

#             if (i + 1) % 2 == 0:
#                 print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], "
#                       f"Loss: {loss.item():.4f}, "
#                       f"Accuracy: {((correct / total) * 100):.2f}%")
    model.eval()
    real_labels, predicted_labels = [], []
    with torch.no_grad():
        correct = 0
        total = 0
        for motions, labels in test_loader:
            motions, labels = motions.to(device), labels.to(device)
            real_labels += list(labels)
            outputs = model(motions)
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels += list(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc_kernels.append((correct / total)*100)
        print(f"Test Accuracy of the model on the test moves: {(correct / total)*100:.3f}%")
        name_labels = np.unique(Labels_name)
        cm = plotConfusionMatrix(real_labels, predicted_labels, name_labels)
# -

acc_kernels

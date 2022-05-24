
"""collection of functions for model instances"""

from os.path import dirname, join as pjoin
import os
import sys
import time
from collections import Counter
import math

# import dlc2kinematics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import scipy.io as sio
import torch
import torch.nn as nn
from torch.nn import MaxPool1d
from torch.utils.data import Dataset
from torch.utils.data import DataLoader




class Mov1DCNN(nn.Module):
    def __init__(self):
        
        super(Mov1DCNN, self).__init__()

        self.layer1 = nn.Sequential(
          nn.Conv1d(in_channels=28, out_channels=250, kernel_size=6, stride=2),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
          nn.Conv1d(in_channels=250, out_channels=124, kernel_size=2),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2))

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(9672, 2000)  
        self.fc2 = nn.Linear(2000, 1000)
        # num_classes = 21
        self.fc3 = nn.Linear(1000, 21)

    def forward(self, x):
        out = self.layer1(x)
#         print(out.size())
        out = self.layer2(out)
#         print(out.size(0))
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        # pick the most likely class:
        out = nn.functional.log_softmax(out, dim=1)
    
        return out




class MotionDataset(Dataset):
    def __init__(self, train=True, data_dict = data_dict ):
        self.input_dict = data_dict
#         self.subjects = Subjects
        
        self.motions_train, self.motions_test, self.labels_train, self.labels_test = train_test_split(self.input_dict['input_model'],
                                                                        self.input_dict['labels'],test_size=0.33, random_state=42)
        
        
        if train:
            self.labels = self.labels_train
            self.input_array = self.motions_train
        else:
            self.labels = self.labels_test
            self.input_array = self.motions_test
            

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (np.float32(np.squeeze(self.input_array[idx,:,:])), self.labels[idx])

        return sample



class Run_model():
    def __init__(self,model,input_dict, reg ="l1"): 
        # super(Run_model,self).__init__()
        self.model = model
        self.device = "cpu"
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.reg = reg
        # Hyperparameters
        self.num_epochs = 200
        self.num_classes = np.unique(input_dict['labels_name']).shape[0]
        self.batch_size = 100
        self.input_dict = input_dict
        motion_train = MotionDataset(train=True, data_dict = self.input_dict )
        motion_test = MotionDataset(train=False, data_dict = self.input_dict)
        # Data loader
        self.train_loader = DataLoader(dataset= motion_train, batch_size=self.batch_size, shuffle=True)
        self.test_loader  = DataLoader(dataset=motion_test, batch_size=self.batch_size, shuffle=False)

    def train(self):
        # motion_train = MotionDataset(train=True, data_dict = self.input_dict )
        total_step = len(self.train_loader)
        loss_list = []
        acc_list = []
        for epoch in range(self.num_epochs):
            for i, (motions, labels) in enumerate(self.train_loader):
                motions, labels = motions.to(self.device), labels.to(self.device)

                # Run the forward pass
                outputs = self.model(motions)
                self.loss = self.loss_fn(outputs, labels)

                # add regularization
                reg_lambda = 0.001
                if self.reg == "l2":
                    reg_norm = sum(p.pow(2.0).sum()for p in self.model.parameters())
                    
                if self.reg == "l1":
                    reg_norm = sum(abs(p).sum()for p in self.model.parameters())
                    
                self.loss = self.loss + reg_lambda * reg_norm
                loss_list.append(self.loss.item())

            # Backprop and perform Adam optimization
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

            # Track the accuracy
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                acc_list.append(correct / total)

                if (i + 1) % 2 == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{total_step}], "
                        f"Loss: {self.loss.item():.4f}, "
                        f"Accuracy: {((correct / total) * 100):.2f}%")


        # plt.plot(loss_list)
        # plt.xlabel("steps")
        # plt.ylabel("loss")
        # plt.show()
        # len(loss_list)

    def test(self):
        self.model.eval()
        self.real_labels, self.predicted_labels = [], []
        with torch.no_grad():
            correct = 0
            total = 0
            for motions, labels in self.test_loader:
                motions, labels = motions.to(self.device), labels.to(self.device)
                self.real_labels += list(labels)
                outputs = self.model(motions)
                _, predicted = torch.max(outputs.data, 1)
                self.predicted_labels += list(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Accuracy of the model on the test moves: {(correct / total)*100:.3f}%")

    
    def plotConfusionMatrix(self):
        self.real_labels = [int(x) for x in self.real_labels]
        self.predicted_labels = [int(x) for x in self.predicted_labels]
        labels_name = np.unique(self.input_dict['labels_name'])
        tick_names = [a.replace("_", " ") for a in labels_name]
        cm = confusion_matrix(self.real_labels, self.predicted_labels, normalize='true')
        plt.figure(figsize=(8,10))
        plt.imshow(cm)
        plt.xticks(range(len(tick_names)),tick_names, rotation=90)
        plt.yticks(range(len(tick_names)),tick_names)
        plt.xlabel('predicted move')
        plt.ylabel('real move')
        plt.show()
        return(cm)
        

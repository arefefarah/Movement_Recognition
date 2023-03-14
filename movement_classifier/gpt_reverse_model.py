
"""collection of functions for model instances"""

# import data

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
import rsatoolbox.data as rsd 
import rsatoolbox
from sklearn.manifold import TSNE
import seaborn as sns
import scipy.io as sio
import torch
import torch.nn as nn
from torch.nn import MaxPool1d
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ReverseMov1DCNN(nn.Module):
    def __init__(self, num_classes,maxpool_indices):
        super(ReverseMov1DCNN, self).__init__()
        
        self.num_classes = num_classes
        self.maxpool_indices = maxpool_indices
        self.fc3 = nn.Linear(num_classes, 1000)
        self.fc2 = nn.Linear(1000, 2000)
        self.fc1 = nn.Linear(2000, 9672)
        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=124, out_channels=250, kernel_size=2),
            nn.ReLU()
        )
        
        self.layer1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=250, out_channels=28, kernel_size=6, stride=2),
            nn.ReLU()
        )
        
        self.unpool2 = nn.MaxUnpool1d(kernel_size=2, stride=2)
        self.unpool1 = nn.MaxUnpool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        out = self.fc3(x)
        out = self.fc2(out)
        out = self.fc1(out)
        print(out.size())
        out = out.reshape(out.size(0), 124, 78)
        
        out = self.unpool2(out,self.maxpool_indices[1])
        out = self.layer2(out)
        
        out = self.unpool1(out, self.maxpool_indices[0])
        out = self.layer1(out)
        
        return out




class MotionDataset(Dataset):
    def __init__(self, data_dict,train=True):
        self.input_dict = data_dict
#         self.subjects = Subjects
        
        self.motions_train, self.motions_test, self.labels_train, self.labels_test = train_test_split(self.input_dict['input_model'],
                                                                        self.input_dict['labels'],test_size=0.30, random_state=42)
        

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
        sample = (np.float32(np.squeeze(self.input_array[idx,:])), self.labels[idx])
        return sample
    
    


"""ModelHandler"""
class ModelHandler():
    def __init__(self,model,input_dict, reg ="l1" ): 
        # super(Run_model,self).__init__()
        self.model = model
        self.device = "cpu"
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.reg = reg
        # Hyperparameters
        self.num_epochs = 200
        # self.num_classes = np.unique(input_dict['labels_name']).shape[0]
        self.num_classes = 20
        self.batch_size = 100
        self.input_dict = input_dict
        self.motion_train = MotionDataset(data_dict = self.input_dict,train=True )
        self.motion_test = MotionDataset(data_dict = self.input_dict,train=False)
        # Data loader
        self.train_loader = DataLoader(dataset= self.motion_train, batch_size=self.batch_size, shuffle=True)
        self.test_loader  = DataLoader(dataset=self.motion_test, batch_size=self.batch_size, shuffle=False)
        self.activation = {}
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.input_dict['labels_name'])
       

    def train(self):
        total_step = len(self.train_loader)
        loss_list = []
        acc_list = []
        for epoch in range(self.num_epochs):
            self.real_train_labels =[]
            for i, (motions, labels) in enumerate(self.train_loader):
                motions, labels = motions.to(self.device), labels.to(self.device)
                
                # print("motions",motions.shape)
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
            # print(len(self.real_train_labels))


        # plt.plot(loss_list)
        # plt.xlabel("steps")
        # plt.ylabel("loss")
        # plt.show()
        # len(loss_list)

    def test(self):
        self.model.eval()
        self.real_test_labels, self.predicted_labels = [], []
        with torch.no_grad():
            correct = 0
            total = 0
            for motions, labels in self.test_loader:
                motions, labels = motions.to(self.device), labels.to(self.device)
                self.real_test_labels += list(labels)
                outputs = self.model(motions)
                _, predicted = torch.max(outputs.data, 1)
                self.predicted_labels += list(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Accuracy of the model on the test moves: {(correct / total)*100:.3f}%")
        return((correct / total)*100)

    def layer_extractor(self,train = False):
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.detach()
            return hook

        self.model.fc1.register_forward_hook(get_activation(name = 'fc1'))
        self.model.fc2.register_forward_hook(get_activation(name ='fc2'))
        self.model.fc3.register_forward_hook(get_activation(name = 'fc3'))
        if train:
            self.real_train_labels=[]
            self.traindata_loader = DataLoader(dataset= self.motion_train, batch_size=self.batch_size, shuffle=False)
            for motions, labels in self.traindata_loader:
                m, l = motions.to(self.device), labels.to(self.device)
                self.real_train_labels += list(l)
            d =vars(self.motion_train)
            labels_name = self.le.inverse_transform(self.real_train_labels)
        else:
            d =vars(self.motion_test)
            labels_name = self.le.inverse_transform(self.real_test_labels)
        x = d["input_array"]
        output = self.model(torch.Tensor(x))
        self.activation["input"] = torch.Tensor(x)
        
        return self.activation,labels_name

    def save_layerOutput(self):
        np.save("../data/03_processed/test_input.npy", self.activation["input"])
        np.save("../data/03_processed/fc1-out.npy", self.activation["fc1"])
        np.save("../data/03_processed/fc2-out.npy", self.activation["fc2"])
        np.save("../data/03_processed/fc3-out.npy", self.activation["fc3"])


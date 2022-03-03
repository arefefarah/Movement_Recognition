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

# #### In this notebook, we train a neural network on Deeplabcut output to get an acceptable accuracy in movement classification problem

# +
sys.path.insert(0, '../')
import movement_classifier.utils as utils


from os.path import dirname, join as pjoin
import os
import sys
import math
import time
from collections import Counter


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import rsatoolbox
import rsatoolbox.data as rsd # abbreviation to deal with dataset


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.utils.data as data


# -

# ### Load Data

# ##### Padding to make eaual length of time series for each movement and delete random movement chosen by participants

Input_array = np.load("../data/03_processed/padding_deletedrandom/Input_model.npy")
Labels = np.load("../data/03_processed/padding_deletedrandom/labels.npy")
Subjects = np.load("../data/03_processed/padding_deletedrandom/subjects.npy")
Labels_name = np.load("../data/03_processed/padding_deletedrandom/labels_name.npy")
movements = np.unique(Labels_name)
Labels_name.shape
Input_array.shape

movements = np.unique(Labels_name)

# ##### Resampling

Input_array = np.load("../data/03_processed/resampling_deletedrandom/Input_model.npy")
Labels = np.load("../data/03_processed/resampling_deletedrandom/labels.npy")
Subjects = np.load("../data/03_processed/resampling_deletedrandom/subjects.npy")
Labels_name = np.load("../data/03_processed/resampling_deletedrandom/labels_name.npy")
Input_array.shape

selected_movements = np.where(Labels_name == "walking")
    # print(selected_movements)
Data = np.dstack(Input_array[selected_movements,:])
Data.shape


def avg_movements(Input,Labels_name,movements):
    for m in movements:
        selected_movements = np.where(Labels_name == m)
    # print(selected_movements)
        Data = np.dstack(Input[selected_movements,:])
    
        if m== movements[0]:
            data_all_movement = np.dstack(np.mean(Data, axis = 0))
        else:
            B = np.dstack(np.mean(Data, axis = 0))
            data_all_movement = np.concatenate([data_all_movement,B])

    return(data_all_movement)


rdm.pattern_descriptors["label"] = np.unique(Labels_name)

# +
indexofmotions = {}
for i in range(Labels.shape[0]):
    k = Labels[i]
    if k in indexofmotions:
        pass
    else:
        indexofmotions[k] = Labels_name[i]
        
indexofmotions   

# +
Motions_train, Motions_test, labels_train, Labels_name_test = train_test_split(Input_array, Labels_name,
                                                                          test_size=0.33, random_state=21)
Motions_train, Motions_test, labels_train, labels_test = train_test_split(Input_array, Labels,
                                                                          test_size=0.33, random_state=21)




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

# +
class Mov1DCNN(nn.Module):
    def __init__(self):
        
        super(Mov1DCNN, self).__init__()

        self.layer1 = nn.Sequential(
          nn.Conv1d(in_channels=28, out_channels=124, kernel_size=6, stride=2),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
          nn.Conv1d(in_channels=124, out_channels=22, kernel_size=2),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2))

       

        self.dropout1 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(1496, 2000)  # fix dimensions
        self.nl = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(2000, 1000)
        self.n2 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size(0))
        out = out.reshape(out.size(0), -1)
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.nl(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.n2(out)
        out = self.dropout3(out)
        out = self.fc3(out)
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

# ## Let's start register hooks

test_loader  = DataLoader(dataset=motion_test, batch_size=278, shuffle=False)

# +
model.eval()
visualisation = {}

def hook_fn(m, i, o):
#     print(m)
    visualisation[m] = o 

def get_all_layers(net):
    for name, layer in net._modules.items():
#         layer.register_forward_hook(hook_fn)
    #If it is a sequential, don't register a hook on it
    # but recursively register hook on all it's module children
        if isinstance(layer, nn.Sequential):
            get_all_layers(layer)
            
        else:
#       # it's a non sequential. Register a hook

            layer.register_forward_hook(hook_fn)

get_all_layers(model)


for motions, labels in test_loader:
        motions, labels = motions.to(device), labels.to(device)
        
#         real_labels += list(labels)
        out = model(motions)
       
# Just to check whether we got all layers

visualisation.keys()
# -

for u in visualisation.keys():
    print(u)

Net_layers = ["conv1","relu1","maxpool1","conv2","relu2","maxpool2","droput1","fc1","relufc1","dropout2","fc2","relufc2","dropout3","fc3"]
out_intermidite_layers = {}
count = 0
for t in visualisation.keys():
#     print(t)
#     print(count)
    out_intermidite_layers[Net_layers[count]] = visualisation[t]
    count+=1



# +
data = Motions_test
conds = movements
# shape = (1,20,28,554)
Data = avg_movements(data,Labels_name_test,movements)
Data = Data.reshape(20,Data.shape[1]*Data.shape[2])
obs_des = {"conds":conds}
des = {'subj': all}
data = rsd.Dataset(measurements=Data,
                    descriptors=des,
                    obs_descriptors=obs_des
                    )

title = "Input for all subjects"
rdm = rsatoolbox.rdm.calc_rdm(data, method="correlation", descriptor=None, noise=None)


fig, ax, ret_val = rsatoolbox.vis.show_rdm(rdm,
                        figsize=(10,10),
                        show_colorbar='panel',
                        rdm_descriptor=title,
                        pattern_descriptor='index',
                        icon_spacing=1)

tick_names = [a.replace("_", " ") for a in conds]
plt.show()
 
fig.savefig('../reports/figures/allsubjects_input_rdm.png', bbox_inches='tight', dpi=300)


# -

def plot_rdm_layer(layer, method ):
    data = out_intermidite_layers[layer]
    conds = movements
    # shape = (1,20,28,554)
    # with avar
    Data = avg_movements(data.detach().numpy(),Labels_name_test,movements)
    Data = Data.reshape(20,Data.shape[1]*Data.shape[2])
    obs_des = {"conds":conds}
    des = {'subj': all}
    data = rsd.Dataset(measurements=Data,
                        descriptors=des,
                        obs_descriptors=obs_des
                        )

    title = "output of {} layer".format(layer)+" for all subjects"
    rdm = rsatoolbox.rdm.calc_rdm(data, method=method, descriptor=None, noise=None)


    fig, ax, ret_val = rsatoolbox.vis.show_rdm(rdm,
                            figsize=(10,10),
                            show_colorbar='panel',
                            rdm_descriptor=title,
                            pattern_descriptor='index',
                            icon_spacing=1)

    tick_names = [a.replace("_", " ") for a in conds]

    plt.show()
    fig.savefig('../reports/figures/allsubjects_{}_rdm.png'.format(layer), bbox_inches='tight', dpi=300) 


methods = 'euclidean' , "mahalanobis" , "correlation "
for l in Net_layers:
    plot_rdm_layer(l,method = "correlation")


# +
## save output of fully connected layers for further analysis:

out = Motions_test.reshape(278,28*554)
with open('../data/03_processed/input_test.pkl','wb') as f:
    pkl.dump(out, f)
out = out_intermidite_layers["fc1"].detach().numpy()
with open('../data/03_processed/fc1_out.pkl','wb') as f:
    pkl.dump(out, f)
out = out_intermidite_layers["fc2"].detach().numpy()
with open('../data/03_processed/fc2_out.pkl','wb') as f:
    pkl.dump(out, f)
out = out_intermidite_layers["fc3"].detach().numpy()
with open('../data/03_processed/fc3_out.pkl','wb') as f:
    pkl.dump(out, f)
with open('../data/03_processed/labels_name_test.pkl','wb') as f:
    pkl.dump(Labels_name_test, f)
    labels_test
with open('../data/03_processed/labels_test.pkl','wb') as f:
    pkl.dump(labels_test, f)

# -

from MapExtrackt import FeatureExtractor
fe = FeatureExtractor(model)

# fe(Input_array[0])
help(fe)

# #### compare RDMs for different layers

# +
obs_des = {"conds":conds}
des = {'subj': all}
data = out_intermidite_layers["fc1"]
conds = movements
Data = avg_movements(data.detach().numpy(),Labels_name_test,movements)
Data = Data.reshape(20,Data.shape[1]*Data.shape[2])

data = rsd.Dataset(measurements=Data,
                    descriptors=des,
                    obs_descriptors=obs_des
                    )
rdm1 = rsatoolbox.rdm.calc_rdm(data, method="correlation", descriptor=None, noise=None)
data = out_intermidite_layers["fc2"]
conds = movements
Data = avg_movements(data.detach().numpy(),Labels_name_test,movements)
Data = Data.reshape(20,Data.shape[1]*Data.shape[2])

data = rsd.Dataset(measurements=Data,
                    descriptors=des,
                    obs_descriptors=obs_des
                    )
rdm2 = rsatoolbox.rdm.calc_rdm(data, method="correlation", descriptor=None, noise=None)
rsatoolbox.rdm.compare_cosine(rdm1,rdm2)
rsatoolbox.rdm.compare_correlation(rdm1,rdm2)




# -

# ### plot RDM

rdm = rsatoolbox.rdm.calc_rdm(data, method='euclidean', descriptor=None, noise=None)
rsatoolbox.vis.show_rdm(rdm,figsize=(7,7),
                       show_colorbar='panel',
                        rdm_descriptor='fc1',
                       icon_spacing=.9)
label_names = np.unique(Labels_name)
tick_names = [a.replace("_", " ") for a in icon_name]
plt.xticks(range(len(tick_names)),tick_names, rotation=90)
plt.yticks(range(len(tick_names)),tick_names)
plt.show()



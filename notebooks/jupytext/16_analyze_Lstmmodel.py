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
#     display_name: base
#     language: python
#     name: python3
# ---

# +
import sys
sys.path.insert(0, '../')
import movement_classifier.utils as utils
import movement_classifier.data_loader as data_loader
import movement_classifier.model_funcs as model_funcs
import movement_classifier.gpt_reverse_model as gpt_reverse_model

from os.path import dirname, join as pjoin
import os
import sys
import math

import dlc2kinematics
# from sequitur.models import CONV_LSTM_AE
# from sequitur.models import LSTM_AE 
from sequitur import quick_train
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch.nn as nn
import numpy as np
from torch.nn import MSELoss
from matplotlib import animation
import copy
from IPython.display import HTML
from celluloid import Camera
# %matplotlib inline
import pandas as pd
import plotly.express as px
import torch
import plotly
from sklearn.decomposition import PCA
import seaborn as sns
import scipy.io as sio
# -

# load data

"""load dataframes for the modelling"""
path_file = "../data/03_processed/interpolation"
data_dict = data_loader.load_data_dict(path_file)
data_dict.keys()
# np.unique(data_dict["labels_name"])
data = data_dict['input_model']
train_input = torch.Tensor(data[0:1250,:,0:633])
#  train_Set should be ==>  [num_examples, seq_len, *num_features]
train_set  = train_input.permute(0,2,1)
val_input = torch.Tensor(data[1250:1319,:,0:633])
val_set  = val_input.permute(0,2,1)
val_set.shape


# Define structure of LSTM autoencode

# +
# model structure:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len, n_features = train_set.shape[1], train_set.shape[2]

class Encoder(nn.Module):

  def __init__(self):
    super(Encoder, self).__init__()

    self.lstm1 = nn.LSTM(input_size=28, hidden_size=14, num_layers=2, batch_first=True)
    self.lstm2 = nn.LSTM(input_size=14, hidden_size=7, num_layers=1, batch_first=True)
  
  def forward(self, x):
    x = x.reshape((1,633, 28))
    encoded, _ = self.lstm1(x)
    encoded, _ = self.lstm2(encoded)

    return encoded
  


class Decoder(nn.Module):

  def __init__(self):
    super(Decoder, self).__init__()

    self.lstm1 = nn.LSTM(input_size=7, hidden_size=14, num_layers=2, batch_first=True)
    self.lstm2 = nn.LSTM(input_size=14, hidden_size=28, num_layers=2, batch_first=True)

  def forward(self, x):
 
    decoded, _ = self.lstm1(x)
    decoded, _ = self.lstm2(decoded)

    return( decoded)
  
  
class RecurrentAutoencoder(nn.Module):

  def __init__(self):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder().to(device)
    self.decoder = Decoder().to(device)

  def forward(self, x):
    latant_x = self.encoder(x)
    # print(x.size, "   x size")
    reconstruct_x = self.decoder(latant_x)

    return reconstruct_x,latant_x


# -

# load trained model

# +

# to load saved model
saved_model = RecurrentAutoencoder()
saved_model.load_state_dict(torch.load('../movement_classifier/lstm_autoencoder.pth'))

# +

#run model on all data:

data = data_dict['input_model']
labels = data_dict["labels_name"]
# we want to delete these labels:  ["cross_legged_sitting" ,"sitting_down" ,"crawling" ]:
# ind1 = np.where(data_dict["labels_name"] == "cross_legged_sitting" )
# ind2 = np.where(data_dict["labels_name"] == "crawling")
# ind3 =np.where(data_dict["labels_name"] == "sitting_down" )
# ind =np.union1d(ind1,ind2)
# ind = np.union1d(ind,ind3)
# labels = np.delete(data_dict["labels_name"],ind)
# data =  np.delete(data_dict["input_model"],ind, axis = 0)


# data = data_dict['input_model']
data_input = torch.Tensor(data[:,:,0:633])
#  should be ==>  [num_examples, seq_len, *num_features]
data_input  = data_input.permute(0,2,1)
output = []
latant_space_data = []
with torch.no_grad():
    for data in data_input:
        reconst_x,latant_x = saved_model(data)
        latant_x = torch.squeeze(latant_x)
        latant_x = torch.permute(latant_x, (1, 0))
        latant_space_data.append(latant_x)
        output.append(reconst_x.squeeze())
        #plot
        # fig, axs = plt.subplots(nrows=7, figsize=(8, 40))
        # for i in range(7):
        #     axs[i].plot(latant_x[i])
        #     axs[i].set_title('Row {}'.format(i+1))
        # plt.tight_layout()
        # plt.show()

latant_3dtensors = torch.stack(latant_space_data)
reconst_out = torch.stack(output)

# -


np.array(reconst_out.view(1319,28,633)).shape

np.save("../data/03_processed/interpolation/reconstructRep_lstm.npy", np.array(reconst_out.view(1319,28,633)))


# Feature visualization

# +
input_size = np.array(latant_3dtensors.view(1319,7,633).shape)
input_size
latant_3d = np.array(latant_3dtensors)
latant_3d.shape

np.unique(labels).shape
legend_labels =[]
data = []
for l in np.unique(labels):
    # if l == "cross_legged_sitting" or l == "sitting_down" or l=="crawling"  :
    #     continue
    legend_labels.append(l)
    # print(l)
    ind = np.where(labels == l)
    mat_label = latant_3d[ind,:,:].squeeze()
    mat = np.mean(mat_label, axis = 0)
    data.append(mat)
   
data_array = np.array(data)
data_array.shape
# -

np.unique(legend_labels)

fig, axs = plt.subplots(7)
# fig.suptitle('Vertically stacked subplots')
for i in range(7):
    for s in range(len(np.unique(legend_labels))):
        axs[i].plot(data_array[s,i,100:500])


# +
colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'brown', 'gray', 'pink', 'teal', 'olive', 'navy', 'gold', 'lime', 'indigo', 'maroon', 'turquoise', 'salmon']

for i in range(7-1):
    for s in range(len(np.unique(legend_labels))):
        plt.plot(data_array[s,i,100:500],data_array[s,i+1,100:500],color=colors[s])
        
    plt.legend(legend_labels, bbox_to_anchor = (1.1,1.05), fontsize = "10")
    
    for s in range(len(np.unique(legend_labels))):
        plt.scatter(data_array[s,i,100],data_array[s,i+1,100],color=colors[s])
    plt.xlabel("Feature: "+str(i))
    plt.ylabel("Feature: "+str(i+1))
    plt.show()   
    #     plt.plot(data_array[s,i,:])
    # plt.legend(legend_labels, bbox_to_anchor = (1.1,1.05), fontsize = "10")
    # plt.xlabel("Time")
    # plt.ylabel("Feature: "+str(i))
    # plt.show()
# -

# PCA on Features

# +


latant_3d = np.array(latant_3dtensors)
latant_3d.shape
# labels = data_dict["labels_name"]

legend_labels =[]
data = []
i = 0
startpoint = []
for l in np.unique(labels):
    # if l == "cross_legged_sitting" or l == "sitting_down" or l=="crawling" or l== "" :
    #     continue
    legend_labels.append(l)
    # print(l)
    ind = np.where(labels == l)
    mat_label = latant_3d[ind,:,:].squeeze()
    mat = np.mean(mat_label, axis = 0)
    data.append(mat)
    
    z = PCA(n_components=2).fit_transform(mat.T)
    # plt.scatter(z[0,0],z[0,1])
    plt.plot(z[0:633,0],z[0:633,1], color = colors[i])
    startpoint.append((z[0,0],z[0,1]))
    # plt.plot(mat[0,:],mat[1,:])
    # plt.title(l)
    # plt.show()
    i+=1
startpoints = np.array(startpoint)
# print(startpoints.shape)


plt.legend(legend_labels, bbox_to_anchor = (1.1,1.05), fontsize = "10")
plt.scatter(startpoints[:,0], startpoints[:,1], color= colors)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
# -

mat

# +
latant_3d = np.array(latant_3dtensors)
latant_3d.shape
# labels = data_dict["labels_name"]

legend_labels =np.unique(labels)
mat = latant_3dtensors.view(1319,7*633)
print(mat.shape)
z = PCA(n_components=2).fit_transform(mat)

legend_labels = []
data = []
i = 0
for l in np.unique(labels):
    
    legend_labels.append(l)
#     # print(l)
    ind = np.where(labels == l)
    mat_label = z[ind,:].squeeze()
    # print(mat_label.shape)
    plt.scatter(mat_label[:,0],mat_label[:,1],color = colors[i])
#     # for vector in mat_label:
    i+=1
    mat = np.mean(mat_label, axis = 0)
    

plt.legend(legend_labels, bbox_to_anchor = (1.1,1.05), fontsize = "10")
plt.show()

# -

# PCA for each timepoints and then combine all in one plot with mean calculation for each single movement

# +
latant_3d = np.array(latant_3dtensors)
latant_3d.shape
# labels = data_dict["labels_name"]

# legend_labels =np.unique(labels)
# mat = latant_3dtensors.view(1319,7*633)
time_data =[]
for t in range(633):
    mat = latant_3d[:,:,t]  #1319*7
    # print(mat.shape)
    z = PCA(n_components=2).fit_transform(mat) # 1319*2
    # print(z.shape)
    time_data.append(z)

time_data = np.array(time_data)
legend_labels = []
data = []
i = 0
melabels = ["crawling", "walking"]
for l in np.unique(labels):
# for l in melabels: 
    legend_labels.append(l)
# #     # print(l)
    ind = np.where(labels == l)
    mat_label = time_data[:,ind,:].squeeze()
    d = np.mean(mat_label, axis = 1)
    # print("mean", d.shape)
    data.append(d)
    # r = np.array(d)

    plt.scatter(d[:,0],d[:,1],color = colors[i], s = 5)
# #     # for vector in mat_label:
    i+=1
#     mat = np.mean(mat_label, axis = 0)
    

plt.legend(legend_labels, bbox_to_anchor = (1.1,1.05), fontsize = "10")
# plt.show()

# -

# plot PCs for each single motion

# +

latant_3d = np.array(latant_3dtensors)
latant_3d.shape
# labels = data_dict["labels_name"]

legend_labels =[]
data = []
i = 0
for l in np.unique(labels):
    
    legend_labels.append(l)
    # print(l)
    ind = np.where(labels == l)
    mat_label = latant_3d[ind,:,:].squeeze()
    # print(mat_label.shape)
    # for vector in mat_label:

    mat = np.mean(mat_label, axis = 0)
    data.append(mat)
    print(mat.shape)
    # z = PCA(n_components=2).fit_transform(mat.T)
    # # plt.scatter(z[0,0],z[0,1])
    # plt.plot(z[0:200,0],z[0:200,1], color = colors[i])
    # # plt.plot(mat[0,:],mat[1,:])
    # # plt.title(l)
    # # plt.show()
    i+=1

data_array = np.array(data)
data_array.shape
# z = PCA(n_components=2).fit_transform(mat.T)

plt.legend(legend_labels, bbox_to_anchor = (1.1,1.05), fontsize = "10")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
# -

# plot PCs versus time

# +
input_size = np.array(latant_3dtensors.view(1319,7,633).shape)
input_size
latant_3d = np.array(latant_3dtensors)
latant_3d.shape
# labels = data_dict["labels_name"]
np.unique(labels).shape
legend_labels =[]
for l in np.unique(labels):
    # if l == "cross_legged_sitting" or l == "sitting_down" or l=="crawling" or l== "" :
    #     continue
    legend_labels.append(l)
    # print(l)
    ind = np.where(labels == l)
    mat_label = latant_3d[ind,:,:].squeeze()
    mat = np.mean(mat_label, axis = 0)
    z = PCA(n_components=3).fit_transform(mat.T)
    # plt.scatter(z[0,0],z[0,1])
    # plt.plot(z[:,0],z[:,1])
    plt.plot(z[0:100,0])
    # plt.plot(mat[0,:],mat[1,:])

plt.legend(legend_labels, bbox_to_anchor = (1.1,1.05), fontsize = "10")
plt.xlabel("Time")
plt.ylabel("PC1")
plt.show()



for l in np.unique(labels):
    # if l == "cross_legged_sitting" or l == "sitting_down" or l=="crawling" or l== "" :
    #     continue
    legend_labels.append(l)
    # print(l)
    ind = np.where(labels == l)
    mat_label = latant_3d[ind,:,:].squeeze()
    mat = np.mean(mat_label, axis = 0)
    z = PCA(n_components=3).fit_transform(mat.T)
    # plt.scatter(z[0,0],z[0,1])
    # plt.plot(z[:,0],z[:,1])
    plt.plot(z[0:100,1])
    # plt.plot(mat[0,:],mat[1,:])

plt.legend(legend_labels, bbox_to_anchor = (1.1,1.05), fontsize = "10")
plt.xlabel("Time")
plt.ylabel("PC2")
plt.show()



for l in np.unique(labels):
    # if l == "cross_legged_sitting" or l == "sitting_down" or l=="crawling" or l== "" :
    #     continue
    legend_labels.append(l)
    # print(l)
    ind = np.where(labels == l)
    mat_label = latant_3d[ind,:,:].squeeze()
    mat = np.mean(mat_label, axis = 0)
    z = PCA(n_components=3).fit_transform(mat.T)
    # plt.scatter(z[0,0],z[0,1])
    # plt.plot(z[:,0],z[:,1])
    plt.plot(z[0:100,2])
    # plt.plot(mat[0,:],mat[1,:])

plt.legend(legend_labels, bbox_to_anchor = (1.1,1.05), fontsize = "10")
plt.xlabel("Time")
plt.ylabel("PC3")
plt.show()


# -

# Classification

# +
# now let's do the classification part

class Classifier(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size[1], 64)
        self.fc2 = nn.Linear(64, 21)
        
        # self.tanh = nn.Tanh()
        # self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        out = nn.functional.log_softmax(x, dim=1)
        return out

# +
# data  = latant_3dtensors.view(1319,7*633)
# input_size = np.array(latant_3dtensors.view(1319,7*633).shape)
# # Set the hyperparameters
# # input_size = np.array(data.shape)
# hidden_size =64
# output_size = 21

# for i in data:
#     print(i.shape)   


def test(net,testloader):
        net.eval()
        test_labels, predicted_labels = [], []
        with torch.no_grad():
            correct = 0
            total = 0
            for motion, labels in testloader:
                
                test_labels += list(labels)
                output= net(motion)
                # print(output,"    ", type(output))
                _, predicted = torch.max(output.data, 1)
                predicted_labels += list(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("Test Accuracy of the model on the test moves: {}".format((correct / total)*100))
        return((correct / total)*100,predicted_labels)
    


# -

# !ls

# +
data  = latant_3dtensors.view(1319,7*633)
labels_motion = data_dict["labels"]
data_train, data_test, labels_train, labels_test = train_test_split(data,labels_motion,test_size=0.25, random_state=42)

train_data = []
for i in range(len(data_train)):
    train_data.append([data_train[i], torch.tensor(labels_train[i])])
test_data = []
for i in range(len(data_test)):
    test_data.append([data_test[i], torch.tensor(labels_test[i])])

trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=20)
testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=20)

# Set the hyperparameters
input_size = np.array(data_train.shape)
hidden_size =64
output_size = 21


# Define the neural network
saved_classifier = Classifier(input_size, hidden_size, output_size)
saved_classifier.load_state_dict(torch.load('../movement_classifier/classifier.pth'))

# Define the neural network
# net = Classifier(input_size, hidden_size, output_size)


acc, predictedlabels = test(saved_classifier, trainloader)
# -

p=0
for i in testloader:
    # print(i[1].shape)
    p+=1
# testloader[0].shape
print(p)

# +
# to load saved model

# Define the neural network
saved_classifier = Classifier(input_size, hidden_size, output_size)
saved_classifier.load_state_dict(torch.load('../movement_classifier/classifier.pth'))

#run model on all data:
data  = latant_3dtensors.view(1319,7*633)
labels = data_dict["labels"]

# input_data = []
# for i in range(len(data)):
#     input_data.append([data[i], torch.tensor(labels[i])])

output_data = []
with torch.no_grad():
    for datapoint in data:
        output= saved_classifier(datapoint)
        print(output.shape)
        # latant_x = torch.squeeze(latant_x)
        # latant_x = torch.permute(latant_x, (1, 0))
        # latant_space_data.append(latant_x)
        #plot
        # fig, axs = plt.subplots(nrows=7, figsize=(8, 40))
        # for i in range(7):
        #     axs[i].plot(latant_x[i])
        #     axs[i].set_title('Row {}'.format(i+1))
        # plt.tight_layout()
        # plt.show()

# latant_3dtensors = torch.stack(latant_space_data)
# -

data.shape

# +

from torch.utils.data import DataLoader

def train(net, trainloader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))

def test(net,testloader):
        net.eval()
        test_labels, predicted_labels = [], []
        with torch.no_grad():
            correct = 0
            total = 0
            for motion, labels in testloader:
                
                test_labels += list(labels)
                output= net(motion)
                print(output,"    ", type(output))
                _, predicted = torch.max(output.data, 1)
                predicted_labels += list(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Accuracy of the model on the test moves: {(correct / total)*100:.3f}%")
        return((correct / total)*100)


# -


from sklearn.model_selection import train_test_split
def main():
    data  = latant_3dtensors.view(1319,7*633)
    labels_motion = data_dict["labels"]
    data_train, data_test, labels_train, labels_test = train_test_split(data,labels_motion,test_size=0.25, random_state=42)

    train_data = []
    for i in range(len(data_train)):
        train_data.append([data_train[i], torch.tensor(labels_train[i])])
    test_data = []
    for i in range(len(data_test)):
        test_data.append([data_test[i], torch.tensor(labels_test[i])])

    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=20)
    testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=20)

    # Set the hyperparameters
    input_size = np.array(data_train.shape)
    hidden_size =64
    output_size = 21
    learning_rate = 0.001
    epochs = 1000

    
    # Define the neural network
    net = Classifier(input_size, hidden_size, output_size)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Train the neural network
    train(net, trainloader, criterion, optimizer, epochs)
    test(net, testloader)

main()




latant_3dtensors.view(1319,7*633).shape



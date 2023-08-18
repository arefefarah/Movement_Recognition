
import sys
# sys.path.insert(0, '../')
# import movement_classifier.utils as utils
import data_loader as data_loader
# import movement_classifier.model_funcs as model_funcs
# import movement_classifier.gpt_reverse_model as gpt_reverse_model

from os.path import dirname, join as pjoin
import os
import sys
import math

# import dlc2kinematics
# from sequitur.models import CONV_LSTM_AE
# from sequitur.models import LSTM_AE 
# from sequitur import quick_train
# from scipy.interpolate import CubicSpline
# import matplotlib.pyplot as plt
# from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import MSELoss
from sklearn import preprocessing
# from matplotlib import animation
from sklearn.metrics import confusion_matrix
import copy
# from IPython.display import HTML
# from celluloid import Camera
# %matplotlib inline
# import pandas as pd
# import plotly.express as px
import torch
# import plotly
# from sklearn.decomposition import PCA
# import seaborn as sns
# import scipy.io as sio

# functions:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# seq_len, n_features = train_set.shape[1], train_set.shape[2]

class Encoder(nn.Module):

  def __init__(self):
    super(Encoder, self).__init__()

    self.lstm1 = nn.LSTM(input_size=28, hidden_size=14, num_layers=1, batch_first=True)
    self.lstm2 = nn.LSTM(input_size=14, hidden_size=7, num_layers=1, batch_first=True)
  
  def forward(self, x):
    x = x.reshape((1,633, 28))
    encoded, _ = self.lstm1(x)
    encoded, _ = self.lstm2(encoded)

    return encoded
  


class Decoder(nn.Module):

  def __init__(self):
    super(Decoder, self).__init__()

    self.lstm1 = nn.LSTM(input_size=7, hidden_size=14, num_layers=1, batch_first=True)
    self.lstm2 = nn.LSTM(input_size=14, hidden_size=28, num_layers=1, batch_first=True)

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

"""load dataframes for the modelling"""
path_file = "../data_interpolation"
data_dict = data_loader.load_data_dict(path_file)
data_dict.keys()
# np.unique(data_dict["labels_name"])
data = data_dict['input_model']
train_input = torch.Tensor(data[0:1250,:,0:633])
#  train_Set should be ==>  [num_examples, seq_len, *num_features]
train_set  = train_input.permute(0,2,1)
val_input = torch.Tensor(data[1250:1319,:,0:633])
val_set  = val_input.permute(0,2,1)


# to load saved model
saved_model = RecurrentAutoencoder()
saved_model.load_state_dict(torch.load('lstm_autoencoder.pth'))


#run model on all data:
data = data_dict['input_model']
data_input = torch.Tensor(data[:,:,0:633])
#  should be ==>  [num_examples, seq_len, *num_features]
data_input  = data_input.permute(0,2,1)

latant_space_data = []
with torch.no_grad():
    for data in data_input:
        reconst_x,latant_x = saved_model(data)
        latant_x = torch.squeeze(latant_x)
        latant_x = torch.permute(latant_x, (1, 0))
        latant_space_data.append(latant_x)
     

latant_3dtensors = torch.stack(latant_space_data)

input_size = np.array(latant_3dtensors.view(1319,7*633).shape)

# now let's do the classification part


class Classifier(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size[1], hidden_size)
        # self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 21)
        
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        # x = self.tanh(self.fc2(x))
        # x = self.fc3(x)
        out = nn.functional.log_softmax(x, dim=1)
        return out
    

def train(net, trainloader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # add regularization
            # reg_lambda = 0.001
            # # reg == "l2":
            # reg_norm_weights = sum(p.pow(2.0).sum()for p in net.parameters())                    
            # loss = loss + reg_lambda * reg_norm_weights 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch {0} loss: {1}'.format(epoch + 1, running_loss / len(trainloader)))

def test(net,testloader):
        net.eval()
        test_labels, predicted_labels = [], []
        with torch.no_grad():
            correct = 0
            total = 0
            for motion, labels in testloader:
                
                test_labels += list(labels)
                output= net(motion)
                
                _, predicted = torch.max(output.data, 1)
                predicted_labels += list(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("Test Accuracy of the model on the test moves: {}".format((correct / total)*100))
        return((correct / total)*100, predicted_labels,test_labels)



from sklearn.model_selection import train_test_split
def main():
    data  = latant_3dtensors.view(1319,7*633)
    labels_motion = data_dict["labels"]
    data_train, data_test, labels_train, labels_test = train_test_split(data,labels_motion,test_size=0.25, random_state=42)
    le = preprocessing.LabelEncoder()
    le.fit(data_dict['labels_name'])
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
    acc, predicted_labels,test_labels = test(net, testloader)


    real_labels = [int(x) for x in test_labels]
    predicted_labels = [int(x) for x in predicted_labels]
    labels_unique = np.unique(test_labels)
    labels_name = le.inverse_transform(labels_unique)
    tick_names = [a.replace("_", " ") for a in labels_name]
    cm = confusion_matrix(real_labels, predicted_labels,labels = labels_unique,normalize='true')
    plt.figure(figsize=(8,10))
    plt.imshow(cm)
    plt.xticks(range(len(tick_names)),tick_names, rotation=90)
    plt.yticks(range(len(tick_names)),tick_names)
    plt.xlabel('predicted move')
    plt.ylabel('real move')
    plt.show()
    plt.savefig("confmat.png")
    



# def plotConfusionMatrix(predicted_labels,test_labels):
#         real_labels = [int(x) for x in test_labels]
#         predicted_labels = [int(x) for x in predicted_labels]
#         labels_unique = np.unique(test_labels)
#         # labels_name = self.le.inverse_transform(labels_unique)
#         # tick_names = [a.replace("_", " ") for a in labels_name]
#         # cm = confusion_matrix(self.real_labels, self.predicted_labels,labels = labels_unique,normalize='true')
#         cm = confusion_matrix(real_labels, predicted_labels,labels = labels_unique,normalize='true')
#         plt.figure(figsize=(8,10))
#         plt.imshow(cm)
#         # plt.xticks(range(len(tick_names)),tick_names, rotation=90)
#         # plt.yticks(range(len(tick_names)),tick_names)
#         plt.xlabel('predicted move')
#         plt.ylabel('real move')
#         plt.show()
#         plt.savefig("confmat.png")
#         # return(cm,tick_names)



main()
# plotConfusionMatrix(predicted_labels,test_labels)
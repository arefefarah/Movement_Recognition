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


# +
# visualize one sample's feature in each plot
sample = np.array(val_input[0])
sample.shape
fig, axs = plt.subplots(nrows=14, ncols=2,figsize=(10, 30))
data = sample
# plot each row in a separate subplot
for i in range(14):
    axs[i,0].plot(data[i*2])
    axs[i,1].plot(data[i*2+1])
    axs[i,0].set_title('Feature {}'.format(i*2+1))
    axs[i,1].set_title('Feature {}'.format(i*2+2))

# adjust the layout of the subplots
plt.tight_layout()

# show the plot
plt.show()

# +
# functions:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len, n_features = train_set.shape[1], train_set.shape[2]

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


# +
#  Training

def train_model(model, train_dataset, val_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  # or nn.L1loss(reduction = "sum")
  # criterion = nn.L1loss(reduction = "sum").to(device)
  criterion = MSELoss().to(device)
  history = dict(train=[], val=[])

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 0.0001
  
  for epoch in range(1, n_epochs + 1):
    model = model.train()

    train_losses = []
    for seq_true in train_dataset:
      optimizer.zero_grad()

      seq_true = seq_true.to(device)
      seq_pred,_ = model(seq_true)
      # print("#######################",seq_true.shape, "shape true seq ")
      # print("#######################",seq_pred.shape, "shape of output")
      loss = criterion(seq_pred.reshape(633,28), seq_true)

      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

    val_losses = []
    val_data_predicted = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:

        seq_true = seq_true.to(device)
        seq_pred,_ = model(seq_true)
        val_data_predicted.append(seq_pred.reshape(633,28))
        loss = criterion(seq_pred.reshape(633,28), seq_true)
        val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

  model.load_state_dict(best_model_wts)
  return model.eval(), history,val_data_predicted


# +
model = RecurrentAutoencoder()
model = model.to(device)

model, history, val_data_predicted = train_model(
  model, 
  train_set, 
  val_set, 
  n_epochs=100
)

#to save the model

torch.save(model.state_dict(), '../data/lstm_autoencoder.pth')

# +

# to load saved model
saved_model = RecurrentAutoencoder()
saved_model.load_state_dict(torch.load('../data/lstm_autoencoder.pth'))

# +
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
        #plot
        # fig, axs = plt.subplots(nrows=7, figsize=(8, 40))
        # for i in range(7):
        #     axs[i].plot(latant_x[i])
        #     axs[i].set_title('Row {}'.format(i+1))
        # plt.tight_layout()
        # plt.show()

latant_3dtensors = torch.stack(latant_space_data)


    

# -





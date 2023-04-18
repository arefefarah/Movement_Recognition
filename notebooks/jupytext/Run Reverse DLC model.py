# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jupytext//py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: 'Python 3.9.7 (''base'': conda)'
#     language: python
#     name: python3
# ---

# #### This notebook is for ruunig the reverse model 

# +
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
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import pandas as pd
import plotly.express as px
import plotly
from sklearn.decomposition import PCA
import seaborn as sns
import scipy.io as sio


# +
"""load dataframes for the modelling"""
path_file = "../data/03_processed/padding"
data_dict = data_loader.load_data_dict(path_file)

model = model_funcs.Mov1DCNN(num_classes = 20)
input_dict = data_dict
reg = "l2"
params = (model , input_dict   , reg )
my_testmodel = model_funcs.ModelHandler(*params)
"""train model"""
model_trained =  my_testmodel.train()
"""test model"""
my_testmodel.test()


# -

input_train= np.load("../data/03_processed/output_train/input.npy")
print(input_train.shape)
out, indices = model_trained(input_train)

# +
visualization_train,labels_name_train,output_train= my_testmodel.layer_extractor(train=True)
output = output_train


reverse_model = gpt_reverse_model.ReverseMov1DCNN(num_classes= 20,maxpool_indices =maxpool_indices)
reconstuct_out = reverse_model(output)

# +
#analyze model for inverse model
out_fc3_train= np.load("../data/03_processed/output_train/fc3-out.npy")
input_train= np.load("../data/03_processed/output_train/input.npy")
labels_name_train = np.load("../data/03_processed/output_train/labels_name.npy")
labels_train = np.load("../data/03_processed/output_train/labels.npy")

# combine with test data:
out_fc3_test= np.load("../data/03_processed/output_test/fc3-out.npy")
input_test= np.load("../data/03_processed/output_test/input.npy")
labels_name_test = np.load("../data/03_processed/output_test/labels_name.npy")
labels_test = np.load("../data/03_processed/output_test/labels.npy")

out_fc3= np.concatenate((out_fc3_train,out_fc3_test), axis=0)
input= np.concatenate((input_train,input_test), axis=0)
labels_name = np.concatenate((labels_name_train,labels_name_test), axis=0)
labels = np.concatenate((labels_train,labels_test), axis=0)


# path_file = "../data/03_processed/padding"
# data_dict = data_loader.load_data_dict(path_file)
input_dict = {}
input_dict["input_model"] = out_fc3
input_dict["labels_name"] = labels_name
input_dict["labels"] = labels

print(input_test.shape)
print(out_fc3.shape)


model1 = gpt_reverse_model.ReverseMov1DCNN(num_classes = 20)
reg = "l2"
params = (model1 , input_dict   , reg )
my_testmodel = reverse_model.ModelHandler(*params)
# """train model"""
my_testmodel.train()
# """test model"""
# my_testmodel.test()

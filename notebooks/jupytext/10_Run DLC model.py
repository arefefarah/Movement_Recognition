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
#     display_name: 'Python 3.9.7 (''base'': conda)'
#     language: python
#     name: python3
# ---

# #### This notebook is for loading data from Deeplabcut and running classification model 

# +
import sys
sys.path.insert(0, '../')
import movement_classifier.utils as utils
import movement_classifier.data_loader as data_loader
import movement_classifier.model_funcs as model_funcs
import movement_classifier.reverse_model as reverse_model

from os.path import dirname, join as pjoin
import os

import math

import dlc2kinematics
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import torch
import pandas as pd
import plotly.express as px
import plotly
from sklearn.decomposition import PCA
import seaborn as sns
import scipy.io as sio

# -

"""Load raw data and create Dataframe of all subjects and their movements and save them"""
min_length,max_length,_,_ = data_loader.timelength_loader("../data/01_raw/F_Subjects")
sub_info,movement_name_list,subjects = data_loader.csvSubject_loader("../data/01_raw/CSV_files",min_length,max_length,method="interpolation")
data_loader.save_data(sub_info, movement_name_list,subjects, method = "padding")

"""load dataframes for the modelling"""
path_file = "../data/03_processed/padding"
data_dict = data_loader.load_data_dict(path_file)
data_dict.keys()
np.unique(data_dict["labels_name"])

# +

### Extract two/three classes data for two-class classifier
# np.unique(data_dict["labels_name"])
ind = np.where(np.logical_or(np.logical_or(data_dict["labels_name"] == "walking" , data_dict["labels_name"] == 'jumping_jacks'), data_dict["labels_name"] == 'jogging'))
p = ['input_model', 'labels', 'labels_name']
input_dict ={}
for k in p:
    print(k)
    input_dict[k]= data_dict[k][ind]
    input_dict[k].shape
input_dict["labels"]


labels = input_dict["labels"]
labels[labels== 7]= 0
labels[labels== 19]= 1
labels[labels== 6]= 2
input_dict["labels"] = labels

# +

""" Run functions for the model"""

model1 = model_funcs.Mov1DCNN(num_classes = 20)
input_dict = data_dict
reg = "l2"
params = (model1 , input_dict   , reg )
my_testmodel = model_funcs.ModelHandler(*params)
"""train model"""
my_testmodel.train()
"""test model"""
my_testmodel.test()
"""plot confusionmatrix"""
# my_testmodel.plotConfusionMatrix()
"""plot RDM for input and fully connected layers"""
# visualization_train,labels_name_train,output_train= my_testmodel.layer_extractor(train=True)
# my_testmodel.save_layerOutput(train = True)
# visualization_test,labels_name_test,output_train= my_testmodel.layer_extractor(train=False)
# my_testmodel.save_layerOutput(train = False)
# out_fc1= np.load("../data/03_processed/fc1-out.npy")
# out_fc2= np.load("../data/03_processed/fc2-out.npy")
# out_fc3= np.load("../data/03_processed/fc3-out.npy")
# my_testmodel.plotRDM(plot_input=True)
# my_testmodel.plotRDM(plot_input=False)

# -

def plot_difconfmat(in1,in2):
    model1 = model_funcs.Mov1DCNN()
    path_file = "../data/03_processed/"+in1
    data_dict = data_loader.load_data_dict(path_file)   
    input_dict = data_dict
    reg = "l2"
    params = (model1 , input_dict   , reg )
    m1 = model_funcs.ModelHandler(*params)
    m1.train()
    m1.test()
    conf1,tick_names = m1.plotConfusionMatrix()

    model2 = model_funcs.Mov1DCNN()
    path_file = "../data/03_processed/"+in2
    data_dict = data_loader.load_data_dict(path_file)   
    input_dict = data_dict
    reg = "l2"
    params = (model2 , input_dict   , reg )
    m2 = model_funcs.ModelHandler(*params)
    m2.train()
    m2.test()
    conf2,tick_names = m2.plotConfusionMatrix()

    plt.figure(figsize=(8,10))
    plt.imshow(conf1-conf2)
    plt.colorbar(orientation="horizontal")
    plt.xticks(range(len(tick_names)),tick_names, rotation=90)
    plt.yticks(range(len(tick_names)),tick_names)
    plt.xlabel('predicted move')
    plt.ylabel('real move')
    plt.show()
    return(conf1,conf2)



conf1,conf2 = plot_difconfmat("interpolation","padding")
plt.figure(figsize=(8,10))
plt.imshow(conf1-conf2)
plt.colorbar()
# plt.xticks(range(len(tick_names)),tick_names, rotation=90)
# plt.yticks(range(len(tick_names)),tick_names)
plt.xlabel('predicted move')
plt.ylabel('real move')
plt.title("dif conf mat")
plt.show()



# +
"""test new library of DLC2kinematics"""

# load dlc2kinematics to add velocity and angular 
df, bodyparts, scorer = dlc2kinematics.load_data("/home/arefe/My Project/data/01_raw/h5files/F_PG1_Subject_21_LDLC_resnet101_myDLC_21_25Nov17shuffle1_103000.h5")
# it gives dataframe of velocity for each 42 channel
df_vel = dlc2kinematics.compute_velocity(df,bodyparts=['all'])
# joint_dict= {}
# joint_dict['R-Elbow']  = ['R_shoulder', 'Right_elbow', 'Right_wrist']
# joint_angles = dlc2kinematics.compute_joint_angles(df,joint_dict)
# joint_vel = dlc2kinematics.compute_joint_velocity(joint_angles)
# pca = dlc2kinematics.compute_pca(joint_vel, plot=True)
dlc2kinematics.plot_3d_pca_reconstruction(df_vel, n_components=10, framenumber=500)





"""plot length distribution of each movement """
_,_,subjects,all_motions_dist= data_loader.timelength_loader("../data/01_raw/F_Subjects")

motions_dist_mean = {}
motions_dist_std = {}
for k in all_motions_dist.keys():
    motions_dist_mean[k] = np.mean(all_motions_dist[k])
    motions_dist_std[k] = np.std(all_motions_dist[k])
positions = all_motions_dist.keys()
plt.figure(figsize=(8,6))
plt.bar(positions, motions_dist_mean.values(), color="Cyan", yerr=motions_dist_std.values())
plt.xticks(rotation=30)
plt.show()

# dlc2kinematics.compute_umap(df, key=['LeftForelimb', 'RightForelimb'], chunk_length=30,fit_transform=True, n_neighbors=30, n_components=3,metric="euclidean")

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

# #### In this notebook, we did tsne method on output of fully connedcted layers of model to visualize featurs and outputs for different movement during the process of model 

# +

sys.path.insert(0, '../')
import movement_classifier.utils as utils
import movement_classifier.data_loader as data_loader
import movement_classifier.model_funcs as model_funcs
# import movement_classifier.df_freq_builder 

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
from sklearn.manifold import TSNE
import plotly.express as px
import plotly
from sklearn.decomposition import PCA
import seaborn as sns
import scipy.io as sio


# -

def plot_tsne(labels_name,visualization,layer,perplexity = 30, iter = 2000 ):
    u_labels = np.unique(labels_name)
    model = TSNE(n_components=2, random_state=1,learning_rate=100,perplexity=perplexity, n_iter=iter)
    if layer == "input":
        input_data = visualization["input"].reshape(visualization["input"].shape[0],visualization["input"].shape[1]*visualization["input"].shape[2])
    else:
        input_data = visualization[layer]
    tsne_data = model.fit_transform(input_data)
    print(tsne_data.shape)
    cmp = ["#00FFFF", "#0000FF","#8A2BE2","#EE3B3B","#7FFF00","#EE7621","#FF1493","#FFD700","#8B2252","#FF6A6A",
    "#BFEFFF","#FFBBFF","#FFB5C5","#00CD66","#008080","#8B8B00","#CDBA96","#8B3626","#8B8989","#5E2612"]
    plt.figure(num=1, figsize=(4, 2), dpi=300, facecolor='w', edgecolor='w')
    i=0
    for ul in u_labels:
        # print(ul)
        plt.scatter(tsne_data[labels_name==ul,0], tsne_data[labels_name==ul, 1],label=ul,s=2,c=cmp[i],cmap=cmp)
        i+=1
    plt.axis('tight')
    plt.yticks([])
    # plt.title("Input of First Layer",fontsize = 5)
    plt.xticks([])
    plt.legend(loc='best', ncol=2,fancybox=True, shadow=False, fontsize = 2)
    # plt.savefig('../reports/figures/input_tsne.png')
    plt.show()
    return(tsne_data)


# +


# with open('../data/03_processed/fc1_out.npy','rb') as f:
#     out_fc1 = pkl.load(f)

# with open('../data/03_processed/fc2_out.pkl','rb') as f:
#     out_fc2 = pkl.load(f)

# with open('../data/03_processed/fc3_out.pkl','rb') as f:
#     out_fc3 = pkl.load(f)

# with open('../data/03_processed/labels_name_test.pkl','rb') as f:
#     labels_name_test = pkl.load(f)
# with open('../data/03_processed/labels_test.pkl','rb') as f:
#     labels_test = pkl.load(f)

# +

"""load dataframes for the modelling"""
path_file = "../data/03_processed/interpolation"
data_dict = data_loader.load_data_dict(path_file)
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
# visualization,labels_name= my_testmodel.layer_extractor(train=True)
my_testmodel.save_layerOutput()
# out_fc1= np.load("../data/03_processed/fc1-out.npy")
# out_fc2= np.load("../data/03_processed/fc2-out.npy")
# out_fc3= np.load("../data/03_processed/fc3-out.npy")
# my_testmodel.plotRDM(plot_input=True)
# my_testmodel.plotRDM(plot_input=False)


# -

# my_testmodel.plot_tsne("input")
# my_testmodel.plot_tsne(labels_test_name,,"fc1")
# my_testmodel.plot_tsne(labels_test_name,,"fc2")
# my_testmodel.plot_tsne(labels_test_name,,"fc3")
visualization,labels_name= my_testmodel.layer_extractor(train=True)
# my_testmodel.plot_tsne(labels_name,visualization,"input")
visualization,labels_name= my_testmodel.layer_extractor(train=False)
plot_tsne(labels_name,visualization,"fc1")
plot_tsne(labels_name,visualization,"fc2")
plot_tsne(labels_name,visualization,"fc3")

rdm = my_testmodel.plotRDM(plot_input=False)


my_testmodel.plotRDM(plot_input=True)

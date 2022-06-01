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

# #### In this notebook, we did tsne method on output of fully connedcted layers of model to visualize featurs and outputs for different movement during the process of model 

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
# sys.path.insert(0, '/home/arefe/My Project/my_project/utils')
# from DLC_functions import *
from sklearn.model_selection import train_test_split
# get some pytorch:
import torch
import torch.nn as nn
import pickle as pkl
from torch.nn import MaxPool1d
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy, rsatoolbox
# confusion matrix from sklearn
from sklearn.metrics import confusion_matrix
# to get some idea of how long stuff will take to complete:
import time
# to see how unbalanced the data is:
from collections import Counter
import rsatoolbox.data as rsd # abbreviation to deal with dataset
from sklearn.manifold import TSNE

# +

with open('../data/03_processed/input_test.pkl','rb') as f:
    input_data = pkl.load(f)

with open('../data/03_processed/fc1_out.pkl','rb') as f:
    out_fc1 = pkl.load(f)

with open('../data/03_processed/fc2_out.pkl','rb') as f:
    out_fc2 = pkl.load(f)

with open('../data/03_processed/fc3_out.pkl','rb') as f:
    out_fc3 = pkl.load(f)

with open('../data/03_processed/labels_name_test.pkl','rb') as f:
    labels_name_test = pkl.load(f)
with open('../data/03_processed/labels_test.pkl','rb') as f:
    labels_test = pkl.load(f)

# +

u_labels = np.unique(labels_name_test)
model = TSNE(n_components=2, random_state=1,learning_rate=100,perplexity=30, n_iter=2000)
tsne_data = model.fit_transform(input_data)
cmp = ["#00FFFF", "#0000FF","#8A2BE2","#EE3B3B","#7FFF00","#EE7621","#FF1493","#FFD700","#8B2252","#FF6A6A",
"#BFEFFF","#FFBBFF","#FFB5C5","#00CD66","#008080","#8B8B00","#CDBA96","#8B3626","#8B8989","#5E2612"]

plt.figure(num=1, figsize=(4, 2), dpi=500, facecolor='w', edgecolor='w')
i=0
for ul in u_labels:
    # print(ul)
    plt.scatter(tsne_data[labels_name_test==ul,0], tsne_data[labels_name_test==ul, 1],label=ul,s=5,c=cmp[i],cmap=cmp)
    i+=1
plt.axis('tight')
plt.yticks([])
plt.title("Input of First Layer",fontsize = 5)
plt.xticks([])
plt.legend(loc='upper right', bbox_to_anchor=(0, 0), ncol=1,fancybox=True, shadow=False, fontsize = 2)
plt.savefig('../reports/figures/input_tsne.png')
plt.show()

# +


u_labels = np.unique(labels_name_test)
model = TSNE(n_components=2, random_state=1,learning_rate=100,perplexity=30, n_iter=2000)
tsne_data = model.fit_transform(out_fc1)
cmp = ["#00FFFF", "#0000FF","#8A2BE2","#EE3B3B","#7FFF00","#EE7621","#FF1493","#FFD700","#8B2252","#FF6A6A",
"#BFEFFF","#FFBBFF","#FFB5C5","#00CD66","#008080","#8B8B00","#CDBA96","#8B3626","#8B8989","#5E2612"]

plt.figure(num=1, figsize=(4, 2), dpi=500, facecolor='w', edgecolor='w')
i=0
for ul in u_labels:
    # print(ul)
    plt.scatter(tsne_data[labels_name_test==ul,0], tsne_data[labels_name_test==ul, 1],label=ul,s=5,c=cmp[i],cmap=cmp)
    i+=1
plt.axis('tight')
plt.yticks([])
plt.title("Output of First Fully Connected Layer",fontsize = 5)
plt.xticks([])
plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=4,fancybox=True, shadow=False, fontsize = 2)
plt.savefig('../reports/figures/fc1_tsne.png')
plt.show()

# +
u_labels = np.unique(labels_name_test)
model = TSNE(n_components=2, random_state=1,learning_rate=100,perplexity=30, n_iter=2000)
tsne_data = model.fit_transform(out_fc2)
cmp = ["#00FFFF", "#0000FF","#8A2BE2","#EE3B3B","#7FFF00","#EE7621","#FF1493","#FFD700","#8B2252","#FF6A6A",
"#BFEFFF","#FFBBFF","#FFB5C5","#00CD66","#008080","#8B8B00","#CDBA96","#8B3626","#8B8989","#5E2612"]

plt.figure(num=1, figsize=(4, 2), dpi=500, facecolor='w', edgecolor='w')
i=0
for ul in u_labels:
    # print(ul)
    plt.scatter(tsne_data[labels_name_test==ul,0], tsne_data[labels_name_test==ul, 1],label=ul,s=5,c=cmp[i],cmap=cmp)
    i+=1
plt.axis('tight')
plt.yticks([])
plt.title("Output of Second Fully Connected Layer", fontsize = 5)
plt.xticks([])
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=4,fancybox=True, shadow=False, fontsize = 2)
plt.savefig('../reports/figures/fc2_tsne.png')
plt.show()
# plt.clf()

# +
u_labels = np.unique(labels_name_test)
model = TSNE(n_components=2, random_state=1,learning_rate=100,perplexity=30, n_iter=2000)
tsne_data = model.fit_transform(out_fc3)
cmp = ["#00FFFF", "#0000FF","#8A2BE2","#EE3B3B","#7FFF00","#EE7621","#FF1493","#FFD700","#8B2252","#FF6A6A",
"#BFEFFF","#FFBBFF","#FFB5C5","#00CD66","#008080","#8B8B00","#CDBA96","#8B3626","#8B8989","#5E2612"]

plt.figure(num=1, figsize=(4, 2), dpi=500, facecolor='w', edgecolor='w')
i=0
for ul in u_labels:
    # print(ul)
    plt.scatter(tsne_data[labels_name_test==ul,0], tsne_data[labels_name_test==ul, 1],label=ul,s=5,c=cmp[i],cmap=cmp)
    i+=1
plt.axis('tight')
plt.yticks([])
plt.title("Output of Third Fully Connected Layer", fontsize = 5)
plt.xticks([])
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=4,fancybox=True, shadow=False, fontsize = 2)
plt.savefig('../reports/figures/fc3_tsne.png')
plt.show()
# -





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

# +
sys.path.insert(0, '../')
import movement_classifier.utils as utils


from os.path import dirname, join as pjoin
import os
import sys
import math
import pickle as pkl
import time
from collections import Counter


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.utils.data as data



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


# -

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


print(u_labels)

# plot each movement versus each other
plt.plot(out_fc2_avg[u_labels== "walking"].reshape(1000),out_fc2_avg[u_labels== "phone_talking"].reshape(1000))
plt.show()
plt.plot(np.arange(1000),out_fc2_avg[u_labels== "phone_talking"].reshape(1000))
plt.plot(np.arange(1000),out_fc2_avg[u_labels== "walking"].reshape(1000))
plt.show()


# +
# u_labels = np.unique(labels_name_test)
# out_fc2_avg = avg_movements(out_fc2,labels_name_test,u_labels)
# out_fc2_avg = out_fc2_avg.reshape(out_fc2_avg.shape[0],out_fc2_avg.shape[1]*out_fc2_avg.shape[2])
# # model = TSNE(n_components=2, random_state=1,learning_rate=100,perplexity=30, n_iter=2000)
# # tsne_data = model.fit_transform(out_fc1_avg)
# cmp = ["#00FFFF", "#0000FF","#8A2BE2","#EE3B3B","#7FFF00","#EE7621","#FF1493","#FFD700","#8B2252","#FF6A6A",
# "#BFEFFF","#FFBBFF","#FFB5C5","#00CD66","#008080","#8B8B00","#CDBA96","#8B3626","#8B8989","#5E2612"]

# plt.figure(num=1, figsize=(4, 2), dpi=500, facecolor='w', edgecolor='w')
# i=0
# for ul in u_labels:
#     # print(ul)
#     plt.plot(out_fc2_avg[u_labels==ul])
#     # plt.scatter(out_fc2_avg[u_labels==ul], out_fc2_avg[u_labels==ul],label=ul,s=5,c=cmp[i],cmap=cmp)
#     i+=1
# plt.axis('tight')
# plt.yticks([])
# plt.title("Input of First Layer",fontsize = 5)
# plt.xticks([])
# plt.legend(loc='upper right', bbox_to_anchor=(0, 0), ncol=1,fancybox=True, shadow=False, fontsize = 2)
# # plt.savefig('../reports/figures/input_tsne.png')
# plt.show()


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
# +
# can we do clustering on movements? does it have any information for us?

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# +
from pyclustering.cluster import kmeans
# initialize initial centers using K-Means++ method
# initial_centers =  kmeans.kmeans_plusplus_initializer(sample, 3).initialize()
# create instance of K-Means algorithm with prepared centers
kmeans_instance =  kmeans(input_data, []])
# run cluster analysis and obtain results
cluster.kmeans_instance.process()
clusters = cluster.kmeans_instance.get_clusters()
final_centers = cluster.kmeans_instance.get_centers()


# # create instance of K-Means algorithm
# kmeans_instance = kmeans(sample, [ [0.0, 0.1], [2.5, 2.6] ])
# # run cluster analysis and obtain results
# kmeans_instance.process()
# clusters = kmeans_instance.get_clusters()
# -




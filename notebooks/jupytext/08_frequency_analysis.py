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
    # print(tsne_data.shape)
    # cmp = ["#00FFFF", "#0000FF","#8A2BE2","#EE3B3B","#7FFF00","#EE7621","#FF1493","#FFD700","#8B2252","#FF6A6A",
    # "#BFEFFF","#FFBBFF","#FFB5C5","#00CD66","#008080","#8B8B00","#CDBA96","#8B3626","#8B8989","#5E2612"]
    # plt.figure(num=1, figsize=(2, 1), dpi=500, facecolor='w', edgecolor='w')
    # i=0
    # for ul in u_labels:
    #     # print(ul)
    #     plt.scatter(tsne_data[labels_name==ul,0], tsne_data[labels_name==ul, 1],label=ul,s=5,c=cmp[i],cmap=cmp)
    #     i+=1
    # plt.axis('tight')
    # plt.yticks([])
    # # plt.title("Input of First Layer",fontsize = 5)
    # plt.xticks([])
    # # plt.legend(loc='upper right', bbox_to_anchor=(0, 0), ncol=1,fancybox=True, shadow=False, fontsize = 2)
    # # plt.savefig('../reports/figures/input_tsne.png')
    # plt.show()
    return(tsne_data)


# +
joint1 = "knee1_x"
freq = 0.6
amp = 100

sub_info,movement_name_list,subjects = data_loader.df_freq( joint1,amp , freq,joint2 = "knee2_x" )
data_loader.save_data(sub_info, movement_name_list,subjects, method = "frequency")
# -

joints = ["elbow1_y","elbow1_x","ankle1_x","ankle1_y","knee1_y","knee1_x","hip1_y","hip1_x","shoulder1_x","shoulder1_y","wrist1_x","wrist1_y"]
# joints = ["knee1_x","knee1_y","knee2_x","knee2_y"]
freqs = [0.1,0.2,0.4,0.6,0.8]
amp = 100
for joint in joints:
    for f in freqs:
        sub_info,movement_name_list,subjects = df_freq( joint,amp , f,joint2 = None ) #joint2 = joint.replace("1","2")
        data_loader.save_data(sub_info, movement_name_list,subjects, method = "frequency")
        path_file = "../data/03_processed/frequency"
        data_dict = data_loader.load_data_dict(path_file)
        model1 = model_funcs.Mov1DCNN()
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
        visualization,labels_test_name= my_testmodel.layer_extractor()
        print("########################","Joint: ", joint , " freq = ",f,"########################")
        tsne_data_freq = plot_tsne(labels_test_name,visualization,"fc3")
        u_labels = np.unique(labels_test_name)
        mean_labels = {}
        dprime_mat = np.empty((len(u_labels), len(u_labels)))
        i_list = []
        for i in range(len(u_labels)):
            for j in range(len(u_labels)):
                if j not in i_list:
                    if i == j:
                        dprime_mat[i,j] = 0
                    else: 
                        kk1 = np.mean(tsne_data_freq[labels_test_name==u_labels[i]],axis = 0)
                        kk2 = np.mean(tsne_data_freq[labels_test_name==u_labels[j]],axis = 0)
                        ss1 = np.std(tsne_data_freq[labels_test_name==u_labels[i]],axis = 0)
                        ss2 = np.std(tsne_data_freq[labels_test_name==u_labels[j]],axis = 0)
                        dprime_x = (kk1[0]-kk2[0])/(np.sqrt(0.5*(ss1[0]**2+ss2[0]**2)))
                        dprime_y = (kk1[1]-kk2[1])/(np.sqrt(0.5*(ss1[1]**2+ss2[1]**2)))
                        d = np.sqrt(dprime_x**2+dprime_y**2)
                        dprime_mat[i,j] = d
                        dprime_mat[j,i] = d
            i_list.append(i)
        plt.imshow(dprime_mat)
        tick_names = [a.replace("_", " ") for a in u_labels]
        
        plt.xticks(range(len(tick_names)),tick_names, rotation=90)
        plt.yticks(range(len(tick_names)),tick_names)
        plt.xlabel('predicted move')
        plt.ylabel('real move')
        plt.title("{}_{}".format(joint,f).replace("1_", "_") )
        plt.clim((0,45))
        plt.colorbar()
        plt.savefig('../reports/figures/freq_analysis/samerange/singlejoint/{}_{}.png'.format(joint,f),bbox_inches='tight')
        plt.show()



# +
# plot matrix of different freq in different joints in two class classification

joints = ['ankle1_x', 'knee1_x', 'hip1_x', 'hip2_x', 'knee2_x', 'ankle2_x',
       'wrist1_x', 'elbow1_x', 'shoulder1_x', 'shoulder2_x', 'elbow2_x',
       'wrist2_x', 'chin_x', 'forehead_x', 'ankle1_y', 'knee1_y', 'hip1_y',
       'hip2_y', 'knee2_y', 'ankle2_y', 'wrist1_y', 'elbow1_y', 'shoulder1_y',
       'shoulder2_y', 'elbow2_y', 'wrist2_y', 'chin_y', 'forehead_y']
joints = ['ankle1_x', 'knee1_x','shoulder1_x',"elbow1_x","wrist1_x"]
# freqs = [0.1,0.2,0.4,0.6,0.8]
freqs = [0.2,0.5,0.8]
amp = 80

acc_arr = np.empty((len(joints), len(freqs)))
for j,joint in enumerate(joints):
    for i,f in enumerate(freqs):
        print(joint)
        joint2 = joint.replace("1","2")
        sub_info,movement_name_list,subjects = data_loader.df_freq( joint,amp , f,joint2 = joint2 ) #joint2 = joint.replace("1","2")
        data_loader.save_data(sub_info, movement_name_list,subjects, method = "frequency")
        path_file = "../data/03_processed/frequency"
        data_dict = data_loader.load_data_dict(path_file)
        ind = np.where(np.logical_or(data_dict["labels_name"] == "walking" , data_dict["labels_name"] == 'jumping_jacks'))
        p = ['input_model', 'labels', 'labels_name']
        input_dict ={}
        for k in p:
            input_dict[k]= data_dict[k][ind]
            
        labels = input_dict["labels"]
        labels[labels== 7]= 0
        labels[labels== 19]= 1
        input_dict["labels"] = labels
        model1 = model_funcs.Mov1DCNN()
        reg = "l2"
        params = (model1 , input_dict   , reg )
        my_testmodel = model_funcs.ModelHandler(*params)
        """train model"""
        my_testmodel.train()
        """test model"""
        acc = my_testmodel.test()
        """plot confusionmatrix"""
        # my_testmodel.plotConfusionMatrix()
        """plot RDM for input and fully connected layers"""
        # visualization,labels_test_name= my_testmodel.layer_extractor()
        print("########################","Joint: ", joint , " freq = ",f,"########################")
        print("accuracy:   ", acc)
        acc_arr[j,i]= acc


# -

plt.imshow(acc_arr)
acc_arr
# plt.imshow(acc_arr, cmap=plt.cm.gray, interpolation='nearest')

# +
# d-prime for baseline model in which padding would be used
path_file = "../data/03_processed/padding"
data_dict = data_loader.load_data_dict(path_file)
model1 = model_funcs.Mov1DCNN()
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
visualization,labels_test_name= my_testmodel.layer_extractor()
# print("########################","Joint: ", joint , " freq = ",f,"########################")
tsne_data = plot_tsne(labels_test_name,visualization,"fc3")
u_labels = np.unique(labels_test_name)
mean_labels = {}
dprime_mat = np.empty((len(u_labels), len(u_labels)))
i_list = []
for i in range(len(u_labels)):
    for j in range(len(u_labels)):
        if j not in i_list:
            if i == j:
                dprime_mat[i,j] = 0
            else: 
                kk1 = np.mean(tsne_data[labels_test_name==u_labels[i]],axis = 0)
                kk2 = np.mean(tsne_data[labels_test_name==u_labels[j]],axis = 0)
                ss1 = np.std(tsne_data[labels_test_name==u_labels[i]],axis = 0)
                ss2 = np.std(tsne_data[labels_test_name==u_labels[j]],axis = 0)
                dprime_x = (kk1[0]-kk2[0])/(np.sqrt(0.5*(ss1[0]**2+ss2[0]**2)))
                dprime_y = (kk1[1]-kk2[1])/(np.sqrt(0.5*(ss1[1]**2+ss2[1]**2)))
                d = np.sqrt(dprime_x**2+dprime_y**2)
                dprime_mat[i,j] = d
                dprime_mat[j,i] = d
    i_list.append(i)
plt.imshow(dprime_mat)
tick_names = [a.replace("_", " ") for a in u_labels]

plt.xticks(range(len(tick_names)),tick_names, rotation=90)
plt.yticks(range(len(tick_names)),tick_names)

plt.xlabel('predicted move')
plt.ylabel('real move')
plt.clim(0, 45)
plt.colorbar()
# plt.savefig('../reports/figures/freq_analysis/{}_{}.png'.format(joint,f),bbox_inches='tight')
plt.show()

# +
tsne_data = plot_tsne(labels_test_name,visualization,"fc3")
u_labels = np.unique(labels_test_name)
mean_labels = {}
dprime_mat = np.empty((len(u_labels), len(u_labels)))
i_list = []
for i in range(len(u_labels)):
    for j in range(len(u_labels)):
        if j not in i_list:
            if i == j:
                dprime_mat[i,j] = 0
            else: 
                kk1 = np.mean(tsne_data[labels_test_name==u_labels[i]],axis = 0)
                kk2 = np.mean(tsne_data[labels_test_name==u_labels[j]],axis = 0)
                ss1 = np.std(tsne_data[labels_test_name==u_labels[i]],axis = 0)
                ss2 = np.std(tsne_data[labels_test_name==u_labels[j]],axis = 0)
                dprime_x = (kk1[0]-kk2[0])/(np.sqrt(0.5*(ss1[0]**2+ss2[0]**2)))
                dprime_y = (kk1[1]-kk2[1])/(np.sqrt(0.5*(ss1[1]**2+ss2[1]**2)))
                d = np.sqrt(dprime_x**2+dprime_y**2)
                dprime_mat[i,j] = d
                dprime_mat[j,i] = d
    i_list.append(i)
plt.imshow(dprime_mat)
tick_names = [a.replace("_", " ") for a in u_labels]

plt.xticks(range(len(tick_names)),tick_names, rotation=90)
plt.yticks(range(len(tick_names)),tick_names)
plt.xlabel('predicted move')
plt.ylabel('real move')
plt.clim(0, 45)
plt.colorbar()
# plt.savefig('../reports/figures/freq_analysis/{}_{}.png'.format(joint,f),bbox_inches='tight')
plt.show()
# -



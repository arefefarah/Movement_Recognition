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

# #### this notebook is for loading and preprocessing AMASS data to be used for model

# +
import pandas as pd
import sys
import os
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly
import math
from sklearn.decomposition import PCA
import seaborn as sns
from os.path import dirname, join as pjoin
import scipy.io as sio
sys.path.insert(0, '/home/arefeh/Motion-Project/My Project/my_project/utils')

from DLC_functions import *
from sklearn.preprocessing import MinMaxScaler
# get some pytorch:
import torch
import torch.nn as nn
from torch.nn import MaxPool1d
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# confusion matrix from sklearn
from sklearn.metrics import confusion_matrix

# to get some idea of how long stuff will take to complete:
import time

# to see how unbalanced the data is:
from collections import Counter
# -

# ### make dataset of AMASS data using Padding for equal length of timeseries

# +
subjects = [1,2,3,4,13,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,
            31,34,35,36,37,38,39,40,42,43,44,45,46,47,48,50,51,52,53,
            54,55,56,58,59,60,61,62,63,64,65,66,67,68,69,
           70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90]
mat_files = []
all_motions={}
motion_list =[]
length_seq=[]
#mke a df
df = pd.DataFrame()
for s in subjects:
#     print(s)
    data_dic = []
    file = "../data/01_raw/AMASS/F_amass_Subject_"+str(s)+".mat"
    inf_dic = mat2dict(file)
    for i in range(21):
        name = "move_"+str(i)
        
        e = inf_dic[name]
        k = e["jointsExpMaps_amass"]
        m = e["description"]
        if not m.endswith("_rm"):
            
#         print(m)
#         print(k.shape[0])
            length_seq.append(k.shape[0])
#     ## delete third angle
#         k = np.delete(k[:][:], 2, axis=2)
#         info = k.reshape(k.shape[0],52*2)
#         print(info.shape)
    
            m = e["description"]
            if m not in motion_list:
                motion_list.append(m)
                all_motions[m] = len(all_motions.keys())


# -

thresh = min(length_seq)
thresh

# +

motion_labels = []
subject_list = []
motion_numbers = []
inf =[]
count=0
indices_motions = []
df = pd.DataFrame(columns=np.arange(104)+1)
# dataframe.columns= list(np.arange(104))
for s in subjects:
    print(s)
    data_dic = []
    file = "../data/01_raw/AMASS/F_amass_Subject_{}.mat".format(s)
    inf_dic = mat2dict(file)
    for i in range(21):
        name = "move_"+str(i)
        
        e = inf_dic[name]
        m = e["description"]
        if not m.endswith("_rm"):
            k = e["jointsExpMaps_amass"]
    ## delete third angle
            k = np.delete(k[:][:], 2, axis=2)
            p = k.reshape(k.shape[0],52*2)
#         print(p.shape)

    ## add velocity
            for t in range(104):
                h= p[:,t]
                h_sub = []
                h_sub.append(h[0])
                h_sub = h_sub +list(h)
                h_sub = h_sub[:-1]
                v = h-h_sub
                p = np.c_[p,v]
            p = np.rollaxis(p,-1)
    ## Padding
            N = max(length_seq) - p.shape[1]
            p = np.pad(p, ((0,0),(0,N)), 'constant')




            count+=1
            inf.append(p)
        
            motion_labels.append(m)
            motion_numbers.append(all_motions[m])
            subject_list.append(s)
   
# -

inf = np.array(inf)
# data = np.dstack(inf)
# data = np.rollaxis(data,-1)
inf.shape

# +

Final_subjects = np.array(subject_list)
Final_motions = np.array(motion_numbers)
Final_data = inf
motion_name = np.array(motion_labels) 


np.save("../data/03_processed/AMASS_velocity/Input_model.npy", Final_data)
np.save("../data/03_processed/AMASS_velocity/labels.npy", Final_motions)
np.save("../data/03_processed/AMASS_velocity/labels_name.npy", motion_labels)
np.save("../data/03_processed/AMASS_velocity/subjects.npy", Final_subjects)
# -

# ## resampling and velocity 

# +
motion_labels = []
subject_list = []
motion_numbers = []
inf =[]
count=0
indices_motions = []
df = pd.DataFrame(columns=np.arange(104)+1)
# dataframe.columns= list(np.arange(104))
for s in subjects:
    print(s)
    data_dic = []
    file = "../data/01_raw/AMASS/F_amass_Subject_{}.mat".format(s)
    inf_dic = mat2dict(file)
    for i in range(21):
        name = "move_"+str(i)
        
        e = inf_dic[name]
        m = e["description"]
        if not m.endswith("_rm"):
            k = e["jointsExpMaps_amass"]
    ## delete third angle
            k = np.delete(k[:][:], 2, axis=2)
            p = k.reshape(k.shape[0],52*2)
#         print(p.shape)

    ## add velocity
            for t in range(104):
                h= p[:,t]
                h_sub = []
                h_sub.append(h[0])
                h_sub = h_sub +list(h)
                h_sub = h_sub[:-1]
                v = h-h_sub
                p = np.c_[p,v]
            
            #resampling
            num_samples = math.ceil((p.shape[0])/thresh)
            for i in range(num_samples):

        #                  print(i)
                if i+1 != num_samples:
                    motion_labels.append(m)
                    motion_numbers.append(all_motions[m])
                    subject_list.append(s)
                    
                    a = p[i*thresh:i*thresh+thresh,:]
                    a = np.rollaxis(a,-1)
                    inf.append(a)
                    

                else:
                    motion_labels.append(m)
                    motion_numbers.append(all_motions[m])
                    subject_list.append(s)
                    
                    a = p[p.shape[0]-thresh:p.shape[0],:]
                    a = np.rollaxis(a,-1)
                    inf.append(a)

inf = np.array(inf)
inf.shape

# +

Final_subjects = np.array(subject_list)
Final_motions = np.array(motion_numbers)
Final_data = inf
motion_name = np.array(motion_labels) 


np.save("../data/03_processed/AMASS_resamplingvelocity/Input_model.npy", Final_data)
np.save("../data/03_processed/AMASS_resamplingvelocity/labels.npy", Final_motions)
np.save("../data/03_processed/AMASS_resamplingvelocity/labels_name.npy", motion_labels)
np.save("../data/03_processed/AMASS_resamplingvelocity/subjects.npy", Final_subjects)
# -

# # Normalization
#

# +

motion_labels = []
subject_list = []
motion_numbers = []
inf =[]
count=0
indices_motions = []
df = pd.DataFrame(columns=np.arange(104)+1)
# dataframe.columns= list(np.arange(104))
for s in subjects:
    print(s)
    data_dic = []
    file = "../data/01_raw/AMASS/F_amass_Subject_{}.mat".format(s)
    inf_dic = mat2dict(file)
    for i in range(21):
        name = "move_"+str(i)
        
        e = inf_dic[name]
        m = e["description"]
        if not m.endswith("_rm"):
            k = e["jointsExpMaps_amass"]
    ## delete third angle
            k = np.delete(k[:][:], 2, axis=2)
            p = k.reshape(k.shape[0],52*2)
#         print(p.shape)
#             p = np.rollaxis(p,-1)
            indices_motions.append(p.shape[0])
            df = df.append(pd.DataFrame(p, columns=df.columns), ignore_index=True)
#             print(p.shape)




            count+=1
            inf.append(p)
        
            motion_labels.append(m)
            motion_numbers.append(all_motions[m])
            subject_list.append(s)
   
# -

plt.plot(df.iloc[:,9])

plt.plot(df_norm[:,9])

#normalization
scaler = MinMaxScaler()
scaler.fit(df)
df_norm= scaler.transform(df)
df_norm

#padding
arg = 0
inf = []
for ind in indices_motions:
    print(ind)
    p = df[arg:arg+ind,:]
    p = np.rollaxis(p,-1)
    print(p.shape)
    #Padding
    N = max(length_seq) - p.shape[1]
    p = np.pad(p, ((0,0),(0,N)), 'constant')
    inf.append(p)
    print(p.shape)
    arg= ind

# +
inf = np.array(inf)
Final_subjects = np.array(subject_list)
Final_motions = np.array(motion_numbers)
Final_data = inf
motion_name = np.array(motion_labels) 


np.save("../data/03_processed/AMASS_normalizedpadding/Input_model.npy", Final_data)
np.save("../data/03_processed/AMASS_normalizedpadding/labels.npy", Final_motions)
np.save("../data/03_processed/AMASS_normalizedpadding/labels_name.npy", motion_labels)
np.save("../data/03_processed/AMASS_normalizedpadding/subjects.npy", Final_subjects)
# -



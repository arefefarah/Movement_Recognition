from os.path import dirname, join as pjoin
import os
import sys
import time
from collections import Counter
import math

import dlc2kinematics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import scipy.io as sio
import torch
import torch.nn as nn
from torch.nn import MaxPool1d
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



def timelength_loader(dir):
    subject_files = []
    subjects = []
    min_length = []
    max_length = []
    all_motions = {}
    all_motions_dist = {}

    
    for dirpathes, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            subject_files.append(filename)
            name, ext = os.path.splitext(filename)
            # print(name)
            name= name.split("_")
            subjects.append(name[-1])
            
            inf_dic = utils.mat2dict(os.path.join(dir, filename))
            m = inf_dic["move"]
            motions_list = m["motions_list"]
            timelabels = m["flags30"]

            for m in motions_list:
                if m in all_motions_dist:
                    pass
                else:
                    # all_motions[m] = len(all_motions.keys())
                    all_motions_dist[m] = []
        
            seq_lengths = []
            for i,val in enumerate(timelabels):

                seq_lengths.append(val[1]-val[0])
                all_motions_dist[motions_list[i]].append(val[1]-val[0])

            tresh_min = min(seq_lengths)
            tresh_max = max(seq_lengths)
            min_length.append(tresh_min)
            max_length.append(tresh_max)
            
    return(min_length,max_length,subjects, all_motions_dist)


def csvSubject_loader(dir,min_length,max_length,method = "padding"):
    movement_list = []
    movement_name_list = []
    all_data_sub = {}
    sub_info = []
    subjects = []
    for dirpathes, dirnames, filenames in os.walk(dir):
        for file in filenames:
            name, ext = os.path.splitext(file)
            # print(name)
            name= name.split("_")
            name  = name[3]
            subjects.append(name)
            df = pd.read_csv(os.path.join(dir, file))

            #preprocessing
            df.columns = (df.iloc[0] + '_' + df.iloc[1])
            df = df.iloc[2:].reset_index(drop=True)
            df = df.set_index(df.columns[0])
            df = df.astype('float64')
            df = df.round(4)

            #normalization
            df_x = df.loc[:, df.columns.str.contains('_x')]
            df_y = df.loc[:, df.columns.str.contains('_y')]
            ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
            ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
            df_x = df_x.sub(ref_x,axis = 0)
            df_y = df_y.sub(ref_y,axis = 0)
            df = pd.concat([df_x, df_y], axis=1, join='inner')

            # add velocity
            # for (columnName, columnData) in df.iteritems():
            #     daa = []
            #     for i in range(len(columnData.values)):
            #         if i ==0:
            #             val= 0
            #         else:
            #             val = (columnData[i]-columnData[i-1])
            #         daa.append(val)
            #     name = columnName+"_velocity"
            #     df[name] = daa
        
        
            # Add column of movements
            inf_dic = utils.mat2dict(os.path.join("../data/01_raw/F_Subjects/F_v3d_Subject_{}.mat".format(name)))
            m = inf_dic["move"]
            motions_list = list(m["motions_list"])
            for idx, element in enumerate(motions_list):
                if element.endswith("arms"):
                    motions_list[idx] = "crossarms"
                if element.startswith("jumping"):
                    motions_list[idx] = "jumping_jacks"
            index = [idx for idx, element in enumerate(motions_list) if element.endswith("_rm")]
            timeflags = list(m["flags30"])
            
            for d in range(len(index)):
                motions_list.pop(index[d])
                timeflags.pop(index[d])
    
            all_data = []
############# 
            if method == "padding":
            #build arrays of all subjects using padding
                
                for i,val in enumerate(timeflags):
                    movement_name_list.append(motions_list[i])
                    list_arrays = []
                    N = max(max_length)-(val[1]-val[0])
            #extract time vector of each specific movement
                    for c in df.columns:
                        a= df[c].iloc[val[0]:val[1]]
                        x = a.to_numpy()
                        #padding
                        val_arry = np.pad(x, (0, N), 'constant')
                        list_arrays.append(val_arry)
                    m =np.array(list_arrays)
                    print(m.shape)
                    all_data.append(m)
#############
            if method == "interpolation":
                for i,val in enumerate(timeflags):
                    movement_name_list.append(motions_list[i])
                    list_arrays = []
                    N = max(max_length)-(val[1]-val[0])
            #extract time vector of each specific movement
                    for c in df.columns:
                        a= df[c].iloc[val[0]:val[1]]
                        x = np.arange(val[1]-val[0])
                        # print("x" , x.shape)
                        y = a.to_numpy()
                        # print("y" , y.shape)
                        cs = CubicSpline(x, y)
                        xs = np.linspace(0, val[1]-val[0], num=max(max_length))
                        ys = cs(xs)
                        list_arrays.append(ys)
                    m =np.array(list_arrays)
                    print(m.shape)
                    all_data.append(m)

#############
#                   
            #build arrays of all subjects using resampling    
            if method == "resampling":
                thresh = min(min_length)
                for ind,val in enumerate(timeflags):
                    num_samples = math.ceil((val[1]-val[0])/thresh)
                    # print("num_samples",num_samples)
                    for i in range(num_samples):
        
                        if i+1 != num_samples:
                            movement_name_list.append(motions_list[ind])
                            list_arrays = []
                            for c in df.columns:
                                a= df[c].iloc[val[0]+i*thresh:val[0]+i*thresh+thresh]
                                val_arry = a.to_numpy()
                                list_arrays.append(val_arry)
                        else:
                            movement_name_list.append(motions_list[ind])
                            list_arrays = []
                            for c in df.columns:
                                a= df[c].iloc[val[1]-thresh:val[1]]
                                val_arry = a.to_numpy()
                                list_arrays.append(val_arry)
                            
                        m =np.array(list_arrays)
                        all_data.append(m)
            
            sub_info.append(np.array(all_data))  

    return(sub_info,movement_name_list,subjects)



def save_data(sub_info, movement_name_list,subjects, method = "padding"):
    p=np.array(sub_info)
    jj = p[0]
    k =p.shape[0]
    for t in range(k-1):
        jj = np.vstack((jj, p[t+1]))
    
# Final_motions = np.array(movement_list)
    final_data = jj
    le = preprocessing.LabelEncoder()
    le.fit(movement_name_list)
    motion_num = le.transform(movement_name_list)
    motion_name = np.array(movement_name_list) 
    if method== "padding":
        np.save("../data/03_processed/padding/input_model.npy", final_data)
        np.save("../data/03_processed/padding/labels.npy", motion_num)
        np.save("../data/03_processed/padding/labels_name.npy", motion_name)
        np.save("../data/03_processed/padding/subjects.npy", subjects)

    if method == "resampling":
        np.save("../data/03_processed/resampling/input_model.npy", final_data)
        np.save("../data/03_processed/resampling/labels.npy", motion_num)
        np.save("../data/03_processed/resampling/labels_name.npy", motion_name)
        np.save("../data/03_processed/resampling/subjects.npy", subjects)

    if method == "interpolation":
        np.save("../data/03_processed/interpolation/input_model.npy", final_data)
        np.save("../data/03_processed/interpolation/labels_name.npy", motion_name)
        np.save("../data/03_processed/interpolation/labels.npy", motion_num)
        np.save("../data/03_processed/interpolation/subjects.npy", subjects)
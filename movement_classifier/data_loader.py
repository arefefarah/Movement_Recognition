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
from sklearn import preprocessing
from scipy.interpolate import CubicSpline
import scipy.io as sio
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from torch.nn import MaxPool1d
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def mat2dict(filename):
    """Converts MoVi mat files to a python nested dictionary.
    This makes a cleaner representation compared to sio.loadmat
    Arguments:
        filename {str} -- The path pointing to the .mat file which contains
        MoVi style mat structs
    Returns:
        dict -- A nested dictionary similar to the MoVi style MATLAB struct
    """
    # Reading MATLAB file
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)

    # Converting mat-objects to a dictionary
    for key in data:
        if key != "__header__" and key != "__global__" and key != "__version__":
            if isinstance(data[key], sio.matlab.mio5_params.mat_struct):
                data_out = matobj2dict(data[key])
    return data_out

def matobj2dict(matobj):
    """A recursive function which converts nested mat object
    to a nested python dictionaries
    Arguments:
        matobj {sio.matlab.mio5_params.mat_struct} -- nested mat object
    Returns:
        dict -- a nested dictionary
    """
    ndict = {}
    for fieldname in matobj._fieldnames:
        attr = matobj.__dict__[fieldname]
        if isinstance(attr, sio.matlab.mio5_params.mat_struct):
            ndict[fieldname] = matobj2dict(attr)
        elif isinstance(attr, np.ndarray) and fieldname == "move":
            for ind, val in np.ndenumerate(attr):
                ndict[
                    fieldname
                    + str(ind).replace(",", "").replace(")", "").replace("(", "_")
                    ] = matobj2dict(val)
        elif fieldname == "skel":
            tree = []
            for ind in range(len(attr)):
                tree.append(matobj2dict(attr[ind]))
            ndict[fieldname] = tree
        else:
            ndict[fieldname] = attr
    return ndict

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
            
            inf_dic = mat2dict(os.path.join(dir, filename))
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
            ddf = pd.concat([df_x, df_y], axis=1, join='inner')
            scaler = MinMaxScaler(feature_range=(-1, 1))
            df = pd.DataFrame(scaler.fit_transform(ddf), columns=ddf.columns)
            print(df.shape)
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
            inf_dic = mat2dict(os.path.join("../data/01_raw/F_Subjects/F_v3d_Subject_{}.mat".format(name)))
            m = inf_dic["move"]
            motions_list = list(m["motions_list"])
            for idx, element in enumerate(motions_list):
                if element.endswith("arms"):
                    motions_list[idx] = "crossarms"
                if element.startswith("jumping"):
                    motions_list[idx] = "jumping_jacks"
            timeflags = list(m["flags30"])
            # indices = np.where(motions_list.endswith("_rm"))
            # for i in indices:
            #     motions_list.pop()
            index = [idx for idx, element in enumerate(motions_list) if element.endswith("_rm")]        
            for d in range(len(index)):
                motions_list.pop(index[d])
                timeflags.pop(index[d])
            index = [idx for idx, element in enumerate(motions_list) if element=="punching_rm"]        
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
                    # print(m.shape)
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
                    # print(m.shape)
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

    print("all Dataframs have been created!")
    return(sub_info,movement_name_list,subjects)



def save_data(sub_info, movement_name_list,subjects, method = "padding", freq = None):
    # p=np.array(sub_info)
    # k = p.shape(0)
    p = sub_info
    jj = p[0]
    k =len(p)
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

    if method == "frequency":
        np.save("../data/03_processed/frequency/input_model.npy", final_data)
        np.save("../data/03_processed/frequency/labels_name.npy", motion_name)
        np.save("../data/03_processed/frequency/labels.npy", motion_num)
        np.save("../data/03_processed/frequency/subjects.npy", subjects)
    print("all Dataframes have been saved properly")



def load_data_dict(dir):
    """Loads numpy data files from a folder and assigns to dict entries based on filename.
    
    Because we return on the first loop iteration, this only loads files in the top-level 
    directory `dir`, not from subdirectories.
    """
    data = dict()
    for _, _, filenames in os.walk(dir):
        for filename in filenames:
            name, ext = os.path.splitext(filename)
            if ext == '.npy':
                data[name] = np.load(os.path.join(dir, filename))

        return data




def df_freq(joint1,amp,freq,joint2 = None):
    sub_info = []
    movement_name_list = []
    min_length,max_length,_,_ = timelength_loader("../data/01_raw/F_Subjects")
    method = "padding"
    dir = "../data/01_raw/CSV_files"
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

                # add specific frequency to one joint
                x = np.arange(df.shape[0])
                freqsin = pd.Series(amp*np.sin(freq*x))
                freqsin.index = df.index
                # print(df["hip1_x"].shape)
                
                df[joint1] = df[joint1].add(freqsin,axis =0)
                if joint2 is not None:
                    df[joint2] = df[joint2].add(freqsin,axis =0)
            
                # plot each channel(joint) for each subject
                # df.plot(subplots=True, layout=(7,4), figsize=(15,10))
                # plt.tight_layout()
                # plt.show()


                inf_dic = mat2dict(os.path.join("../data/01_raw/F_Subjects/F_v3d_Subject_{}.mat".format(name)))
                m = inf_dic["move"]
                motions_list = list(m["motions_list"])
                for idx, element in enumerate(motions_list):
                    if element.endswith("arms"):
                        motions_list[idx] = "crossarms"
                    if element.startswith("jumping"):
                        motions_list[idx] = "jumping_jacks"
            
                timeflags = list(m["flags30"])
                index = [idx for idx, element in enumerate(motions_list) if element.endswith("_rm")]        
                for d in range(len(index)):
                    motions_list.pop(index[d])
                    timeflags.pop(index[d])
                index = [idx for idx, element in enumerate(motions_list) if element=="punching_rm"]        
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
                        all_data.append(m)
                
                sub_info.append(np.array(all_data))  

    print("all Dataframes have been created!")
    return(sub_info,movement_name_list,subjects)
    
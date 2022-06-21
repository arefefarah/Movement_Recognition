import utils as utils
import data_loader as data_loader
import model_funcs as model_funcs

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



def df_freq(joint,amp,freq):
    sub_info = []
    movement_name_list = []
    min_length,max_length,_,_ = data_loader.timelength_loader("../data/01_raw/F_Subjects")
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
                df["knee1_x"] = df["knee1_x"].add(freqsin,axis =0)
                df[joint] = df[joint].add(freqsin,axis =0)
            
                # plot each channel(joint) for each subject
                # df.plot(subplots=True, layout=(7,4), figsize=(15,10))
                # plt.tight_layout()
                # plt.show()


                inf_dic = data_loader.mat2dict(os.path.join("../data/01_raw/F_Subjects/F_v3d_Subject_{}.mat".format(name)))
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
    
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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# #### This notebook is for loading data from Deeplabcut and preprocessing data to be used in our main model

# ## Data Loading....

# +
sys.path.insert(0, '../')
import movement_classifier.utils as utils
import movement_classifier.utils as utils


from os.path import dirname, join as pjoin
import os
import sys

import dlc2kinematics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly
from sklearn.decomposition import PCA
import seaborn as sns
import scipy.io as sio

# -

# ### Load Mat files

# +
subjects = [1,2,3,4,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
            31,34,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]
mat_files = []
for s in subjects:
    file = "F_v3d_Subject_{}.mat".format(s)
    mat_files.append(file)

mat_files   

# +
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

# dlc2kinematics.compute_umap(df, key=['LeftForelimb', 'RightForelimb'], chunk_length=30,fit_transform=True, n_neighbors=30, n_components=3,metric="euclidean")
# -

df_vel

# ### Observation of min and max length of movements

# +
all_motions = {}
# all_timelabels = {}

for f in mat_files:

    inf_dic = utils.mat2dict("../data/01_raw/F_Subjects/{}".format(f))
    m = inf_dic["move"]
    motions_list = m["motions_list"]
    timelabels = m["flags30"]
    
    p = []
    for i,val in enumerate(timelabels):

        p.append(val[1]-val[0])
#         print(val[1],val[0])
    if min(p)==34:
        print(motions_list[i])
    print("min: ",min(p), "     max: ", max(p))
#     plt.hist(p)
#     plt.show()
    
    for m in motions_list:
        if m in all_motions:
            pass
        else:
            all_motions[m] = len(all_motions.keys())

# -

all_motions

# ###  find min of time sequence for truncate all movements 

p = 0
min_length = []
for s in subjects:
#     file= [filename for filename in os.listdir("../data/01_raw/CSV_files") if filename.startswith("F_PG1_Subject_{}_".format(s))]
    inf_dic = utils.mat2dict("../data/01_raw/F_Subjects/F_v3d_Subject_{}.mat".format(s))
    m = inf_dic["move"]

    timeflags = m["flags30"]
            
    seq_lengths = []
    for i,val in enumerate(timeflags):
        seq_lengths.append(val[1]-val[0])
    treshold = min(seq_lengths)
    min_length.append(treshold)
#     p = p+i
#     print(i)
# print(p)

len(min_length)

all_motions

# ## 1- Build arrays based on Truncation 
#

# ### load csv files of each subject, Normalization, Truncate all time sequences, Build array of all data

    file= [filename for filename in os.listdir("../data/01_raw/CSV_files") if filename.startswith("F_PG1_Subject_{}_".format(21))]
    file = file[0]
    df = pd.read_csv("../data/01_raw/CSV_files/"+file)
    df.columns = (df.iloc[0] + '_' + df.iloc[1])
    df = df.iloc[2:].reset_index(drop=True)
    df = df.set_index(df.columns[0])
    df = df.astype('float64')
    df = df.round(4)
    df


for s in subjects:
    file= [filename for filename in os.listdir("../data/01_raw/CSV_files") if filename.startswith("F_PG1_Subject_{}_".format(s))]
    file = file[0]
     
    #read csv file 
    df = pd.read_csv("../data/01_raw/CSV_files/"+file)
    df.columns = (df.iloc[0] + '_' + df.iloc[1])
    df = df.iloc[2:].reset_index(drop=True)
    df = df.set_index(df.columns[0])
    df = df.astype('float64')
    df = df.round(4)
    
    #Normalization
    df_x = df.loc[:, df.columns.str.contains('_x')]
    df_y = df.loc[:, df.columns.str.contains('_y')]
    ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
    ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
    df_x = df_x.sub(ref_x,axis = 0)
    df_y = df_y.sub(ref_y,axis = 0)
    df = pd.concat([df_x, df_y], axis=1, join='inner')
    
    # Add column of movements
    inf_dic = utils.mat2dict("../data/01_raw/F_Subjects/F_v3d_Subject_{}.mat".format(s))
    m = inf_dic["move"]
    motions_list = m["motions_list"]
    timeflags = m["flags30"]
    
    #build arrays of all subjects
    all_data = []
    for i,val in enumerate(timeflags):
        subject_list.append(s)
        movement_list.append(all_motions[motions_list[i]])
        movement_name_list.append(motions_list[i])
        list_arrays = []
        N = max(max_length)-(val[1]-val[0])
        for c in df.columns:
            a= df[c].iloc[val[0]:val[1]]
            x = a.to_numpy()
            #padding
            val_arry = np.pad(x, (0, N), 'constant')
            list_arrays.append(val_arry)
        m =np.array(list_arrays)
        print(m.shape)
        all_data.append(m)
    h = np.array(all_data)
    print(h.shape)
        
    all_data_sub[s] = np.array(all_data)
    Sub_info.append(np.array(all_data))    

# +
subject_list = []
movement_list = []
movement_name_list = []
all_data_sub = {}
Sub_info = []
for s in subjects:
    file= [filename for filename in os.listdir("../data/01_raw/CSV_files") if filename.startswith("F_PG1_Subject_{}_".format(s))]
    file = file[0]
            
            #read csv file 
    df = pd.read_csv("../data/01_raw/CSV_files/"+file)
    df.columns = (df.iloc[0] + '_' + df.iloc[1])
    df = df.iloc[2:].reset_index(drop=True)
    df = df.set_index(df.columns[0])
    df = df.astype('float64')
    df = df.round(4)
    
    #Normalization
    df_x = df.loc[:, df.columns.str.contains('_x')]
    df_y = df.loc[:, df.columns.str.contains('_y')]
    ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
    ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
    df_x = df_x.sub(ref_x,axis = 0)
    df_y = df_y.sub(ref_y,axis = 0)
    df = pd.concat([df_x, df_y], axis=1, join='inner')
    
    # Add column of movements
    inf_dic = utils.mat2dict("../data/01_raw/F_Subjects/F_v3d_Subject_{}.mat".format(s))
    m = inf_dic["move"]
    motions_list = m["motions_list"]
#           b = [0]*df.shape[0]
#           df["Movement"] = b
    timeflags = m["flags30"]
#           for i,val in enumerate(timeflags):
#               df.iloc[val[0]:val[1],df.columns.get_loc("Movement")] = all_motions[motions_list[i]]
                
                
            # find min of time sequence for trucate process 
#             seq_lengths = []
#             for i,val in enumerate(timeflags):
#                 seq_lengths.append(val[1]-val[0])
#                 treshold = min(seq_lengths)
                
            
            #build arrays of all subjects
    all_data = []
    for i,val in enumerate(timeflags):
#               print(i)
        subject_list.append(s)
        movement_list.append(all_motions[motions_list[i]])
        movement_name_list.append(motions_list[i])
        list_arrays = []
        for c in df.columns:
            a= df[c].iloc[val[0]:val[0]+min(min_length)]
            val_arry = a.to_numpy()
            list_arrays.append(val_arry)
        m =np.array(list_arrays)
        all_data.append(m)
    print(np.array(all_data).shape)
        
    all_data_sub[s] = np.array(all_data)
    Sub_info.append(np.array(all_data))    
#             print("df has been completed")
            
            
        
# -

def unique(list1):
 
    # initialize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
#     for x in unique_list:
#         print(x)
unique(movement_list)

# +

p=np.array(Sub_info)
jj = p[0]
k =p.shape[0]
for t in range(k-1):
    jj = np.vstack((jj, p[t+1]))
# -

jj.shape

# ### save arrays

# +
Final_subjects = np.array(subject_list)
Final_motions = np.array(movement_list)
Final_data = jj
motion_name = np.array(movement_name_list) 


np.save("../data/03_processed/truncation/Input_model.npy", Final_data)
np.save("../data/03_processed/truncation/labels.npy", Final_motions)
np.save("../data/03_processed/truncation/labels_name.npy", motion_name)
np.save("../data/03_processed/truncation/subjects.npy", Final_subjects)
# -

Final_data.shape



# ## 2- Build arrays based on Resampling 
# (extract equal length of different movements with moving window)

# +
subject_list = []
movement_list = []
movement_name_list = []
all_data_sub = {}
Sub_info = []
for s in subjects:
    file= [filename for filename in os.listdir("../data/01_raw/CSV_files") if filename.startswith("F_PG1_Subject_{}_".format(s))]
    file = file[0]
               
    #read csv file 
    df = pd.read_csv("../data/01_raw/CSV_files/"+file)
    df.columns = (df.iloc[0] + '_' + df.iloc[1])
    df = df.iloc[2:].reset_index(drop=True)
    df = df.set_index(df.columns[0])
    df = df.astype('float64')
    df = df.round(4)
    
    #Normalization
    df_x = df.loc[:, df.columns.str.contains('_x')]
    df_y = df.loc[:, df.columns.str.contains('_y')]
    ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
    ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
    df_x = df_x.sub(ref_x,axis = 0)
    df_y = df_y.sub(ref_y,axis = 0)
    df = pd.concat([df_x, df_y], axis=1, join='inner')
    
    # Add column of movements
    inf_dic = utils.mat2dict("../data/01_raw/F_Subjects/F_v3d_Subject_{}.mat".format(s))
    m = inf_dic["move"]
    motions_list = m["motions_list"]
#           b = [0]*df.shape[0]
#           df["Movement"] = b
    timeflags = m["flags30"]
#           for i,val in enumerate(timeflags):
#               df.iloc[val[0]:val[1],df.columns.get_loc("Movement")] = all_motions[motions_list[i]]
        
        


                
    thresh = min(min_length)
    #build arrays of all subjects
    all_data = []
    for ind,val in enumerate(timeflags):
        num_samples = math.ceil((val[1]-val[0])/thresh)
        print("num_samples",num_samples)
        for i in range(num_samples):
            print(ind)
        
#                   print(i)
            if i+1 != num_samples:
                subject_list.append(s)
                movement_list.append(all_motions[motions_list[ind]])
                movement_name_list.append(motions_list[ind])
                list_arrays = []
                for c in df.columns:
                    a= df[c].iloc[val[0]+i*thresh:val[0]+i*thresh+thresh]
                    val_arry = a.to_numpy()
                    list_arrays.append(val_arry)
            else:
                subject_list.append(s)
                movement_list.append(all_motions[motions_list[ind]])
                movement_name_list.append(motions_list[ind])
                list_arrays = []
                for c in df.columns:
                    a= df[c].iloc[val[1]-thresh:val[1]]
                    val_arry = a.to_numpy()
                    list_arrays.append(val_arry)
                
            m =np.array(list_arrays)
            print(m.shape)
            all_data.append(m)
    print("np.array(all_data).shape",np.array(all_data).shape)
        
#           all_data_sub[s] = np.array(all_data)
    Sub_info.append(np.array(all_data))    
#           print("df has been completed")
            
            


# +
# movement_name_list

# +
p=np.array(Sub_info)
jj = p[0]
k =p.shape[0]
for t in range(k-1):
    jj = np.vstack((jj, p[t+1]))
    
jj.shape
# -



Final_subjects = np.array(subject_list)
Final_motions = np.array(movement_list)
Final_data = jj
motion_name = np.array(movement_name_list) 



# +


np.save("../data/03_processed/resampling/Input_model.npy", Final_data)
np.save("../data/03_processed/resampling/labels.npy", Final_motions)
np.save("../data/03_processed/resampling/labels_name.npy", motion_name)
np.save("../data/03_processed/resampling/subjects.npy", Final_subjects)
# -

df.columns

# ## 3- Build arrays based on Padding 
# (add 0 values to vectors with lower length)

# ### find the maximum length

# +
p = 0
max_length = []
for s in subjects:
#     file= [filename for filename in os.listdir("../data/01_raw/CSV_files") if filename.startswith("F_PG1_Subject_{}_".format(s))]
    inf_dic = utils.mat2dict("../data/01_raw/F_Subjects/F_v3d_Subject_{}.mat".format(s))
    m = inf_dic["move"]

    timeflags = m["flags30"]
            
    seq_lengths = []
    for i,val in enumerate(timeflags):
        seq_lengths.append(val[1]-val[0])
        treshold = max(seq_lengths)
        max_length.append(treshold)
#     p = p+i
#     print(i)
# print(p)

max(max_length)   
# -



subject_list = []
movement_list = []
movement_name_list = []
all_data_sub = {}
Sub_info = []
for s in subjects:
    file= [filename for filename in os.listdir("../data/01_raw/CSV_files") if filename.startswith("F_PG1_Subject_{}_".format(s))]
    file = file[0]
     
    #read csv file 
    df = pd.read_csv("../data/01_raw/CSV_files/"+file)
    df.columns = (df.iloc[0] + '_' + df.iloc[1])
    df = df.iloc[2:].reset_index(drop=True)
    df = df.set_index(df.columns[0])
    df = df.astype('float64')
    df = df.round(4)
    
    #Normalization
    df_x = df.loc[:, df.columns.str.contains('_x')]
    df_y = df.loc[:, df.columns.str.contains('_y')]
    ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
    ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
    df_x = df_x.sub(ref_x,axis = 0)
    df_y = df_y.sub(ref_y,axis = 0)
    df = pd.concat([df_x, df_y], axis=1, join='inner')
    
    # Add column of movements
    inf_dic = utils.mat2dict("../data/01_raw/F_Subjects/F_v3d_Subject_{}.mat".format(s))
    m = inf_dic["move"]
    motions_list = m["motions_list"]
    timeflags = m["flags30"]
    
    #build arrays of all subjects
    all_data = []
    for i,val in enumerate(timeflags):
        subject_list.append(s)
        movement_list.append(all_motions[motions_list[i]])
        movement_name_list.append(motions_list[i])
        list_arrays = []
        N = max(max_length)-(val[1]-val[0])
        for c in df.columns:
            a= df[c].iloc[val[0]:val[1]]
            x = a.to_numpy()
            #padding
            val_arry = np.pad(x, (0, N), 'constant')
            list_arrays.append(val_arry)
        m =np.array(list_arrays)
        print(m.shape)
        all_data.append(m)
    h = np.array(all_data)
    print(h.shape)
        
    all_data_sub[s] = np.array(all_data)
    Sub_info.append(np.array(all_data))    
#             print("df has been completed")


# +
p=np.array(Sub_info)
jj = p[0]
k =p.shape[0]
# print(jj.shape)
for t in range(k-1):
    jj = np.vstack((jj, p[t+1]))
#     print(jj.shape)
    
jj.shape

# +
Final_subjects = np.array(subject_list)
Final_motions = np.array(movement_list)
Final_data = jj
motion_name = np.array(movement_name_list) 


np.save("../data/03_processed/padding/Input_model.npy", Final_data)
np.save("../data/03_processed/padding/labels.npy", Final_motions)
np.save("../data/03_processed/padding/labels_name.npy", motion_name)
np.save("../data/03_processed/padding/subjects.npy", Final_subjects)
# -

# ## remove all random movements in Resampling with velocity

# +
all_motions = {}
# all_timelabels = {}

for f in mat_files:

    inf_dic = utils.mat2dict("../data/01_raw/F_Subjects/{}".format(f))
    m = inf_dic["move"]
    motions_list = list(m["motions_list"])
    index = [idx for idx, element in enumerate(motions_list) if element.endswith("_rm")]
    timeflags = list(m["flags30"])
    
    for d in range(len(index)):
        motions_list.pop(index[d])
        timeflags.pop(index[d])
    
    p = []
    for i,val in enumerate(timeflags):

        p.append(val[1]-val[0])
#             print(val[1],val[0])
    print("min: ",min(p), "     max: ", max(p))
#     plt.hist(p)
#     plt.show()
    
    for m in motions_list:
        if m in all_motions:
            pass
        else:
            all_motions[m] = len(all_motions.keys())
            


# +
            
            
p = 0
min_length = []
for s in subjects:
#     file= [filename for filename in os.listdir("../data/01_raw/CSV_files") if filename.startswith("F_PG1_Subject_{}_".format(s))]
    inf_dic = utils.mat2dict("../data/01_raw/F_Subjects/F_v3d_Subject_{}.mat".format(s))
    m = inf_dic["move"]
    motions_list = list(m["motions_list"])
    index = [idx for idx, element in enumerate(motions_list) if element.endswith("_rm")]
    timeflags = list(m["flags30"])
    
    for d in range(len(index)):
        motions_list.pop(index[d])
        timeflags.pop(index[d])
            
    seq_lengths = []
    for i,val in enumerate(timeflags):
        seq_lengths.append(val[1]-val[0])
    treshold = min(seq_lengths)
    min_length.append(treshold)
    
min(min_length)
# -



# +
subject_list = []
movement_list = []
movement_name_list = []
all_data_sub = {}
Sub_info = []
# s = 1
# if s==1:
for s in subjects:
    file= [filename for filename in os.listdir("../data/01_raw/CSV_files") if filename.startswith("F_PG1_Subject_{}_".format(s))]
    file = file[0]
               
    #read csv file 
    df = pd.read_csv("../data/01_raw/CSV_files/"+file)
    df.columns = (df.iloc[0] + '_' + df.iloc[1])
    df = df.iloc[2:].reset_index(drop=True)
    df = df.set_index(df.columns[0])
    df = df.astype('float64')
    df = df.round(4)
    
    #Normalization
    df_x = df.loc[:, df.columns.str.contains('_x')]
    df_y = df.loc[:, df.columns.str.contains('_y')]
    ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
    ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
    df_x = df_x.sub(ref_x,axis = 0)
    df_y = df_y.sub(ref_y,axis = 0)
    df = pd.concat([df_x, df_y], axis=1, join='inner')
    df.columns  = df.columns 
    
    
    # add velocity
    for (columnName, columnData) in df.iteritems():
        daa = []

        for i in range(len(columnData.values)):
            if i ==0:
                val= 0
            else:
                val = (columnData[i]-columnData[i-1])
            daa.append(val)
        name = columnName+"_velocity"
        df[name] = daa
    
    # Add column of movements
    inf_dic = utils.mat2dict("../data/01_raw/F_Subjects/F_v3d_Subject_{}.mat".format(s))
    m = inf_dic["move"]
    motions_list = list(m["motions_list"])
    index = [idx for idx, element in enumerate(motions_list) if element.endswith("_rm")]
    timeflags = list(m["flags30"])
    
    for d in range(len(index)):
        motions_list.pop(index[d])
        timeflags.pop(index[d])
    
                
    thresh = min(min_length)
    #build arrays of all subjects
    all_data = []
    for ind,val in enumerate(timeflags):
    
        num_samples = math.ceil((val[1]-val[0])/thresh)
        print("num_samples",num_samples)
        for i in range(num_samples):
            print(ind)
        
    #                  print(i)
            if i+1 != num_samples:
                subject_list.append(s)
                movement_list.append(all_motions[motions_list[ind]])
                movement_name_list.append(motions_list[ind])
                list_arrays = []
                for c in df.columns:
                    a= df[c].iloc[val[0]+i*thresh:val[0]+i*thresh+thresh]
                    val_arry = a.to_numpy()
                    list_arrays.append(val_arry)
            else:
                subject_list.append(s)
                movement_list.append(all_motions[motions_list[ind]])
                movement_name_list.append(motions_list[ind])
                list_arrays = []
                for c in df.columns:
                    a= df[c].iloc[val[1]-thresh:val[1]]
                    val_arry = a.to_numpy()
                    list_arrays.append(val_arry)
                
            m =np.array(list_arrays)
            print(m.shape)
            all_data.append(m)
    print("np.array(all_data).shape",np.array(all_data).shape)
        
#           all_data_sub[s] = np.array(all_data)
    Sub_info.append(np.array(all_data))    
#           print("df has been completed")


# -


p=np.array(Sub_info)
jj = p[0]
k =p.shape[0]
for t in range(k-1):
    jj = np.vstack((jj, p[t+1]))



# +
    
Final_subjects = np.array(subject_list)
Final_motions = np.array(movement_list)
Final_data = jj
motion_name = np.array(movement_name_list) 



np.save("../data/03_processed/resampling_deletedrandom/Input_model.npy", Final_data)
np.save("../data/03_processed/resampling_deletedrandom/labels.npy", Final_motions)
np.save("../data/03_processed/resampling_deletedrandom/labels_name.npy", motion_name)
np.save("../data/03_processed/resampling_deletedrandom/subjects.npy", Final_subjects)
# -

Final_data.shape

# ## remove all random movements in Padding with velocity

# +
all_motions = {}
# all_timelabels = {}

for f in mat_files:

    inf_dic = utils.mat2dict("../data/01_raw/F_Subjects/{}".format(f))
    m = inf_dic["move"]
    motions_list = list(m["motions_list"])
    for idx, element in enumerate(motions_list):
        if element.endswith("arms"):
            motions_list[idx] = "crossarms"
        if element.startswith("jumping"):
            motions_list[idx] = "jumping_jacks"
            
    index = [idx for idx, element in enumerate(motions_list) if element.endswith("_rm")]
    timelabels = list(m["flags30"])
    
    for d in range(len(index)):
        motions_list.pop(index[d])

        timelabels.pop(index[d])
        
    p = []
    for i,val in enumerate(timelabels):

        p.append(val[1]-val[0])
#         print(val[1],val[0])
    if min(p)==34:
        print(motions_list[i])
#     print("min: ",min(p), "     max: ", max(p))

    
    for m in motions_list:
        if m in all_motions:
            pass
        else:
            print(m)
            all_motions[m] = len(all_motions.keys())

# +
p = 0
max_length = []
for s in subjects:
#     file= [filename for filename in os.listdir("../data/01_raw/CSV_files") if filename.startswith("F_PG1_Subject_{}_".format(s))]
    inf_dic = utils.mat2dict("../data/01_raw/F_Subjects/F_v3d_Subject_{}.mat".format(s))
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
            
    seq_lengths = []
    for i,val in enumerate(timeflags):
        seq_lengths.append(val[1]-val[0])
        treshold = max(seq_lengths)
        max_length.append(treshold)
#     p = p+i
#     print(i)
# print(p)

max(max_length)   
# -

all_motions

# +
subject_list = []
movement_list = []
movement_name_list = []
all_data_sub = {}
Sub_info = []
for s in subjects:
    file= [filename for filename in os.listdir("../data/01_raw/CSV_files") if filename.startswith("F_PG1_Subject_{}_".format(s))]
    file = file[0]
     
    #read csv file 
    df = pd.read_csv("../data/01_raw/CSV_files/"+file)
    df.columns = (df.iloc[0] + '_' + df.iloc[1])
    df = df.iloc[2:].reset_index(drop=True)
    df = df.set_index(df.columns[0])
    df = df.astype('float64')
    df = df.round(4)
    
    #Normalization
    df_x = df.loc[:, df.columns.str.contains('_x')]
    df_y = df.loc[:, df.columns.str.contains('_y')]
    ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
    ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
    df_x = df_x.sub(ref_x,axis = 0)
    df_y = df_y.sub(ref_y,axis = 0)
    df = pd.concat([df_x, df_y], axis=1, join='inner')
    
    # add velocity
#     for (columnName, columnData) in df.iteritems():
#         daa = []

#         for i in range(len(columnData.values)):
#             if i ==0:
#                 val= 0
#             else:
#                 val = (columnData[i]-columnData[i-1])
#             daa.append(val)
#         name = columnName+"_velocity"
#         df[name] = daa
        
        
    # Add column of movements
    inf_dic = utils.mat2dict("../data/01_raw/F_Subjects/F_v3d_Subject_{}.mat".format(s))
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
    
    #build arrays of all subjects
    all_data = []
    for i,val in enumerate(timeflags):
        subject_list.append(s)
        movement_list.append(all_motions[motions_list[i]])
        movement_name_list.append(motions_list[i])
        list_arrays = []
        N = max(max_length)-(val[1]-val[0])
        for c in df.columns:
            a= df[c].iloc[val[0]:val[1]]
            x = a.to_numpy()
            #padding
            val_arry = np.pad(x, (0, N), 'constant')
            list_arrays.append(val_arry)
        m =np.array(list_arrays)
        print(m.shape)
        all_data.append(m)
    h = np.array(all_data)
    print(h.shape)
        
    all_data_sub[s] = np.array(all_data)
    Sub_info.append(np.array(all_data))    
#             print("df has been completed")
# -


p=np.array(Sub_info)
jj = p[0]
k =p.shape[0]
for t in range(k-1):
    jj = np.vstack((jj, p[t+1]))



# +
    
Final_subjects = np.array(subject_list)
Final_motions = np.array(movement_list)
Final_data = jj
motion_name = np.array(movement_name_list) 



np.save("../data/03_processed/padding_deletedrandom/Input_model.npy", Final_data)
np.save("../data/03_processed/padding_deletedrandom/labels.npy", Final_motions)
np.save("../data/03_processed/padding_deletedrandom/labels_name.npy", motion_name)
np.save("../data/03_processed/padding_deletedrandom/subjects.npy", Final_subjects)
# -

Final_data.shape



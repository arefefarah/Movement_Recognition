# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jupytext//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Data Loading....

import pandas as pd
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
from utils_fun import *

# ### Load Mat files

# +
subjects = [1,10,11,12,13,15]
mat_files = []
for s in subjects:
    file = "F_v3d_Subject_{}.mat".format(s)
    mat_files.append(file)

mat_files   
# -

all_motions = {}
# all_timelabels = {}
for f in mat_files:
    inf_dic = mat2dict("F_Subjects_1_45/{}".format(f))
    m = inf_dic["move"]
    motions_list = m["motions_list"]
    timelabels = m["flags30"]
    
    for m in motions_list:
        if m in all_motions:
            pass
        else:
            all_motions[m] = len(all_motions.keys())+1
all_motions

timelabels

for t,d in enumerate(motions_list):
    print(d)

# ### load csv files of each subject

# +
for filename in os.listdir("CSV_files"):
    for s in subjects:
        if filename.startswith("F_PG1_Subject_{}_".format(s)):
            file = filename
            df = pd.read_csv("CSV_files/"+file)
            df.columns = (df.iloc[0] + '_' + df.iloc[1])
            df = df.iloc[2:].reset_index(drop=True)
            df = df.set_index(df.columns[0])
            df = df.astype('float64')
            df = df.round(4)
            
            df_x = df.loc[:, df.columns.str.contains('_x')]

            df_y = df.loc[:, df.columns.str.contains('_y')]

            ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
            ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
            df_x = df_x.sub(ref_x,axis = 0)
            df_y = df_y.sub(ref_y,axis = 0)
            df = pd.concat([df_x, df_y], axis=1, join='inner')
            # Add column of movements
            inf_dic = mat2dict("F_Subjects_1_45/F_v3d_Subject_{}.mat".format(s))
            m = inf_dic["move"]
            motions_list = m["motions_list"]
            timeflags = m["flags30"]
            b = [0]*df.shape[0]
            df["Movement"] = b
            for i,val in enumerate(timeflags):
                df.iloc[val[0]:val[1],df.columns.get_loc("Movement")] = all_motions[motions_list[i]]
                
            print("df has been completed")
            
            
        
# -

df[4841:5370]



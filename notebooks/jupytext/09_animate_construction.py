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
from IPython.display import HTML
import celluloid
# -

joints = ['ankle1_x', 'knee1_x', 'hip1_x', 'hip2_x', 'knee2_x', 'ankle2_x',
       'wrist1_x', 'elbow1_x', 'shoulder1_x', 'shoulder2_x', 'elbow2_x',
       'wrist2_x', 'chin_x', 'forehead_x', 'ankle1_y', 'knee1_y', 'hip1_y',
       'hip2_y', 'knee2_y', 'ankle2_y', 'wrist1_y', 'elbow1_y', 'shoulder1_y',
       'shoulder2_y', 'elbow2_y', 'wrist2_y', 'chin_y', 'forehead_y']


# +
 
def df_animate(joint1,amp,freq,joint2 = None):
    sub_info = []
    movement_name_list = []
    min_length,max_length,_,_ = data_loader.timelength_loader("../data/01_raw/F_Subjects")
    method = "padding"
    dir = "../data/01_raw/CSV_files"
    subjects = []
    ### specify just one person to test animation
    dir = "../data/01_raw/CSV_files/F_PG1_Subject_15_LDLC_resnet101_shuffle1_103000_filtered.csv"
    name = 15
    if name == 15:
        df = pd.read_csv(dir)
    # for dirpathes, dirnames, filenames in os.walk(dir):
    #         for file in filenames:
    #             name, ext = os.path.splitext(file)
    #             # print(name)
    #             name= name.split("_")
    #             name  = name[3]
    #             subjects.append(name)
    #            df = pd.read_csv(os.path.join(dir, file))
                

                #preprocessing
        df.columns = (df.iloc[0] + '_' + df.iloc[1])
        df = df.iloc[2:].reset_index(drop=True)
        df = df.set_index(df.columns[0])
        df = df.astype('float64')
        df = df.round(4)

        #we don't have normalization for visualization
        df_x = df.loc[:, df.columns.str.contains('_x')]
        df_y = df.loc[:, df.columns.str.contains('_y')]
        # ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
        # ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
        # df_x = df_x.sub(ref_x,axis = 0)
        # df_y = df_y.sub(ref_y,axis = 0)
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
############# here for animation construction padding should replace the last value with 0 for the rest of the vector 
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
                    # val_arry = np.pad(x, (0, N), 'constant',constant_values=x[-1])
                    # list_arrays.append(val_arry)
                    #without padding for animation
                    list_arrays.append(x)
                m =np.array(list_arrays)
                all_data.append(m)
        
        # sub_info.append(np.array(all_data))  

    print("all Dataframes have been created!")
    return(all_data,movement_name_list)

# -

# %matplotlib inline
plt.rcParams['animation.ffmpeg_path'] = '../../../../usr/bin/ffmpeg'
# %matplotlib inline


# +

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from celluloid import Camera
# %matplotlib inline

amp = 30
joints_test =  ['ankle1_x', 'knee1_x','shoulder1_x',"elbow1_x","wrist1_x"]

# freqs = [0.1,0.2,0.4,0.6,0.8]
freqs = [0.2,0.5,0.8]
for joint1 in joints_test:
    for freq in freqs:
        joint2 = joint1.replace("1","2")
        all_data,movement_name_list = df_animate(joint1,amp,freq,joint2 )

        # Animate a single trajectory
        pick_traj = 5      # Select a trajectory to simulate

        # Set up the graph using Matplotlib
        fig, ax = plt.subplots(figsize=(8,3))
        ax.set(xlim=(0, 700), ylim=(700, 0))
        # ax.set_ylabel('Height/m', fontsize=15)
        # ax.set_xlabel('Range/m', fontsize=15)
        ax.get_xaxis().set_label_coords(0.5, 0.12)

        # Initiate camera
        # camera = Camera(fig)

        #specify just two motion
        movement_list = ["walking", "jumping_jacks"]
        
        for i,motion in enumerate(movement_name_list):
            if motion in movement_list:
                camera = Camera(fig)
                print(motion)
                data = all_data[i]
                df = pd.DataFrame(data).transpose()
                
                df.columns = joints
                for t in range(df.shape[0]):

                    # Projectile's trajectory
                    x = {}
                    y = {}
                    for j in range(int(len(joints)/2)):
                            x[joints[j]] = df[joints[j]].to_numpy()
                            y[joints[j+14]] = df[joints[j+14]].to_numpy()
                            # Show Projectile's location
                            ax.plot(x[joints[j]][t], y[joints[j+14]][t], marker='o', markersize=4, markeredgecolor='r', markerfacecolor='r')


                    # Show Projectile's trajectory
                    ax.plot([x["shoulder1_x"][t],x["chin_x"][t], x["shoulder2_x"][t]],[y["shoulder1_y"][t], y["chin_y"][t],y["shoulder2_y"][t]],'ro-')
                    ax.plot([x["shoulder1_x"][t],x["elbow1_x"][t], x["wrist1_x"][t]],[y["shoulder1_y"][t], y["elbow1_y"][t],y["wrist1_y"][t]],'ro-')
                    ax.plot([x["shoulder2_x"][t],x["elbow2_x"][t], x["wrist2_x"][t]],[y["shoulder2_y"][t], y["elbow2_y"][t],y["wrist2_y"][t]],'ro-')
                    ax.plot([x["shoulder1_x"][t],x["hip1_x"][t], x["hip2_x"][t], x["shoulder2_x"][t]],[y["shoulder1_y"][t], y["hip1_y"][t],y["hip2_y"][t],
                    y["shoulder2_y"][t]],'ro-')
                    ax.plot([x["hip1_x"][t],x["knee1_x"][t], x["ankle1_x"][t]],[y["hip1_y"][t],y["knee1_y"][t], y["ankle1_y"][t]],'ro-')
                    ax.plot([x["hip2_x"][t],x["knee2_x"][t], x["ankle2_x"][t]],[y["hip2_y"][t],y["knee2_y"][t], y["ankle2_y"][t]],'ro-')
                    ax.plot([x["forehead_x"][t],x["chin_x"][t]],[y["forehead_y"][t],y["chin_y"][t]],'ro-') 
            
                    # Capture frame
                    camera.snap()
                anim = camera.animate(interval = 30, repeat = False)
                # if motion == "throw/catch":
                #     motion = "throw&catch"
                anim.save('../reports/animations/symmetryjoints/{}_{}_{}.mp4'.format(joint1,freq,motion))
            else:
                print(motion, "not calculated")
            



# anim = camera.animate(interval = 40, repeat = True, repeat_delay = 500)

    

# -

path_file = "../data/03_processed/frequency"
data_dict = data_loader.load_data_dict(path_file)
data_array = data_dict["input_model"]
data_dict["labels_name"].shape
sample = data_array[1,:,:]
sample.shape


# %matplotlib inline

# !pwd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython import display
from IPython.display import HTML
from celluloid import Camera
# %matplotlib inline
plt.rcParams['animation.ffmpeg_path'] = '../../../../usr/bin/ffmpeg'
# %matplotlib inline

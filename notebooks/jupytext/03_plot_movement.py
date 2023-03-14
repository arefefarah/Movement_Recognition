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

# +
sys.path.insert(0, '../')
import movement_classifier.utils as utils
import movement_classifier.model_funcs as model_funcs



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
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import scipy.io as sio
import torch
import torch.nn as nn
from torch.nn import MaxPool1d
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# +
def plot_movements(x = None, y = None ,hist = False, axis_colors = "white"):

    fig, ax = plt.subplots()
    # plt.figure(figsize=(8,6))
    if hist:
        xlabels = np.unique(y)
        ax.hist(y)
        ax.set_xticklabels(xlabels, rotation=30)
    else:
        ax.plot(x,y)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.spines['top'].set_color(axis_colors)
    ax.spines['left'].set_color(axis_colors)
    ax.spines['right'].set_color(axis_colors)
    ax.xaxis.label.set_color(axis_colors)
    ax.yaxis.label.set_color(axis_colors)
    ax.tick_params(axis='y', colors=axis_colors)
    ax.tick_params(axis='x', colors=axis_colors)
    plt.show()




plot_movements(x = None, y = data_dict['labels_name'], hist = True,axis_colors = "white")


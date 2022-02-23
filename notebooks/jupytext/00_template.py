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

import pandas as pd
import sys
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
# sys.path.insert(0, '/home/arefe/My Project/my_project/utils')
# from DLC_functions import *
from sklearn.model_selection import train_test_split
# get some pytorch:
import torch
import torch.nn as nn
import pickle as pkl
from torch.nn import MaxPool1d
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy, rsatoolbox
# confusion matrix from sklearn
from sklearn.metrics import confusion_matrix
# to get some idea of how long stuff will take to complete:
import time
# to see how unbalanced the data is:
from collections import Counter
import rsatoolbox.data as rsd # abbreviation to deal with dataset
from sklearn.manifold import TSNE



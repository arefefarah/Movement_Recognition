# -*- coding: utf-8 -*-
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

# #### This notebook is for loading raw data of MotionDataset and visualization of data for all movements with different tools

# +
sys.path.insert(0, '../')
import movement_classifier.utils as utils
import movement_classifier.utils as utils

from os.path import dirname, join as pjoin
import os


import sklearn
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly
from sklearn.decomposition import PCA
import seaborn as sns
import scipy.io as sio
import pandas as pd

# +

mat_file = "F_v3d_Subject_1.mat"
sample = utils.mat2dict(mat_file)
sample.keys()
# -

inf_dict = pretty_dict(sample, print_type=False, indent=1)

# inf_dict.keys
m = sample["move"]
motions_list = m["motions_list"]
timeflags = m["flags30"]

motions_list
all_motions = {}
for m in motions_list:
    if m in all_motions:
        pass
    else:
        
        all_motions[m] = len(all_motions.keys())+1
all_motions

# ## Building DataFrame 

# +
df = pd.read_csv("F_PG1_Subject_1_LDLC_resnet101_myDLC_motion_fullmovieOct26shuffle1_103000.csv")
df.columns = (df.iloc[0] + '_' + df.iloc[1])
df = df.iloc[2:].reset_index(drop=True)
df = df.set_index(df.columns[0])
df = df.astype('float64')
df = df.round(4)

df_x = df.loc[:, df.columns.str.contains('_x')]

df_y = df.loc[:, df.columns.str.contains('_y')]

# df = pd.concat([df_x, df_y], axis=1, join='inner')
ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
df_x = df_x.sub(ref_x,axis = 0)
df_y = df_y.sub(ref_y,axis = 0)
df = pd.concat([df_x, df_y], axis=1, join='inner')
# -

Add one column

# +

b = [0]*df.shape[0]
df["Movement"] = b
for i in timeflags:
#     print(timeflags[i])
#     print(type(i))
    df.iloc[i[0]:i[1],df.columns.get_loc("Movement")] = 1
# df.iloc[414:525,df.columns.get_loc('Movement')] =2
# df.iloc[526:598,df.columns.get_loc('Movement')] =3
# df.iloc[617:702,df.columns.get_loc('Movement')] =4
# df.iloc[720:824,df.columns.get_loc('Movement')] =5
# df.iloc[826:935,df.columns.get_loc('Movement')] =6
# df.iloc[936:1027,df.columns.get_loc('Movement')] =7
# df.iloc[1028:1134,df.columns.get_loc("Movement")] =8
# df.iloc[1137:1340,df.columns.get_loc("Movement")] =9
# df.iloc[1341:1517,df.columns.get_loc("Movement")] =10
# df.iloc[1522:1599,df.columns.get_loc("Movement")] =11
# df.iloc[1624:1758, df.columns.get_loc('Movement')] = 12
# df.iloc[1760:1931,df.columns.get_loc("Movement")] =13
# df.iloc[1936:2050,df.columns.get_loc("Movement")] =14
# df.iloc[2058:2304,df.columns.get_loc("Movement")] =15
# df.iloc[2310:2421,df.columns.get_loc("Movement")] =16
# df.iloc[2433:2710,df.columns.get_loc("Movement")] =17
# df.iloc[2716:2872,df.columns.get_loc("Movement")] =18
# df.iloc[2884:3089,df.columns.get_loc("Movement")] =19
# df.iloc[3093:3211,df.columns.get_loc("Movement")] =20
# df.iloc[3236:3391,df.columns.get_loc("Movement")] =21
# -

# ## Load Dataframe of DeepLabCut 

df = pd.read_csv("F_PG1_Subject_1_LDLC_resnet101_myDLC_motion_fullmovieOct26shuffle1_103000.csv")
df.columns = (df.iloc[0] + '_' + df.iloc[1])
df = df.iloc[2:].reset_index(drop=True)
df = df.set_index(df.columns[0])
df = df.astype('float64')
df = df.round(4)
# df

# ## Ref point for Normalization 

m = (df["hip1_y"]-df["hip2_y"])/2
m.plot()
plt.show()

df["hip2_y"].plot()


df["chin_x"].plot()


df["wrist1_x"].plot()

# ## Normalization
# ### Method1
# #### 1 - Choose avarage position of two hips during the time as a fixed refrence point for all markers during the time
# #### 2- Normalize each feature(marker) during the time

# +
m = df["hip1_x"]
hip1_x_mean = np.mean(m)
print(hip1_x_mean)
m = df["hip2_x"]
hip2_x_mean = np.mean(m)
print(hip2_x_mean)
ref_x = hip1_x_mean+((hip2_x_mean - hip1_x_mean)/2)
print("Refrence point for X cordination:  ", ref_x)


m = df["hip1_y"]
hip1_y_mean = np.mean(m)
print(hip1_y_mean)
m = df["hip2_y"]
hip2_y_mean = np.mean(m)
print(hip2_y_mean)
ref_y = hip1_y_mean+ ((hip2_y_mean - hip1_y_mean)/2)
print("Refrence point for Y cordination:  ", ref_y)
# -

df_x = df.loc[:, df.columns.str.contains('_x')]
df_x = df_x.sub(ref_x)
df_y = df.loc[:, df.columns.str.contains('_y')]
df_y = df_y.sub(ref_y)
df = pd.concat([df_x, df_y], axis=1, join='inner')


df_x = df.loc[:, df.columns.str.contains('_x')]
df_x = df_x.sub(ref_x)

# +
# df_y

# +
normalized_df=(df-df.mean())/df.std()

pca = PCA(n_components=4)
pca.fit(normalized_df)
y = pca.fit_transform(normalized_df)
df = pd.DataFrame(y, columns = ['pc1','pc2',"pc3","pc4"])
# -
# ### add one column to indicate time label of different movements

# +

b = [0]*3441
df["Movement"] = b
df.iloc[268:413,df.columns.get_loc('Movement')] =1
df.iloc[414:525,df.columns.get_loc('Movement')] =2
df.iloc[526:598,df.columns.get_loc('Movement')] =3
df.iloc[617:702,df.columns.get_loc('Movement')] =4
df.iloc[720:824,df.columns.get_loc('Movement')] =5
df.iloc[826:935,df.columns.get_loc('Movement')] =6
df.iloc[936:1027,df.columns.get_loc('Movement')] =7
df.iloc[1028:1134,df.columns.get_loc("Movement")] =8
df.iloc[1137:1340,df.columns.get_loc("Movement")] =9
df.iloc[1341:1517,df.columns.get_loc("Movement")] =10
df.iloc[1522:1599,df.columns.get_loc("Movement")] =11
df.iloc[1624:1758, df.columns.get_loc('Movement')] = 12
df.iloc[1760:1931,df.columns.get_loc("Movement")] =13
df.iloc[1936:2050,df.columns.get_loc("Movement")] =14
df.iloc[2058:2304,df.columns.get_loc("Movement")] =15
df.iloc[2310:2421,df.columns.get_loc("Movement")] =16
df.iloc[2433:2710,df.columns.get_loc("Movement")] =17
df.iloc[2716:2872,df.columns.get_loc("Movement")] =18
df.iloc[2884:3089,df.columns.get_loc("Movement")] =19
df.iloc[3093:3211,df.columns.get_loc("Movement")] =20
df.iloc[3236:3391,df.columns.get_loc("Movement")] =21
# -



# ## Sequence of Movements in Subject 1

# (1) kicking,
# (2) running in place,
# (3) pointing, 
# (4) clapping hands,
# (5) jumping jacks,
# (6) stretching,
# (7) crossing arms, 
# (8) jogging, 
# (9) crawling, 
# (10) walking, 
# (11) waving,
# (12) checking one’s watch
# (13) side gallop, 
# (14) vertical jumping,  
# (15) sitting down on a chair,
# (16) taking a picture, 
# (17) crossing legs while sitting, 
# (18) throwing and catching, 
# (19) performing a self-chosen movement. 
# (20) scratching one’s head, 
# (21) talking on the phone, 

# +
# df_plot = df.loc[df["Movement"] ==9]
df_plot = df
columns = ['pc1','pc2',"pc3","pc4"]
features = columns

fig = px.scatter_matrix(
    df_plot,
    dimensions=features,
color = "Movement")
fig.update_traces(diagonal_visible=False)
fig.show()
# -

# ## Normalization
# ### Method2
# #### 1 - Choose avarage position of two hips in each time for each cordinate as a  refrence point for all markers in that specific time
# #### 2- Normalize each feature(marker) during the time

# +
df = pd.read_csv("F_PG1_Subject_1_LDLC_resnet101_myDLC_motion_fullmovieOct26shuffle1_103000.csv")
df.columns = (df.iloc[0] + '_' + df.iloc[1])
df = df.iloc[2:].reset_index(drop=True)
df = df.set_index(df.columns[0])
df = df.astype('float64')
df = df.round(4)

df_x = df.loc[:, df.columns.str.contains('_x')]

df_y = df.loc[:, df.columns.str.contains('_y')]

# df = pd.concat([df_x, df_y], axis=1, join='inner')
ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
df_x = df_x.sub(ref_x,axis = 0)
df_y = df_y.sub(ref_y,axis = 0)
df = pd.concat([df_x, df_y], axis=1, join='inner')
# normalized_df=(df-df.mean())/df.std()
# normalized_df = df

# +
# Scaling

# in rows:
normalizedrow_df = df.div(df.sum(axis=1), axis=0)

#in columns:
normalizedcolumn_df=df.div(df.sum(axis=0), axis=1)
# normalizedcolumn_df

# +
df["wrist1_x"].plot()
plt.show()
normalizedrow_df["wrist1_x"].plot()
plt.show()

normalizedcolumn_df["wrist1_x"].plot()
plt.show()


# -

# ## PCA 

pca = PCA(n_components=4)
pca.fit(df)
y = pca.fit_transform(df)
df = pd.DataFrame(y, columns = ['pc1','pc2',"pc3","pc4"])

# ### Sequence of Movements in Subject 1

# (1) kicking,
# (2) 'dancing_rm',
# (3) pointing, 
# (4) clapping hands,
# (5) jumping jacks,
# (6) stretching,
# (7) crossing arms, 
# (8) 'running_in_spot', 
# (9) crawling, 
# (10) walking, 
# (11) hand waving,
# (12) checking one’s watch
# (13) side gallop, 
# (14) vertical jumping,  
# (15) sitting down on a chair,
# (16) taking a picture, 
# (17) crossing legs while sitting, 
# (18) throwing and catching, 
# (19) jogging. 
# (20) scratching one’s head, 
# (21) talking on the phone, 

# ### Add one column to indicate time label of different movements

# +

b = [0]*3441
df["Movement"] = b
df.iloc[268:413,df.columns.get_loc('Movement')] =1
df.iloc[414:525,df.columns.get_loc('Movement')] =2
df.iloc[526:598,df.columns.get_loc('Movement')] =3
df.iloc[617:702,df.columns.get_loc('Movement')] =4
df.iloc[720:824,df.columns.get_loc('Movement')] =5
df.iloc[826:935,df.columns.get_loc('Movement')] =6
df.iloc[936:1027,df.columns.get_loc('Movement')] =7
df.iloc[1028:1134,df.columns.get_loc("Movement")] =8
df.iloc[1137:1340,df.columns.get_loc("Movement")] =9
df.iloc[1341:1517,df.columns.get_loc("Movement")] =10
df.iloc[1522:1599,df.columns.get_loc("Movement")] =11
df.iloc[1624:1758, df.columns.get_loc('Movement')] = 12
df.iloc[1760:1931,df.columns.get_loc("Movement")] =13
df.iloc[1936:2050,df.columns.get_loc("Movement")] =14
df.iloc[2058:2304,df.columns.get_loc("Movement")] =15
df.iloc[2310:2421,df.columns.get_loc("Movement")] =16
df.iloc[2433:2710,df.columns.get_loc("Movement")] =17
df.iloc[2716:2872,df.columns.get_loc("Movement")] =18
df.iloc[2884:3089,df.columns.get_loc("Movement")] =19
df.iloc[3093:3211,df.columns.get_loc("Movement")] =20
df.iloc[3236:3391,df.columns.get_loc("Movement")] =21
df_sub1 = df
columns = ['pc1','pc2',"pc3","pc4"]
features = columns

fig = px.scatter_matrix(
    df_sub1,
    dimensions=features,
color = "Movement")
fig.update_traces(diagonal_visible=False)
fig.show()
# -

# ### Sequence of Movements in Subject 15
#

# 'sitting_down'15
# 'sideways'13
# 'kicking'1
# 'squatting_rm'2
# 'vertical_jumping'14
# 'running_in_spot'8
# 'jumping_jack'5
# 'crawling'9
# 'pointing'3
# 'taking_photo'16
# 'cross_legged_sitting'17
# 'phone_talking'21
# 'scratching_head'20
# 'jogging'19
# 'throw/catch'18
# 'walking'10
# 'stretching'6
# 'checking_watch'12
# 'crossarms'7
# 'hand_clapping'4
# 'hand_waving'11

# +
df = pd.read_csv("F_PG1_Subject_15_LDLC_resnet101_subject15Oct31shuffle1_103000_filtered.csv")
df.columns = (df.iloc[0] + '_' + df.iloc[1])
df = df.iloc[2:].reset_index(drop=True)
df = df.set_index(df.columns[0])
df = df.astype('float64')
df = df.round(4)

df_x = df.loc[:, df.columns.str.contains('_x')]

df_y = df.loc[:, df.columns.str.contains('_y')]

# df = pd.concat([df_x, df_y], axis=1, join='inner')
ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
df_x = df_x.sub(ref_x,axis = 0)
df_y = df_y.sub(ref_y,axis = 0)
df= pd.concat([df_x, df_y], axis=1, join='inner')
# pca = PCA(n_components=4)
# pca.fit(df)
y = pca.fit_transform(df)
df = pd.DataFrame(y, columns = ['pc1','pc2',"pc3","pc4"])

b = [0]*4462
df["Movement"] = b
df.iloc[138:358,df.columns.get_loc('Movement')] =15
df.iloc[382:666,df.columns.get_loc('Movement')] =13
df.iloc[686:835,df.columns.get_loc('Movement')] =1
df.iloc[855:1001,df.columns.get_loc('Movement')] =2
df.iloc[1030:1135,df.columns.get_loc('Movement')] =14
df.iloc[1229:1382,df.columns.get_loc('Movement')] =8
df.iloc[1398:1526,df.columns.get_loc('Movement')] =5
df.iloc[1569:1935,df.columns.get_loc("Movement")] =9
df.iloc[1944:2048,df.columns.get_loc("Movement")] =3
df.iloc[2103:2228,df.columns.get_loc("Movement")] =16
df.iloc[2259:2489,df.columns.get_loc("Movement")] =17
df.iloc[2496:2713, df.columns.get_loc('Movement')] = 21
df.iloc[2722:2829,df.columns.get_loc("Movement")] =20
df.iloc[2885:3125,df.columns.get_loc("Movement")] =19
df.iloc[3137:3357,df.columns.get_loc("Movement")] =18
df.iloc[3373:3644,df.columns.get_loc("Movement")] =10
df.iloc[3661:3798,df.columns.get_loc("Movement")] =6
df.iloc[3819:3907,df.columns.get_loc("Movement")] =12
df.iloc[3938:4068,df.columns.get_loc("Movement")] =7
df.iloc[4118:4235,df.columns.get_loc("Movement")] =4
df.iloc[4264:4352,df.columns.get_loc("Movement")] =11
df_sub15 = df
columns = ['pc1','pc2',"pc3","pc4"]
features = columns

fig = px.scatter_matrix(
    df_sub15,
    dimensions=features,
color = "Movement")
fig.update_traces(diagonal_visible=False)
fig.show()
# -

# ### Seperate X & Y cordination

# # X
#

# + active=""
#
# pca = PCA(n_components=4)
# pca.fit(df_x)
# out = pca.fit_transform(df_x)
# df_x = pd.DataFrame(out, columns = ['pc1','pc2',"pc3","pc4"])
#
# b = [0]*3441
# df_x["Movement"] = b
# df_x.iloc[268:413,df_x.columns.get_loc('Movement')] =1
# df_x.iloc[414:525,df_x.columns.get_loc('Movement')] =2
# df_x.iloc[526:598,df_x.columns.get_loc('Movement')] =3
# df_x.iloc[617:702,df_x.columns.get_loc('Movement')] =4
# df_x.iloc[720:824,df_x.columns.get_loc('Movement')] =5
# df_x.iloc[826:935,df_x.columns.get_loc('Movement')] =6
# df_x.iloc[936:1027,df_x.columns.get_loc('Movement')] =7
# df_x.iloc[1028:1134,df_x.columns.get_loc("Movement")] =8
# df_x.iloc[1137:1340,df_x.columns.get_loc("Movement")] =9
# df_x.iloc[1341:1517,df_x.columns.get_loc("Movement")] =10
# df_x.iloc[1522:1599,df_x.columns.get_loc("Movement")] =11
# df_x.iloc[1624:1758,df_x.columns.get_loc('Movement')] = 12
# df_x.iloc[1760:1931,df_x.columns.get_loc("Movement")] =13
# df_x.iloc[1936:2050,df_x.columns.get_loc("Movement")] =14
# df_x.iloc[2058:2304,df_x.columns.get_loc("Movement")] =15
# df_x.iloc[2310:2421,df_x.columns.get_loc("Movement")] =16
# df_x.iloc[2433:2710,df_x.columns.get_loc("Movement")] =17
# df_x.iloc[2716:2872,df_x.columns.get_loc("Movement")] =18
# df_x.iloc[2884:3089,df_x.columns.get_loc("Movement")] =19
# df_x.iloc[3093:3211,df_x.columns.get_loc("Movement")] =20
# df_x.iloc[3236:3391,df_x.columns.get_loc("Movement")] =21
#
#
# columns = ['pc1','pc2',"pc3","pc4"]
# features = columns
#
# fig = px.scatter_matrix(
#     df_x,
#     dimensions=features,
# color = "Movement")
# fig.update_traces(diagonal_visible=False)
# fig.show()
# -

# # Y

# + active=""
# normalized_df_y=(df_y-df_y.mean())/df_y.std()
#
# pca = PCA(n_components=4)
# pca.fit(normalized_df_y)
# out = pca.fit_transform(normalized_df_y)
# df_y = pd.DataFrame(out, columns = ['pc1','pc2',"pc3","pc4"])
#
# b = [0]*3441
# df_y["Movement"] = b
# df_y.iloc[268:413,df_y.columns.get_loc('Movement')] =1
# df_y.iloc[414:525,df_y.columns.get_loc('Movement')] =2
# df_y.iloc[526:598,df_y.columns.get_loc('Movement')] =3
# df_y.iloc[617:702,df_y.columns.get_loc('Movement')] =4
# df_y.iloc[720:824,df_y.columns.get_loc('Movement')] =5
# df_y.iloc[826:935,df_y.columns.get_loc('Movement')] =6
# df_y.iloc[936:1027,df_y.columns.get_loc('Movement')] =7
# df_y.iloc[1028:1134,df_y.columns.get_loc("Movement")] =8
# df_y.iloc[1137:1340,df_y.columns.get_loc("Movement")] =9
# df_y.iloc[1341:1517,df_y.columns.get_loc("Movement")] =10
# df_y.iloc[1522:1599,df_y.columns.get_loc("Movement")] =11
# df_y.iloc[1624:1758,df_y.columns.get_loc('Movement')] = 12
# df_y.iloc[1760:1931,df_y.columns.get_loc("Movement")] =13
# df_y.iloc[1936:2050,df_y.columns.get_loc("Movement")] =14
# df_y.iloc[2058:2304,df_y.columns.get_loc("Movement")] =15
# df_y.iloc[2310:2421,df_y.columns.get_loc("Movement")] =16
# df_y.iloc[2433:2710,df_y.columns.get_loc("Movement")] =17
# df_y.iloc[2716:2872,df_y.columns.get_loc("Movement")] =18
# df_y.iloc[2884:3089,df_y.columns.get_loc("Movement")] =19
# df_y.iloc[3093:3211,df_y.columns.get_loc("Movement")] =20
# df_y.iloc[3236:3391,df_y.columns.get_loc("Movement")] =21
#
#
# columns = ['pc1','pc2',"pc3","pc4"]
# features = columns
#
# fig = px.scatter_matrix(
#     df_y,
#     dimensions=features,
# color = "Movement")
# fig.update_traces(diagonal_visible=False)
# fig.show()
# -

# ## Method 2 of Normalization has been chosen  :)
# ## ---------------------------------------------------------------------------------------------------------------------------

# # NonLinear Dimention Reduction 
# ##  t-SNE 
# perplexity = 50
# ,n = 2

# +
df = pd.read_csv("F_PG1_Subject_1_LDLC_resnet101_myDLC_motion_fullmovieOct26shuffle1_103000.csv")
df.columns = (df.iloc[0] + '_' + df.iloc[1])
df = df.iloc[2:].reset_index(drop=True)
df = df.set_index(df.columns[0])
df = df.astype('float64')
df = df.round(4)

df_x = df.loc[:, df.columns.str.contains('_x')]

df_y = df.loc[:, df.columns.str.contains('_y')]

# df = pd.concat([df_x, df_y], axis=1, join='inner')
ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
df_x = df_x.sub(ref_x,axis = 0)
df_y = df_y.sub(ref_y,axis = 0)
df = pd.concat([df_x, df_y], axis=1, join='inner')



# +
from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=0,perplexity=50, n_iter=2000)

tsne_data = model.fit_transform(df)

# +
df= pd.DataFrame(tsne_data, columns = ['first','second'])
b = [0]*3441
df["Movement"] = b
df.iloc[268:413,df.columns.get_loc('Movement')] =1
df.iloc[414:525,df.columns.get_loc('Movement')] =2
df.iloc[526:598,df.columns.get_loc('Movement')] =3
df.iloc[617:702,df.columns.get_loc('Movement')] =4
df.iloc[720:824,df.columns.get_loc('Movement')] =5
df.iloc[826:935,df.columns.get_loc('Movement')] =6
df.iloc[936:1027,df.columns.get_loc('Movement')] =7
df.iloc[1028:1134,df.columns.get_loc("Movement")] =8
df.iloc[1137:1340,df.columns.get_loc("Movement")] =9
df.iloc[1341:1517,df.columns.get_loc("Movement")] =10
df.iloc[1522:1599,df.columns.get_loc("Movement")] =11
df.iloc[1624:1758, df.columns.get_loc('Movement')] = 12
df.iloc[1760:1931,df.columns.get_loc("Movement")] =13
df.iloc[1936:2050,df.columns.get_loc("Movement")] =14
df.iloc[2058:2304,df.columns.get_loc("Movement")] =15
df.iloc[2310:2421,df.columns.get_loc("Movement")] =16
df.iloc[2433:2710,df.columns.get_loc("Movement")] =17
df.iloc[2716:2872,df.columns.get_loc("Movement")] =18
df.iloc[2884:3089,df.columns.get_loc("Movement")] =19
df.iloc[3093:3211,df.columns.get_loc("Movement")] =20
df.iloc[3236:3391,df.columns.get_loc("Movement")] =21
columns = ['first','second']
features = columns

fig = px.scatter_matrix(
    df,
    dimensions=features,
color = "Movement")
fig.update_traces(diagonal_visible=False)
fig.show()
# -

# ### tSNE for other subject

# +
df = pd.read_csv("F_PG1_Subject_15_LDLC_resnet101_subject15Oct31shuffle1_103000_filtered.csv")
df.columns = (df.iloc[0] + '_' + df.iloc[1])
df = df.iloc[2:].reset_index(drop=True)
df = df.set_index(df.columns[0])
df = df.astype('float64')
df = df.round(4)

df_x = df.loc[:, df.columns.str.contains('_x')]

df_y = df.loc[:, df.columns.str.contains('_y')]

# df = pd.concat([df_x, df_y], axis=1, join='inner')
ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
df_x = df_x.sub(ref_x,axis = 0)
df_y = df_y.sub(ref_y,axis = 0)
df= pd.concat([df_x, df_y], axis=1, join='inner')

model = TSNE(n_components=2, random_state=0,perplexity=50, n_iter=2000)

tsne_data = model.fit_transform(df)
df= pd.DataFrame(tsne_data, columns = ['first','second'])

b = [0]*4462
df["Movement"] = b
df.iloc[138:358,df.columns.get_loc('Movement')] =15
df.iloc[382:666,df.columns.get_loc('Movement')] =13
df.iloc[686:835,df.columns.get_loc('Movement')] =1
df.iloc[855:1001,df.columns.get_loc('Movement')] =2
df.iloc[1030:1135,df.columns.get_loc('Movement')] =14
df.iloc[1229:1382,df.columns.get_loc('Movement')] =8
df.iloc[1398:1526,df.columns.get_loc('Movement')] =5
df.iloc[1569:1935,df.columns.get_loc("Movement")] =9
df.iloc[1944:2048,df.columns.get_loc("Movement")] =3
df.iloc[2103:2228,df.columns.get_loc("Movement")] =16
df.iloc[2259:2489,df.columns.get_loc("Movement")] =17
df.iloc[2496:2713, df.columns.get_loc('Movement')] = 21
df.iloc[2722:2829,df.columns.get_loc("Movement")] =20
df.iloc[2885:3125,df.columns.get_loc("Movement")] =19
df.iloc[3137:3357,df.columns.get_loc("Movement")] =18
df.iloc[3373:3644,df.columns.get_loc("Movement")] =10
df.iloc[3661:3798,df.columns.get_loc("Movement")] =6
df.iloc[3819:3907,df.columns.get_loc("Movement")] =12
df.iloc[3938:4068,df.columns.get_loc("Movement")] =7
df.iloc[4118:4235,df.columns.get_loc("Movement")] =4
df.iloc[4264:4352,df.columns.get_loc("Movement")] =11
df_sub15 = df
columns = ['first','second']
features = columns

fig = px.scatter_matrix(
    df_sub15,
    dimensions=features,
color = "Movement")
fig.update_traces(diagonal_visible=False)
fig.show()
# -

# #### All time labels of  one specific movement should not be similar because there is no repeated pattern! it is just a process of movement.

# +
df = pd.read_csv("F_PG1_Subject_1_LDLC_resnet101_myDLC_motion_fullmovieOct26shuffle1_103000.csv")
df.columns = (df.iloc[0] + '_' + df.iloc[1])
df = df.iloc[2:].reset_index(drop=True)
df = df.set_index(df.columns[0])
df = df.astype('float64')
df = df.round(4)
df_x = df.loc[:, df.columns.str.contains('_x')]
df_y = df.loc[:, df.columns.str.contains('_y')]

# df = pd.concat([df_x, df_y], axis=1, join='inner')
ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
df_x = df_x.sub(ref_x,axis = 0)
df_y = df_y.sub(ref_y,axis = 0)
df = pd.concat([df_x, df_y], axis=1, join='inner')

b = [0]*3441
df["Movement"] = b
df.iloc[268:413,df.columns.get_loc('Movement')] =1
df.iloc[414:525,df.columns.get_loc('Movement')] =2
df.iloc[526:598,df.columns.get_loc('Movement')] =3
df.iloc[617:702,df.columns.get_loc('Movement')] =4
df.iloc[720:824,df.columns.get_loc('Movement')] =5
df.iloc[826:935,df.columns.get_loc('Movement')] =6
df.iloc[936:1027,df.columns.get_loc('Movement')] =7
df.iloc[1028:1134,df.columns.get_loc("Movement")] =8
df.iloc[1137:1340,df.columns.get_loc("Movement")] =9
df.iloc[1341:1517,df.columns.get_loc("Movement")] =10
df.iloc[1522:1599,df.columns.get_loc("Movement")] =11
df.iloc[1624:1758, df.columns.get_loc('Movement')] = 12
df.iloc[1760:1931,df.columns.get_loc("Movement")] =13
df.iloc[1936:2050,df.columns.get_loc("Movement")] =14
df.iloc[2058:2304,df.columns.get_loc("Movement")] =15
df.iloc[2310:2421,df.columns.get_loc("Movement")] =16
df.iloc[2433:2710,df.columns.get_loc("Movement")] =17
df.iloc[2716:2872,df.columns.get_loc("Movement")] =18
df.iloc[2884:3089,df.columns.get_loc("Movement")] =19
df.iloc[3093:3211,df.columns.get_loc("Movement")] =20
df.iloc[3236:3391,df.columns.get_loc("Movement")] =21
df_sub1 = df

df = pd.read_csv("F_PG1_Subject_15_LDLC_resnet101_subject15Oct31shuffle1_103000_filtered.csv")
df.columns = (df.iloc[0] + '_' + df.iloc[1])
df = df.iloc[2:].reset_index(drop=True)
df = df.set_index(df.columns[0])
df = df.astype('float64')
df = df.round(4)

df_x = df.loc[:, df.columns.str.contains('_x')]

df_y = df.loc[:, df.columns.str.contains('_y')]

# df = pd.concat([df_x, df_y], axis=1, join='inner')
ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
df_x = df_x.sub(ref_x,axis = 0)
df_y = df_y.sub(ref_y,axis = 0)
df= pd.concat([df_x, df_y], axis=1, join='inner')

b = [0]*4462
df["Movement"] = b
df.iloc[138:358,df.columns.get_loc('Movement')] =15
df.iloc[382:666,df.columns.get_loc('Movement')] =13
df.iloc[686:835,df.columns.get_loc('Movement')] =1
df.iloc[855:1001,df.columns.get_loc('Movement')] =2
df.iloc[1030:1135,df.columns.get_loc('Movement')] =14
df.iloc[1229:1382,df.columns.get_loc('Movement')] =8
df.iloc[1398:1526,df.columns.get_loc('Movement')] =5
df.iloc[1569:1935,df.columns.get_loc("Movement")] =9
df.iloc[1944:2048,df.columns.get_loc("Movement")] =3
df.iloc[2103:2228,df.columns.get_loc("Movement")] =16
df.iloc[2259:2489,df.columns.get_loc("Movement")] =17
df.iloc[2496:2713, df.columns.get_loc('Movement')] = 21
df.iloc[2722:2829,df.columns.get_loc("Movement")] =20
df.iloc[2885:3125,df.columns.get_loc("Movement")] =19
df.iloc[3137:3357,df.columns.get_loc("Movement")] =18
df.iloc[3373:3644,df.columns.get_loc("Movement")] =10
df.iloc[3661:3798,df.columns.get_loc("Movement")] =6
df.iloc[3819:3907,df.columns.get_loc("Movement")] =12
df.iloc[3938:4068,df.columns.get_loc("Movement")] =7
df.iloc[4118:4235,df.columns.get_loc("Movement")] =4
df.iloc[4264:4352,df.columns.get_loc("Movement")] =11
df_sub15 = df

# +

fig = px.scatter(
    df_sub1["wrist1_x"],color = df_sub1["Movement"])

fig.show()
fig = px.scatter(
    df_sub15["wrist1_x"],color = df_sub15["Movement"])

fig.show()

# +

fig = px.scatter(
    df_sub1["knee1_y"],color = df_sub1["Movement"])
# fig.update_traces(diagonal_visible=False)
fig.show()
fig = px.scatter(
    df_sub15["knee1_y"],color = df_sub15["Movement"])
# fig.update_traces(diagonal_visible=False)
fig.show()
# -

# ### using PCA

# +



df = pd.read_csv("F_PG1_Subject_1_LDLC_resnet101_myDLC_motion_fullmovieOct26shuffle1_103000.csv")
df.columns = (df.iloc[0] + '_' + df.iloc[1])
df = df.iloc[2:].reset_index(drop=True)
df = df.set_index(df.columns[0])
df = df.astype('float64')
df = df.round(4)

df_x = df.loc[:, df.columns.str.contains('_x')]

df_y = df.loc[:, df.columns.str.contains('_y')]

# df = pd.concat([df_x, df_y], axis=1, join='inner')
ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
df_x = df_x.sub(ref_x,axis = 0)
df_y = df_y.sub(ref_y,axis = 0)
df= pd.concat([df_x, df_y], axis=1, join='inner')
pca = PCA(n_components=4)
pca.fit(df)
y = pca.fit_transform(df)
df = pd.DataFrame(y, columns = ['pc1','pc2',"pc3","pc4"])

b = [0]*3441
df["Movement"] = b
df.iloc[268:413,df.columns.get_loc('Movement')] =1
df.iloc[414:525,df.columns.get_loc('Movement')] =2
df.iloc[526:598,df.columns.get_loc('Movement')] =3
df.iloc[617:702,df.columns.get_loc('Movement')] =4
df.iloc[720:824,df.columns.get_loc('Movement')] =5
df.iloc[826:935,df.columns.get_loc('Movement')] =6
df.iloc[936:1027,df.columns.get_loc('Movement')] =7
df.iloc[1028:1134,df.columns.get_loc("Movement")] =8
df.iloc[1137:1340,df.columns.get_loc("Movement")] =9
df.iloc[1341:1517,df.columns.get_loc("Movement")] =10
df.iloc[1522:1599,df.columns.get_loc("Movement")] =11
df.iloc[1624:1758, df.columns.get_loc('Movement')] = 12
df.iloc[1760:1931,df.columns.get_loc("Movement")] =13
df.iloc[1936:2050,df.columns.get_loc("Movement")] =14
df.iloc[2058:2304,df.columns.get_loc("Movement")] =15
df.iloc[2310:2421,df.columns.get_loc("Movement")] =16
df.iloc[2433:2710,df.columns.get_loc("Movement")] =17
df.iloc[2716:2872,df.columns.get_loc("Movement")] =18
df.iloc[2884:3089,df.columns.get_loc("Movement")] =19
df.iloc[3093:3211,df.columns.get_loc("Movement")] =20
df.iloc[3236:3391,df.columns.get_loc("Movement")] =21
df_sub1 = df








df = pd.read_csv("F_PG1_Subject_15_LDLC_resnet101_subject15Oct31shuffle1_103000_filtered.csv")
df.columns = (df.iloc[0] + '_' + df.iloc[1])
df = df.iloc[2:].reset_index(drop=True)
df = df.set_index(df.columns[0])
df = df.astype('float64')
df = df.round(4)

df_x = df.loc[:, df.columns.str.contains('_x')]

df_y = df.loc[:, df.columns.str.contains('_y')]

# df = pd.concat([df_x, df_y], axis=1, join='inner')
ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
df_x = df_x.sub(ref_x,axis = 0)
df_y = df_y.sub(ref_y,axis = 0)
df= pd.concat([df_x, df_y], axis=1, join='inner')
pca = PCA(n_components=4)
pca.fit(df)
y = pca.fit_transform(df)
df = pd.DataFrame(y, columns = ['pc1','pc2',"pc3","pc4"])

b = [0]*4462
df["Movement"] = b
df.iloc[138:358,df.columns.get_loc('Movement')] =15
df.iloc[382:666,df.columns.get_loc('Movement')] =13
df.iloc[686:835,df.columns.get_loc('Movement')] =1
df.iloc[855:1001,df.columns.get_loc('Movement')] =2
df.iloc[1030:1135,df.columns.get_loc('Movement')] =14
df.iloc[1229:1382,df.columns.get_loc('Movement')] =8
df.iloc[1398:1526,df.columns.get_loc('Movement')] =5
df.iloc[1569:1935,df.columns.get_loc("Movement")] =9
df.iloc[1944:2048,df.columns.get_loc("Movement")] =3
df.iloc[2103:2228,df.columns.get_loc("Movement")] =16
df.iloc[2259:2489,df.columns.get_loc("Movement")] =17
df.iloc[2496:2713, df.columns.get_loc('Movement')] = 21
df.iloc[2722:2829,df.columns.get_loc("Movement")] =20
df.iloc[2885:3125,df.columns.get_loc("Movement")] =19
df.iloc[3137:3357,df.columns.get_loc("Movement")] =18
df.iloc[3373:3644,df.columns.get_loc("Movement")] =10
df.iloc[3661:3798,df.columns.get_loc("Movement")] =6
df.iloc[3819:3907,df.columns.get_loc("Movement")] =12
df.iloc[3938:4068,df.columns.get_loc("Movement")] =7
df.iloc[4118:4235,df.columns.get_loc("Movement")] =4
df.iloc[4264:4352,df.columns.get_loc("Movement")] =11
df_sub15 = df

# +

fig = px.scatter(
    df_sub1["pc1"],color = df_sub1["Movement"])
# fig.update_traces(diagonal_visible=False)
fig.show()
fig = px.scatter(
    df_sub15["pc1"],color = df_sub15["Movement"])
# fig.update_traces(diagonal_visible=False)
fig.show()

# +

fig = px.scatter(
    df_sub1["pc2"],color = df_sub1["Movement"])
# fig.update_traces(diagonal_visible=False)
fig.show()
fig = px.scatter(
    df_sub15["pc2"],color = df_sub15["Movement"])
# fig.update_traces(diagonal_visible=False)
fig.show()

# +

fig = px.scatter(
    df_sub1["pc3"],color = df_sub1["Movement"])
# fig.update_traces(diagonal_visible=False)
fig.show()
fig = px.scatter(
    df_sub15["pc3"],color = df_sub15["Movement"])
# fig.update_traces(diagonal_visible=False)
fig.show()
# -

# ## plot Liklihood Distribution

# +


df = pd.read_csv("F_PG1_Subject_1_LDLC_resnet101_myDLC_motion_fullmovieOct26shuffle1_103000.csv")
df.columns = (df.iloc[0] + '_' + df.iloc[1])
df = df.iloc[2:].reset_index(drop=True)
df = df.set_index(df.columns[0])
df = df.astype('float64')
df = df.round(4)

# df_x = df.loc[:, df.columns.str.contains('_x')]

# df_y = df.loc[:, df.columns.str.contains('_y')]
df = df.loc[:, df.columns.str.contains('_likelihood')]
# # df = pd.concat([df_x, df_y], axis=1, join='inner')
# ref_x = ((df_x["hip2_x"]-df_x["hip1_x"])/2)+ df_x["hip1_x"]
# ref_y = ((df_y["hip2_y"]-df_y["hip1_y"])/2)+ df_y["hip1_y"]
# df_x = df_x.sub(ref_x,axis = 0)
# df_y = df_y.sub(ref_y,axis = 0)
# df_sub1 = pd.concat([df_x, df_y], axis=1, join='inner')
# # df_sub1=(df-df.mean())/df.std()
b = [0]*3441
df["Movement"] = b
df.iloc[268:413,df.columns.get_loc('Movement')] =1
df.iloc[414:525,df.columns.get_loc('Movement')] =2
df.iloc[526:598,df.columns.get_loc('Movement')] =3
df.iloc[617:702,df.columns.get_loc('Movement')] =4
df.iloc[720:824,df.columns.get_loc('Movement')] =5
df.iloc[826:935,df.columns.get_loc('Movement')] =6
df.iloc[936:1027,df.columns.get_loc('Movement')] =7
df.iloc[1028:1134,df.columns.get_loc("Movement")] =8
df.iloc[1137:1340,df.columns.get_loc("Movement")] =9
df.iloc[1341:1517,df.columns.get_loc("Movement")] =10
df.iloc[1522:1599,df.columns.get_loc("Movement")] =11
df.iloc[1624:1758, df.columns.get_loc('Movement')] = 12
df.iloc[1760:1931,df.columns.get_loc("Movement")] =13
df.iloc[1936:2050,df.columns.get_loc("Movement")] =14
df.iloc[2058:2304,df.columns.get_loc("Movement")] =15
df.iloc[2310:2421,df.columns.get_loc("Movement")] =16
df.iloc[2433:2710,df.columns.get_loc("Movement")] =17
df.iloc[2716:2872,df.columns.get_loc("Movement")] =18
df.iloc[2884:3089,df.columns.get_loc("Movement")] =19
df.iloc[3093:3211,df.columns.get_loc("Movement")] =20
df.iloc[3236:3391,df.columns.get_loc("Movement")] =21
# df_sub1 = df
df_liklihood =df 
# -

df_liklihood









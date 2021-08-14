# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 00:23:31 2021

@author: kiraz
"""
import numpy as np 
import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt

directory = 'C:/Users/kiraz/Documents/McGill/Data/Stimulus-WT/SignalsDetectedClampfit/20200722'
#create directory for saving plots
# os.mkdir(directory+'/plots')
#read
spatialFreq = np.load (directory + "/spatialFreqM3L2.npy") #read from HistRast_ClampfitTrialsSpatialFrequency
#create dataframe
neuron = ['H8']
label_spatial = []                
for i in ['0.02', '0.04', '0.08', '1.6', '3.2']:
    for j in range(8):
        label_spatial.append(i)
label_temporal = []             
for i in ['0.3', '0.75', '1.2', '1.6', '2.4']:
    for j in range(8):
        label_temporal.append(i)
array = np.linspace(0,360,9)[0:-1]
directions = [str(int(i)) for i in array]
df = pd.DataFrame(columns = ["neuron", "direction", "label_spatial","fr_spatial"])
df['direction'] = directions*5
df['neuron'] = neuron*40
df['fr_spatial'] = spatialFreq
df['label_spatial'] = label_spatial


#guardar el data frame a pickle
df.to_pickle(directory+'/SpatDataFrameM3L2.pkl')
#read dataframe
df_m = pd.read_pickle(directory + '/SpatDataFrameM3L2.pkl')


#with
heat_data = df_m.pivot("direction", "label_spatial", "fr_spatial")
heat_data.index = [0,135,180,225,270,315,45,90]
heat_data1= heat_data.sort_index()
heat_data1 = heat_data1/max(heat_data1.max())
#
# array= df_m[["direction", "fr_spatial"]].groupby("direction").agg(list).values
# array = [i[0] for i in array]
plt.figure()
sns.heatmap(heat_data1)
plt.show()
plt.savefig(directory+'/plots/heatmap.png')
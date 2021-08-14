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
"""
Input: numpy from HistRast_ClampfitTrialsSpatialFrequency
Output: dataframe with firing rate by spatial frequency. Each frequency has several directions.
"""
directory = 'C:/Users/kiraz/Documents/McGill/Data/Stimulus-WT/SignalsDetectedClampfit/20200722'
#create directory for saving plots
os.mkdir(directory+'/plots')
#read
numneur = 'L6'
spatialFreq = np.load (directory + "/spatialFreq" + numneur + ".npy") #read from HistRast_ClampfitTrialsSpatialFrequency
#create dataframe
neuron = ['L6']
label_spatial = []                
for i in ['0.02', '0.04', '0.08', '1.6', '3.2']:
    for j in range(8):
        label_spatial.append(i)
array = np.linspace(0,360,9)[0:-1]
directions = [str(int(i)) for i in array]
df = pd.DataFrame(columns = ["neuron", "direction", "label","FR"])
df['direction'] = directions*5
df['neuron'] = neuron*40
df['FR'] = spatialFreq
df['label'] = label_spatial

#save dataframe as pickle
df.to_pickle(directory+'/SpatDataFrame' +numneur + '.pkl')

#with
heat_data = df.pivot("direction", "label", "FR")
heat_data.index = [0,135,180,225,270,315,45,90]
heat_data1= heat_data.sort_index(ascending=False)
heat_data1 = heat_data1/max(heat_data1.max())
#
# array= df_m[["direction", "fr_spatial"]].groupby("direction").agg(list).values
# array = [i[0] for i in array]

#Make plots
plt.figure()
sns.heatmap(heat_data1)
plt.show()
plt.savefig(directory+'/plots/heatmap.png')
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
numneur = '174C'
path = directory + "/SFTF_spatialFreq" + numNeur
SFTF = np.load (path + ".npy") #read from HistRast_ClampfitTrialsSpatialFrequency
ftemporal = ['0.3', '0.75', '1.2', '1.6', '2.4']
fspatial = ['0.02', '0.04', '0.08', '1.6', '3.2']
#create dataframe
label_spatial = []                
for i in fspatial:
    for j in range(len(ftemporal)):
        label_spatial.append(i)
label_temporal = []
for j in range(len(ftemporal)):             
    for i in ftemporal:
        label_temporal.append(i)
df = pd.DataFrame(columns = ["label_spatial",'label_temporal',"FR"])
df['FR'] = SFTF 
df['label_spatial'] = label_spatial
df['label_temporal'] = label_temporal 

#save dataframe as pickle
df.to_pickle(directory+'/SFTF' +numneur + '.pkl')

#with
heat_data = df.pivot("label_temporal", "label_spatial", "FR")
heat_data1= heat_data.sort_index(ascending=False)
heat_data1 = heat_data1/max(heat_data1.max())
#
# array= df_m[["direction", "fr_spatial"]].groupby("direction").agg(list).values
# array = [i[0] for i in array]

plt.figure()
sns.heatmap(heat_data1, square=True, vmin=0, vmax =1)
plt.show()

plt.savefig(directory+'/plots/heatmap.png')
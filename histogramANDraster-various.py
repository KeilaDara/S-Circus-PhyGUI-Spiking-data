#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 01:36:11 2020

@author: vite
"""


#importing libraries 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import *
import os

"""
Read data
"""
#Select data directory
# directory = 'C:/Users/kiraz/Documents/McGill/Data/Stimulus-WT/20200722/data_WT_Grating8Dir_202007221537156Th5.25/data_WT_Grating8Dir_202007221537156.GUI'
directory = 'C:/Users/kiraz/Documents/McGill/Data/Stimulus-WT/20200722/EulerBaden/EB1527116CC1Th6/data_WT_EulerBadenStimulus_202007221527116.GUI'
#create a pandas data frame with the information coming from the cluster_group file
df = pd.read_csv(directory +"/cluster_group.tsv",  sep="\t")
#Select unique values from data frame
labels=df.iloc[:,0].unique()
# Create a column of Boolean values. All the "good" values will have the label True. 
df['label']=df['group']=='good'
#Take the cluster id corresponding to the templates marked as good
neurons = df[df['label']].cluster_id.values
#load spike data/todos los spike times combinados de todos las neuronas detectadas
neuralData = np.load (directory + "/spike_times.npy")
#load clusters/son los clusters id ordenados segun la ocurrencia de los spike times, se repiten por cada spike time
klusters= np.load (directory + "/spike_clusters.npy")
#stack arrays coming from the spikes and klusters/une clusters y spike times en un arreglo np
data=np.stack([neuralData,klusters], axis=1)
#make a data frame with the two arrays/se hace el data frame a partir del arreglo que unio las variables
df=pd.DataFrame(data, columns=['spikes','klusters'])

#select a template/elige la neurona de la posicion solicitada en neurons
neuron =neurons[1]
#find the values in the dataframe that corresponds to the template chosen/crea una columna en df con valores boobleanos 'true' cada vez que encuentre el cluster id de la neurona elegida y su spike time corresp. y falso en los otros id's y st
df['label'] = df['klusters']==neuron
#Take just the True values corresponding to the template (neuron) chosen
data =df[df['label']]['spikes']


spikes=[]
for i,j in enumerate(neurons):
   #select a template
    neuron =neurons[i]
    print(i)
    #find the values in the dataframe that corresponds to the template chosen
    df['label'] = df['klusters']==neuron
    #Take just the True values corresponding to the template (neuron) chosen
    data =df[df['label']]['spikes']
    
    #Transform data to ms
    data = data/20
    spikes.append(data.values)

spikes_sel = np.asarray(spikes) #in ms


n=1 #select neuron
binsize = 1000 #in ms
nbins = (spikes_sel[n][-1]-spikes_sel[n][0]) / binsize
linesize = 0.1


file= directory + "/bin.txt"
import pandas as pd
df_stim=pd.read_csv(file, header=None)
df_stim.columns=["bloque", "nombre", "tiempo", "u"] 
df_stim["tiempo"]=df_stim["tiempo"]/20

fig, (ax1,ax2) = plt.subplots(2, 1, sharex = True, figsize = [16,12])
#left, bottom, width, height = (dcf_stim["tiempo"][2]/20, 0, 3000, 5)
# fig, ax = plt.subplots()
#shadow for white chirps to moving bar     
for i in range(0, 26, 2):
    left=df_stim["tiempo"][i]
    height = len(spikes)
    width=df_stim["tiempo"][i+1] - df_stim["tiempo"][i]
    rect = plt.Rectangle((left, 0), width, height, facecolor="lightsalmon", alpha=0.1)
    ax1.add_patch(rect)
#shadow for white noise   
left=df_stim["tiempo"][26]
height = len(spikes)
#last block of EB simulus + white noise duration in min*...
width=df_stim["tiempo"][26]+ 4*60*1000
rect = plt.Rectangle((left, 0), width, height, facecolor="lightblue", alpha=0.1)
ax1.add_patch(rect)
#ax.plot()


ax1.eventplot(spikes_sel[n], linelengths = linesize, color='black')
ax1.set_title('Neuron ' + str(n))
plt.box(False)
ax2.hist(spikes_sel[n], bins = int(nbins), color = "sienna")
ax2.set_xlabel("time (ms)")

plt.box(False)
plt.show()

# All at once!
#Create bins
bins = np.arange(begin, end)
# get the spike count per bin
for n in neurons:
spike_count, _ = np.histogram(spk, bins)



fig, (ax1,ax2) = plt.subplots(2, 1, sharex = True, figsize = [16,12])
lista = ax1.eventplot(spikes_sel, linelengths = linesize, color='black')
nbins = 20
n, bins, patches = ax2.hist(spikes, bins=nbins)

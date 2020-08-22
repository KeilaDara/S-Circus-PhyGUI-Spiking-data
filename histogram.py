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
directory = '/Users/vite/navigation_system/Rudo/156Th5.25/'
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

n=6 #select neuron
binsize = 500 #in ms
nbins = (spikes_sel[n][-1]-spikes_sel[n][0]) / binsize
plt.figure()
plt.hist(spikes_sel[n], bins = int(nbins), color = "tan")
plt.title("Histogram for neuron " + str(n))
plt.xlabel("time (ms)")


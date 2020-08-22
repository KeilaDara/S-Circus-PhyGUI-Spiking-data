# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:04:16 2020

@author: kiraz
"""

#importing libraries 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import *
import os
from guidata.qt.compat import getexistingdirectory
"""
Read data
"""
#Select data directory
# directory= getexistingdirectory()
directory = 'C:/Users/kiraz/Documents/McGill/Data/Stimulus-WT/20191111/Retina2/BarCode_1111426317/BarCode_1111426317.GUI'
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



#Make a loop to load the data from all the neurons and save everything in a dataframe

spikes={}

"""
Paste this for loading data:

# load data
spikes = pickle.load(open(directory + "/spikes.pickle", 'rb'))
"""

#select neurons for the raster
neuronsr = neurons
for i,j in enumerate(neuronsr):
   #select a template
    neuron =neuronsr[i]
    print(i)
    #find the values in the dataframe that corresponds to the template chosen
    df['label'] = df['klusters']==neuron
    #Take just the True values corresponding to the template (neuron) chosen
    data =df[df['label']]['spikes']
    #Transform data to ms
    data = data/20/1000
    spikes[j] = data.values
    
import pickle 
with open(directory + "/spikes.pickle", 'wb') as handle:
    pickle.dump(spikes, handle, protocol=pickle.HIGHEST_PROTOCOL)

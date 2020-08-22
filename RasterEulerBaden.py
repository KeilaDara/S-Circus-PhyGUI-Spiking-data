# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:21:37 2020

@author: kiraz
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
directory = 'C:/Users/kiraz/Documents/McGill/Data/Stimulus-WT/20200304/data_WT_EulerBadenStimulus_202003042122525.GUI'
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


#select a template/elige la neurona de la posicion solicitada en neurons
neuron =neurons[0]
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
    #Transform data from timesteps to s by dividing them by sampling frequency: 20000 Hz, or in ms dividing by 20
    data = data/20
    spikes.append(data.values)

spikes= np.asarray(spikes)


#RASTER PLOTS WITHOUT STIMULUS
colors1 = ['C{}'.format(i) for i in range(len(neurons))]

plt.figure()
plt.eventplot(spikes/1000, linelengths=0.1, colors ='k')
plt.title('Raster plot')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron')
positions = [0,1,2,3,4]
labels =["1","2","3","4","5", "6"]
plt.yticks(positions, labels)
plt.box(False)

plt.savefig(directory + '/plots' + '/raster'  + '.pdf')

 

"""
Pandas way
"""
#RASTER PLOT WITH STIMULUS

#read pickled bin file that contains stimulus TTL metadata 
#import pickle
#stimulus= directory + "/stimulus.bin"

#df_stim=pd.read_csv(stimulus, header=None)
#infile = open(stimulus,'rb')
#new_dict = pickle.load(infile)
#infile.close()
#stimdata = pickle.load(open(stimulus, "rb"))

#read txt file that contains stimulus TTL metadata 
file= directory + "/stimulus.txt"
df_stim=pd.read_csv(file, header=None)
df_stim.columns=["bloque", "nombre", "tiempo", "u"] 
df_stim["tiempo"]=df_stim["tiempo"]/20


#Shadows for stimulus times: from bright step to moving bar
#left, bottom, width, height = (df_stim["tiempo"][2]/20, 0, 3000, 5)
from matplotlib.pyplot import locator_params
fig, ax = plt.subplots()
for i in range(0, 26, 2):
    left=df_stim["tiempo"][i]
    height = len(spikes)
    width=df_stim["tiempo"][i+1] - df_stim["tiempo"][i]
    rect = plt.Rectangle((left, 0), width, height, facecolor="turquoise", alpha=0.1)
    ax.add_patch(rect)
    
    
#shadow for white noise   
left=df_stim["tiempo"][26]
height = len(spikes)
#last block of EB simulus + white noise duration in min*...
width=df_stim["tiempo"][26]+ 4*60*1000
rect = plt.Rectangle((left, 0), width, height, facecolor="pink", alpha=0.1)
ax.add_patch(rect)
#ax.plot()

ax.eventplot(spikes, linelengths=0.1, colors ='k')
ax.set_title('Raster plot')
ax.set_xlabel('Time (s)')
ax.set_xticklabels([0,100,200,300,400,500,600])#Comentar para corroborar escala X original
ax.set_ylabel('neuron')
ax.set_yticks(range(1,len(spikes)+1))
ax.legend(["Stimulation"])
positions = [0,1,2,3]
labels =["1","2","3","4"]
plt.yticks(positions, labels)
plt.box(False)
plt.show()

plt.savefig(directory + '/plots' + '/raster'  + '.jpg')





#fig = plt.figure()
#for i in range(0,32,4):
#    left=df_stim["tiempo"][i]
#    height = len(spikes)
#    width=df_stim["tiempo"][i+1] - df_stim["tiempo"][i]
#    plt.Rectangle((left, 0), width, height, facecolor="turquoise", alpha=0.1)
##ax.plot()
#plt.eventplot(spikes, linelengths=0.1, colors ='k')

"Aqui"


#Raster plot for one neuron
#select a template
neuron =neurons[0]
#find the values in the dataframe that corresponds to the template chosen
df['label'] = df['klusters']==neuron
#Take just the True values corresponding to the template (neuron) chosen
data =df[df['label']]['spikes']
#Transform data to ms
data = data/20
plt.eventplot(data, linelengths=0.1, colors ='k')
plt.title('Raster plot')
plt.xlabel('Time (ms)')
plt.ylabel('neuron')
#plt.yticks(range(1,5))
plt.savefig(directory + '/plots' + '/raster'  + '.pdf')


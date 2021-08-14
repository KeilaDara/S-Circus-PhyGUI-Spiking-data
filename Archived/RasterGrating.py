# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:02:51 2020

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
directory = 'C:/Users/kiraz/Documents/McGill/Data/Stimulus-WT/20200722/data_WT_Grating8Dir_202007221537156Th5.25/data_WT_Grating8Dir_202007221537156.GUI'
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

spikes= np.asarray(spikes)


#RASTER PLOTS WITHOUT STIMULI
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
#RASTER PLOT CON ESTIMULO
#read txt with stimulus TTL metadata information
file= directory + "/bin.txt"
import pandas as pd
df_stim=pd.read_csv(file, header=None)
df_stim.columns=["bloque", "nombre", "tiempo", "u"] 
df_stim["tiempo"]=df_stim["tiempo"]/20
#left, bottom, width, height = (dcf_stim["tiempo"][2]/20, 0, 3000, 5)

fig, ax = plt.subplots()
for i in range(2,34,4):
    left=df_stim["tiempo"][i]
    height = len(spikes)
    width=df_stim["tiempo"][i+1] - df_stim["tiempo"][i]
    rect = plt.Rectangle((left, 0), width, height, facecolor="turquoise", alpha=0.1)
    ax.add_patch(rect)
#ax.plot()
ax.eventplot(spikes, linelengths=0.1, colors ='k')
ax.set_title('Raster plot')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('neuron')
ax.set_yticks(range(1,len(spikes)+1))
ax.legend(["Stimulation"])
positions = [0,1,2,3]
labels =["1","2","2","3"]
plt.yticks(positions, labels)
plt.box(False)
plt.show()

plt.savefig(directory + '/plots' + '/raster'  + '.jpg')


#,7,8
#,"8","9"

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




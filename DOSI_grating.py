# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:02:51 2020

@author: vite
"""

import numpy as np
import pandas as pd
from functions import *
import os
import matplotlib.pyplot as plt
import seaborn as sns

"""
Read data
"""
#Select data directory
directory = '/Users/vite/OneDrive - McGill University/archivos_mamifero/data_WT_Grating8Dir_202007221537156Th5CC1R/data_WT_Grating8Dir_202007221537156.GUI'
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


spikesd={}
for i,j in enumerate(neurons):
   #select a template
    neuron =neurons[i]
    print(i)
    #find the values in the dataframe that corresponds to the template chosen
    df['label'] = df['klusters']==neuron
    #Take just the True values corresponding to the template (neuron) chosen
    data =df[df['label']]['spikes']
    #Transform data to us
    data = (data/0.02) 
    spikesd[neuron] = nts.Ts(data.values)


spikes=[]
for i,j in enumerate(neurons):
   #select a template
    neuron =neurons[i]
    print(i)
    #find the values in the dataframe that corresponds to the template chosen
    df['label'] = df['klusters']==neuron
    #Take just the True values corresponding to the template (neuron) chosen
    data =df[df['label']]['spikes']
    #Transform data to us
    data = (data/0.02)
    spikes.append(data.values)
spikes= np.asarray(spikes)


 

"""
Pandas way
"""
#RASTER PLOT CON ESTIMULO
#read txt with stimulus TTL metadata information
file= directory + "/bin.txt"
df_stim=pd.read_csv(file, header=None)
df_stim.columns=["bloque", "nombre", "tiempo", "u"] 
df_stim["tiempo"]=(df_stim["tiempo"]/0.02)
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
ax.set_xlabel('Time (us)')
ax.set_ylabel('neuron')
ax.set_yticks(range(1,len(spikes)+1))
ax.legend(["Stimulation"])
positions = [0,1]
labels =["1","2"]
plt.yticks(positions, labels)
plt.box(False)
plt.show()

plt.savefig(directory + '/plots' + '/raster'  + '.jpg')




"""
DSI
"""

df_o = pd.DataFrame(columns = ['FR', 'neuron', 'orientation', 'X', 'Y'])
# df_o["orientation"] = np.linspace(0, 2*np.pi, 9)[0:-1]
c = 0
orientations = np.linspace(0, 2*np.pi, 9)[0:-1]

for n in neurons:
    for i in range(2,34,4):
        #createa a time interval
        interval = nts.IntervalSet(start = df_stim['tiempo'][i], end = df_stim['tiempo'][i+1])
        #restrict the spikes to the defined time interval
        spk = spikesd[n].restrict(interval)
        #compute firing rate
        fr = len(spk)/interval.tot_length('s')
        df_o.at[c,'FR'] = fr
        df_o.at[c,'neuron'] = "n" + str(n)
        c+=1
    df_o.loc[df_o['neuron']== "n" + str(n), 'orientation']= orientations
    
df_o.orientation = df_o.orientation.astype(float)
df_o.FR = df_o.FR.astype(float)


plt.figure()
sns.lineplot(x = "orientation", y="FR", hue = "neuron", data=df_o)
#polar plot
plt.figure()
g = sns.FacetGrid (df_o, col='neuron', hue = "neuron", subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)
g.map(sns.scatterplot, "orientation", "FR")
g.map(plt.plot, "orientation", "FR")

#Calculate x and y coordinates
df_o['X'] = df_o['FR']*np.cos(df_o['orientation'])
df_o['Y'] = df_o['FR']*np.sin(df_o['orientation'])
#fix for x 
df_o.loc[df_o['orientation'] ==orientations[4], 'X'] = 0
#fix for y
df_o.loc[df_o['orientation'] ==orientations[2], 'Y'] = 0
df_o.loc[df_o['orientation'] ==orientations[6], 'Y'] = 0

DSI_lista = []
for c, n in enumerate(neurons):
    X = df_o[df_o['neuron']== "n" + str(n)]['X'].sum()
    Y = df_o[df_o['neuron']== "n" + str(n)]['Y'].sum()

    Xa = df_o[df_o['neuron']== "n" + str(n)]['X'].abs().sum()
    Ya = df_o[df_o['neuron']== "n" + str(n)]['Y'].abs().sum()
    DSI_lista.append(np.sqrt( X**2 + Y**2)/np.sqrt( Xa**2 + Ya**2))
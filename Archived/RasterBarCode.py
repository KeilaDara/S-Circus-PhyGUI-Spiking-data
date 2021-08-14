# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:21:11 2020

@author: kiraz
"""


#importing libraries 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import *
import os
import pickle
from guidata.qt.compat import getexistingdirectory
"""
Read data
"""
#Select data directory
# directory= getexistingdirectory()
directory = 'C:/Users/kiraz/Documents/McGill/Data/Stimulus-WT/20191111/Retina2/BarCode_1111426317/BarCode_1111426317.GUI'

# load data
spikes = pickle.load(open(directory + "/spikes.pickle", 'rb'))

#Store number of neurons in a single variable 
neurons = [*spikes.keys()]

lista_n = []
for i in neurons:
    lista_n.append(spikes[i]/1000)
    
#RASTER PLOTS WITHOUT STIMULI
colors1 = ['C{}'.format(i) for i in range(len(neuronsr))]
plt.figure()
plt.eventplot(lista_n, linelengths=0.1, colors ='k')
plt.title('Raster plot')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron')
labels =[i for i in range(1,len(neuronsr)+1)]
positions = [i-1 for i in range(1,len(neuronsr)+1)]
plt.yticks(positions, labels)
plt.box(False)
plt.show()

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
#left, bottom, width, height = (df_stim["tiempo"][2]/20, 0, 3000, 5)

fig, ax = plt.subplots()
for i in range(0,1,1):
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
positions = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
labels =["1","2","3","4","5","6","7","8","9","10","11","12","13","14"]
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


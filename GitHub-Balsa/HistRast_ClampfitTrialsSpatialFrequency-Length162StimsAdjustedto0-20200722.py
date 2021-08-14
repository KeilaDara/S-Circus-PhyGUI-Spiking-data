# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 22:50:27 2021

@author: kiraz
"""

import numpy as np
import pandas as pd
from functions import *
from my_functions import *
import os
import matplotlib.pyplot as plt
import seaborn as sns

directory = 'C:/Users/kiraz/Documents/McGill/Data/Stimulus-WT/SignalsDetectedClampfit/20200722'
#get the number of trials
# trials = [*range( int(len(os.listdir(directory))/2)) ]
trials = [*range(3)]#cambiar dependiendo del numero de trials o señales a cargar
numNeur= 'L6'
#Son 5 frecuencias temporales

#H8
# trial1 = np.loadtxt(directory + '/SpatialFrequency_258_H8.txt')
# trial2 = np.loadtxt(directory + '/SpatialFrequency_408_H8.txt')
# trial3 = np.loadtxt(directory + '/SpatialFrequency_528_H8.txt')

# H9H8
# trial1 = np.loadtxt(directory + '/SpatialFrequency_258_H9H8.txt')
# trial2 = np.loadtxt(directory + '/SpatialFrequency_408_H9H8.txt')
# trial3 = np.loadtxt(directory + '/SpatialFrequency_528_H9H8.txt')
# trial4 = np.loadtxt(directory + '/SpatialFrequency_595_H8H9.txt')

# #I5
# trial1 = np.loadtxt(directory + '/SpatialFrequency_258_I5.txt')
# trial2 = np.loadtxt(directory + '/SpatialFrequency_408_I5.txt')
# trial3 = np.loadtxt(directory + '/SpatialFrequency_528_I5.txt')
# trial4 = np.loadtxt(directory + '/SpatialFrequency_595_I5r.txt')

# #I8
trial1 = np.loadtxt(directory + '/SpatialFrequency_258_I8.txt')
trial2 = np.loadtxt(directory + '/SpatialFrequency_408_I8.txt')
trial3 = np.loadtxt(directory + '/SpatialFrequency_528_I8.txt')


# #K4K5
# trial1 = np.loadtxt(directory + '/SpatialFrequency_258_K5K4.txt')
# trial2 = np.loadtxt(directory + '/SpatialFrequency_408_K4K5.txt')
# trial3 = np.loadtxt(directory + '/SpatialFrequency_528_K4K5.txt')
# trial4 = np.loadtxt(directory + '/SpatialFrequency_528_K5.txt')

# #K6N1
# trial1 = np.loadtxt(directory + '/SpatialFrequency_258_K6N1.txt')
# trial2 = np.loadtxt(directory + '/SpatialFrequency_408_N1K6.txt')
# trial3 = np.loadtxt(directory + '/SpatialFrequency_528_N1K6.txt')

# #L3
# trial1 = np.loadtxt(directory + '/SpatialFrequency_258_L3.txt')
# trial2 = np.loadtxt(directory + '/SpatialFrequency_408_L3.txt')
# trial3 = np.loadtxt(directory + '/SpatialFrequency_528_L3.txt')
# trial4 = np.loadtxt(directory + '/SpatialFrequency_595_L3.txt')

# #L6
trial1 = np.loadtxt(directory + '/SpatialFrequency_408_L6.txt')
trial2 = np.loadtxt(directory + '/SpatialFrequency_258_L6.txt')
trial3 = np.loadtxt(directory + '/SpatialFrequency_528_L6.txt')

# #M3-L2
# trial1 = np.loadtxt(directory + '/SpatialFrequency_408_M3corto.txt')
# trial2 = np.loadtxt(directory + '/SpatialFrequency_528_M3L2.txt')
# trial3 = np.loadtxt(directory + '/SpatialFrequency_595_M3L2.txt')
# trial4 = np.loadtxt(directory + '/SpatialFrequency_595_M2.txt')

# trial1 = np.loadtxt(directory + '/SpatialFrequency_408_L2medio.txt')
# trial2 = np.loadtxt(directory + '/SpatialFrequency_408_M2largo.txt')
# trial3 = np.loadtxt(directory + '/SpatialFrequency_528_M2.txt')
# trial4 = np.loadtxt(directory + '/SpatialFrequency_258_M3L2r.txt')
# trial5 = np.loadtxt(directory + '/SpatialFrequency_258_M2.txt')

"AJUSTADOS A 0"


spikes=[]
spikes.append(trial1)
spikes.append(trial2)
spikes.append(trial3)
# spikes.append(trial4)
# spikes.append(trial5)
# spikes.append(trial6)
# spikes.append(trial7)
              # spikes.append(trial5)
spikes = [i*1000 for i in spikes]
spikes= np.asarray(spikes)

spikesd={}
for i in trials:
    spikesd[i] = nts.Ts(spikes[i])

file1= directory + "/bin258.txt"
file2= directory + "/bin408.txt"
file3= directory + "/bin528.txt"
file4= directory + "/bin595.txt"
file5= directory + "/bin074.txt"
#Read stimulus times
files = [file1, file2, file3]


# trials= [1,2,3,4,5]

numBlocks = 160#eran  162 (en c/ nuevo arch txt de stim) pero elimino prev la primera y ultima linea manualm

df_stimes = pd.DataFrame(index = [*range(numBlocks)], columns = range(len(trials)))
for i in range(len(trials)):
    df_stim=pd.read_csv(files[i], header=None, usecols = [2])#files era i, pero un solo arch como ref, AQUI SE CORRIGIO PARA EVITAR PRIMERA Y ULTIMA FILA DE ARCHIVO DEL ESTIMULO
    df_stimes.loc[:,i] = df_stim.values.flatten()/0.02#transform ttl timesteps a us
 
#restrict spike times to the stimulation time    
spikes_hist = []
for i in range(len(trials)):
    startS = int(df_stimes.iloc[2,i])
    endS = int(df_stimes.iloc[-1,i] + 1e6)
    if spikesd[i].index[0]>startS and spikesd[i].index[-1]<endS:
        spikes_hist.append(spikesd[i].index.values - startS)
    else:
        interval = nts.IntervalSet(startS, endS)
        t = spikesd[i].restrict(interval).index.values - startS
        spikes_hist.append(t)

# plt.figure()
# plt.plot(spikesd[i].index, '.')
# plt.axhline(y=low)
# plt.axhline(y=up)
# plt.show()



# All at once!
#generate data for the histogram
begin = int(np.amin(np.concatenate(spikes_hist).flatten()))
end = int(np.amax(np.concatenate(spikes_hist).flatten()))
binsize = 1e6 #in us
bins = np.arange(begin, end+binsize, binsize)
# get the spike count per bin
spk_counts= []
for i in range(len(trials)):
    spike_count, _ = np.histogram(spikes_hist[i], bins)
    spk_counts.append(spike_count)
spk_hist= np.sum(spk_counts, axis=0)

#plot the raster
fig, (ax1,ax2) = plt.subplots(2, 1, sharex = True, figsize = [16,12])
begin = min([spikes_hist[i][0] for i in range(len(trials))])
stims = []
c = begin
stims.append(c)
array = df_stimes.diff().mean(axis=1).values[1::]
for i in array:
    c+= i
    stims.append(c)
stims = stims-stims[2]    
stim_list = []
colors =  ['darkslategray','cadetblue','dimgray','cadetblue','darkgray']
for p,i in enumerate(range(2,len(stims),4)):
    left=stims[i]
    stim_list.append(int(left))
    height = len(trials)
    width=stims[i+1]- stims[i]
    rect = plt.Rectangle((left, 0), width, height, facecolor=colors[p//8], alpha=0.1)
    ax1.add_patch(rect)
linesize = 0.1
ax1.eventplot(spikes_hist, linelengths = 0.2, color='black')
ax1.set_title('Trials')
plt.box(False)
# ax2.hist(spikes_sel[n], bins = int(nbins))
ax2.set_xlabel("time (min)")
ax2.bar(bins[:-1], spk_hist, width = binsize, color = "sienna")
plt.box(False)
plt.show()

#Calculate the peak for every stimulation block (every dir thorugh 5 SF)
stim_list = np.asarray(stim_list)/1e6
stim_list.round(out=stim_list)
stim_list = stim_list.astype('int')
fr_stim = []
for i in stim_list:
    val = np.max(spk_hist[i:i+5])
    fr_stim.append(val)   
plt.figure()
plt.bar([*range(40)],fr_stim)
plt.show()
path = directory + "/spatialFreq" + numNeur
np.save(path, np.asarray(fr_stim))

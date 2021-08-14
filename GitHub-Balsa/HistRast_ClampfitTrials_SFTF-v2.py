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

def getspikes(directory, numNeur, recordingsList):
    spikes = []
    for num in recordingsList:
        spiks = np.loadtxt(directory + '/SFTN' + num + '-' +numNeur +'.txt')
        spikes.append(spiks)
    
    spikes = [i*1000 for i in spikes] #convert to us
    spikes= np.asarray(spikes)
    spikesd={}
    for i in range(len(recordingsList)):
        spikesd[i] = nts.Ts(spikes[i])
    return spikesd
def getStimTimes(directory, recordingsList):
    files = []
    for i in recordingsList:
        files.append(directory+'/bin'+i+'.txt')
    df_stims = pd.DataFrame(columns = [*range(len(files))])
    for i,file in enumerate (files): 
        df_stim=pd.read_csv(file, header=None)
        df_stim.columns=["bloque", "nombre", "tiempo", "u"] 
        df_stim["tiempo"]=df_stim["tiempo"]/20
        df_stims.loc[:,i] = df_stim['tiempo'].values
         # df_stim['tiempo'].values
    df_stimsMean = df_stims.mean(axis=1) 
    return df_stimsMean
"""
Load Data
"""
# directory = 'C:/Users/kiraz/Documents/McGill/Data/Stimulus-WT/SignalsDetectedClampfit/20200722'
directory = '/Users/vite/navigation_system/Rudo/GitHub-Balsa'
numNeur = '174C'
recordingsList = ['06159','09212', '12282','15354'] #recordings just for numNeur
# dictionary with the spikes corresponding to all the trials from one neuron
spikes = getspikes(directory, numNeur, recordingsList) 
#DataFrame with the times of the stimulation
df_stimtimes = getStimTimes(directory, recordingsList)

#length of raws in the stimulation file
numBlocks = 160#eran  162 (en c/ nuevo arch txt de stim) pero elimino prev la primera y ultima linea manualm

df_stimes = pd.DataFrame(index = [*range(numBlocks)], columns = range(len(trials)))
for i in range(len(trials)):
    df_stim=pd.read_csv(files[i], header=None, usecols = [2])#files era i, pero un solo arch como ref, AQUI SE CORRIGIO PARA EVITAR PRIMERA Y ULTIMA FILA DE ARCHIVO DEL ESTIMULO
    df_stimes.loc[:,i] = df_stim.values.flatten()/0.02#transform ttl timesteps a us
 
#restrict spike times to the stimulation time    
spikes_hist = []
for i in range(len(recordingsList)):
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

#Align the stimulation times to the spike times
begin = min([spikes_hist[i][0] for i in range(len(trials))]) #time of the first spike
#list with the stimulation times + the time of the first spike
stims = [] 
c = begin
stims.append(c)
#array with the mean times of the differences between starts and ends of events in df_stimes
array = df_stimes.diff().mean(axis=1).values[1::] 
for i in array:
    c+= i
    stims.append(c)
stims = stims-stims[2] #rest the end time of the first wait interval
path = directory + "/MeanStimTimes" + numNeur +'.npy'
np.save(path, stims)

#plot the raster
fig, (ax1,ax2) = plt.subplots(2, 1, sharex = True, figsize = [16,12])
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

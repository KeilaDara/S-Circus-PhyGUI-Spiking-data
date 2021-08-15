#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:39:18 2020

@author: vite
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

directory = 'C:/Users/kiraz/Documents/McGill/Data/Stimulus-WT/SignalsDetectedClampfit/20210706'

spikes = []
numNeur = '174C'
recordingsList = ['06159','09212', '12282','15354','59480']
for num in recordingsList:
    spiks = np.loadtxt(directory + '/SFTN' + num + '-' +numNeur +'.txt')
    spikes.append(spiks)


spikes= np.asarray(spikes)
spikes_sel = spikes

n=1 #select neuron
binsize = 1300 #in ms
nbins = (spikes_sel[n][-1]-spikes_sel[n][0]) / binsize
linesize = 0.2
neurons= list(range(len(spikes)))

# All at once!
#Create bins
begin = int(np.amin(np.concatenate(spikes).flatten()))
end = int(np.amax(np.concatenate(spikes).flatten()))
bins = np.arange(begin, end+binsize, binsize)
# get the spike count per bin
spk_counts= []

for i in range(len(neurons)):
    spike_count, _ = np.histogram(spikes_sel[i], bins)
    spk_counts.append(spike_count)
    
spk_hist= np.sum(spk_counts, axis=0)

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
  
fig, (ax1,ax2) = plt.subplots(2, 1, sharex = True, figsize = [16,12])
#Shadows for stimulus times: from bright step to moving bar
#left, bottom, width, height = (df_stim["tiempo"][2]/20, 0, 3000, 5)
from matplotlib.pyplot import locator_params
#chirp to color
for i in range(0, 26, 2):
    left=df_stimsMean[i]
    height = len(spikes)
    width=df_stimsMean[i+1] - df_stimsMean[i]
    rect = plt.Rectangle((left, 0), width, height, facecolor="turquoise", alpha=0.1)
    ax1.add_patch(rect)
    
#shadow for white noise   
left=df_stimsMean[26]
height = len(spikes)
#last block of EB simulus + white noise duration in min*...
width=df_stimsMean[26]+ 4*60*1000
rect = plt.Rectangle((left, 0), width, height, facecolor="pink", alpha=0.1)
ax1.add_patch(rect)
ax1.plot()

ax1.eventplot(spikes, linelengths=0.3, colors ='k')
ax1.set_title('Raster plot')
ax1.set_xlabel('Time (s)')
ax1.set_xticklabels([0,100,200,300,400,500,600])#Comentar para corroborar escala X original
ax1.set_ylabel('trials')
ax1.set_yticks(range(1,len(spikes)+1))


plt.box(False)
# ax2.hist(spikes_sel[n], bins = int(nbins))
ax2.set_xlabel("time (ms)")
ax2.bar(bins[:-1], spk_hist, width = binsize, color = "sienna")
plt.box(False)
plt.show()

# plt.savefig(directory + '/plots'  + '.png')

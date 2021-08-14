#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:39:19 2020

@author: vite
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
trials = [*range( int(len(os.listdir(directory))/2)) ]

trial1 = np.loadtxt(directory + '/Grating156-H8.txt')
trial2 = np.loadtxt(directory + '/Grating345-H8.txt')
trial3 = np.loadtxt(directory + '/Grating475-H8.txt')
trial4 = np.loadtxt(directory + '/Grating336-H8.txt')
trial5 = np.loadtxt(directory + '/Grating074-H8.txt')


spikes=[]
spikes.append(trial1)
spikes.append(trial2)
spikes.append(trial3)
spikes.append(trial4)
spikes.append(trial5)
spikes = [i*1000 for i in spikes]
spikes= np.asarray(spikes)

spikesd={}
for i in trials:
    spikesd[i] = nts.Ts(spikes[i])

file1= directory + "/bin156.txt"
file2= directory + "/bin345.txt"
file3= directory + "/bin475.txt"
file4= directory + "/bin336.txt"
file5= directory + "/bin074.txt"
#Read stimulus times
files = [file1, file2, file3, file4, file5]

# trials= [1,2,3,4,5]
df_stimes = pd.DataFrame(index = [*range(32)], columns = range(len(trials)))
for i in range(len(trials)):
    df_stim=pd.read_csv(files[i], header=None, usecols = [2])
    df_stimes.loc[:,i] = df_stim.values.flatten()/0.02
    
spikes_hist = []
for i in range(len(trials)):
    low = (df_stimes.iloc[0,i])
    up = (df_stimes.iloc[-1,i] + 1e6)
    interval = nts.IntervalSet(low, up)
    t = spikesd[i].restrict(interval).index.values - low
    spikes_hist.append(t)

# All at once!
#generate data for the histogram
begin = int(np.amin(np.concatenate(spikes_hist).flatten()))
end = int(np.amax(np.concatenate(spikes_hist).flatten()))
binsize = 1e6 #in ms
bins = np.arange(begin, end+binsize, binsize)
# get the spike count per bin
spk_counts= []
for i in range(len(trials)):
    spike_count, _ = np.histogram(spikes_hist[i], bins)
    spk_counts.append(spike_count)
spk_hist= np.sum(spk_counts, axis=0)


fig, (ax1,ax2) = plt.subplots(2, 1, sharex = True, figsize = [16,12])

begin = min([spikes_hist[i][0] for i in range(len(trials))])
stims = []
c = begin
stims.append(c)
array = df_stimes.diff().mean(axis=1).values[1::]
for i in array:
    c+= i
    stims.append(c)
for i in range(2,34,4):
    left=stims[i]
    height = len(trials)
    width=stims[i+1]- stims[i]
    rect = plt.Rectangle((left, 0), width, height, facecolor="lightsalmon", alpha=0.1)
    ax1.add_patch(rect)
linesize = 0.1
ax1.eventplot(spikes_hist, linelengths = linesize, color='black')
ax1.set_title('Trials')
plt.box(False)
# ax2.hist(spikes_sel[n], bins = int(nbins))
ax2.set_xlabel("time (ms)")
ax2.bar(bins[:-1], spk_hist, width = binsize, color = "sienna")
plt.box(False)
plt.show()

    


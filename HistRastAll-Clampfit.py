#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:39:18 2020

@author: vite
"""
import pandas as pd

# directory = 'C:/Users/kiraz/Documents/McGill/Data/Stimulus-WT/SignalsDetectedClampfit/20200722'
directory = '/Users/vite/navigation_system/Rudo/20200722/'
spikes1 = np.loadtxt(directory + '/Grating156-H8.txt')
spikes2 = np.loadtxt(directory + '/Grating156-I5.txt')
spikes3 = np.loadtxt(directory + '/Grating156-I8.txt')
spikes4 = np.loadtxt(directory + '/Grating156-K5.txt')
spikes5 = np.loadtxt(directory + '/Grating156-L2.txt')
spikes6 = np.loadtxt(directory + '/Grating156-L3.txt')
spikes7 = np.loadtxt(directory + '/Grating156-M2.txt')
spikes8 = np.loadtxt(directory + '/Grating156-M3.txt')

spikes=[]
spikes.append(spikes1)
spikes.append(spikes2)
spikes.append(spikes3)
spikes.append(spikes4)
spikes.append(spikes5)
# spikes.append(spikes6)
spikes.append(spikes7)
spikes.append(spikes8)

spikes= np.asarray(spikes)
spikes_sel = spikes


binsize = 1000 #in ms
nbins = (spikes_sel[n][-1]-spikes_sel[n][0]) / binsize
linesize = 0.2


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

fig, (ax1,ax2) = plt.subplots(2, 1, sharex = True, figsize = [16,12])

file= directory + "bin.txt"
import pandas as pd
df_stim=pd.read_csv(file, header=None)
df_stim.columns=["bloque", "nombre", "tiempo", "u"] 
df_stim["tiempo"]=df_stim["tiempo"]/20
#left, bottom, width, height = (dcf_stim["tiempo"][2]/20, 0, 3000, 5)
# fig, ax = plt.subplots()
for i in range(2,34,4):
    left=df_stim["tiempo"][i]
    height = len(spikes)
    width=df_stim["tiempo"][i+1] - df_stim["tiempo"][i]
    rect = plt.Rectangle((left, 0), width, height, facecolor="lightsalmon", alpha=0.1)
    ax1.add_patch(rect)

ax1.eventplot(spikes_sel, linelengths = linesize, color='black')
ax1.set_title('Neurons')
plt.box(False)
ax2.hist(spikes_sel[n], bins = int(nbins))
ax2.set_xlabel("time (ms)")
ax2.bar(bins[:-1], spk_hist, width = binsize, color = "sienna")
plt.box(False)
plt.show()
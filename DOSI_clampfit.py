#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:49:19 2020

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

spikes = [i*1000 for i in spikes]
spikes= np.asarray(spikes)



spikesd={}
neurons = [*range(len(spikes))]
for i in neurons:
    spikesd[i] = nts.Ts(spikes[i])
    
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
file= directory + "bin.txt"
df_stim=pd.read_csv(file, header=None)
df_stim.columns=["bloque", "nombre", "tiempo", "u"] 
df_stim["tiempo"]=(df_stim["tiempo"]/0.02)
#left, bottom, width, height = (dcf_stim["tiempo"][2]/20, 0, 3000, 5)

# fig, ax = plt.subplots()
# for i in range(2,34,4):
#     left=df_stim["tiempo"][i]
#     height = len(spikes)
#     width=df_stim["tiempo"][i+1] - df_stim["tiempo"][i]
#     rect = plt.Rectangle((left, 0), width, height, facecolor="turquoise", alpha=0.1)
#     ax.add_patch(rect)
# #ax.plot()
# ax.eventplot(spikes, linelengths=0.1, colors ='k')
# ax.set_title('Raster plot')
# ax.set_xlabel('Time (us)')
# ax.set_ylabel('neuron')
# ax.set_yticks(range(1,len(spikes)+1))
# ax.legend(["Stimulation"])
# positions = [0,1]
# labels =["1","2"]
# plt.yticks(positions, labels)
# plt.box(False)
# plt.show()

# plt.savefig(directory + '/plots' + '/raster'  + '.jpg')






df_base = pd.DataFrame(columns = ['FR', 'neuron', 'orientation'])
# df_base["orientation"] = np.linspace(0, 2*np.pi, 9)[0:-1]
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
        df_base.at[c,'FR'] = fr
        df_base.at[c,'neuron'] = "n" + str(n)
        c+=1
    df_base.loc[df_base['neuron']== "n" + str(n), 'orientation']= orientations
    
df_base.orientation = df_base.orientation.astype(float)
df_base.FR = df_base.FR.astype(float)

#linear plot
plt.figure()
sns.lineplot(x = "orientation", y="FR", hue = "neuron", data=df_base)
#polar plot
g = sns.FacetGrid (df_base, col='neuron', hue = "neuron", subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)
g.map(sns.scatterplot, "orientation", "FR")
g.map(plt.plot, "orientation", "FR")

#polar plot just for one neuron
n = 3
neuron = "n" + str(n)
df_n = df_base[df_base['neuron']== neuron]
df_n = df_n.iloc[:,0:3]
df_n.loc[8] = [df_base[df_base['neuron']== neuron]['FR'].values[0], neuron, 0]
g = sns.FacetGrid (df_n, col='neuron', hue = "neuron", subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)
g.map(sns.scatterplot, "orientation", "FR")
g.map(plt.plot, "orientation", "FR")


"""
DSI
"""
df_dsi = df_base
#Calculate x and y coordinates
df_dsi['X'] = df_dsi['FR']*np.cos(df_dsi['orientation'])
df_dsi['Y'] = df_dsi['FR']*np.sin(df_dsi['orientation'])
#fix for x 
df_dsi.loc[df_dsi['orientation'] ==orientations[4], 'X'] = 0
#fix for y
df_dsi.loc[df_dsi['orientation'] ==orientations[2], 'Y'] = 0
df_dsi.loc[df_dsi['orientation'] ==orientations[6], 'Y'] = 0

DSI_lista = []
for c, n in enumerate(neurons):
    X = df_dsi[df_dsi['neuron']== "n" + str(n)]['X'].sum()
    Y = df_dsi[df_dsi['neuron']== "n" + str(n)]['Y'].sum()

    Xa = df_dsi[df_dsi['neuron']== "n" + str(n)]['X'].abs().sum()
    Ya = df_dsi[df_dsi['neuron']== "n" + str(n)]['Y'].abs().sum()
    DSI_lista.append(np.sqrt( X**2 + Y**2)/np.sqrt( Xa**2 + Ya**2))

    
"""
OSI
"""
axis = ['horizontal', 'ur-dl', 'vertical', 'ul-dr']
df_ohsi = pd.DataFrame(columns = [ 'neuron', 'axis', 'FRt' ])

c = 0
for n in neurons:
    for i in range(4):
        df_ohsi.at[c,'neuron'] = "n" + str(n)
        df_ohsi.at[c,'FRt'] = (df_base['FR'].iloc[c]+df_base['FR'].iloc[c+4])/2
        c+=1
    df_ohsi.loc[df_ohsi['neuron'] == 'n' + str(n), 'axis'] = axis

OSI = []  
for n in neurons:
    horizontal = df_ohsi[df_ohsi['neuron']==  "n" + str(n) ][df_ohsi['axis']== 'horizontal']['FRt'].values
    vertical = df_ohsi[df_ohsi['neuron']==  "n" + str(n) ][df_ohsi['axis']== 'vertical']['FRt'].values
    ur_dl = df_ohsi[df_ohsi['neuron']==  "n" + str(n) ][df_ohsi['axis']== 'ur-dl']['FRt'].values 
    ul_dr = df_ohsi[df_ohsi['neuron']==  "n" + str(n) ][df_ohsi['axis']== 'ul-dr']['FRt'].values
    X = horizontal - vertical
    Y = ur_dl - ul_dr
    Xa = horizontal + vertical
    Ya = ur_dl + ul_dr
    # OSI.append(np.sqrt(( X**2 + Y**2)[0])/np.sqrt( (Xa**2 + Ya**2)[0]))
    OSI.append(np.sqrt(( X**2 + Y**2)[0])/(Xa + Ya)[0])
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 02:36:59 2021

@author: kiraz

"""

import numpy as np
import pandas as pd
from functionsk import *
import os
import matplotlib.pyplot as plt
import seaborn as sns

"""
Read data
"""
directory = 'C:/Users/kiraz/Documents/McGill/Data/Stimulus-WT/SignalsDetectedClampfit/20200722'


######

df_m = pd.read_pickle(directory + '/SpatTempDataFrame.pkl')

#Read the stim file (previous grating)
df_stim=pd.read_csv(file1, header=None)
df_stim.columns=["bloque", "nombre", "tiempo", "u"] 
df_stim["tiempo"]=(df_stim["tiempo"]/0.02)

df_base = pd.DataFrame(columns = ['FR', 'trial', 'orientation'])
# df_base["orientation"] = np.linspace(0, 2*np.pi, 9)[0:-1]
c = 0
orientations = np.linspace(0, 2*np.pi, 9)[0:-1]

#PEDAZO DE KEILA para tener una lista de 40 orientaciones (8 para cada spatial frequency)
num_spatial_frequencies=5
orientations_list = []

for i in range(0, num_spatial_frequencies):
    for o in orientations:
        orientations_list.append(o)
        
orientations = orientations_list

for n in neurons:
    for i in range(2,159,4):
        #createa a time interval
        interval = nts.IntervalSet(start = df_stim['tiempo'][i], end = df_stim['tiempo'][i+1])
        #restrict the spikes to the defined time interval
        spk = spikesd[n].restrict(interval)
        #compute firing rate
        fr = len(spk)/interval.tot_length('s')
        df_base.at[c,'FR'] = fr
        df_base.at[c,'trial'] = "t" + str(n)
        c+=1            
    df_base.loc[df_base['trial']== "t" + str(n), 'orientation']= orientations
    
df_base.orientation = df_base.orientation.astype(float)
df_base.FR = df_base.FR.astype(float)

#####################################
"Or load data from the excel that contains the averages of FR de los trials por cada neurona"
# path = '/Users/kiraz/Documents/McGill/Tesis 2/Results/Gratings trials/Promedios-FR-trials-pneurona-concatenadedForPython.xlsx'

# df = pd.read_excel(path, names = ["FR", "trial", "orientation"], header = None)
# df_base=df


"""
La manera correcta de resolver lo anterior fue tomando el peak amp de df_base
el valor de FR de n0-n8-n16 y n1-n9-n17 y así etc. tres FRs si son 3 trials
"""

path = 'C:/Users/kiraz/Documents/McGill/Data/Stimulus-WT/PolarPlots/SpatialFrequency/20200722/Promedios-FR-trials-pneurona-concatenadedForPython-M3L2.xlsx'

df = pd.read_excel(path, names = ["FR", "SF", "orientation"], header = None)
df_base=df

path = 'C:/Users/kiraz/Documents/McGill/Data/Stimulus-WT/PolarPlots/Temporal Frequency/Promedios-FR-trials-pneurona-concatenadedForPython-H8H9.xlsx'

df = pd.read_excel(path, names = ["FR", "SF", "orientation"], header = None)
df_base=df

#####################################

"""
POLAR PLOTS
"""
#linear plot
plt.figure()
sns.lineplot(x = "orientation", y="FR", hue = "SF", data=df_base)
#polar plot
g = sns.FacetGrid (df_base, col='SF', hue = "SF", subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)
g.map(sns.scatterplot, "orientation", "FR")
g.map(plt.plot, "orientation", "FR")


#polar plot just for one neuron
n = 1
neuron = "n" + str(n)
df_n = df_base[df_base['SF']== neuron]
df_n = df_n.iloc[:,0:3]
df_n.loc[8] = [df_base[df_base['SF']== neuron]['FR'].values[0], neuron, 0]
g = sns.FacetGrid (df_n, col='SF', hue = "SF", subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)
g.map(sns.scatterplot, "orientation", "FR")
g.map(plt.plot, "orientation", "FR")


"""
DSI
"""
df_peak = df_base.groupby(["orientation"]).max()
df_peak.reset_index(level=0, inplace=True)
df_dsi = df_peak
#Calculate x and y coordinates
orientations = df_dsi['orientation'].values
df_dsi['X'] = df_dsi['FR']*np.cos(df_dsi["orientation"])
df_dsi['Y'] = df_dsi['FR']*np.sin(df_dsi['orientation'])
#fix for x 
df_dsi.loc[df_dsi['orientation'] ==orientations[2], 'X'] = 0
df_dsi.loc[df_dsi['orientation'] ==orientations[6], 'X'] = 0
#fix for y
df_dsi.loc[df_dsi['orientation'] ==orientations[0], 'Y'] = 0
df_dsi.loc[df_dsi['orientation'] ==orientations[4], 'Y'] = 0


X = df_dsi['X'].sum()
Y = df_dsi['Y'].sum()

Xa = df_dsi['X'].abs().sum()
Ya = df_dsi['Y'].abs().sum()
elDSI = (np.sqrt( X**2 + Y**2)/np.sqrt( Xa**2 + Ya**2))


#DSI by trial
# DSI_lista = []
# for c, n in enumerate(neurons):
#     X = df_dsi[df_dsi['trial']== "t" + str(n)]['X'].sum()
#     Y = df_dsi[df_dsi['trial']== "t" + str(n)]['Y'].sum()

#     Xa = df_dsi[df_dsi['trial']== "t" + str(n)]['X'].abs().sum()
#     Ya = df_dsi[df_dsi['trial']== "t" + str(n)]['Y'].abs().sum()
#     DSI_lista.append(np.sqrt( X**2 + Y**2)/np.sqrt( Xa**2 + Ya**2))


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
    

file='C:/Users/kiraz/Documents/McGill/Data/Stimulus-WT/20200722/BIN/data_WT_SpatialFrequencyTuning_202007221628528.txt'

    
    
    if os.path.splitext(i)[1]=='.csv':
        if i.split('_')[1][0] != 'T': 
            events.append( i.split('_')[1][0])





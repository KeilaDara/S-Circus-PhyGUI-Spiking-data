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

def CalcDOSI(df_base):
    """

    Parameters
    ----------
    df_base : DataFrame containing a label (like neuron or spatial frequency) and FR by direction

    Returns
    -------
    elDSI : Direction selectivity index.

    """
    df_peak = df_base.groupby(["direction"]).max()
    df_peak.reset_index(level=0, inplace=True)
    ##########################################################################
    # Calculate DSI
    df_dsi = df_peak
    #Calculate x and y coordinates
    orientations = df_dsi['direction'].values
    df_dsi['X'] = df_dsi['FR']*np.cos(df_dsi["direction"])
    df_dsi['Y'] = df_dsi['FR']*np.sin(df_dsi['direction'])
    #fix for x 
    df_dsi.loc[df_dsi['direction'] ==orientations[2], 'X'] = 0
    df_dsi.loc[df_dsi['direction'] ==orientations[6], 'X'] = 0
    #fix for y
    df_dsi.loc[df_dsi['direction'] ==orientations[0], 'Y'] = 0
    df_dsi.loc[df_dsi['direction'] ==orientations[4], 'Y'] = 0
    X = df_dsi['X'].sum()
    Y = df_dsi['Y'].sum()
    Xa = df_dsi['X'].abs().sum()
    Ya = df_dsi['Y'].abs().sum()
    elDSI = (np.sqrt( X**2 + Y**2)/np.sqrt( Xa**2 + Ya**2))
    print('your DSI: ', elDSI)
    ##########################################################################
    # Calculate OSI
    axis = ['horizontal', 'ur-dl', 'vertical', 'ul-dr'] # u=up, d=down, l=left, r=right
    df_ohsi = pd.DataFrame(columns = [ 'axis'], data = axis)
    vlength=[]
    for i in range(len(axis)):
        vlength.append((df_peak['FR'].iloc[i]+df_peak['FR'].iloc[i+4])/2)
    df_ohsi['V-lenght']= vlength
    horizontal = df_ohsi[df_ohsi['axis']=='horizontal'].values[0][1]
    vertical = df_ohsi[df_ohsi['axis']=='vertical'].values[0][1]
    ur_dl = df_ohsi[df_ohsi['axis']=='ur-dl'].values[0][1]
    ul_dr = df_ohsi[df_ohsi['axis']=='ul-dr'].values[0][1]
    X = horizontal - vertical
    Y = ur_dl - ul_dr
    vectorSum = np.sqrt(( X**2 + Y**2))
    totalVsum = horizontal + vertical + ur_dl + ul_dr
    elOSI = vectorSum/totalVsum
    return (elDSI, elOSI)

"""
Read data
"""
#directory = 'C:/Users/kiraz/Documents/McGill/Data/Stimulus-WT/SignalsDetectedClampfit/20200722'
directory = '/Users/vite/navigation_system/Rudo/GitHub-Balsa
######
path_neuron = directory+'/SpatDataFrame' +numneur + '.pkl'
df_FRbySpatialFrec = pd.read_pickle(path_neuron)
path_stims = directory + "/MeanStimTimes" + numNeur +'.npy'
stims =  np.load(path_stims) #in us



"""
POLAR PLOTS
"""

#Create a list of orientations in radians. Otherwise, seaborn won't make the plots properly
orientations = np.linspace(0, 2*np.pi, 9)[0:-1]
num_labels=len(df_base['label'].unique()) #it can be your spatial frequencies or neurons
orientations_list = []
for i in range(0, num_labels):
    for o in orientations:
        orientations_list.append(o)      
orientations = orientations_list

#linear plot
df_FRbySpatialFrec.drop(['neuron', 'direction'], axis=1, inplace=True)
df_FRbySpatialFrec['direction'] = orientations
df_FRbySpatialFrec = df_FRbySpatialFrec.astype({'direction':float, 'FR':float, 'label':str})
plt.figure(figsize = (20,12), tight_layout=True)
sns.lineplot(x = "direction", y="FR", hue = "label", data=df_FRbySpatialFrec)
#polar plot
g = sns.FacetGrid (df_FRbySpatialFrec, col='label', hue = "label", subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)
g.map(sns.scatterplot, "direction", "FR")
g.map(plt.plot, "direction", "FR")

#polar plot just for one neuron
frequency = str(0.02)
df_n = df_FRbySpatialFrec[df_FRbySpatialFrec['label']== frequency]
df_n = df_n.append(df_n.loc[0])
g = sns.FacetGrid (df_n, col='label', hue = "label", subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)
g.map(sns.scatterplot, "direction", "FR")
g.map(plt.plot, "direction", "FR")


"""
Calculate DOSIs
"""

(DSI,OSI) = CalcDOSI(df_FRbySpatialFrec)







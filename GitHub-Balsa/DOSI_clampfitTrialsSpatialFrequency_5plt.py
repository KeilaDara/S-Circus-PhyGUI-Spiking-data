# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 02:36:59 2021

@author: kiraz

Outputs
-Polar plots by each spatial frequency
"""

import numpy as np
import pandas as pd
from functions import *
import os
import matplotlib.pyplot as plt
import seaborn as sns

def CalcDOSI(df_base, peak=False):
    """

    Parameters
    ----------
    df_base : DataFrame containing a label (like neuron or spatial frequency) and FR by direction
    peak : Bool, if False, it willl require just one set of directions

    Returns
    -------
    elDSI : Direction selectivity index.

    """
    if not peak:
        df_base.reset_index(level=0, inplace=True)
    else: 
        df_base = df_base.groupby(["direction"]).max()
        df_base.reset_index(level=0, inplace=True)
    ##########################################################################
    # Calculate DSI
    df_dsi = df_base
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
    ##########################################################################
    # Calculate OSI
    axis = ['horizontal', 'ur-dl', 'vertical', 'ul-dr'] # u=up, d=down, l=left, r=right
    df_ohsi = pd.DataFrame(columns = [ 'axis'], data = axis)
    vlength=[]
    for i in range(len(axis)):
        vlength.append((df_base['FR'].iloc[i]+df_base['FR'].iloc[i+4])/2)
    df_ohsi['V-lenght']= vlength
    horizontal = df_ohsi.loc[df_ohsi['axis']=='horizontal'].values[0][1]
    vertical = df_ohsi.loc[df_ohsi['axis']=='vertical'].values[0][1]
    ur_dl = df_ohsi.loc[df_ohsi['axis']=='ur-dl'].values[0][1]
    ul_dr = df_ohsi.loc[df_ohsi['axis']=='ul-dr'].values[0][1]
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
directory = '/Users/vite/navigation_system/Rudo/GitHub-Balsa'
######
numNeur = 'L6'
path_neuron = directory+'/SpatDataFrame' +numNeur + '.pkl'
df_base = pd.read_pickle(path_neuron)
path_stims = directory + "/MeanStimTimes" + numNeur +'.npy'
stims =  np.load(path_stims) #in us
pathxDOSI = directory + '/df_DOSI.pkl'
df_DOSIs = pd.read_pickle(pathxDOSI)

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
df_base.drop(['neuron', 'direction'], axis=1, inplace=True)
df_base['direction'] = orientations
df_base = df_base.astype({'direction':float, 'FR':float, 'label':str})
plt.figure(figsize = (20,12), tight_layout=True)
sns.lineplot(x = "direction", y="FR", hue = "label", data=df_base)
#polar plot
g = sns.FacetGrid (df_base, col='label', hue = "label", subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)
g.map(sns.scatterplot, "direction", "FR")
g.map(plt.plot, "direction", "FR")

#polar plot just for one neuron
frequency = str(0.02)
df_n = df_base[df_base['label']== frequency]
df_n = df_n.append(df_n.loc[0])
g = sns.FacetGrid (df_n, col='label', hue = "label", subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)
g.map(sns.scatterplot, "direction", "FR")
g.map(plt.plot, "direction", "FR")
 


"""
Calculate DOSIs
"""
# df_DOSIs = pd.DataFrame(columns = ['neuron', 'frequency', 'DSI', 'OSI']) #Uncomment just the first time you are running this script
for l in df_base['label'].unique():
    (DSI,OSI) = CalcDOSI(df_base.loc[df_base['label']==l])
    df_row = {'neuron': numNeur, 'frequency': l, 'DSI': DSI, 'OSI':OSI} 
    df_DOSIs = df_DOSIs.append(df_row, ignore_index=True)

#save results
df_DOSIs.to_pickle(pathxDOSI)

##############################################################################
# Pandas way
### DSI
#plot histograms by frequency
for freq in df_base['label'].unique():
    plt.figure()
    df_DOSIs.loc[df_DOSIs['frequency']==freq]['DSI'].hist(alpha=0.5)
    plt.xlabel = 'DSI'
    plt.ylabel='count'
    plt.show()
#Plot for all
df_DOSIs.loc[:,['frequency','DSI']].hist(by='frequency',alpha=0.5)
### OSI
#plot histograms by frequency
for freq in df_base['label'].unique():
    plt.figure()
    df_DOSIs.loc[df_DOSIs['frequency']==freq]['OSI'].hist(alpha=0.5, color='r')
    plt.xlabel = 'OSI'
    plt.ylabel='count'
    plt.show()
#Plot for all
df_DOSIs.loc[:,['frequency','OSI']].hist(by='frequency',alpha=0.5, color='r')

##############################################################################
# Seaborn way
### DSI
#plot histograms by frequency
sns.set_theme(style="darkgrid")
for freq in df_DOSIs['frequency'].unique():
    plt.figure()
    sns.histplot(df_DOSIs.loc[df_DOSIs['frequency']==freq]['DSI'], edgecolor=".3")
    plt.title(freq)
    plt.xlim(0,1)
    plt.xlabel = 'DSI'
    plt.ylabel='count'
    plt.show()
#Plot for all
plt.figure()
sns.histplot(data=df_DOSIs.loc[:,['frequency','DSI']], x="DSI",hue='frequency')
plt.xlim(0,1)
### OSI
#plot histograms by frequency
for freq in df_base['label'].unique():
    plt.figure()
    sns.histplot(df_DOSIs.loc[df_DOSIs['frequency']==freq]['OSI'], color='r', edgecolor=".3",)
    plt.title(freq)
    plt.xlim(0,1)
    plt.xlabel = 'OSI'
    plt.ylabel='count'
    plt.show()
#Plot for all
plt.figure()
sns.histplot(data=df_DOSIs.loc[:,['frequency','OSI']], x="OSI",hue='frequency')
plt.xlim(0,1)
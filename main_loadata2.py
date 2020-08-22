#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:59:02 2020

@author: vite
"""

#importing libraries 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import *


"""
Read data
"""
#Select data directory
directory = '/Users/vite/navigation_system/Data_Kei'
#create a pandas data frame with the information coming from the cluster_group file
df = pd.DataFrame.from_csv(directory +"/cluster_group.tsv", sep="\t")
#Select unique values from data frame
labels=df.iloc[:,0].unique()
# Create a column of Boolean values. All the "good" values will have the label True. 
df['label']=df=='good'
#Take the index corresponding to the templates marked as good
neurons = df[df['label']].index.values
#load soike data
neuralData = np.load (directory + "/spike_times.npy")
#load clusters
klusters= np.load (directory + "/spike_clusters.npy")
#stack arrays coming from the spikes and klusters 
data=np.stack([neuralData,klusters], axis=1)
#make a pd with the two arrays
df=pd.DataFrame(data, columns=['spikes','klusters'])


#Make a loop to load the data from all the neurons and save everything in a dataframe


# Let's say you want to compute the autocorr with 5 ms bins
binsize = 4
# with 200 bins
nbins = 200
# Now we can call the function crossCorr
from functions import crossCorr

#Autocorrelogram for one neuron

#select a template
neuron =neurons[0]
#find the values in the dataframe that corresponds to the template chosen
df['label'] = df['klusters']==neuron
#Take just the True values corresponding to the template (neuron) chosen
data =df[df['label']]['spikes']
#Transform data to ms
data = data/20
aucorr = crossCorr(data.values, data.values, binsize, nbins)
#Compute mean firing rate
rec_duration = (data.iloc[-1] - data.iloc[0])/1000
meanfirate = len(data)/rec_duration
aucorr=aucorr/meanfirate
aucorr [int(nbins/2)] = meanfirate
#aucorr= smooth_corr(aucorr, nbins, binsize, meanfirate , window = 7, stdv = 5.0, plot = True
# The corresponding times can be computed as follow 
times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
# Let's make a time series
aucorr = pd.Series(index = times, data = aucorr)
plt.plot(aucorr.values)
plt.title ("Neuron " + str(neuron))
plt.show()



plt.figure(figsize= [20,30])
for i,j in enumerate(neurons):
    #select a template
    neuron =neurons[i]
    #find the values in the dataframe that corresponds to the template chosen
    df['label'] = df['klusters']==neuron
    #Take just the True values corresponding to the template (neuron) chosen
    data =df[df['label']]['spikes']
    #Transform data to ms
    data = data/20
    aucorr = crossCorr(data.values, data.values, binsize, nbins)
    #Compute mean firing rate
    rec_duration = (data.iloc[-1] - data.iloc[0])/1000
    meanfirate = len(data)/rec_duration
    aucorr=aucorr/meanfirate
    aucorr [int(nbins/2)] = meanfirate
    #aucorr= smooth_corr(aucorr, nbins, binsize, meanfirate , window = 7, stdv = 5.0, plot = True
    # The corresponding times can be computed as follow 
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
    # Let's make a time series
    aucorr = pd.Series(index = times, data = aucorr)
    ax = plt.subplot(2,3,i+1)
    ax.plot(aucorr)
    plt.title ("Neuron " + str(neuron))
plt.tight_layout()
plt.show()
plt.savefig(directory + '/plots' + '/autocorrelogram'  + '.pdf')



    
#Make the raster plots and histograms for all neurons in one plot
#plots
l=[]
for i,j in enumerate(neurons):
   #select a template
    neuron =neurons[i]
    print(i)
    #find the values in the dataframe that corresponds to the template chosen
    df['label'] = df['klusters']==neuron
    #Take just the True values corresponding to the template (neuron) chosen
    data =df[df['label']]['spikes']
    #Transform data to ms
    data = data/20
    l.append(data.values)

l= np.asarray(l)
np.asarray

colors1 = ['C{}'.format(i) for i in range(len(neurons))]

plt.figure()
plt.eventplot(l/1000, linelengths=0.1, colors ='k')
plt.title('Raster plot')
plt.xlabel('Time (s)')
plt.ylabel('Neuron')
positions = [0,1,2,3,4]
labels =["1","2","3","4","5"]
plt.yticks(positions, labels)
plt.box(False)
plt.savefig(directory + '/plots' + '/raster'  + '.pdf')


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
plt.xlabel('Time (s)')
plt.ylabel('neuron')
#plt.yticks(range(1,5))
plt.savefig(directory + '/plots' + '/raster'  + '.pdf')



#Distribution of spikes 
my_neuron = data*1000
first_spike = my_neuron.iloc[0]
last_spike = my_neuron.iloc[-1]
#Determine bin size in us
bin_size=1000000 # = 1s
 # Observe the -1 for the value at the end of an array
duration = last_spike - first_spike
# it's the time of the last spike
# with a bin size of 1 second, the number of points is 
nb_points = duration/bin_size  
nb_points = int(nb_points)
#Determine the bins of your data and apply digitize to get a classification index
bins = np.arange(first_spike, last_spike, bin_size)
index = np.digitize(my_neuron.index.values, bins, right=False)
#Create a pd
df_n = pd.DataFrame(index = index)
df_n['firing_time']=my_neuron.index.values
#count the number of spikes per bin
df_n_grouped=df_n.groupby(df_n.index).size().reset_index(name='counts')
df_n_grouped.set_index('index', inplace=True)
df_n_grouped.hist(bins=100)


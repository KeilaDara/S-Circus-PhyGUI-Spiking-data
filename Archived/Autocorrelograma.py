# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 23:35:51 2020

@author: kiraz
"""


#importing libraries 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import *
from guidata.qt.compat import getexistingdirectory


"""
Read data
"""
#Select data directory

directory= getexistingdirectory()

#directory = 'C:/Users/kiraz/Documents/McGill/Data/Espontanea-RD1/20191219/ND4LB/Data0125/Data0125.GUI'
#create a pandas data frame with the information coming from the cluster_group file
df = pd.read_csv(directory +"/cluster_group.tsv", sep="\t")
#Select unique values from data frame
labels=df.iloc[:,0].unique()
# Create a column of Boolean values. All the "good" values will have the label True. 
df['label']=df['group']=='good'
#Take the cluster id corresponding to the templates marked as good
neurons = df[df['label']].cluster_id.values
#load spike data/todos los spike times combinados de todos las neuronas detectadas
neuralData = np.load (directory + "/spike_times.npy")
#load clusters/son los clusters id ordenados segun la ocurrencia de los spike times, se repiten por cada spike time
klusters= np.load (directory + "/spike_clusters.npy")
#stack arrays coming from the spikes and klusters/une clusters y spike times en un arreglo np
data=np.stack([neuralData,klusters], axis=1)
#make a data frame with the two arrays/se hace el data frame a partir del arreglo que unio las variables
df=pd.DataFrame(data, columns=['spikes','klusters'])


#Make a loop to load the data from all the neurons and save everything in a dataframe



#AUTOCORRELOGRAM SETTINGS
# Let's say you want to compute the autocorr with 5 ms bins
binsize = 5
# with 200 bins
nbins = 300 #con 200 bines de 5ms tendremos 1s en todo el AuCorr(500 izq y 500 der)
# Now we can call the function crossCorr
from functions import crossCorr

#####################Autocorrelogram for one neuron##################################################################
#select a template/elige la neurona de la posicion solicitada en neurons
neuron =neurons[2]
#find the values in the dataframe that corresponds to the template chosen/crea una columna en df con valores boobleanos 'true' cada vez que encuentre el cluster id de la neurona elegida y su spike time corresp. y falso en los otros id's y st
df['label'] = df['klusters']==neuron
#Take just the True values corresponding to the template (neuron) chosen
data =df[df['label']]['spikes']

#Transform data to ms
data = data/20
#Compute autocorrelogram
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
aucorr[0.0]=0
aucorr[-5.0, 5.0]=2
aucorr[-10.0, 10.0]=3
aucorr[-15.0, 15.0]=8.5
aucorr[-20.0, 20.0]=8.6
plt.figure()
aucorr.plot(color="C0", linewidth=1.5, linestyle="-")
plt.title ("Neuron " + str(neuron))
plt.xlabel('time (ms)')
plt.show()

neurona0=aucorr
neurona1=aucorr
neurona2=aucorr
neurona3=aucorr
neurona4=aucorr
neurona5=aucorr
#neurona6=aucorr
neurona7=aucorr
neurona8=aucorr
neurona9=aucorr
neurona10=aucorr

Average=(neurona0+neurona2+neurona3+neurona4)/4
#Average=(neurona2+neurona4+neurona5+neurona10)/4

plt.figure()
neurona0.plot(color="silver")
#neurona1.plot(color="brown")
neurona2.plot(color="mistyrose")
neurona3.plot(color="papayawhip")
neurona4.plot(color="lightcyan")
#neurona5.plot(color="lightyellow")
#neurona6.plot(color="silver")
#neurona7.plot(color="silver")
#neurona8.plot(color="silver")
#neurona9.plot(color="silver")
Average.plot(color="k", linewidth=2.0, linestyle="-")
plt.title ("Autocorrelogram")
plt.xlabel('time (ms)')
plt.show()
#plt.box(False)



#######################################################################################################################

##############################Autocorrelogram for more neurons###########################################33

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
    ax = plt.subplot(5,5,i+1)
    aucorr[0.0]=0
    ax.plot(aucorr)
    plt.title ("Neuron " + str(neuron))
plt.tight_layout()
plt.show()
plt.savefig(directory + '/plots' + '/autocorrelogram'  + '.jpg')






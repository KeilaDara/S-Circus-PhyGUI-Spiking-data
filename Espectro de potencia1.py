# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 23:49:34 2020

@author: kiraz
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt, exp


"""
Read data
"""
#Select data directory
directory = 'C:/Users/kiraz/Documents/McGill/Data/Espontanea-RD1/20191219/ND5LB/Data0130/Data0130.GUI'
# directory = '/Users/vite/navigation_system/Rudo/156Th5.25'
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
#select a template/elige la neurona de la posicion solicitada en neurons
neuron =neurons[0]
#find the values in the dataframe that corresponds to the template chosen/
df['label'] = df['klusters']==neuron
#Take just the True values corresponding to the template (neuron) chosen
data =df[df['label']]['spikes']
data = data/20

"""
Read neural data from clampfit
"""
directory = '/Users/vite/navigation_system/Data_Kei'
data = np.loadtxt(directory + '/D11Data002.txt')

#Convolution with Gaussian Kernel to transform data to continuous
desirefreq=20 #Hz
binsize=((1/desirefreq)/10)*1000
sigma=1.5
cutoff= 1000/binsize/(2*np.pi*sigma)

def gauss(n=11,sigma=sigma):
    r = np.linspace(-int(n/2)+0.5,int(n/2)-0.5, n)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)*2/(2*sigma*2)) for x in r]
#data=spikes[0].restrict(data).as_units('ms').index.values
minv = int(data.min())
maxv = int(data.max())
bins = np.linspace( minv, maxv, int((maxv - minv)/binsize))
hist, edges = np.histogram(
    data,
    bins= bins,
    density=False)
kernel =  gauss()

from scipy.signal import convolve
signal=convolve(hist,kernel)

plt.figure()
plt.plot(signal)

signaldf=pd.DataFrame(signal)
signaldf.to_csv (directory+'/signal_Data0130_13.csv')

"""
FFT
"""

##Apply FFT to the continuous convolved data to obtain the power spectrum:
import scipy.fftpack
srate=1000/binsize #In Hz
npnts=srate*2 #2s
signalX = scipy.fftpack.fft(signal)
signalAmp=2*np.abs(np.sqrt(signalX))/npnts#Eje Y

#Obtaining frequencies
hz = np.linspace(0,srate/2,int(np.floor(npnts/2)+1)) #freq spectrum method 1,  Eje X

t= 1/20000 #sampling space
freq = np.fft.fftfreq(len(signal), t)#freq spectrum method 2,  Eje X
# Get positive half of frequencies
i = freq>0
positFreq=freq[i]

#Plots
plt.plot((hz),signalAmp[0:len(hz)],'k')
#plt.plot(positFreq, signalAmp [i], 'k')
plt.xlim([0,100])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Power Spectrum')
plt.show()  

#hz.to_csv (directory+'/hz.csv')
#signalAmp.to_csv (directory+'/signalAmp.csv')

#power
#signalPow=2*np.abs(signalX)**2/npnts
#plt.plot(hz,signalPow[0:len(hz)],'k')
#plt.xlim([3,30])
#plt.xlabel('Frequency (Hz)')
#plt.ylabel('Amplitude')
#plt.title('Frequency domain')
#plt.show()
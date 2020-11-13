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
directory = 'C:/Users/kiraz/Documents/McGill/Data/Gnat-espontanea/05082020'
# directory = '/Users/vite/navigation_system/Rudo/156Th5.25'
#create a pandas data frame with the information coming from the cluster_group file

"""
Read neural data from clampfit
"""

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

##Apply FFT to the continuous convolved data to obtain the power spectrum:NO
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
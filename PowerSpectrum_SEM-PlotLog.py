#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 00:00:03 2020

@author: vite
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

path = '/Users/vite/Downloads/RD1-MediaPowerSpectrumPerNeuronConcatenatedForPython.xlsx'

df = pd.read_excel(path, names = ["freq", "data", "neuron"], header = None)

# #way 1
# df.plot(logx=True)

# #way 2
# plt.figure()
# plt.semilogx(df.index, df[1])
# plt.show()

#seaborn vite modo
f, ax = plt.subplots(figsize=(7, 7))
ax.set(xscale="log")
sns.lineplot(x = "freq", y = "data", ci =  'sd', data = df, ax=ax)


#Rudo way
n = np.sqrt(126)
std = df.groupby('freq').agg(np.std)
err = std/n
mean = df.groupby('freq').agg(np.mean)


array1 = (mean-err)['data'].values.flatten()
array2 = (mean+err)['data'].values.flatten()
mean = mean.values.flatten()
freqs = df[df['neuron']==df['neuron'][0]]['freq'].values.flatten()

f, ax = plt.subplots(figsize=(7, 7))
ax.set(xscale="log")
ax.fill_between(freqs, array1, array2, alpha = 0.5, facecolor='lightsteelblue')
ax.plot(freqs, mean, 'k')
ax.tick_params(axis=u'both', which=u'both',length=0)
plt.show()
plt.box(None)
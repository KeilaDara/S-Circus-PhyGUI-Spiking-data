#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:18:38 2020

@author: vite
"""

import sounddevice as sd

wave = []
length = 100
for i in range(len(spk_counts[0])):
    if spk_counts[0][i] == 0:
        for j in range(length):
            wave.append(0)
    else:
        for j in range(length):
            wave.append(10000)

# Convert it to wav format (16 bits)
wav_wave = np.array(wave, dtype=np.int16)

sd.play(wav_wave, blocking=True)

plt.figure()
plt.plot(wave)
plt.show()




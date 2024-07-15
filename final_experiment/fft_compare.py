# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 10:34:17 2024

@author: fra01
"""

import numpy as np
import scipy.signal

import matplotlib.pyplot as plt


def abs2(x):
    return x.real**2 + x.imag**2


framelength=1.0
N=1000
x=np.linspace(0,framelength,N,endpoint=False)
y=np.sin(44*2*np.pi*x)+2
#y=y-np.mean(y)
ffty=np.fft.fft(y)
#power spectrum, after real2complex transfrom (factor )
scale=2.0/(len(y)*len(y))
power=scale*abs2(ffty)
freq=np.fft.fftfreq(len(y) , framelength/len(y) )

# power spectrum, via scipy welch. 'boxcar' means no window, nperseg=len(y) so that fft computed on the whole signal.
freq2,power2=scipy.signal.welch(y, fs=len(y)/framelength,window='boxcar',nperseg=len(y),detrend=False,scaling='spectrum', axis=-1, average='mean')
    
for i in range(len(freq2)):
    print(i, freq2[i], power2[i], freq[i], power[i])
print(np.sum(power2))

L=np.arange(0, np.floor(n/2), dtype='int') 
    
plt.figure()
plt.plot(freq[L],power[L],label='np.fft.fft()')
plt.plot(freq2,power2,label='scipy.signal.welch()')
plt.xlim(0,np.max(freq[L]))
plt.legend()


plt.show()
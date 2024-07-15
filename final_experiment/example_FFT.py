import numpy as np 
import matplotlib.pyplot as plt
import scipy

dt=0.001                #period of sampling
f_sampling=1/dt         #sampling frequency
t=np.arange(0,2,dt)

#Create a simple signal with 2 frequencies
f=np.sin(2*np.pi*50*t) + 4*np.sin(2*np.pi*120*t)+2
f_clean=f

#Add some noise
noise=2.5*np.random.randn(len(t))
f=f+noise

plt.close('all')
plt.plot(t, f, color='c', linewidth=1.5, label='Noisy')
plt.plot(t, f_clean, color='k', linewidth=2, label='Signal')
plt.xlim(t[0], t[-1])
plt.xlabel('t [s]')
plt.ylabel('Voltage [V]')
plt.grid()
plt.legend()

#%%
#FFT

n=len(t)
f_hat=np.fft.fft(f,n)                           #Compute the FFT
PSD=2*np.real(f_hat * np.conj(f_hat)) / (f_sampling*n)                  #Power spectrum
freq=(1/(dt*n))*np.arange(n)                    #create x-axis of frequencies
L=np.arange(0, np.floor(n/2), dtype='int')      #Only plot the first half of the values
PSD[0]=PSD[0]/2

# plt.close('all')
fig, axs=plt.subplots(2,1, num=2)

plt.sca(axs[0])
plt.plot(t, f, color='c', linewidth=1.5, label='Noisy')
plt.plot(t, f_clean, color='k', linewidth=2, label='Signal')
plt.xlim(t[0], t[-1])
plt.suptitle('fft into my own version of PSD')
plt.xlabel('t [s]')
plt.ylabel('Voltage [V]')
plt.grid()
plt.legend()


plt.sca(axs[1])
plt.plot(freq[L], PSD[L], color='c', linewidth=1.5, label='Noisy')
# plt.xlim(freq[L[0]], freq[L[-1]])

plt.xlabel('f [Hz]')
plt.ylabel('PSD [$V^2/Hz$]')
plt.grid()
plt.legend()




freq_new, Pxx=scipy.signal.welch(f, f_sampling, window='boxcar',nperseg=n, detrend=False, noverlap=0)


fig, axs=plt.subplots(2,1, num=3)

plt.sca(axs[0])
plt.plot(t, f, color='c', linewidth=1.5, label='Noisy')
plt.plot(t, f_clean, color='k', linewidth=2, label='Signal')
plt.xlim(t[0], t[-1])
plt.suptitle('welch')
plt.xlabel('t [s]')
plt.ylabel('Voltage [V]')
plt.grid()
plt.legend()


plt.sca(axs[1])
plt.plot(freq_new, Pxx, color='c', linewidth=1.5, label='Noisy')
# plt.xlim(freq[L[0]], freq[L[-1]])
plt.xlabel('f [Hz]')
plt.ylabel('PSD [$V^2/Hz$]')
plt.grid()
plt.legend()



#%%
#Use the PSD to filter out the noise 

indices=PSD>100         #find out all the freqs with large power 
PSDclean=PSD*indices    #zero out all the others
f_hat=indices*f_hat     #zero out small fourier coeff in Y
f_filt=np.fft.ifft(f_hat)   #inverse FFt for filtered time signal 




plt.close('all')
fig, axs=plt.subplots(3,1)

plt.sca(axs[0])
plt.plot(t, f, color='c', linewidth=1.5, label='Noisy')
plt.plot(t, f_clean, color='k', linewidth=2, label='Signal')
plt.xlim(t[0], t[-1])
plt.xlabel('t [s]')
plt.ylabel('Voltage [V]')
plt.grid()
plt.legend()


plt.sca(axs[1])
plt.plot(t, f_filt, color='k', linewidth=2, label='Filtered')
plt.xlim(t[0], t[-1])
plt.xlabel('t [s]')
plt.ylabel('Voltage [V]')
plt.grid()
plt.legend()



plt.sca(axs[2])
plt.plot(freq[L], PSD[L], color='c', linewidth=2, label='Noisy')
plt.plot(freq[L], PSDclean[L], color='k', linewidth=1.5, label='Filtered')
plt.xlim(freq[L[0]], freq[L[-1]])
plt.xlabel('f [Hz]')
plt.ylabel('PSD [$V^2/Hz$]')
plt.grid()
plt.legend()






import numpy as np
import matplotlib.pyplot as plt
import scipy
from myfunctions import Circuit, do_graphs
import t0_AB
import pandas as pd

df1 = pd.read_csv('final_experiment/data/DCmeasurement/totalDC1.csv')

C1= Circuit(V_PP_str_gau_value=15, f_str_gau_value=625, V_PP_source_value=15, f_source_value=100*10**(-3),
           i_0_rec_value=0.995, time_constant_value=300*10**(-3), FS_value=500*10**(-6),
           t_values=df1['Time'].to_numpy(), V_CH1_values=df1['Channel1'].to_numpy(), V_CH2_values=df1['Channel2'].to_numpy(), V_CH3_values=df1['Channel3'].to_numpy())



f_sampling=C1.output["F_sampling [Hz]"]
f_source=C1.source_parameters["f_source [Hz]"]

t=C1.output["t [s]"]
F=C1.output["F_t [N]"]
F_QUAD=C1.output["F_QUAD_t [N]"]
V_CH1=C1.output["V_CH1_lock_in [V]"]
V_CH2=C1.output["V_CH2_lock_in [V]"]
V_CH3=C1.output["V_CH3 [V]"]

f=C1.output["frequencies [Hz]"]
S_F_sqrt=np.sqrt(C1.output["S_F [N^2/Hz]"])
S_F_QUAD_sqrt=np.sqrt(C1.output["S_F_QUAD [N^2/Hz]"])
S_V_sqrt=np.sqrt(C1.output["S_V_CH1_lock_in [V^2/Hz]"])
S_V_QUAD_sqrt=np.sqrt(C1.output["S_V_CH2_lock_in [V^2/Hz]"])


samples_tot=C1.output["N_samples"]
samples_per_div=samples_tot//2

freq_new, Pxx=scipy.signal.welch(F, f_sampling, window='blackmanharris', nperseg=samples_per_div)#, detrend=False)
S_F_ave_sqrt=np.sqrt(Pxx)
freq_new, Pxx=scipy.signal.welch(F_QUAD, f_sampling, window='blackmanharris', nperseg=samples_per_div)
S_F_QUAD_ave_sqrt=np.sqrt(Pxx)
freq_new, Pxx=scipy.signal.welch(V_CH1, f_sampling, window='blackmanharris', nperseg=samples_per_div)
S_V_ave_sqrt=np.sqrt(Pxx)
freq_new, Pxx=scipy.signal.welch(V_CH2, f_sampling, window='blackmanharris', nperseg=samples_per_div)
S_V_QUAD_ave_sqrt=np.sqrt(Pxx)

plt.figure()
plt.plot(t[:1000], F[:1000], linestyle='None', marker='.', label='Force')
plt.plot(t[:1000], F_QUAD[:1000], linestyle='None', marker='.', label='Quadrature')
plt.xlabel('t [s]')
plt.ylabel('V [V]')
plt.legend()
plt.grid()
plt.show()

Y_list=[F*1000, F_QUAD*1000, V_CH1*1000, V_CH2*1000, S_F_sqrt, S_F_QUAD_sqrt, S_V_sqrt, S_V_QUAD_sqrt, S_F_ave_sqrt, S_F_QUAD_ave_sqrt, S_V_ave_sqrt, S_V_QUAD_ave_sqrt]
plt_params_list=[['None', '.', 'purple', 'force (in phase)'],
                 ['None', '.', 'red', 'quadrature'],
                 ['None', '.', 'green', 'X'],
                 ['None', '.', 'blue', 'Y'],
                  ['-', None, 'purple', '(in phase)'],
                  ['-', None, 'red', '(in quadrature)'],
                  ['-', None, 'green', 'X'],
                  ['-', None, 'blue', 'Y'],
                  ['-', None, 'purple', '(in phase)'],
                  ['-', None, 'red', '(in quadrature)'],
                  ['-', None, 'green', 'X (average)'],
                  ['-', None, 'blue', 'Y (average)']
                 ]

graph_settings_list=[['t [s]', 'F [mN]', 'upper right', 'linear', 'linear'],
                 ['t [s]', 'V [mV]', 'upper right', 'linear', 'linear'],
                  ['f [Hz]', '$S_F^{1/2}\ \ [N/Hz^{1/2}]$', 'upper right', 'log', 'log'],
                  ['f [Hz]', '$S_V^{1/2}\ \ [V/Hz^{1/2}]$', 'upper right', 'log', 'log'],
                  ['f [Hz]', '$S_F^{1/2}\ \ [N/Hz^{1/2}]$', 'upper right', 'log', 'log'],
                  ['f [Hz]', '$S_V^{1/2}\ \ [V/Hz^{1/2}]$', 'upper right', 'log', 'log'],
                 ['$V^{PP}=15 \ V \ f_{BR}=625, \ V^{source}_{PP}=2 \ V \ f_{MOD}=0.05 \ Hz, \ I_{DC}=0.995 \ A$', 'xx-large'],
                 ]


do_graphs(nrows=3, ncols=2, comb=[2,2,2,2,2,2], X=[t,t, f, f, freq_new, freq_new], Y=Y_list, plot_params=plt_params_list, graph_settings=graph_settings_list)


t0, f_MOD=t0_AB.determine_t0_fmod_function(time=t, channel3=df1['Channel3'].to_numpy(), f_MOD=f_source, n_divisions=30, plot=True)

t0 += 0.3

A_mean, A_std, B_mean, B_std= t0_AB.determine_A_B_func(time=t, channel=F, t_0=t0, f_MOD=f_MOD, n_divisions=30, plot=True)

A_mean, A_std, B_mean, B_std= t0_AB.determine_A_B_func(time=t, channel=df1['Channel3'].to_numpy(), t_0=t0, f_MOD=f_MOD, n_divisions=30, plot=True)

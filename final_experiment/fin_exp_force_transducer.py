# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from scipy import odr
import scipy
import myfunctions


#-------------------------------  
'1)DC/quasi-DC current configuration'


#1) Source 50 mHz Vpp=2V
data=np.loadtxt('final_experiment/50mHz2V_101_v.dat', dtype=str, skiprows=3)#, delimiter=';')
data=data.astype(np.float64)

C1= myfunctions.Circuit(V_PP_str_gau_value=15, f_str_gau_value=625, V_PP_source_value=2, f_source_value=50*10**(-3),
           i_0_rec_value=0.995, time_constant_value=300*10**(-3), FS_value=500*10**(-6),
           t_values=data[:,0], V_CH1_values=data[:,1], V_CH2_values=data[:,2], V_CH3_values=data[:,3])



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

# do_graphs(1,1, comb=[1], X=t, Y=V_CH3, ngraph=1, plot_params=['None', '.', 'blue', 'Channel 3'], 
#           graph_settings=[['t [s]', 'V [V]', 'upper right', 'linear', 'linear'],['$f_{MOD}=0.05 Hz, V_{PP}^{source}=2 V$', 'xx-large']])



#---------------------------------
'PLOTS'
plt.close('all')
fig, ax = plt.subplots(dpi=120)
ax.plot(t, V_CH3, linestyle='None', marker='.', label='Channel 3')
ax.set_title('$V^{PP}=15 \ V \ f_{BR}=625, \ V^{source}_{PP}=15 \ V \ f_{MOD}=0.05 \ Hz, \ I_{DC}=0.995 \ A$')
ax.set_xlabel('t [s]')
ax.set_ylabel('V [V]')
ax.legend(loc='upper right')
ax.grid()

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


myfunctions.do_graphs(nrows=3, ncols=2, comb=[2,2,2,2,2,2], X=[t,t, f, f, freq_new, freq_new], Y=Y_list, plot_params=plt_params_list, graph_settings=graph_settings_list)



t0, f_MOD=myfunctions.determine_t0_fmod_function(x=t, y=V_CH3, f_MOD=f_source, n_divisions=5, plot=True)

A_mean, A_std, B_mean, B_std= myfunctions.determine_A_B_func(time=t, channel=F, t_0=t0, f_MOD=f_MOD, plot=True)







#%%
#2) Source 100 mHz Vpp=2V
data=np.loadtxt('final_experiment/forcenoise_DC1A_100mHz15V_105_v.dat', dtype=str, skiprows=3)#, delimiter=';')
data=data.astype(np.float64)

C2=myfunctions.Circuit(V_PP_str_gau_value=15, f_str_gau_value=625, V_PP_source_value=15, f_source_value=100*10**(-3),
           i_0_rec_value=0.995, time_constant_value=300*10**(-3), FS_value=500*10**(-6),
           t_values=data[:,0], V_CH1_values=data[:,1], V_CH2_values=data[:,2], V_CH3_values=data[:,3])



f_source=C2.source_parameters["f_source [Hz]"]
f_sampling=C2.output["F_sampling [Hz]"]

t=C2.output["t [s]"]
F=C2.output["F_t [N]"]
F_QUAD=C2.output["F_QUAD_t [N]"]
V_CH1=C2.output["V_CH1_lock_in [V]"]
V_CH2=C2.output["V_CH2_lock_in [V]"]
V_CH3=C2.output["V_CH3 [V]"]


f=C2.output["frequencies [Hz]"]
S_F_sqrt=np.sqrt(C2.output["S_F [N^2/Hz]"])
S_F_QUAD_sqrt=np.sqrt(C2.output["S_F_QUAD [N^2/Hz]"])
S_V_sqrt=np.sqrt(C2.output["S_V_CH1_lock_in [V^2/Hz]"])
S_V_QUAD_sqrt=np.sqrt(C2.output["S_V_CH2_lock_in [V^2/Hz]"])


samples_tot=C2.output["N_samples"]
samples_per_div=7200

freq_new, Pxx=scipy.signal.welch(F, f_sampling, window='blackmanharris', nperseg=samples_per_div)#, detrend=False)
S_F_ave_sqrt=np.sqrt(Pxx)
freq_new, Pxx=scipy.signal.welch(F_QUAD, f_sampling, window='blackmanharris', nperseg=samples_per_div)
S_F_QUAD_ave_sqrt=np.sqrt(Pxx)
freq_new, Pxx=scipy.signal.welch(V_CH1, f_sampling, window='blackmanharris', nperseg=samples_per_div)
S_V_ave_sqrt=np.sqrt(Pxx)
freq_new, Pxx=scipy.signal.welch(V_CH2, f_sampling, window='blackmanharris', nperseg=samples_per_div)
S_V_QUAD_ave_sqrt=np.sqrt(Pxx)


# start_point=0
end_point=int(samples_tot/100)
steps_point=1
# iterations_values=range(start_point, end_point, steps_point)





#-------------------------
'PLOTS'
plt.close('all')
fig, ax = plt.subplots(dpi=120)
ax.plot(t[:end_point:steps_point], V_CH3[:end_point:steps_point], label='Channel 3')
ax.set_title('$V^{PP}=15 \ V \ f_{BR}=625, \ V^{source}_{PP}=15 \ V \ f_{MOD}=0.1 \ Hz, \ I_{DC}=0.995 \ A$')
ax.set_xlabel('t [s]')
ax.set_ylabel('V [V]')
ax.legend(loc='upper right')
ax.set_xscale('log')
ax.grid()

X_list=[t[:end_point:steps_point],t[:end_point:steps_point], f, f, freq_new, freq_new]
Y_list=[F[:end_point:steps_point], F_QUAD[:end_point:steps_point], V_CH1[:end_point:steps_point], 
        V_CH2[:end_point:steps_point], S_F_sqrt, S_F_QUAD_sqrt, S_V_sqrt, S_V_QUAD_sqrt, 
        S_F_ave_sqrt, S_F_QUAD_ave_sqrt, S_V_ave_sqrt, S_V_QUAD_ave_sqrt]

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

graph_settings_list=[['t [s]', 'F [N]', 'upper right', 'linear', 'linear'],
                 ['t [s]', 'V [V]', 'upper right', 'linear', 'linear'],
                  ['f [Hz]', '$S_F^{1/2}\ \ [N/Hz^{1/2}]$', 'upper right', 'log', 'log'],
                  ['f [Hz]', '$S_V^{1/2}\ \ [V/Hz^{1/2}]$', 'upper right', 'log', 'log'],
                  ['f [Hz]', '$S_F^{1/2}\ \ [N/Hz^{1/2}]$', 'upper right', 'log', 'log'],
                  ['f [Hz]', '$S_V^{1/2}\ \ [V/Hz^{1/2}]$', 'upper right', 'log', 'log'],
                 ['$V^{PP}=15 \ V \ f_{BR}=625, \ V^{source}_{PP}=15 \ V \ f_{MOD}=0.1 \ Hz, \ I_{DC}=0.995 \ A$', 'xx-large'],
                 ]


myfunctions.do_graphs(nrows=3, ncols=2, comb=[2,2,2,2,2,2], X=X_list, Y=Y_list, plot_params=plt_params_list, graph_settings=graph_settings_list)



f_MOD=f_source
t0, f_MOD=myfunctions.determine_t0_fmod_function(x=t, y=V_CH3, f_MOD=f_source, n_divisions=5, plot=True)


# t0=0
A_mean, A_std, B_mean, B_std= myfunctions.determine_A_B_func(time=t, channel=F, t_0=t0, f_MOD=f_MOD, plot=True)





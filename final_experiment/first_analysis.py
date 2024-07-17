import numpy as np
import matplotlib.pyplot as plt
import scipy
import myfunctions
import pandas as pd

df1 = pd.read_csv('final_experiment/data/weekend4.csv')

C1= myfunctions.Circuit(V_PP_str_gau_value=15, f_str_gau_value=625, V_PP_source_value=2, f_source_value=50*10**(-3),
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

# do_graphs(1,1, comb=[1], X=t, Y=V_CH3, ngraph=1, plot_params=['None', '.', 'blue', 'Channel 3'], 
#           graph_settings=[['t [s]', 'V [V]', 'upper right', 'linear', 'linear'],['$f_{MOD}=0.05 Hz, V_{PP}^{source}=2 V$', 'xx-large']])



#---------------------------------
'PLOTS'
'''plt.close('all')
fig, ax = plt.subplots(dpi=120)
ax.plot(t, V_CH3, linestyle='None', marker='.', label='Channel 3')
ax.set_title('$V^{PP}=15 \ V \ f_{BR}=625, \ V^{source}_{PP}=2 \ V \ f_{MOD}=0.05 \ Hz, \ I_{DC}=0.995 \ A$')
ax.set_xlabel('t [s]')
ax.set_ylabel('V [V]')
ax.legend(loc='upper right')
ax.grid()'''

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



#t0, f_MOD=myfunctions.determine_t0_fmod_function(x=t, y=V_CH3, f_MOD=f_source, n_divisions=5, plot=True)

#A_mean, A_std, B_mean, B_std= myfunctions.determine_A_B_func(x=t, y=F, t0=t0, f_MOD=f_MOD, plot=True)

'''d_F = np.diff(F)
d_FQUAD = np.diff(F_QUAD)

first_disc_f = np.argmax(d_F[:len(d_F)//2])
second_disc_f = np.argmax(d_F[len(d_F)//2:])
print(f'\nFirst discontinuty in F is {t[first_disc_f]} s')
print(f'Second discontinuty in F is {t[second_disc_f]} s')
print(f'Time elapsed {t[second_disc_f]-t[first_disc_f]} s')

first_disc_fquad = np.argmin(d_FQUAD[:len(d_F)//2])
second_disc_fquad = np.argmin(d_FQUAD[len(d_F)//2:])
print(f'\nFirst discontinuty in F_quad is {t[first_disc_fquad]} s')
print(f'Second discontinuty in F_quad is {t[second_disc_fquad]} s')
print(f'Time elapsed {t[second_disc_fquad]-t[first_disc_fquad]} s')
'''
plt.figure()
plt.plot(df1['Time'], df1['Channel1'], linestyle='None', marker='+', label='Force')
plt.plot(df1['Time'], df1['Channel2'], label='Quadrature')
plt.xlabel('t [s]')
plt.ylabel('V [V]')
plt.legend()
plt.grid()
plt.show()

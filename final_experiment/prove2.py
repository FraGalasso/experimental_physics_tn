import pandas as pd
import numpy as np
import t0_AB
import t0_ABCD
import conversion as conv

df = pd.read_csv(
    'final_experiment/data/DCmeasurement/totalDC1.csv')

time = df['Time'].to_numpy()
ch1 = conv.compvolt_to_force(df['Channel1'].to_numpy(), 0.0005)
ch3 = df['Channel3'].to_numpy()

t0, f_MOD = t0_AB.determine_t0_fmod_function(time, ch3, 0.1, False, 31)

t0 += 0.3

print('old approach:')
A_mean, A_std, B_mean, B_std = t0_AB.determine_A_B_func(
    time=time, channel=ch1, t_0=t0, f_MOD=f_MOD, interval_length=300, plot=True)

print('new approach:')
A_mean, A_std, B_mean, B_std = t0_AB.determine_A_B_func_alt(
    time=time, channel=ch1, t_0=t0, f_MOD=f_MOD, interval_length=300, plot=True)

d = 0.0165  # m
i_r = 0.996  # A
i_s = 0.13486  # A
delta_ir = 0.012
delta_is = 0.00012
delta_d = 0.0005

delta_f = t0_AB.dc_force_estimate()*np.sqrt((delta_ir/i_r)**2+(delta_is/i_s)**2)
print(f'F = {t0_AB.dc_force_estimate()} +/- {delta_f} N')

df = pd.read_csv(
    f'final_experiment/data/ACmeasurement_firstbatch/ac_first2.csv')

time = df['Time'].to_numpy()
ch1 = conv.compvolt_to_force(df['Channel1'].to_numpy(), 1e-4)
ch3 = df['Channel3'].to_numpy()

t0, f_MOD = t0_ABCD.determine_t0_fmod_function(time, ch3, 0.05, False, 27)
t0 += 0.3

print('old approach:')
A_mean, A_std, B_mean, B_std, C_mean, C_std, D_mean, D_std = t0_ABCD.determine_A_B_C_D_func(
    time=time, channel=ch1, t_0=t0, f_MOD=f_MOD, interval_length=300, plot=True, freq=300)

print('new approach:')
A_mean, A_std, B_mean, B_std, C_mean, C_std, D_mean, D_std = t0_ABCD.determine_A_B_C_D_func_alt(
    time=time, channel=ch1, t_0=t0, f_MOD=f_MOD, interval_length=300, plot=True, freq=300)

import pandas as pd
import t0_ABCD
import t0_AB

df = pd.read_csv(
    'final_experiment/data/DCmeasurement/totalDC1.csv')

time = df['Time'].to_numpy()
ch1 = df['Channel1'].to_numpy()
ch3 = df['Channel3'].to_numpy()

print(f'Elapsed time: {df["Time"].iloc[-1]-df["Time"].iloc[0]} s')

t0, f_MOD = t0_AB.determine_t0_fmod_function(time, ch3, 0.1, False, 31)

t0 += 0.3

A_mean, A_std, B_mean, B_std = t0_AB.determine_A_B_func(
    time=time, channel=df['Channel1'].to_numpy(), t_0=t0, f_MOD=f_MOD, n_divisions=31, plot=True)


A_mean, A_std, B_mean, B_std = t0_ABCD.determine_A_B_func(
    time=time, channel=df['Channel1'].to_numpy(), t_0=t0, f_MOD=f_MOD, interval_length=300, plot=True)

df = pd.read_csv('final_experiment/data/ACmeasurement_firstbatch/ac_first2.csv')

print(f'Elapsed time: {df["Time"].iloc[-1]-df["Time"].iloc[0]} s')

time = df['Time'].to_numpy()
ch1 = df['Channel1'].to_numpy()
ch3 = df['Channel3'].to_numpy()

t0, f_MOD = t0_AB.determine_t0_fmod_function(time, ch3, 0.05, False, 28)

t0 +=0.3

A_mean, A_std, B_mean, B_std, C_mean, C_std, D_mean, D_std = t0_ABCD.determine_A_B_C_D_func(
    time=time, channel=df['Channel1'].to_numpy(), t_0=t0, f_MOD=f_MOD, interval_length=300, plot=True)

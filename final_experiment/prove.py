import pandas as pd
import t0_ABCD
import t0_AB

'''df = pd.read_csv(
    'final_experiment/data/DCmeasurement/totalDC1.csv')

time = df['Time'].to_numpy()
ch1 = df['Channel1'].to_numpy()
ch3 = df['Channel3'].to_numpy()

print(f'Elapsed time: {df["Time"].iloc[-1]-df["Time"].iloc[0]} s')

t0, f_MOD = t0_AB.determine_t0_fmod_function(time, ch3, 0.1, False, 31)

t0 += 0.3

A_mean, A_std, B_mean, B_std = t0_AB.determine_A_B_func(
    time=time, channel=ch1, t_0=t0, f_MOD=f_MOD, n_divisions=31, plot=True)'''

df_list = []

for i in range(1, 7):
    df_list.append(pd.read_csv(
        f'final_experiment/data/ACmeasurement_firstbatch/ac_first{i}.csv'))

for i in range(1, 7):
    df_list.append(pd.read_csv(
        f'final_experiment/data/ACmeasurement_secondbatch/ac_second{i}.csv'))

for item in df_list:
    print(f'Elapsed time: {item["Time"].iloc[-1]-item["Time"].iloc[0]} s')

    time = item['Time'].to_numpy()
    ch1 = item['Channel1'].to_numpy()
    ch3 = item['Channel3'].to_numpy()

    t0, f_MOD = t0_ABCD.determine_t0_fmod_function(time, ch3, 0.05, False, 28)

    t0 += 0.3

    A_mean, A_std, B_mean, B_std, C_mean, C_std, D_mean, D_std = t0_ABCD.determine_A_B_C_D_func(
        time=time, channel=ch1, t_0=t0, f_MOD=f_MOD, interval_length=300, plot=True)

'''
df = pd.read_csv(
    f'final_experiment/data/ACmeasurement_firstbatch/ac_first2.csv')

time = df['Time'].to_numpy()
ch3 = df['Channel3'].to_numpy()

print('Using model A*sin(2*pi*f*t)+B*cos(2*pi*f*t):')
t0, f_MOD = t0_AB.determine_t0_fmod_function(time, ch3, 0.05, True, 28)
print('Using model A*sin(2*pi*f*t)+B*cos(2*pi*f*t)+C*sin(2*pi*(2*f)*t)+D*cos(2*pi*(2*f)*t)+E:')
print('t0 = (1/2*pi*f)arctan(-B/A)')
t0, f_MOD = t0_ABCD.determine_t0_fmod_function(time, ch3, 0.05, True, 28)
print('Using model A*sin(2*pi*f*t)+B*cos(2*pi*f*t)+C*sin(2*pi*(2*f)*t)+D*cos(2*pi*(2*f)*t)+E:')
print('t0 = (1/4*pi*f)arctan(C/D)')
t0, f_MOD = CD.determine_t0_fmod_function_CD(time, ch3, 0.05, True, 28)'''


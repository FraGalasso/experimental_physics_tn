import matplotlib.pyplot as plt
import pandas as pd
import t0_AB


def ch1ch2_plot(df, conversion):
    starting_time = df['Time'].iloc[0]

    plt.figure()
    plt.plot(df['Time']-starting_time, df['Channel1']*conversion,
             linestyle='None', marker='.', label='CH1')
    plt.plot(df['Time']-starting_time, df['Channel2']*conversion,
             linestyle='None', marker='.', label='CH2')
    plt.xlabel('t [s]')
    plt.ylabel('V [$\mu V$]')
    plt.legend()
    plt.grid()
    plt.show()


def ch3_plot(df, iter=1,  s=0, title=None):
    plt.figure()
    plt.plot(df['Time']-iter * 8400-s, df['Channel3'],
             linestyle='None', marker='.', label='CH1')
    plt.xlabel('t [s]')
    plt.ylabel('V [V]')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


conversion = 10

'''dftot = pd.read_csv(
    'final_experiment/data/ACmeasurement_firstbatch/total_ac1.csv')
starting_time = dftot['Time'].iloc[0]

df_ch3_1 = dftot[(dftot['Time'] > starting_time + 8400-300) &
                 (dftot['Time'] < starting_time + 8400+300)]
df_ch3_2 = dftot[(dftot['Time'] > starting_time + 2*8400-300)
                 & (dftot['Time'] < starting_time + 2*8400+300)]
df_ch3_3 = dftot[(dftot['Time'] > starting_time + 3*8400-300)
                 & (dftot['Time'] < starting_time + 3*8400+300)]
df_ch3_4 = dftot[(dftot['Time'] > starting_time + 4*8400-300)
                 & (dftot['Time'] < starting_time + 4*8400+300)]
df_ch3_5 = dftot[(dftot['Time'] > starting_time + 5*8400-300)
                 & (dftot['Time'] < starting_time + 5*8400+300)]

ch3_plot(df_ch3_1, 1, starting_time, '100-300 Hz')
ch3_plot(df_ch3_2, 2, starting_time, '300-500 Hz')
ch3_plot(df_ch3_3, 3, starting_time, '500-700 Hz')
ch3_plot(df_ch3_4, 4, starting_time, '700-800 Hz')
ch3_plot(df_ch3_5, 5, starting_time, '800-900 Hz')'''


df1 = pd.read_csv('final_experiment/data/ACmeasurement_firstbatch/ac_first1.csv')
df2 = pd.read_csv('final_experiment/data/ACmeasurement_firstbatch/ac_first2.csv')
df3 = pd.read_csv('final_experiment/data/ACmeasurement_firstbatch/ac_first3.csv')
df4 = pd.read_csv('final_experiment/data/ACmeasurement_firstbatch/ac_first4.csv')
df5 = pd.read_csv('final_experiment/data/ACmeasurement_firstbatch/ac_first5.csv')
df6 = pd.read_csv('final_experiment/data/ACmeasurement_firstbatch/ac_first6.csv')

dataframes = [df1, df2, df3, df4, df5, df6]
t0_list = []
f_MOD_list = []
A_list = []
A_std_list = []
B_list = []
B_std_list = []

for i in range(6):
    time = dataframes[i]['Time'].to_numpy()
    t0, f_MOD = t0_AB.determine_t0_fmod_function(time, dataframes[i]['Channel3'].to_numpy(), 0.05, True, 30)

    t0 += 0.3
    t0_list.append(t0)
    f_MOD_list.append(f_MOD)

    A_mean, A_std, B_mean, B_std = t0_AB.determine_A_B_func(
        time=time, channel=(dataframes[i]['Channel1']*conversion).to_numpy(), t_0=t0, f_MOD=f_MOD, n_cycl_div=500, plot=True)
    
    A_list.append(A_mean)
    A_std_list.append(A_std)
    B_list.append(A_mean)
    B_std_list.append(A_std)

for i in range(6):
    ch1ch2_plot(dataframes[i], conversion)
    print(f'Iteration {i+1}')
    print(f't0 = {t0_list[i]} s')
    print(f'f_MOD = {f_MOD_list[i]} Hz')
    print(f'A = {A_list[i]} +/- {A_std_list[i]} uV$')
    print(f'A = {B_list[i]} +/- {B_std_list[i]} uV$')

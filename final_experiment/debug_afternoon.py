import numpy as np
import matplotlib.pyplot as plt
import scipy
import myfunctions
import pandas as pd

conversion = 10**(-5)

df1 = pd.read_csv(
    'final_experiment/data/debugging_15afternoon/AC_fmod50mHz_fac700Hz_1139_v.dat', delimiter='\t')

starting_time = df1['Time'].iloc[0]
plt.figure()
plt.plot(df1['Time']-starting_time, df1['Channel1']*conversion,
         linestyle='None', marker='.', label='CH1')
plt.plot(df1['Time']-starting_time, df1['Channel2']*conversion,
         linestyle='None', marker='.', label='CH2')
plt.xlabel('t [s]')
plt.ylabel('V [V]')
plt.legend()
plt.grid()
plt.show()

df2 = pd.read_csv(
    'final_experiment/data/debugging_15afternoon/total_debug1.csv')
plt.figure()
plt.plot(df2['Time'], df2['Channel1']*conversion,
         linestyle='None', marker='.', label='CH1')
plt.plot(df2['Time'], df2['Channel2']*conversion,
         linestyle='None', marker='.', label='CH2')
plt.xlabel('t [s]')
plt.ylabel('V [V]')
plt.legend()
plt.grid()
plt.show()

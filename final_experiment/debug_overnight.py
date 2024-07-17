import matplotlib.pyplot as plt
import pandas as pd

conversion = 10

df1 = pd.read_csv(
    'final_experiment/data/ACmeasurement_secondbatch/total_ac2_1.csv')

starting_time = df1['Time'].iloc[0]
print(df1['Time'].iloc[-1]-starting_time)
print(df1['Time'].iloc[-1])
plt.figure()
plt.plot(df1['Time']-starting_time, df1['Channel1']*conversion,
         linestyle='None', marker='.', label='CH1')
plt.plot(df1['Time']-starting_time, df1['Channel2']*conversion,
         linestyle='None', marker='.', label='CH2')
plt.xlabel('t [s]')
plt.ylabel('V [$\mu V$]')
plt.legend()
plt.grid()
plt.show()

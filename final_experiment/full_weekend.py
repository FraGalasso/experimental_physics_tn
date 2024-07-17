import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv('final_experiment/data/weekend/total_weekend1.csv')

plt.figure()
plt.plot(df1['Time']-4391592, df1['Channel1'], linestyle='None', marker='+', label='Force')
plt.plot(df1['Time']-4391592 , df1['Channel2'], label='Quadrature')
plt.xlabel('t [s]')
plt.ylabel('V [V]')
plt.legend()
plt.grid()
plt.show()

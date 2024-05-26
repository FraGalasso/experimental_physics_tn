import numpy as np
# from io import StringIO
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def lin_func(x, a, b):
    return a+(x*b)

data=np.loadtxt('monochromator/original_data/new_red_600.csv', dtype=str, skiprows=4, delimiter=';')
data=np.char.replace(data, ',','.')
data
data=data.astype(np.float64)
t_data_lockin, V_data_lockin=data[:,0], data[:,1]

t_data_poten, V_data_poten=data[:,2], data[:,3]

peaks, _ = find_peaks(V_data_lockin)


half_size_V_lock=int(np.size(V_data_lockin)/2)
max_V_lock_red=max(V_data_lockin[:half_size_V_lock])
for i in range(np.size(V_data_lockin)):
    if V_data_lockin[i]==max_V_lock_red:
        cont=i
        t_maxV_lock_red=t_data_lockin[cont]
        max_V_poten_red=V_data_poten[cont]
        break
# for i in range(np.size(t_data_poten)):
    
popt_red, pcov_red=curve_fit(lin_func, t_data_poten[:4700], V_data_poten[:4700])

x_red_theo=np.linspace(min(t_data_poten[:4700]), max(t_data_poten[:4700]), 100)
y_red_theo=popt_red[0]+popt_red[1]*x_red_theo
new_max_V_poten_red=popt_red[0]+popt_red[1]*t_maxV_lock_red   
    
# plt.close('all')
plt.figure(num=3, dpi=120)
# plt.plot(t_data_lockin, V_data_lockin, label='lock-in')
plt.plot(t_data_poten, V_data_poten, label='potentiometer')
plt.plot(x_red_theo, y_red_theo, label='fit')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('Voltage [V]')
plt.title('red')
plt.grid()

plt.figure(num=4, dpi=120)
plt.plot(t_data_lockin, V_data_lockin, label='lock-in')
# plt.plot(t_data_lockin[peaks], V_data_lockin[peaks], marker='+')
# plt.plot(t_data_poten, V_data_poten, label='potentiometer')
plt.plot(t_maxV_lock_red, max_V_lock_red, marker='+')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('Voltage [V]')
plt.title('red')
plt.grid()
plt.show()

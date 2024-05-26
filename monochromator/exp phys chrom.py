import numpy as np
# from io import StringIO
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def lin_func(x, a, b):
    return a+(x*b)


#yellow
data=np.loadtxt('monochromator/original_data/yellow.csv', dtype=str, skiprows=4, delimiter=';')
data=np.char.replace(data, ',','.')
data
data=data.astype(np.float64)
t_data_lockin, V_data_lockin=data[:,0], data[:,1]

t_data_poten, V_data_poten=data[:,2], data[:,3]

peaks, _ = find_peaks(V_data_lockin)


half_size_V_lock=int(np.size(V_data_lockin)/2)
max_V_lock_yellow=max(V_data_lockin[:half_size_V_lock])
for i in range(np.size(V_data_lockin)):
    if V_data_lockin[i]==max_V_lock_yellow:
        cont=i
        t_maxV_lock_yellow=t_data_lockin[cont]
        max_V_poten_yellow=V_data_poten[cont]
        break
# for i in range(np.size(t_data_poten)):
    

popt_yellow, pcov_yellow=curve_fit(lin_func, t_data_poten[:9300], V_data_poten[:9300])

x_yellow_theo=np.linspace(min(t_data_poten[:9300]), max(t_data_poten[:9300]), 100)
y_yellow_theo=popt_yellow[0]+popt_yellow[1]*x_yellow_theo
new_max_V_poten_yellow=popt_yellow[0]+popt_yellow[1]*t_maxV_lock_yellow
    
plt.close('all')
plt.figure(num=1, dpi=120)
# plt.plot(t_data_lockin, V_data_lockin, label='lock-in')
plt.plot(t_data_poten, V_data_poten, label='potentiometer')
plt.plot(x_yellow_theo, y_yellow_theo, label='fit')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('Voltage [V]')
plt.title('yellow')
plt.grid()

plt.figure(num=2, dpi=120)
plt.plot(t_data_lockin, V_data_lockin, label='lock-in')
# plt.plot(t_data_lockin[peaks], V_data_lockin[peaks], marker='+')
# plt.plot(t_data_poten, V_data_poten, label='potentiometer')
plt.plot(t_maxV_lock_yellow, max_V_lock_yellow, marker='+')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('Voltage [V]')
plt.title('yellow')
plt.grid()

# # plt.close('all')
# plt.figure(num=10)
# plt.plot(V_data_poten, V_data_lockin)
# plt.title('try')

#red
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


#green
data=np.loadtxt('monochromator/original_data/green.csv', dtype=str, skiprows=4, delimiter=';')
data=np.char.replace(data, ',','.')
data
data=data.astype(np.float64)
t_data_lockin, V_data_lockin=data[:,0], data[:,1]

t_data_poten, V_data_poten=data[:,2], data[:,3]

peaks, _ = find_peaks(V_data_lockin)


half_size_V_lock=int(np.size(V_data_lockin)/2)
max_V_lock_green=max(V_data_lockin[:half_size_V_lock])
for i in range(np.size(V_data_lockin)):
    if V_data_lockin[i]==max_V_lock_green:
        cont=i
        t_maxV_lock_green=t_data_lockin[cont]
        max_V_poten_green=V_data_poten[cont]
        break
# for i in range(np.size(t_data_poten)):
    
    
popt_green, pcov_green=curve_fit(lin_func, t_data_poten[:6700], V_data_poten[:6700])

x_green_theo=np.linspace(min(t_data_poten[:6700]), max(t_data_poten[:6700]), 100)
y_green_theo=popt_green[0]+popt_green[1]*x_green_theo
new_max_V_poten_green=popt_green[0]+popt_green[1]*t_maxV_lock_green

    
# plt.close('all')
plt.figure(num=5, dpi=120)
# plt.plot(t_data_lockin, V_data_lockin, label='lock-in')
plt.plot(t_data_poten, V_data_poten, label='potentiometer')
plt.plot(x_green_theo, y_green_theo, label='fit')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('Voltage [V]')
plt.title('green')
plt.grid()

plt.figure(num=6, dpi=120)
plt.plot(t_data_lockin, V_data_lockin, label='lock-in')
# plt.plot(t_data_lockin[peaks], V_data_lockin[peaks], marker='+')
# plt.plot(t_data_poten, V_data_poten, label='potentiometer')
plt.plot(t_maxV_lock_green, max_V_lock_green, marker='+')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('Voltage [V]')
plt.title('green')
plt.grid()



#505 filter
data=np.loadtxt('monochromator/original_data/505_2.csv', dtype=str, skiprows=4, delimiter=';')
data=np.char.replace(data, ',','.')
data
data=data.astype(np.float64)
t_data_lockin, V_data_lockin=data[:,0], data[:,1]

t_data_poten, V_data_poten=data[:,2], data[:,3]

peaks, _ = find_peaks(V_data_lockin)


half_size_V_lock=int(np.size(V_data_lockin)/2)
max_V_lock_505_fil=max(V_data_lockin[:half_size_V_lock])
for i in range(np.size(V_data_lockin)):
    if V_data_lockin[i]==max_V_lock_505_fil:
        cont=i
        t_maxV_lock_505_fil=t_data_lockin[cont]
        max_V_poten_505_fil=V_data_poten[cont]
        break
# for i in range(np.size(t_data_poten)):
    
    
popt_505, pcov_505=curve_fit(lin_func, t_data_poten[:5700], V_data_poten[:5700])

x_505_theo=np.linspace(min(t_data_poten[:5700]), max(t_data_poten[:5700]), 100)
y_505_theo=popt_505[0]+popt_505[1]*x_505_theo
new_max_V_poten_505_fil=popt_505[0]+popt_505[1]*t_maxV_lock_505_fil


# plt.close('all')
plt.figure(num=7, dpi=120)
# plt.plot(t_data_lockin, V_data_lockin, label='lock-in')
plt.plot(t_data_poten, V_data_poten, label='potentiometer')
plt.plot(x_505_theo, y_505_theo, label='fit')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('Voltage [V]')
plt.title('505 filter')
plt.grid()

plt.figure(num=8, dpi=120)
plt.plot(t_data_lockin, V_data_lockin, label='lock-in')
# plt.plot(t_data_lockin[peaks], V_data_lockin[peaks], marker='+')
# plt.plot(t_data_poten, V_data_poten, label='potentiometer')
plt.plot(t_maxV_lock_505_fil, max_V_lock_505_fil, marker='+')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('Voltage [V]')
plt.title('505 filter')
plt.grid()


#linear regression
data_V_lock=[max_V_lock_yellow, max_V_lock_red, max_V_lock_green, max_V_lock_505_fil]
data_V_res=[max_V_poten_yellow, max_V_poten_red, max_V_poten_green, max_V_poten_505_fil]
new_data_V_res=[new_max_V_poten_yellow, new_max_V_poten_red, new_max_V_poten_green, new_max_V_poten_505_fil]
colour_nm=[578, 592, 552, 505]



popt, pcov = curve_fit(lin_func, colour_nm, data_V_res)

x_lin=np.linspace(min(colour_nm), max(colour_nm), 100)
y_lin=popt[0]+popt[1]*x_lin

popt_new, pcov_new = curve_fit(lin_func, colour_nm, new_data_V_res)

x_lin_new=np.linspace(min(colour_nm), max(colour_nm), 100)
y_lin_new=popt_new[0]+popt_new[1]*x_lin_new

# plt.close('all')
plt.figure(num=9, dpi=120)
plt.plot(colour_nm, data_V_res, linestyle='None', marker='.')
plt.plot(x_lin, y_lin)
plt.plot(colour_nm, new_data_V_res, linestyle='None', marker='.')
plt.plot(x_lin_new, y_lin_new)
plt.xlabel('wavelength [nm]')
plt.ylabel('Voltage [V]')
plt.title('linear regression')
plt.grid()

plt.show()

err_intercept=np.sqrt(pcov[0,0])
err_slope=np.sqrt(pcov[1,1])

percentage_err_intercept=err_intercept*100/popt[0]
percentage_err_slope=err_slope*100/popt[1]

print('percentage error intercept:%f ' %percentage_err_intercept)
print('percentage error slope:%f ' %percentage_err_slope)

'''
data=np.loadtxt('monochromator lab/spectrum.csv', dtype=str, skiprows=4, delimiter=';')
data=np.char.replace(data, ',','.')
data
data=data.astype(np.float64)
t_data_lockin, V_data_lockin=data[:,0], data[:,1]

t_data_poten, V_data_poten=data[:,2], data[:,3]

peaks, _ = find_peaks(V_data_lockin)


plt.figure(num=10, dpi=120)

plt.plot(t_data_lockin, V_data_lockin)
plt.title('Spectrum')
plt.xlabel('time [s]')
plt.ylabel('Voltage [V]')
plt.grid()
'''
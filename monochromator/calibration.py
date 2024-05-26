from utilities import read_data, linear, plot_color, lin_fit_peak
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# loading data
red_data = read_data('new_red_600.csv')
yellow_data = read_data('yellow.csv')
green_data = read_data('green.csv')
extra_filter_data = read_data('505_2.csv')

# fitting first (forward) sweep of the potentiometer
popt_y, pcov_y = curve_fit(
    linear, yellow_data['Time1'][:9700], yellow_data['Potentiometer'][:9700])
popt_r, pcov_r = curve_fit(
    linear, red_data['Time1'][:4700], red_data['Potentiometer'][:4700])
popt_g, pcov_g = curve_fit(
    linear, green_data['Time1'][:6700], green_data['Potentiometer'][:6700])
popt_5, pcov_5 = curve_fit(
    linear, extra_filter_data['Time1'][:5700], extra_filter_data['Potentiometer'][:5700])

# plotting all the data
# plot_color(yellow_data, 'Yellow', popt_y)
# plot_color(red_data, 'Red', popt_r)
# plot_color(green_data, 'Green', popt_g)
# plot_color(extra_filter_data, '505', popt_5)

# finding peaks in lock-in data and corresponding potentiometer distribution
peak_y = lin_fit_peak(yellow_data, 'Yellow', popt_y)
peak_r = lin_fit_peak(red_data, 'Red', popt_r)
peak_g = lin_fit_peak(green_data, 'Green', popt_g)
peak_5 = lin_fit_peak(extra_filter_data, '505', popt_5)

peaks = np.array([peak_y, peak_r, peak_g, peak_5])
wavelength = [578, 592, 552, 505]

# linear fit of potentiometer and wavelength
popt_calibration, pcov_calibration = curve_fit(linear, wavelength, peaks[:, 2])

print(
    f'Intercept = {popt_calibration[0]} +/- {np.sqrt(pcov_calibration[0,0])} V')
print(f'Slope = {popt_calibration[1]} +/- {np.sqrt(pcov_calibration[1,1])} V/nm')

fitlabel = f'a={popt_calibration[0]}V, b={popt_calibration[1]}V/nm'

x_calib = np.linspace(min(wavelength), max(wavelength), 100)
y_calib = popt_calibration[0] + popt_calibration[1] * x_calib

# plotting the fit
plt.figure(dpi=200)
plt.xlabel('Wavelenght [nm]')
plt.ylabel('Potentiometer voltage [V]')
plt.title('Calibration')
plt.plot(wavelength, peaks[:, 2], linestyle='None', marker='.', label='data')
plt.plot(x_calib, y_calib, label=fitlabel)
plt.legend()
plt.grid()

plt.show()

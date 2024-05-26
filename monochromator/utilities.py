import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def linear(x, a, b):
    return a + b * x


def read_data(filename):
    data = pd.read_csv(f'monochromator/data/{filename}', sep=';')
    return data


def lin_fit_peak(data, filter, popt):
    '''Returns a list with the time of the peak in V_lockin, the peak in
    V_lockin and the corresponding potentiometer value (obtained with linear fitting)'''
    if filter == 'Yellow':
        data = data.iloc[:9700]
    elif filter == 'Green':
        data = data.iloc[:6700]
    elif filter == '505':
        data = data.iloc[:5700]
    elif filter == 'Red':
        data = data.iloc[:4700]

    max_row = data.query(f"{'Lock_in'} == {data['Lock_in'].max()}")
    better_potentiometer = popt[0] + popt[1] * max_row.iloc[0, 0]
    return [max_row.iloc[0, 0], max_row.iloc[0, 1], better_potentiometer]


def plot_color(data, filter, popt):
    '''Produces 3 plots, potentiometer in time (with linear fit), lock-in in
    time, and potentiometer vs lock-in (only fitted part)'''
    if filter == 'Yellow':
        x_fit = np.linspace(min(data['Time1'][:9700]), max(
            data['Time1'][:9700]), 100)
    elif filter == 'Red':
        x_fit = np.linspace(min(data['Time1'][:4700]), max(
            data['Time1'][:4700]), 100)
    elif filter == 'Green':
        x_fit = np.linspace(min(data['Time1'][:6700]), max(
            data['Time1'][:6700]), 100)
    elif filter == '505':
        x_fit = np.linspace(min(data['Time1'][:5700]), max(
            data['Time1'][:5700]), 100)

    y_fit = popt[0] + popt[1] * x_fit

    plt.figure(num=1, dpi=200)
    plt.title('Potentiometer - ' + filter)
    data.plot(kind='scatter', x='Time1', y='Potentiometer',
              ax=plt.gca(), label='potentiometer')
    plt.plot(x_fit, y_fit, label='fit', color='orange')
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    plt.legend()
    plt.grid()

    plt.figure(num=2, dpi=200)
    plt.title('Lock-in - ' + filter)
    data.plot(kind='scatter', x='Time1', y='Lock_in',
              ax=plt.gca(), label='lock-in')
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    plt.grid()

    if filter == 'Yellow':
        data = data.iloc[:9700]
    elif filter == 'Green':
        data = data.iloc[:6700]
    elif filter == '505':
        data = data.iloc[:5700]
    elif filter == 'Red':
        data = data.iloc[:4700]

    plt.figure(num=3, dpi=200)
    plt.title('Potentiometer vs Lock-in - ' + filter)
    data.plot(kind='scatter', x='Potentiometer', y='Lock_in',
              ax=plt.gca())
    plt.xlabel('Potentiometer voltage [V]')
    plt.ylabel('Lock-in voltage [V]')
    plt.grid()
    plt.show()


def plot_spectrum(data, title=None, label=None, filename=None):
    '''Takes as input a pandas DataFrame and plots it with given title
    and label.'''

    calib_result = [3.213650130473219, 0.004381576754374339]
    '''results from calibration, first element is intercept in V,
    second one is slope in V/nm'''

    data['Wavelength'] = (data['Potentiometer'] -
                          calib_result[0]) / calib_result[1]

    plt.figure(dpi=200)
    plt.title(title)
    data.plot(kind='scatter', x='Wavelength', y='Lock_in',
                   ax=plt.gca(), label=label)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Voltage [V]')
    plt.grid()

    if filename is not None:
        plt.savefig(filename)

    plt.show()

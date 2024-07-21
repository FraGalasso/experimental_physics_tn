import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import plot_functions as pltfunc

# COMMON FUNCTIONS


def lin_func(x, m, q):
    '''Just a linear function.'''
    return m * x + q


def stats(vec):
    '''Function which takes a vector and returns mean and std dev of the mean.'''
    return np.mean(vec), np.std(vec) / np.sqrt(len(vec))

# DC MEASUREMENT : only A, B


def sin_cos_func(x, A, B, f, x0):
    '''Sinusoidal model function.'''
    return A * np.sin(2 * np.pi * f * (x - x0)) + B * np.cos(2 * np.pi * f * (x - x0))

def sin_cos_func_off(x, A, B, C, f, x0):
    '''Sinusoidal model function, with an offset.'''
    return A * np.sin(2 * np.pi * f * (x - x0)) + B * np.cos(2 * np.pi * f * (x - x0)) + C


def determine_A_B_in_interval(time, channel, t_0, f_MOD):
    '''Determines A and B for a given time interval.'''
    def model(time, A, B): return sin_cos_func(time, A, B, f=f_MOD, x0=t_0)
    popt, pcov = curve_fit(model, time, channel)
    A = popt[0]
    B = popt[1]
    return A, B

def determine_A_B_in_interval_off(time, channel, t_0, f_MOD):
    '''Determines A and B for a given time interval. (offset model function)'''
    def model(time, A, B, C): return sin_cos_func_off(time, A, B, C, f=f_MOD, x0=t_0)
    popt, pcov = curve_fit(model, time, channel)
    A = popt[0]
    B = popt[1]
    return A, B


def determine_t0_fmod_function(time, channel3, f_MOD, plot, n_divisions):
    '''Determines t0, f_MOD.'''
    n_samples = len(time)
    n_samples_per_division = int(n_samples/n_divisions)

    xmean_vec = np.zeros(n_divisions)
    phi_vec = np.zeros(n_divisions)

    m = 0
    delta = 0
    error = 0

    # 2 iterations should be sufficient
    while (error <= np.abs(m)):
        # correct f_MOD from previous step
        f_MOD = f_MOD-delta
        def model(x, A, B): return sin_cos_func(x, A, B, f=f_MOD, x0=0)

        # split the sample in n blocks
        for i in range(n_divisions):
            start_point = int(i*n_samples_per_division)
            end_point = int((i+1)*n_samples_per_division)

            x_new = time[start_point:end_point]
            y_new = channel3[start_point:end_point]

            popt, pcov = curve_fit(model, x_new, y_new)

            xmean_vec[i] = np.mean(x_new)
            phi_vec[i] = np.arctan2(-popt[1], popt[0])  # -B/A

        # linear fit in order to find dphi/dt
        popt, pcov = curve_fit(lin_func, xmean_vec, phi_vec)
        m = popt[0]  # dphi/dt
        q = popt[1]
        delta = m/(2*np.pi)  # correction to the next f_MOD
        error = np.sqrt(pcov[0, 0])

        t0 = np.mean(phi_vec)/(2*np.pi*f_MOD)

        print(f'dphi_dt = {m} +/- {np.sqrt(pcov[0,0])}')
        print(f'phi = {np.mean(phi_vec)}')
        print(f't0 = {t0}')
        print(f'f_mod = {f_MOD}')
        print(f'delta f = {delta} +/- {np.sqrt(pcov[0,0])/(2*np.pi)}\n')

        if plot:
            x_fit = np.linspace(time[0], time[-1], 1000)
            y_fit = m * x_fit + q
            plt.figure(dpi=120)
            plt.plot(xmean_vec, phi_vec, linestyle='None',
                     marker='.', label='data of intervals')
            plt.plot(x_fit, y_fit, label='fit')
            plt.xlabel('t [s]')
            plt.ylabel('phi [radians]')
            plt.legend()
            plt.grid()
            plt.show()

    return t0, f_MOD


def determine_A_B_func(interval_length, time, channel, f_MOD, t_0, plot, is_force=True):
    '''Determines A, B for a long data collection by splitting the data into intervals of interval_length seconds.
    The function returns the mean and std dev/sqrt(n) of the mean of the amplitudes of A and B.
    Assumes data to be already in force units, if not use is_force=False.'''
    # define parameters
    cycles_in_interval = interval_length * f_MOD
    one_cycle_time = 1/f_MOD  # s

    if not cycles_in_interval.is_integer():
        cycles_in_interval = int(cycles_in_interval)
        interval_length = one_cycle_time * cycles_in_interval
        print('ERROR: interval length was not a multiple of the oscillation period. %f s used instead' % interval_length)

    sampling_rate = 10  # Hz
    n_samples = len(time)
    n_samples_per_interval = int(sampling_rate * interval_length)
    n_intervals = n_samples // n_samples_per_interval

    # Initialize lists to store A and B values
    A_vec = []
    B_vec = []
    xmean_vec = []

    # Find A, B for each interval and append to vectors A_vec and B_vec
    # SAREBBE BELLO CAMBIARE QUESTI IN BASE AL TIMESTAMP INVECE CHE IN BASE ALL'INDICE DEL SAMPLE
    for i in range(0, n_intervals):
        start_point = int(i * n_samples_per_interval)
        end_point = int((i + 1) * n_samples_per_interval)

        time_int = time[start_point:end_point]
        channel_int = channel[start_point:end_point]

        A, B = determine_A_B_in_interval(time_int, channel_int, t_0, f_MOD)

        A_vec.append(A)
        B_vec.append(B)
        xmean_vec.append(np.mean(time_int))

    # Calculate mean and std devs and print
    A_mean, A_err = stats(A_vec)
    B_mean, B_err = stats(B_vec)
    print('A: %f +/- %f' % (A_mean, A_err))
    print('B: %f +/- %f' % (B_mean, B_err))

    # Plot the values
    if plot:
        pltfunc.generate_plot_A_B(xmean_vec, A_vec, B_vec, is_force)

    return A_mean, A_err, B_mean, B_err


def dc_force_estimate():
    '''Gives an estimate for the DC force in Newton.'''
    # CONSTANTS
    D_b = 0.235   # m
    d = 0.0165  # m
    N_windings = 84
    i_r = 0.996  # A
    i_s = 0.13486  # A
    return 0.2e-6 * i_r * i_s * (N_windings**2) * np.pi * D_b / d


def determine_A_B_func_alt(interval_length, time, channel, f_MOD, t_0, plot, is_force=True):
    '''Determines A, B for a long data collection by splitting the data into intervals of interval_length seconds.
    The function returns the mean and std dev/sqrt(n) of the mean of the amplitudes of A and B.
    Assumes data to be already in force units, if not use is_force=False.'''
    df = pd.DataFrame({'Time': time, 'Channel1': channel})
    df['interval'] = np.floor(
        (df['Time']-df['Time'].iloc[0]) / interval_length)

    def apply_func(group):
        t = group['Time'].values
        y = group['Channel1'].values
        avg_t = np.mean(t)
        A, B = determine_A_B_in_interval_off(
            time=t, channel=y, t_0=t_0, f_MOD=f_MOD)
        return pd.Series({'A': A, 'B': B, 'avg_t': avg_t})

    results = df.groupby('interval').apply(apply_func).reset_index()

    A_list = results['A'].tolist()
    B_list = results['B'].tolist()
    avg_t_list = results['avg_t'].tolist()

    # Calculate mean and std devs and print
    A_mean, A_err = stats(A_list)
    B_mean, B_err = stats(B_list)
    print('A: %f +/- %f' % (A_mean, A_err))
    print('B: %f +/- %f' % (B_mean, B_err))

    # Plot the values
    if plot:
        pltfunc.generate_plot_A_B(avg_t_list, A_list, B_list, is_force)

    return A_mean, A_err, B_mean, B_err

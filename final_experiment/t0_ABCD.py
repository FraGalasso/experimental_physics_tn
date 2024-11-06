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

# AC MEASUREMENT; with offset

def sin_cos_func_off(x, A, B, E, f, x0):
    '''Sinusoidal model function.'''
    return A * np.sin(2 * np.pi * f * (x - x0)) + B * np.cos(2 * np.pi * f * (x - x0)) + E

def sin_cos_func_bimod(x, A, B, C, D, E, f, x0):
    '''Bisinusoidal model function.'''
    return A * np.sin(2 * np.pi * f * (x - x0)) + B * np.cos(2 * np.pi * f * (x - x0)) + C * np.sin(2 * np.pi * 2*f * (x - x0)) + D * np.cos(2 * np.pi * 2*f * (x - x0)) + E


def determine_A_B_C_D_in_interval_off(time, channel, t_0, f_MOD):
    '''Determine A B C D for a given time interval.'''
    def model(time, A, B, C, D, E): return sin_cos_func_bimod(
        time, A, B, C, D, E, f=f_MOD, x0=t_0)
    popt, pcov = curve_fit(model, time, channel)
    A = popt[0]
    B = popt[1]
    C = popt[2]
    D = popt[3]
    return A, B, C, D


def determine_t0_fmod_function(time, channel3, f_MOD, plot, n_divisions):
    '''DetermineS t0, fmod using the A B C D with offset model.'''
    n_samples = len(time)
    n_samples_per_division = n_samples // n_divisions

    xmean_vec = np.zeros(n_divisions)
    phi_vec = np.zeros(n_divisions)

    m = 0
    delta = 0
    error = 0

    # 2 iterations should be sufficient
    while (error <= np.abs(m)):
        # correct f_MOD from previous step
        f_MOD = f_MOD-delta

        def model(x, A, B, E): return sin_cos_func_off(
            x, A, B, E, f=f_MOD, x0=0)

        # split the  sample in n blocks
        for i in range(n_divisions):
            start_point = int(i * n_samples_per_division)
            end_point = int((i + 1) * n_samples_per_division)

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


def determine_A_B_C_D_func(interval_length, time, channel, f_MOD, t_0, plot, freq=0, is_force=True):
    '''Determines A, B, C, D for a long data collection by splitting the data into intervals of interval_length seconds.
    The function returns the mean and std dev/sqrt(n) of the mean of the amplitudes of A B C and D.
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
    n_intervals = n_samples//n_samples_per_interval

    # Initialize lists to store A, B, C, D values
    A_vec = []
    B_vec = []
    C_vec = []
    D_vec = []
    xmean_vec = []

    # Find A, B, C, D for each interval and append to vectors A_vec, B_vec, C_vec, D_vec
    # SAREBBE BELLO CAMBIARE QUESTI IN BASE AL TIMESTAMP INVECE CHE IN BASE ALL'INDICE DEL SAMPLE
    for i in range(0, n_intervals):
        start_point = int(i * n_samples_per_interval)
        end_point = int((i + 1) * n_samples_per_interval)

        time_int = time[start_point:end_point]
        channel_int = channel[start_point:end_point]

        A, B, C, D = determine_A_B_C_D_in_interval_off(
            time_int, channel_int, t_0, f_MOD)

        A_vec.append(A)
        B_vec.append(B)
        C_vec.append(C)
        D_vec.append(D)
        xmean_vec.append(np.mean(time_int))

    # Calculate mean and std devs and print
    A_mean, A_err = stats(A_vec)
    B_mean, B_err = stats(B_vec)
    C_mean, C_err = stats(C_vec)
    D_mean, D_err = stats(D_vec)
    print('A: %f +/- %f' % (A_mean, A_err))
    print('B: %f +/- %f' % (B_mean, B_err))
    print('C: %f +/- %f' % (C_mean, C_err))
    print('D: %f +/- %f' % (D_mean, D_err))

    # Plot the values
    if plot:
        pltfunc.generate_plot_A_B_C_D(freq, xmean_vec, A_vec, B_vec, C_vec, D_vec, is_force)

    return A_mean, A_err, B_mean, B_err, C_mean, C_err, D_mean, D_err


# takes in the pars_list and std_list which are lists of lists. returns the separate lists
def extract_values(df_freq, pars_list, stds_list):
    # A is the first element in each sublist
    a_values = [pars[0] for pars in pars_list]
    a_std = [stds[0] for stds in stds_list]
    b_values = [pars[1] for pars in pars_list]
    b_std = [stds[1] for stds in stds_list]
    c_values = [pars[2] for pars in pars_list]
    c_std = [stds[2] for stds in stds_list]
    d_values = [pars[3] for pars in pars_list]
    d_std = [stds[3] for stds in stds_list]
    return a_values, a_std, b_values, b_std, c_values, c_std, d_values, d_std


def determine_A_B_C_D_func_alt(interval_length, time, channel, f_MOD, t_0, plot, freq=0, is_force=True):
    '''Determines A, B, C, D for a long data collection by splitting the data into intervals of interval_length seconds.
    The function returns the mean and std dev/sqrt(n) of the mean of the amplitudes of A B C and D.
    Assumes data to be already in force units, if not use is_force=False.'''
    
    df = pd.DataFrame({'Time': time, 'Channel1': channel})
    df['interval'] = np.floor(
        (df['Time']-df['Time'].iloc[0]) / interval_length)

    def apply_func(group):
        t = group['Time'].values
        y = group['Channel1'].values
        avg_t = np.mean(t)
        A, B, C, D = determine_A_B_C_D_in_interval_off(
            time=t, channel=y, t_0=t_0, f_MOD=f_MOD)
        return pd.Series({'A': A, 'B': B, 'C': C, 'D': D, 'avg_t': avg_t})

    results = df.groupby('interval').apply(apply_func).reset_index()

    A_list = results['A'].tolist()
    B_list = results['B'].tolist()
    C_list = results['C'].tolist()
    D_list = results['D'].tolist()
    avg_t_list = results['avg_t'].tolist()

    # Calculate mean and std devs and print
    A_mean, A_err = stats(A_list)
    B_mean, B_err = stats(B_list)
    C_mean, C_err = stats(C_list)
    D_mean, D_err = stats(D_list)
    print(f'Frequency = {freq} Hz')
    print('A: %f +/- %f microN' % (1e6*A_mean, 1e6*A_err))
    print('B: %f +/- %f microN' % (1e6*B_mean, 1e6*B_err))
    print('C: %f +/- %f microN' % (1e6*C_mean, 1e6*C_err))
    print('D: %f +/- %f microN' % (1e6*D_mean, 1e6*D_err))

    # Plot the values
    if plot:
        pltfunc.generate_plot_A_B_C_D(freq, avg_t_list, A_list, B_list, C_list, D_list, is_force)

    return A_mean, A_err, B_mean, B_err, C_mean, C_err, D_mean, D_err
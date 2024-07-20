import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# COMMON FUNCTIONS


# Just a linear function
def lin_func(x, m, q):
    return m * x + q


# Function which takes a vector and returns mean and std dev of the mean
def stats(vec):
    return np.mean(vec), np.std(vec) / np.sqrt(len(vec))

# AC MEASUREMENT; with offset


# Define the bisinusoidal model function
def sin_cos_func_off(x, A, B, C, D, E, f, x0):
    return A * np.sin(2 * np.pi * f * (x - x0)) + B * np.cos(2 * np.pi * f * (x - x0)) + C * np.sin(2 * np.pi * 2*f * (x - x0)) + D * np.cos(2 * np.pi * 2*f * (x - x0)) + E


# Determine A B C D for a given time interval
def determine_A_B_C_D_in_interval_off(time, channel, t_0, f_MOD):
    def model(time, A, B, C, D, E): return sin_cos_func_off(
        time, A, B, C, D, E, f=f_MOD, x0=t_0)
    popt, pcov = curve_fit(model, time, channel)
    A = popt[0]
    B = popt[1]
    C = popt[2]
    D = popt[3]
    return A, B, C, D


# Determine to, fmod using the a b c d with offset model
def determine_t0_fmod_function(time, channel3, f_MOD, plot, n_divisions):
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

        def model(x, A, B, C, D, E): return sin_cos_func_off(
            x, A, B, C, D, E, f=f_MOD, x0=0)

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


# Generate plot of A B C D
def generate_plot_A_B_C_D(xmean_vec, A_vec, B_vec, C_vec, D_vec):
    plt.figure(dpi=120)
    plt.plot(xmean_vec, A_vec, linestyle='None', marker='.', label='$A_{sin}$')
    plt.plot(xmean_vec, B_vec, linestyle='None', marker='.', label='$B_{cos}$')
    plt.plot(xmean_vec, C_vec, linestyle='None',
             marker='.', label='$C_{sin2}$')
    plt.plot(xmean_vec, D_vec, linestyle='None',
             marker='.', label='$D_{cos2}$')
    plt.title('Values of force amplitudes A and B of sinusoidal model')
    A_mean, A_err = stats(A_vec)
    B_mean, B_err = stats(B_vec)
    C_mean, C_err = stats(C_vec)
    D_mean, D_err = stats(D_vec)
    plt.suptitle('A: %f +/- %f, B: %f +/- %f, \n C: %f +/- %f, D: %f +/- %f' %
                 (A_mean, A_err, B_mean, B_err, C_mean, C_err, D_mean, D_err))
    plt.xlabel('Time stamp (s)')
    plt.ylabel('Voltage amplitude coefficients')
    plt.legend()
    plt.grid()
    plt.show()


# determines A, B, C, D for a long data collection by splitting the data into intervals of interval_length seconds
# the function returns the mean and std dev/sqrt(n) of the mean of the amplitudes of A B C and D
def determine_A_B_C_D_func(interval_length, time, channel, f_MOD, t_0, plot):

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
        generate_plot_A_B_C_D(xmean_vec, A_vec, B_vec, C_vec, D_vec)

    return A_mean, A_err, B_mean, B_err, C_mean, C_err, D_mean, D_err
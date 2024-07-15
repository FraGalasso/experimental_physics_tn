import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd


def sin_cos_func(x, A, B, f, x0):
    return A*np.sin(2*np.pi*f*(x-x0))+B*np.cos(2*np.pi*f*(x-x0))


def lin_func(x, m, q):
    return m*x+q


def determine_t0_fmod_function(time, channel3, f_MOD, plot, n_divisions):
    n_samples = len(time)
    n_samples_per_division = int(n_samples/n_divisions)

    xmean_vec = np.zeros(n_divisions)
    phi_vec = np.zeros(n_divisions)

    delta = 0

    for i in range(2):
        f_MOD = f_MOD-delta
        def model(x, A, B): return sin_cos_func(x, A, B, f=f_MOD, x0=0)

        for i in range(n_divisions):
            start_point = int(i*n_samples_per_division)
            end_point = int((i+1)*n_samples_per_division)

            x_new = time[start_point:end_point]
            y_new = channel3[start_point:end_point]

            popt, pcov = curve_fit(model, x_new, y_new)

            xmean_vec[i] = np.mean(x_new)
            phi_vec[i] = np.arctan2(-popt[1], popt[0])  # -B/A

        popt, pcov = curve_fit(lin_func, xmean_vec, phi_vec)
        m = popt[0]
        q = popt[1]
        delta = m/(2*np.pi)

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
                     marker='+', label='data of intervals')
            plt.plot(x_fit, y_fit, label='fit')
            plt.xlabel('t [s]')
            plt.ylabel('phi [radians]')
            plt.legend()
            plt.grid()
            plt.show()

    return t0, f_MOD


'''df = pd.read_csv(
    'final_experiment/forcenoise_DC1A_100mHz15V_105_v.dat', delimiter='\t')

time = df['Time'].to_numpy()
channel3 = df['Channel3'].to_numpy()

determine_t0_fmod_function(time, channel3, 0.1, True, 10)'''
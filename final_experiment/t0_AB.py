import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Define the sinusoidal model function
def sin_cos_func(x, A, B, f, x0):
    return A*np.sin(2*np.pi*f*(x-x0))+B*np.cos(2*np.pi*f*(x-x0))


# Just a linear function
def lin_func(x, m, q):
    return m*x+q

def determine_t0_fmod_function(time, channel3, f_MOD, plot, n_divisions):
    n_samples = len(time)
    n_samples_per_division = int(n_samples/n_divisions)

    xmean_vec = np.zeros(n_divisions)
    phi_vec = np.zeros(n_divisions)

    m = 0
    delta = 0
    error = 0

    # 2 iterations should be sufficient
    while(error <= np.abs(m)):
        # correct f_MOD from previous step
        f_MOD = f_MOD-delta
        def model(x, A, B): return sin_cos_func(x, A, B, f=f_MOD, x0=0)

        # split the  sample in n blocks
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
        m = popt[0] # dphi/dt
        q = popt[1]
        delta = m/(2*np.pi) # correction to the next f_MOD
        error = np.sqrt(pcov[0,0])

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

# Determine A and B for a given time interval
def determine_A_B_in_interval(time, channel, t_0, f_MOD):
    model = lambda time, A, B: sin_cos_func(time, A, B, f=f_MOD, x0=t_0)
    popt, _ = curve_fit(model, time, channel)
    A = popt[0]
    B = popt[1]
    return A, B

# Generate plot of A and B
def generate_plot_A_B(xmean_vec, A_vec, B_vec, A_mean, A_std, B_mean, B_std):
    plt.figure(dpi=120)
    plt.plot(xmean_vec, A_vec, linestyle='None', marker='.', label='$A_{sin}$')
    plt.plot(xmean_vec, B_vec, linestyle='None', marker='.', label='$B_{cos}$')
    plt.title('Values of force amplitudes A and B of sinusoidal model')
    plt.suptitle('A: %f +/- %f, B: %f +/- %f' % (A_mean, A_std, B_mean, B_std))
    plt.xlabel('Interval index')
    plt.ylabel('Force amplitude coefficients')
    plt.legend()
    plt.grid()
    plt.show()

def determine_A_B_func(n_divisions, time, channel, f_MOD, t_0, plot):
    # Generate intervals
    n_samples = len(time)
    n_samples_per_division = int(n_samples / n_divisions)  # Here we are always rounding down, so we will cut out some data points at the end

    # Initialize lists to store A and B values
    A_vec = []
    B_vec = []
    xmean_vec = []

    # Find A, B for each interval and append to vectors A_vec and B_vec
    for i in range(0, n_divisions):
        start_point = int(i * n_samples_per_division)
        end_point = int((i + 1) * n_samples_per_division)

        time_int = time[start_point:end_point]
        channel_int = channel[start_point:end_point]

        A, B = determine_A_B_in_interval(time_int, channel_int, t_0, f_MOD)

        A_vec.append(A)
        B_vec.append(B)
        xmean_vec.append(np.mean(time_int))

    # Calculate mean and std devs and print
    A_mean = np.mean(A_vec)
    A_std = np.std(A_vec)/np.sqrt(n_divisions)
    B_mean = np.mean(B_vec)
    B_std = np.std(B_vec)/np.sqrt(n_divisions)

    print('A: %f +/- %f' % (A_mean, A_std))
    print('B: %f +/- %f' % (B_mean, B_std))

    # Plot the values
    if plot:
        generate_plot_A_B(xmean_vec, A_vec, B_vec, A_mean, A_std, B_mean, B_std)

    return A_mean, A_std, B_mean, B_std


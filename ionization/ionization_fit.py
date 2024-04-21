import numpy as np
import matplotlib.pyplot as plt
import ionization_data as data
from fitting_functions_ionization import odr_linear_fitter
from constants import e_charge, N_A, energy_ion_N_theo, energy_ion_Ar_theo, energy_ion_CO2_theo, energy_ion_He_theo


def ion_fit_N(thres=25):
    v_threshold = thres  # V

    indices = np.flatnonzero(data.V_g_N > v_threshold)
    volts_N = data.V_g_N[indices]
    delta_volts_N = data.delta_V_g_N[indices]
    y = data.ratio_N[indices]
    delta_y = data.delta_ratio_N[indices]

    # np.array with [[a, b], [d_a, d_b]] from y = a + b * x
    result = odr_linear_fitter(
        volts_N, y, delta_volts_N, delta_volts_N, delta_y)

    # numerically equal to ionization energy in eV
    ion_eV_N = -result[0, 0] / result[0, 1]  # -a/b
    # delta_ion_volt = -a/b * sqrt((d_a/a)^2 + (d_b/b)^2)
    delta_ion_eV_N = ion_eV_N * np.sqrt(((result[1, 0]/result[0, 0])**2) +
                                        ((result[1, 1]/result[0, 1])**2))

    print_ion_results(result, ion_eV_N, delta_ion_eV_N)
    plot_ion_results(result, data.V_g_N, data.ratio_N, data.delta_V_g_N, data.delta_ratio_N)


def print_ion_results(result, ion_eV, delta_ion_eV, species='N'):
    print('From linear regression we get:')
    print('a=', result[0, 0], '$\pm$', result[1, 0])
    print('b=', result[0, 1], '$\pm$', result[1, 1])

    print('Ionization energy is given for y=0=a+b*x  --> x=-a/b')
    print('Ionization energy:', ion_eV, '$\pm$', delta_ion_eV, 'eV')
    if species == 'N':
        print('Theorical ionization energy of Nitrogen is:',
              energy_ion_N_theo, 'eV, per single molecule')
    elif species == 'Ar':
        print('Theorical ionization energy of Argon is:',
              energy_ion_Ar_theo/(e_charge*N_A), 'eV, per single particle')
    elif species == 'He':
        print('Theorical ionization energy of Helium is:',
              energy_ion_He_theo/(e_charge*N_A), 'eV, per single particle')
    elif species == 'CO2':
        print('Theorical ionization energy of CO2 is:',
              energy_ion_CO2_theo/(e_charge*N_A), 'eV, per single particle')


def plot_ion_results(result, x, y, dx=None, dy=None, species='N'):
    plt.figure(dpi=200)

    if species == 'N':
        x_model = np.linspace(min(data.V_g_N), max(data.V_g_N), 100)
        plt.title('Nitrogen')
        filename = 'Nitrogen.png'
    elif species == 'Ar':
        x_model = np.linspace(min(data.V_g_Ar), max(data.V_g_Ar), 100)
        plt.title('Argon')
        filename = 'Argon.png'
    elif species == 'He':
        x_model = np.linspace(min(data.V_g_He), max(data.V_g_He), 100)
        plt.title('Helium')
        filename = 'Helium.png'
    elif species == 'CO2':
        x_model = np.linspace(min(data.V_g_CO2), max(data.V_g_CO2), 100)
        plt.title('CO2')
        filename = 'CO2.png'
    else:
        plt.close()
        return

    y_model = result[0, 0] + result[0, 1] * x_model
    plt.plot(x_model, y_model, label='data')
    plt.errorbar(x, y, xerr=dx, yerr=dy, label='data',linestyle='None', marker='.')
    plt.xlabel('$V_g$ [V]')
    plt.ylabel('$I^+/I^-$')
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.show()

    plt.close()

ion_fit_N()

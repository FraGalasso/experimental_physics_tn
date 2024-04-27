import numpy as np
import matplotlib.pyplot as plt
import ionization_data as data
from fitting_functions_ionization import odr_linear_fitter
from constants import e_charge, N_A, energy_ion_N_theo, energy_ion_Ar_theo, energy_ion_CO2_theo, energy_ion_He_theo


def ion_fit(v_min=25, v_max=60, species='N'):
    '''Fits our data. Returns an array with best fit parameters
    on the first row and their standard deviations on the second row.
    We can decide which is data to fit: N for Nitrogen, Ar for Argon,
    He for Helium, CO2 for CO2.'''

    if (v_min < 0) | (v_max < 0) | (v_max < v_min):
        raise ValueError("Invalid voltages")

    if species == 'N':
        indices = np.where((data.V_g_N < v_max) & (data.V_g_N > v_min))
        volts = data.V_g_N[indices]
        delta_volts = data.delta_V_g_N[indices]
        y = data.ratio_N[indices]
        delta_y = data.delta_ratio_N[indices]
    elif species == 'Ar':
        indices = np.where((data.V_g_Ar < v_max) & (data.V_g_Ar > v_min))
        volts = data.V_g_Ar[indices]
        delta_volts = data.delta_V_g_Ar[indices]
        y = data.ratio_Ar[indices]
        delta_y = data.delta_ratio_Ar[indices]
    elif species == 'CO2':
        indices = np.where((data.V_g_CO2 < v_max) & (data.V_g_CO2 > v_min))
        volts = data.V_g_CO2[indices]
        delta_volts = data.delta_V_g_CO2[indices]
        y = data.ratio_CO2[indices]
        delta_y = data.delta_ratio_CO2[indices]
    elif species == 'He':
        indices = np.where((data.V_g_He < v_max) & (data.V_g_He > v_min))
        volts = data.V_g_He[indices]
        delta_volts = data.delta_V_g_He[indices]
        y = data.ratio_He[indices]
        delta_y = data.delta_ratio_He[indices]

    # np.array with [[a, b], [d_a, d_b]] from y = a + b * x
    result = odr_linear_fitter(
        volts, y, delta_volts, delta_volts, delta_y)

    # numerically equal to ionization energy in eV
    ion_eV = -result[0, 0] / result[0, 1]  # -a/b
    # delta_ion_volt = -a/b * sqrt((d_a/a)^2 + (d_b/b)^2)
    delta_ion_eV = ion_eV * np.sqrt(((result[1, 0]/result[0, 0])**2) +
                                    ((result[1, 1]/result[0, 1])**2))

    print_ion_results(result, ion_eV, delta_ion_eV, species)

    if species == 'N':
        plot_ion_results(result, v_min, v_max, data.V_g_N, data.ratio_N,
                         data.delta_V_g_N, data.delta_ratio_N, species)
    elif species == 'Ar':
        plot_ion_results(result, v_min, v_max, data.V_g_Ar, data.ratio_Ar,
                         data.delta_V_g_Ar, data.delta_ratio_Ar, species)
    elif species == 'CO2':
        plot_ion_results(result, v_min, v_max, data.V_g_CO2, data.ratio_CO2,
                         data.delta_V_g_CO2, data.delta_ratio_CO2, species)
    elif species == 'He':
        plot_ion_results(result, v_min, v_max, data.V_g_He, data.ratio_He,
                         data.delta_V_g_He, data.delta_ratio_He, species)

    return result


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


def plot_ion_results(result, v_min, v_max, x, y, dx=None, dy=None, species='N'):
    plt.figure(dpi=200)

    if species == 'N':
        x_model = np.linspace(min(data.V_g_N), max(data.V_g_N), 100)
        plt.title('Nitrogen, $V_{min}$=' + str(v_min) +
                  'V, $V_{max}$=' + str(v_max) + 'V')
        filename = f'Nitrogen_{v_min}_{v_max}.png'
    elif species == 'Ar':
        x_model = np.linspace(min(data.V_g_Ar), max(data.V_g_Ar), 100)
        plt.title('Argon, $V_{min}$=' + str(v_min) +
                  'V, $V_{max}$=' + str(v_max) + 'V')
        filename = f'Argon_{v_min}_{v_max}.png'
    elif species == 'He':
        x_model = np.linspace(min(data.V_g_He), max(data.V_g_He), 100)
        plt.title('Helium, $V_{min}$=' + str(v_min) +
                  'V, $V_{max}$=' + str(v_max) + 'V')
        filename = f'Helium_{v_min}_{v_max}.png'
    elif species == 'CO2':
        x_model = np.linspace(min(data.V_g_CO2), max(data.V_g_CO2), 100)
        plt.title('CO2, $V_{min}$=' + str(v_min) +
                  'V, $V_{max}$=' + str(v_max) + 'V')
        filename = f'CO2_{v_min}_{v_max}.png'
    else:
        plt.close()
        return

    y_model = result[0, 0] + result[0, 1] * x_model
    plt.plot(x_model, y_model,
             label=f'fit, ionization energy = {(-result[0, 0] / result[0, 1]):.3g} eV')
    plt.errorbar(x, y, xerr=dx, yerr=dy, label='data',
                 linestyle='None', marker='.')
    plt.xlabel('$V_g$ [V]')
    plt.ylabel('$I^+/I^-$')
    plt.legend()
    plt.grid()
    # plt.savefig(filename)
    plt.show()

    plt.close()

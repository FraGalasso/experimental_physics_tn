import numpy as np
import matplotlib.pyplot as plt
import constants as cst
import child_data as data
from fitting_functions_child import odr_child_fitter, curve_fit_child_fitter


def child_fit(V_cutoff=13, temp=1, lin=True, use_odr=True):
    '''Fits our data, we can decide which subroutine we want (scipy.odr or scipy.curve_fit)
    and whether we want a linear or a power law fit. Returns an array with best fit parameters
    on the first row and their standard deviations on the second row'''

    if temp not in [1, 2, 3]:
        raise ValueError("temp must be 1, 2, or 3")
    if V_cutoff < 0:
        raise ValueError("Negative V_cutoff")

    voltages = (data.V_g1, data.V_g2, data.V_g3)
    delta_voltages = (data.delta_V_g1, data.delta_V_g2, data.delta_V_g3)
    currents = (data.I_g1, data.I_g2, data.I_g3)
    delta_currents = (data.delta_I_g1, data.delta_I_g2, data.delta_I_g3)

    # selecting elements with v < V_cutoff and their corresponding errors
    indices = np.flatnonzero(voltages[temp-1] < V_cutoff)

    volts = voltages[temp-1][indices]
    delta_volts = delta_voltages[temp-1][indices]
    curr = currents[temp-1][indices]
    delta_curr = delta_currents[temp-1][indices]
    # J = I / (2*pi*rg*lg)
    # consider a current that is perpendicular to the side surface of the cilinder
    curr_dens = curr / (2 * np.pi * data.grid_radius * data.grid_len)
    delta_curr_dens = curr_dens * np.sqrt((delta_curr/curr)**2 + (
        data.delta_grid_radius/data.grid_radius)**2 + (data.delta_grid_len/data.grid_len)**2)

    # not really proud of this, it's a bit convoluted
    if lin:
        log_j = np.log(curr_dens)
        delta_log_j = delta_curr_dens / curr_dens
        log_v = np.log(volts)
        delta_log_v = delta_volts / volts
        # best guess for constant term in linear fit
        bg_a = np.log(4 * cst.eps_0 * np.sqrt(2 * cst.e_charge /
                      cst.e_mass) / (9 * (data.grid_radius**2)))
        if use_odr:
            output = odr_child_fitter(
                log_v, log_j, delta_log_v, delta_log_j, bg_a)
            print_child_result(output, lin)
            plot_child_result(V_cutoff, output, log_v, log_j, delta_log_v,
                              delta_log_j, temp=temp)
        else:
            output = curve_fit_child_fitter(log_v, log_j, delta_log_j, bg_a)
            print_child_result(output, lin)
            plot_child_result(V_cutoff, output, log_v, log_j,
                              delta_y=delta_log_j, temp=temp)
    else:  # lin = False
        # best guess for constant term in linear fit
        bg_a = np.log(4 * cst.eps_0 * np.sqrt(2 * cst.e_charge /
                      cst.e_mass) / (9 * (data.grid_radius**2)))
        if use_odr:
            output = odr_child_fitter(volts, curr_dens, delta_volts,
                                      delta_curr_dens, bg_a, False)
            print_child_result(output, lin)
            plot_child_result(V_cutoff, output, volts, curr_dens, delta_volts,
                              delta_curr_dens, False, temp=temp)
        else:
            output = curve_fit_child_fitter(volts, curr_dens,
                                            delta_curr_dens, bg_a, False)
            print_child_result(output, lin)
            plot_child_result(V_cutoff, output, volts, curr_dens,
                              delta_y=delta_curr_dens, linear_fit=False, temp=temp)

    return output


def print_child_result(output, linear_fit):
    '''Prints results from the fit'''

    ctm_theo = cst.e_charge / cst.e_mass

    if linear_fit:
        print("linear fit:")
        print("log(4/9*eps_0*sqrt(2e/m)*1/rg^2)=",
              output[0, 0], "$\pm$", output[1, 0])
        print("1.5=", output[0, 1], "$\pm$", output[1, 1])
        # charge to mass ratio
        ctm_ratio = 0.5 * np.exp(2*output[0, 0]) * \
            (9*(data.grid_radius**2) / (4*cst.eps_0))**2
    else:
        print("power law fit:")
        print("4/9*eps_0*sqrt(2e/m)*1/rg^2=(",
              output[0, 0], "$\pm$", output[1, 0], ")A / (m^2 V^3/2)")
        print("1.5=", output[0, 1], "$\pm$", output[1, 1])
        # charge to mass ratio
        ctm_ratio = 0.5 * (9*(data.grid_radius**2) *
                           output[0, 0] / (4*cst.eps_0))**2

    print("ctm obtained:", ctm_ratio, "C/kg")
    print("ctm expected:", ctm_theo, "C/kg")
    print("ratio:", ctm_ratio / ctm_theo, '\n')


def plot_child_result(v, output, x_data, y_data, delta_x=None, delta_y=None, linear_fit=True, temp=1):
    '''Plots results from the fit and saves a .png file'''
    x = np.linspace(min(x_data), max(x_data), 100)
    plt.figure(dpi=200)
    if linear_fit:
        y_theo = np.log(4 * cst.eps_0 * np.sqrt(2 * cst.e_charge /
                        cst.e_mass) / (9 * (data.grid_radius**2))) + 1.5 * x
        y_fit = output[0, 0] + x * output[0, 1]
        plt.xlabel('log($V_g$)')
        plt.ylabel('log($J_g$)')
        filename = "plot_T" + str(temp) + "_linear_" + str(v) + ".png"
    else:
        y_theo = 4 * cst.eps_0 * \
            np.sqrt(2 * cst.e_charge / cst.e_mass) / \
            (9 * (data.grid_radius**2))*(x**1.5)
        y_fit = output[0, 0]*(x**output[0, 1])
        plt.xlabel('Grid Voltage $V_g$ [V]')
        plt.ylabel('Grid Density Current $J_g$ [A/m$^2$]')
        filename = "plot_T" + str(temp) + "_power_" + v + ".png"

    fit_label = f'a = {output[0, 0]:.2f}, b = {output[0, 1]:.2f}'

    plt.plot(x, y_theo, label='model')
    plt.plot(x, y_fit, label=fit_label)
    plt.errorbar(x_data, y_data, xerr=delta_x, yerr=delta_y, label='data',
                 linestyle='None', marker='.')
    plt.title(
        '(T$\simeq$' + str(np.ceil(data.T[temp-1]))+' $\pm$ ' + str(np.ceil(data.delta_T[temp-1]))+' Celsius)' + "$V_{cutoff}=$" + str(v) + "V")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.show()

    plt.close()
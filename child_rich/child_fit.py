import numpy as np
import matplotlib.pyplot as plt
import constants as cst
import child_data as data
from fitting_functions_child import child_fitter


def child_fit(V_min=3, V_max=13, temp=1, func='V'):
    '''Fits our data. Returns an array with best fit parameters
    on the first row and their standard deviations on the second row.
    We can decide which is function to use: V for voltage shift, J for
    current shift, F to fix the value 1.5, L for a plain linear fit,
    P for a plain power fit'''

    if temp not in [1, 2, 3]:
        raise ValueError("temp must be 1, 2 or 3")
    if ((V_max < 0) | (V_min < 0)):
        raise ValueError("Invalid voltages")

    voltages = (data.V_g1, data.V_g2, data.V_g3)
    delta_voltages = (data.delta_V_g1, data.delta_V_g2, data.delta_V_g3)
    currents = (data.I_g1, data.I_g2, data.I_g3)
    delta_currents = (data.delta_I_g1, data.delta_I_g2, data.delta_I_g3)

    # selecting elements with v < V_cutoff and their corresponding errors
    indices = np.where((voltages[temp-1] > V_min)
                       & ((voltages[temp-1] < V_max)))

    volts = voltages[temp-1][indices]
    delta_volts = delta_voltages[temp-1][indices]
    curr = currents[temp-1][indices]
    delta_curr = delta_currents[temp-1][indices]
    # J = I / (2*pi*rg*lg)
    # consider a current that is perpendicular to the side surface of the cilinder
    curr_dens = curr / (2 * np.pi * data.grid_radius * data.grid_len)
    delta_curr_dens = curr_dens * np.sqrt((delta_curr/curr)**2 + (
        data.delta_grid_radius/data.grid_radius)**2 + (data.delta_grid_len/data.grid_len)**2)

    if func == 'L':
        # log(J) = a + b * log(V)
        log_j = np.log(curr_dens)
        delta_log_j = delta_curr_dens / curr_dens
        log_v = np.log(volts)
        delta_log_v = delta_volts / volts
        bg_a = np.log(4 * cst.eps_0 * np.sqrt(2 * cst.e_charge /
                      cst.e_mass) / (9 * (data.grid_radius**2)))
        beta = [bg_a, 1.5]

        output = child_fitter(log_v, log_j, delta_log_v, delta_log_j,
                              beta, func)
        print_child_result(output, func)
        plot_child_result(V_min, V_max, output, log_v, log_j, delta_log_v,
                          delta_log_j, func, temp)
    elif func == 'P':
        # J = a * (v ^ b)
        bg_a = 4 * cst.eps_0 * np.sqrt(2 * cst.e_charge /
                                       cst.e_mass) / (9 * (data.grid_radius**2))
        beta = [bg_a, 1.5]

        output = child_fitter(volts, curr_dens, delta_volts, delta_curr_dens,
                              beta, func)
        print_child_result(output, func)
        plot_child_result(V_min, V_max, output, volts, curr_dens,
                          delta_volts, delta_curr_dens, func, temp)
    elif func == 'F':
        # log(J) = a + 1.5 log(V + c)
        log_j = np.log(curr_dens)
        delta_log_j = delta_curr_dens / curr_dens
        bg_a = np.log(4 * cst.eps_0 * np.sqrt(2 * cst.e_charge /
                      cst.e_mass) / (9 * (data.grid_radius**2)))
        beta = [bg_a, 0]

        output = child_fitter(volts, log_j, delta_volts, delta_log_j,
                              beta, func)
        print_child_result(output, func)
        plot_child_result(V_min, V_max, output, volts, log_j,
                          delta_volts, delta_log_j, func, temp=temp)
    elif func == 'V':
        # log(J) = a + b * log(V + c)
        log_j = np.log(curr_dens)
        delta_log_j = delta_curr_dens / curr_dens
        bg_a = np.log(4 * cst.eps_0 * np.sqrt(2 * cst.e_charge /
                      cst.e_mass) / (9 * (data.grid_radius**2)))
        beta = [bg_a, 1.5, 0]

        output = child_fitter(volts, log_j, delta_volts, delta_log_j,
                              beta, func)
        print_child_result(output, func)
        plot_child_result(V_min, V_max, output, volts, log_j,
                          delta_volts, delta_log_j, func, temp=temp)
    elif func == 'J':
        # J = a * (v ^ b) + c
        bg_a = 4 * cst.eps_0 * np.sqrt(2 * cst.e_charge /
                                       cst.e_mass) / (9 * (data.grid_radius**2))
        beta = [bg_a, 1.5, 0]

        output = child_fitter(volts, curr_dens, delta_volts, delta_curr_dens,
                              beta, func)
        print_child_result(output, func)
        plot_child_result(V_min, V_max, output, volts, curr_dens,
                          delta_volts, delta_curr_dens, func, temp=temp)

    return output


def print_child_result(output, func):
    '''Prints results from the fit'''

    ctm_theo = cst.e_charge / cst.e_mass

    if func == 'L':
        print("Linear fit: log(J) = a + b * log(V)")
        print("log(4/9*eps_0*sqrt(2e/m)*1/rg^2) =",
              output[0, 0], "$\pm$", output[1, 0])
        print("1.5 =", output[0, 1], "$\pm$", output[1, 1])
        # charge to mass ratio
        ctm_ratio = 0.5 * np.exp(2*output[0, 0]) * \
            (9*(data.grid_radius**2) / (4*cst.eps_0))**2
    elif func == 'P':
        print("Power law fit: J = a * (V ^ b)")
        print("4/9*eps_0*sqrt(2e/m)*1/rg^2 = (",
              output[0, 0], "$\pm$", output[1, 0], ") A / (m^2 V^3/2)")
        print("1.5 =", output[0, 1], "$\pm$", output[1, 1])
        # charge to mass ratio
        ctm_ratio = 0.5 * (9*(data.grid_radius**2) *
                           output[0, 0] / (4*cst.eps_0))**2
    elif func == 'F':
        print("Linear fit, forcing 1.5 coefficient and adding a voltage shift:")
        print("log(J) = a + 1.5 log(V + c)")
        print("log(4/9*eps_0*sqrt(2e/m)*1/rg^2) =",
              output[0, 0], "$\pm$", output[1, 0])
        print("c = (", output[0, 1], "$\pm$", output[1, 1], ") V")
        # charge to mass ratio
        ctm_ratio = 0.5 * np.exp(2*output[0, 0]) * \
            (9*(data.grid_radius**2) / (4*cst.eps_0))**2
    elif func == 'V':
        print("Fit of logaritm, adding a voltage shift:")
        print("log(J) = a + b * log(V + c)")
        print("log(4/9*eps_0*sqrt(2e/m)*1/rg^2) =",
              output[0, 0], "$\pm$", output[1, 0])
        print("1.5 =", output[0, 1], "$\pm$", output[1, 1])
        print("V_off = (", output[0, 2], "$\pm$", output[1, 2], ") V")
        # charge to mass ratio
        ctm_ratio = 0.5 * np.exp(2*output[0, 0]) * \
            (9*(data.grid_radius**2) / (4*cst.eps_0))**2
    elif func == 'J':
        print("Fit adding a current shift:")
        print("J = a * (v ^ b) + c")
        print("4/9*eps_0*sqrt(2e/m)*1/rg^2 = (",
              output[0, 0], "$\pm$", output[1, 0], ") A / (m^2 V^3/2)")
        print("1.5 =", output[0, 1], "$\pm$", output[1, 1])
        print("J_off = (", output[0, 2], "$\pm$", output[1, 2], ") V")
        # charge to mass ratio
        ctm_ratio = 0.5 * np.exp(2*output[0, 0]) * \
            (9*(data.grid_radius**2) / (4*cst.eps_0))**2

    print("ctm obtained:", ctm_ratio, "C/kg")
    print("ctm expected:", ctm_theo, "C/kg")
    print("ratio:", ctm_ratio / ctm_theo, '\n')


def plot_child_result(v_min, v_max, output, x_data, y_data, delta_x, delta_y, func, temp):
    '''Plots results from the fit and saves a .png file'''
    x = np.linspace(min(x_data), max(x_data), 100)
    plt.figure(dpi=200)
    if func == 'L':
        # log_j = a + b * log_v
        y_theo = np.log(4 * cst.eps_0 * np.sqrt(2 * cst.e_charge /
                        cst.e_mass) / (9 * (data.grid_radius**2))) + 1.5 * x
        y_fit = output[0, 0] + x * output[0, 1]
        plt.xlabel('log($V_g$)')
        plt.ylabel('log($J_g$)')
        filename = f'plot_T{temp}_linear_{v_min}_{v_max}.png'
        fit_label = f'a = {output[0, 0]:.2g}, b = {output[0, 1]:.2g}'
    elif func == 'P':
        # J = a * (v ^ b)
        y_theo = 4 * cst.eps_0 * \
            np.sqrt(2 * cst.e_charge / cst.e_mass) / \
            (9 * (data.grid_radius**2))*(x**1.5)
        y_fit = output[0, 0]*(x**output[0, 1])
        plt.xlabel('Grid Voltage $V_g$ [V]')
        plt.ylabel('Grid Density Current $J_g$ [A/m$^2$]')
        filename = f'plot_T{temp}_power_{v_min}_{v_max}.png'
        fit_label = f'a = {output[0, 0]:.2g}, b = {output[0, 1]:.2g}'
    elif func == 'F':
        # log(J) = a + 1.5 log(V + c)
        y_theo = np.log(4 * cst.eps_0 * np.sqrt(2 * cst.e_charge /
                        cst.e_mass) / (9 * (data.grid_radius**2))) + 1.5 * np.log(x)
        y_fit = output[0, 0] + 1.5 * np.log(x + output[0, 1])
        plt.xlabel('Grid Voltage $V_g$ [V]')
        plt.ylabel('log($J_g$)')
        filename = f'plot_T{temp}_forced_{v_min}_{v_max}.png'
        fit_label = f'a = {output[0, 0]:.2g}, c = {output[0, 1]:.2g}'
    elif func == 'V':
        # log(J) = a + b * log(V + c)
        y_theo = np.log(4 * cst.eps_0 * np.sqrt(2 * cst.e_charge /
                        cst.e_mass) / (9 * (data.grid_radius**2))) + 1.5 * np.log(x)
        y_fit = output[0, 0] + output[0, 1] * np.log(x + output[0, 2])
        plt.xlabel('Grid Voltage $V_g$ [V]')
        plt.ylabel('log($J_g$)')
        filename = f'plot_T{temp}_volt_{v_min}_{v_max}.png'
        fit_label = f'a = {output[0, 0]:.2g}, b = {output[0, 1]:.2g}, c = {output[0, 2]:.2f}'
    elif func == 'J':
        # J = a * (v ^ b) + c
        y_theo = 4 * cst.eps_0 * \
            np.sqrt(2 * cst.e_charge / cst.e_mass) / \
            (9 * (data.grid_radius**2))*(x**1.5)
        y_fit = output[0, 0] * (x**output[0, 1]) + output[0, 2]
        plt.xlabel('Grid Voltage $V_g$ [V]')
        plt.ylabel('Grid Density Current $J_g$ [A/m$^2$]')
        filename = f'plot_T{temp}_curr_{v_min}_{v_max}.png'
        fit_label = f'a = {output[0, 0]:.2g}, b = {output[0, 1]:.2g}, c = {output[0, 2]:.2f}'

    # plt.plot(x, y_theo, label='model')
    plt.plot(x, y_fit, label=fit_label)
    plt.errorbar(x_data, y_data, xerr=delta_x, yerr=delta_y, label='data',
                 linestyle='None', marker='.')
    plt.title(
        '(T$\simeq$' + str(np.ceil(data.T[temp-1])) +
        ' $\pm$ ' + str(np.ceil(data.delta_T[temp-1]))
        + ' Celsius), ' + "$V_{min}=$" + str(v_min) + "V, " + "$V_{max}=$" + str(v_max) + "V")
    plt.legend()
    plt.grid()
    # plt.savefig(filename)
    plt.show()

    plt.close()

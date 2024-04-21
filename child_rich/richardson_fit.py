import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import odr
import constants as cst
import richardson_data as data


def richardson_log(pp, x):
    return np.log(pp[0]) + (2 * np.log(x)) - (pp[1] * cst.e_charge / (cst.KB * x))


def richardson_func(pp, x):
    return pp[0] * (x**2) * np.exp(-pp[1] * cst.e_charge / (cst.KB * x))


def richardson_fit(log=False):
    '''Fits our data with scipy.odr, we can decide which whether we want
     a to fit Richardson law or its log. Returns an array with best fit
     parameters on the first row (work function is in eV) and their standard
     deviations on the second row'''

    # different data and functions depending on what we want
    if log:
        log_j = np.log(data.j_g)
        delta_log_j = data.delta_j_g / data.j_g

        data_to_fit = odr.RealData(
            data.T_kelv, log_j, data.delta_T_kelv, delta_log_j)
        model = odr.Model(richardson_log)
    else:
        data_to_fit = odr.RealData(
            data.T_kelv, data.j_g, data.delta_T_kelv, data.delta_j_g)
        model = odr.Model(richardson_func)

    # best guesses
    bg_A = 6e5  # A / (m^2 * k^2)
    bg_W = 4.5  # eV

    myodr = odr.ODR(data_to_fit, model, beta0=[bg_A, bg_W])
    myodr.set_job(fit_type=0)
    output = myodr.run()
    result = np.array([output.beta, output.sd_beta])

    print_richardson_result(result, log)
    plot_richardson_result(result, log)
    plot_richardson_residuals(result, log)

    return result


def print_richardson_result(result, log=False):
    '''Prints results from the fit'''

    if log:
        print("\nFit of log(J)=log(A)+2log(T)-W/(kB*T):")
        print("A=(", result[0, 0], "$\pm$", result[1, 0], ") A/(m^2 K^2)")
        print("W=(", result[0, 1], "$\pm$", result[1, 1], ") eV")
    else:
        print("\nFit of J=A*(T^2)*exp(-W/(kB*T)):")
        print("A=(", result[0, 0], "$\pm$", result[1, 0], ") A/(m^2 K^2)")
        print("W=(", result[0, 1], "$\pm$", result[1, 1], ") eV")


def plot_richardson_result(result, log=False):
    '''Plots results from the fit and saves a .png file'''

    # Commented lines plot the graph of our best guess (i.e.the theoretical prediction).
    # I don't recommend including them because the plot of theoretical prediction has
    # much bigger values on the y axis. This might be due to our overestimation of W.

    x = np.linspace(min(data.T_kelv), max(data.T_kelv), 100)
    plt.figure(dpi=200)

    # bg_A = 6e5
    # bg_W = 4.5

    if log:
        # y_theo = richardson_log([bg_A, bg_W], x)
        y_fit = richardson_log(result[0], x)
        plt.xlabel('T')
        plt.ylabel('log($J_g$)')
        filename = "plot_rich_log.png"
        plt.errorbar(data.T_kelv, np.log(data.j_g), xerr=data.delta_T_kelv, yerr=(
            data.delta_j_g / data.j_g), label='data', linestyle='None', marker='.')
        plt.title("$log(J_g)=log(A_g)+2log(T)-\\frac{W}{k_B T}$")
    else:
        # y_theo = richardson_func([bg_A, bg_W], x)
        y_fit = richardson_func(result[0], x)
        plt.xlabel('T')
        plt.ylabel('$J_g$')
        filename = "plot_rich.png"
        plt.errorbar(data.T_kelv, data.j_g, xerr=data.delta_T_kelv,
                     yerr=data.delta_j_g, label='data', linestyle='None', marker='.')
        plt.title("$J_g=A_gT^2exp(\\frac{W}{k_B T})$")

    # plt.plot(x, y_theo, label='model')
    plt.plot(x, y_fit, label='fit')
    plt.legend()
    plt.grid()

    plt.savefig(filename)
    plt.show()
    plt.close()


def plot_richardson_residuals(result, log=False):
    '''Plots residuals from the fit and saves a .png file'''

    plt.figure(dpi=200)

    if log:
        plt.xlabel('T')
        plt.ylabel('res of log($J_g$)')
        filename = "plot_rich_log_residuals.png"

        # residuals
        res = np.log(data.j_g) - richardson_log(result[0], data.T_kelv)

        # uncertainty on richardson_log(result[0], data.T_kelv)
        delta_est = np.sqrt((2 - result[0, 1] * cst.e_charge / (cst.KB*data.T_kelv))**2 *
                            (data.delta_T_kelv / data.T_kelv)**2 +
                            (result[1, 1] * cst.e_charge / (cst.KB*data.T_kelv))**2 +
                            (result[1, 0] / result[0, 0])**2)
        '''delta_est = sqrt((2 - W / (kB * T))^2 * (delta_T / T)^2 + 
                            (delta_W / (kB * T))^2 + (delta_A / A)^2)'''

        # uncertainty on residuals
        delta_res = np.sqrt(delta_est**2 + (data.delta_j_g/data.j_g)**2)

        plt.errorbar(data.T_kelv, res, xerr=data.delta_T_kelv,
                     yerr=delta_res, label='data', linestyle='None', marker='.')
        plt.title("Residuals of $log(J_g)=log(A_g)+2log(T)-\\frac{W}{k_B T}$")
    else:
        plt.xlabel('T')
        plt.ylabel('res of $J_g$')
        filename = "plot_rich_residuals.png"

        # residuals
        res = data.j_g - richardson_func(result[0], data.T_kelv)

        # uncertainty on richardson_func(result[0], data.T_kelv)
        delta_est = data.j_g * np.sqrt((2 + result[0, 1] * cst.e_charge / (cst.KB*data.T_kelv))**2 *
                                       (data.delta_T_kelv / data.T_kelv)**2 +
                                       (result[1, 1] * cst.e_charge / (cst.KB*data.T_kelv))**2 +
                                       (result[1, 0] / result[0, 0])**2)
        '''delta_est = J_g * sqrt((2 + W / (kB * T))^2 * (delta_T / T)^2 + 
                                  (delta_W / (kB * T))^2 + (delta_A / A)^2)'''

        # uncertainty on residuals
        delta_res = np.sqrt(delta_est**2 + data.delta_j_g**2)

        plt.errorbar(data.T_kelv, res, xerr=data.delta_T_kelv,
                     yerr=delta_res, label='data', linestyle='None', marker='.')
        plt.title("Residuals of $J_g=A_gT^2exp(\\frac{W}{k_B T})$")

    plt.grid()
    plt.savefig(filename)
    plt.show()
    plt.close()
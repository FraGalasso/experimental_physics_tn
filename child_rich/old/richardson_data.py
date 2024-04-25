import numpy as np
import matplotlib.pyplot as plt
from child_data import r_rest, r0, delta_r_rest, delta_r0, T_0, delta_T_0
from child_data import grid_radius, delta_grid_radius, grid_len, delta_grid_len
from constants import alpha

# accuracy on the programmed current is 0.1% of output programmed current +5mA
amp_system = np.array([4.105, 4.005, 3.905, 3.705, 3.605,
                      3.504, 4.155, 3.805, 4.115, 4.125, 3.755, 3.655, 3.554])  # A
delta_amp_system = (0.1/100)*amp_system+(5*10**(-3))  # A

# accuracy on the read voltage is 0.05% of output programmed voltage +10mV
volt_system = np.array([3.214, 3.078, 2.947, 2.708, 2.595,
                       2.467, 3.27, 2.83, 3.211, 3.228, 2.783, 2.641, 2.538])  # V
delta_volt_system = (0.05/100) * volt_system+(10*10**(-3))  # V

res_system = volt_system / amp_system  # Ohm
delta_res_system = res_system * np.sqrt((delta_volt_system / volt_system) **
                                        2 + (delta_amp_system / amp_system)**2)  # Ohm

r_f = res_system - r_rest  # Ohm
delta_rf = np.sqrt((delta_res_system ** 2) + (delta_r_rest ** 2))  # Ohm

T_cels = T_0 + (1 / alpha) * ((r_f / r0) - 1)  # Celsius
T_kelv = T_cels + 273.15  # Kelvin
delta_T_kelv = np.sqrt((delta_T_0 ** 2) + ((delta_rf / (alpha * r0)) ** 2)
                       + ((r_f * delta_r0 / (alpha * r0 ** 2)) ** 2))

# accuracy is equal to 0.05% of reading + 0.02% of range
# in case of our measurement the minimum range is 10 mA
i_g = np.array([1.18800, 0.70000, 0.37810, 0.085, 0.0383, 0.0168, 2.37, 0.203,
                1.876, 2.0045, 0.1395, 0.0613, 0.0275])*10**(-3)  # A
delta_i_g = (0.05/100) * i_g + (0.02/100) * (10*10**(-3))  # A

j_g = i_g/(2 * np.pi * grid_len * grid_radius)
delta_j_g = j_g * np.sqrt((delta_i_g/i_g)**2 + (delta_grid_radius /
                          grid_radius)**2 + (delta_grid_len/grid_len)**2)


def plot_richardson_data():
    plt.figure(dpi=200)
    plt.errorbar(T_kelv, j_g, xerr=delta_T_kelv, yerr=delta_j_g,
                 marker='.', linestyle='None', label='data')
    plt.xlabel('Tungsten filament temperature $T_f$ [K]')
    plt.ylabel('Grid current density $J_g$ [A/m^2]')
    plt.legend()
    plt.grid()
    plt.savefig("data_richardson.png")
    plt.show()
    plt.close()

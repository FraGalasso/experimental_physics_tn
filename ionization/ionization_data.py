import numpy as np
import matplotlib.pyplot as plt
from select_sigma_ion_currents import select_sigma_ion_currents

# Nitrogen


i_f = 4.106  # A
v_f = 3.195  # V

v_collector = 4.093  # V

# accuracy is equal to 0.05%output voltage + 100mV offset
V_g_N = np.array([10.03, 11.03, 12.02, 13.03, 14.03, 15.02, 16.03, 17.03, 18.04, 19.03, 20.03, 21.04, 22.04, 23.04, 24.04, 25.05, 26.04,
                 27.05, 28.05, 29.04, 30.05, 31.05, 32.05, 33.04, 34.04, 35.05, 36.04, 37.05, 38.06, 39.05, 40.05, 41.05, 42.06, 45.05, 50.05])
delta_V_g_N = (0.05/100) * V_g_N + (100*10**(-3))  # V

# accuracy is equal to 0.05% of reading + 0.02% of range
# in case of our measurement the minimum range is 10 mA
I_minus_N = np.array([0.05869, 0.02646, 0.64, 0.6504, 0.6575, 0.663, 0.6688, 0.673, 0.677, 0.6818, 0.686, 0.6904, 0.694, 0.6906, 0.6935, 0.6963, 0.7028,
                     0.7025, 0.7086, 0.7124, 0.7152, 0.7176, 0.7201, 0.7229, 0.7254, 0.7278, 0.7323, 0.7345, 0.7369, 0.7396, 0.7427, 0.7452, 0.7506, 0.7566, 0.7636])*10**(-3)
delta_I_minus_N = (0.05/100) * I_minus_N + (0.02/100) * (10*10**(-3))  # A


I_plus_N = np.array([0.08, 0.17, 0.065, 0.1086, 0.15518, 0.265, 0.46, 0.74, 1.34, 6.7, 9.6, 13.22, 17.23, 22.51, 27.69, 33.32, 39.38, 45.64,
                    52.4, 59.45, 66.87, 74.61, 82.52, 90.77, 99.06, 106.5, 116.08, 124.65, 133.27, 141.92, 150.75, 159.58, 168.75, 194.99, 238.16])*10**(-9)
# errors on the ionization current given depends on used range
# autoscale option will automatically change the range every
# time the input current is greater than 1.05 times the current range
delta_I_plus_N = select_sigma_ion_currents(I_plus_N)

# ratio of i+ and i-
ratio_N = I_plus_N / I_minus_N
delta_ratio_N = ratio_N * np.sqrt((delta_I_plus_N/I_plus_N)**2
                                  + (delta_I_minus_N/I_minus_N)**2)


# Argon
# gas correction factor = 0.7
# Argon pressure = 0.7 * read pressure (equivalent pressure for Nitrogen)

# accuracy is equal to 0.05%output voltage + 100mV offset
V_g_Ar = np.array([10.03, 11.03, 12.02, 13.03, 14.03, 15.02, 16.03, 17.03, 18.04, 19.03, 20.03, 21.04, 22.04, 23.04, 24.04,
                  25.05, 26.04, 27.05, 28.05, 29.04, 30.05, 31.05, 32.05, 33.04, 34.04, 35.05, 36.04, 38.06, 40.05, 45.05, 50.05])  # V
delta_V_g_Ar = (0.05/100) * V_g_Ar + (100*10**(-3))  # V

# accuracy is equal to 0.05% of reading + 0.02% of range
# in case of our measurement the minimum range is 10 mA
I_minus_Ar = np.array([0.6685, 0.7385, 0.8072, 0.9218, 0.997, 1.07, 1.146, 1.211, 1.2738, 1.334, 1.3757, 1.4037, 1.424, 1.441, 1.456,
                      1.469, 1.481, 1.493, 1.507, 1.519, 1.5291, 1.538, 1.548, 1.57, 1.579, 1.5909, 1.5998, 1.63, 1.645, 1.67, 1.692])*10**(-3)  # A
delta_I_minus_Ar = (0.05/100) * I_minus_Ar + (0.02/100) * (10*10**(-3))  # A

I_plus_Ar = np.array([0.0036, 0.0085, 0.016, 0.032, 0.0933, 0.21, 0.45, 0.9672, 6.266, 10.76, 18.23, 27.78, 39.16, 52.32, 66.71, 82.18,
                     98.56, 115.53, 133.31, 151.19, 169.53, 188.137, 208.13, 228.23, 248.62, 269.45, 290, 334.09, 376.94, 477.52, 570.29])*10**(-9)  # A
# errors on the ionization current given depends on used range
# autoscale option will automatically change the range every
# time the input current is greater than 1.05 times the current range
delta_I_plus_Ar = select_sigma_ion_currents(I_plus_Ar)

ratio_Ar = I_plus_Ar / I_minus_Ar
delta_ratio_Ar = ratio_Ar * np.sqrt((delta_I_plus_Ar/I_plus_Ar)**2
                                    + (delta_I_minus_Ar/I_minus_Ar)**2)


# Helium

# accuracy is equal to 0.05%output voltage + 100mV offset
V_g_He = np.array([10.03, 11.03, 12.02, 13.03, 14.03, 15.02, 16.03, 17.03, 18.04, 19.03, 20.03, 21.04, 22.04, 23.04,
                  24.04, 25.05, 26.04, 27.05, 28.05, 29.04, 30.05, 31.05, 32.05, 33.04, 34.04, 35.05, 37.05, 40.05, 45.05, 50.05])  # V
delta_V_g_He = (0.05/100) * V_g_He + (100*10**(-3))  # V

# accuracy is equal to 0.05% of reading + 0.02% of range
# in case of our measurement the minimum range is 10 mA
I_minus_He = np.array([0.6916, 0.764, 0.836, 0.908, 0.976, 1.41, 1.103, 1.156, 1.189, 1.212, 1.227, 1.24, 1.25, 1.253, 1.262,
                      1.271, 1.279, 1.282, 1.293, 1.295, 1.302, 1.308, 1.314, 1.316, 1.317, 1.319, 1.331, 1.343, 1.341, 1.358])*10**(-3)  # A
delta_I_minus_He = (0.05/100) * I_minus_He + (0.02/100) * (10*10**(-3))  # A

I_plus_He = np.array([0.016, 0.069, 0.141, 0.073, 0.237, 0.234, 0.48, 0.976, 6.05, 10.37, 16.68, 24.69, 34.63, 45.17, 56.54,
                     68.64, 81.11, 94.3, 107.38, 122.16, 136.34, 151.02, 165.72, 180.5, 196.05, 213.94, 245.16, 292.16, 368.4, 437.4])*10**(-9)  # A
# errors on the ionization current given depends on used range
# autoscale option will automatically change the range every
# time the input current is greater than 1.05 times the current range
delta_I_plus_He = select_sigma_ion_currents(I_plus_He)

ratio_He = I_plus_He / I_minus_He
delta_ratio_He = ratio_He * np.sqrt((delta_I_plus_He/I_plus_He)**2
                                    + (delta_I_minus_He/I_minus_He)**2)

# C02

V_g_CO2 = np.array([10.03, 11.03, 12.02, 13.03, 14.03, 15.02, 16.03, 17.03, 18.04, 19.03, 20.03, 21.04, 22.04, 23.04, 24.04,
                   25.05, 26.04, 27.05, 28.05, 29.04, 30.05, 31.05, 32.05, 33.04, 34.04, 35.05, 36.04, 37.05, 40.05, 45.05, 50.05])  # V
delta_V_g_CO2 = (0.05/100) * V_g_CO2 + (100*10**(-3))  # V

I_minus_CO2 = np.array([0.641, 0.6964, 0.74, 0.759, 0.777, 0.7845, 0.7893, 0.791, 0.794, 0.7971, 0.794, 0.796, 0.798, 0.8, 0.808,
                       0.813, 0.815, 0.818, 0.82, 0.821, 0.823, 0.825, 0.826, 0.827, 0.828, 0.829, 0.831, 0.832, 0.836, 0.843, 0.8499])*10**(-3)  # A
delta_I_minus_CO2 = (0.05/100) * I_minus_CO2 + (0.02/100) * (10*10**(-3))  # A

I_plus_CO2 = np.array([0.019, 0.053, 0.012, 0.0495, 0.1617, 0.418, 1.04, 4.77, 7.86, 11.65, 16.18, 21.15, 26.56, 32.52, 38.16, 45.08,
                      52.44, 60.47, 69.05, 77.9, 87.27, 96.84, 106.51, 116.27, 126.95, 135.82, 145.46, 155.26, 184.44, 232.28, 279.67])*10**(-9)  # A
delta_I_plus_CO2 = select_sigma_ion_currents(I_plus_CO2)

ratio_CO2 = I_plus_CO2 / I_minus_CO2
delta_ratio_CO2 = ratio_CO2 * np.sqrt((delta_I_plus_CO2/I_plus_CO2)**2
                                      + (delta_I_minus_CO2/I_minus_CO2)**2)

def plot_ionization_data():
    plt.figure(dpi=200)
    plt.errorbar(V_g_N, ratio_N, xerr=delta_V_g_N, yerr= delta_ratio_N, linestyle='-.', label='Nitrogen', marker='.')
    plt.errorbar(V_g_Ar, ratio_Ar, xerr=delta_V_g_Ar, yerr= delta_ratio_Ar, linestyle='-.', label='Argon', marker='.')
    plt.errorbar(V_g_He, ratio_He, xerr=delta_V_g_He, yerr= delta_ratio_He, linestyle='-.', label='Helium', marker='.')
    plt.errorbar(V_g_CO2, ratio_CO2, xerr=delta_V_g_CO2, yerr= delta_ratio_CO2, linestyle='-.', label='CO2', marker='.')

    plt.xlabel('$V_g$ [V]')
    plt.ylabel('$I^+/I^-$')
    plt.legend()
    plt.grid()
    plt.savefig("data_ionization.png")
    plt.show()
    plt.close()

import numpy as np


def select_sigma_ion_currents(current):
    sigma_curr = np.ones(np.size(current))

    # ranges of multimeter
    Keithley_amp_range1 = 20 * 10**(-12)    # A
    Keithley_amp_range2 = 200 * 10**(-12)   # A
    Keithley_amp_range3 = 2 * 10**(-9)      # A
    Keithley_amp_range4 = 20 * 10**(-9)     # A
    Keithley_amp_range5 = 200 * 10**(-9)    # A
    # Keithley_amp_range6 = 2 * 10**(-6)      # A

    # selcting the correct ranges
    for i in range(np.size(sigma_curr)):

        # accuracy is given in %rdg + counts
        if current[i] < 1.05 * Keithley_amp_range1:
            sigma_curr[i] = (1/100)*current[i] + (30*10**(-16))

        elif current[i] < 1.05 * Keithley_amp_range2:
            sigma_curr[i] = (1/100)*current[i] + (5*10**(-15))

        elif current[i] < 1.05 * Keithley_amp_range3:
            sigma_curr[i] = (0.2/100)*current[i] + (30*10**(-14))

        elif current[i] < 1.05 * Keithley_amp_range4:
            sigma_curr[i] = (0.2/100)*current[i] + (5*10**(-13))

        elif current[i] < 1.05*Keithley_amp_range5:
            sigma_curr[i] = (0.2/100)*current[i] + (5*10**(-12))

        else:
            sigma_curr[i] = (0.1/100)*current[i] + (10*10**(-11))

    return sigma_curr

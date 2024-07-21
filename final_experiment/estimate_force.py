import numpy as np
import matplotlib.pyplot as plt

L = 0.00434     # H
M = 0.00208     # H
R_l = 4.51      # Ohm
R_gen = 50      # Ohm


def C_s(w):
    fact_1 = 1 - (1j*w * (M**2)/L) / (1j*w*L + R_l)
    return 1 / (R_gen + R_l + 1j*w*L * fact_1)


def C_r(w):
    fact = -(1j*w*M) / (1j*w*L + R_l)
    return fact * C_s(w)


def ac_force_estimate():
    '''Computes an estimate of the A coefficient as a function of the frequency
    in the range [80, 5000] Hz and returns a np.array. D can be obtained dividing
    by -4. Constants have to be manually tuned inside the function definition.
    Notice: to match the sign of our data, this function doesn't output A, but rather
    -A.'''
    N_windings = 84
    D_B = 0.235     # m
    d = 0.0165      # m
    lambda_factor = 0.9
    V0 = 7.5
    beta = (0.2e-6) * N_windings**2 * np.pi * D_B / d * lambda_factor

    f_AC = np.arange(80, 5000, 1)
    w_AC = 2 * np.pi * f_AC
    amp_Cs_AC = np.abs(C_s(w_AC))
    amp_Cr_AC = np.abs(C_r(w_AC))
    phase_shift = 0  # np.pi
    phase_Cs_Cr_AC = (np.angle(C_s(w_AC))-np.angle(C_r(w_AC))) + phase_shift

    return -beta * ((V0**2)/4) * amp_Cs_AC * amp_Cr_AC * np.cos(phase_Cs_Cr_AC)

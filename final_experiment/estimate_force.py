import numpy as np
import matplotlib.pyplot as plt

L = 0.00434     # H
M = 0.00208     # H
R_l = 4.516      # Ohm
R_gen = 50      # Ohm


def C_s(w):
    fact_1 = 1 - (1j*w * (M**2)/L) / (1j*w*L + R_l)
    return 1 / (R_gen + R_l + 1j*w*L * fact_1)


def C_s_sheet(w, l, r, k):
    k_sheet = k
    L_sheet = l
    M_sheet = k_sheet*np.sqrt(L*L_sheet)
    R_sheet = r
    return 1 / ((1j * w * L + R_l + R_gen) +
                ((w*M_sheet)**2/(1j*w*L_sheet + R_sheet)))


def C_r(w):
    fact = -(1j*w*M) / (1j*w*L + R_l)
    return fact * C_s(w)


def C_r_sheet(w, l, r, k):
    k_sheet = k
    L_sheet = l
    M_sheet = k_sheet*np.sqrt(L*L_sheet)
    R_sheet = r
    fact = (-1j * w * M_sheet)/(1j*w*L_sheet + R_sheet)
    return fact * C_s(w)


def ac_force_estimate():
    '''Computes an estimate of the A coefficient as a function of the frequency
    in the range [80, 5000] Hz and returns a np.array. D can be obtained dividing
    by -4. Constants have to be manually tuned inside the function definition.
    Notice: to match the sign of our data, this function doesn't output A, but rather
    -A.'''
    f_AC = np.arange(80, 5000, 1)
    w_AC = 2 * np.pi * f_AC
    amp_Cs_AC = np.abs(C_s(w_AC))
    amp_Cr_AC = np.abs(C_r(w_AC))
    N_windings = 84
    D_B = 0.235     # m
    d_ = 0.0195      # m
    lam_ = 0.951
    '''d_d = 0.00005
    d_lam = 0.005'''

    '''d_L = 0.00008
    d_M = 0.00003'''

    '''first_csl = 2 * (R_gen+(1+((w_AC**2)*(M**2))/((w_AC**2)*(L**2)+R_l**2))
                     * R_l) * (w_AC**2)*(M**2)/(((w_AC**2)*(L**2)+R_l**2)**2) * 2 * R_l*L*w_AC**2
    second_csl = 2 * w_AC**2 * L * \
        (1 - ((w_AC**2)*(M**2)) / ((w_AC**2)*(L**2)+R_l**2)) ** 2
    third_csl = (w_AC**2 * L * M)**2 / (((w_AC**2)*(L**2)+R_l**2)**2) * 2 * L
    dCs_dl = 0.5*(amp_Cs_AC)**3*(first_csl+second_csl+third_csl)

    first_csm = 2 * (R_gen+(1+((w_AC**2)*(M**2))/((w_AC**2)*(L**2)+R_l**2))
                     * R_l) * R_l*2 * (w_AC**2 * L*M)**2 / ((w_AC**2)*(L**2)+R_l**2)**2
    second_csm = 2*(w_AC**4)*L**2*M * \
        (1 - ((w_AC**2)*(M**2)) / ((w_AC**2)*(L**2)+R_l**2))
    dCs_dm = 0.5*(amp_Cs_AC)**3*(first_csm+second_csm)

    delta_amp_cs = np.sqrt((d_M * dCs_dm)**2 + (d_L * dCs_dl)**2)

    dCr_dCs = np.sqrt((w_AC**2)*(M**2) / ((w_AC**2)*(L**2)+R_l**2))
    dCr_dl = amp_Cs_AC**2/amp_Cr_AC * \
        (w_AC**4)*(M**2)*L / ((w_AC**2)*(L**2)+R_l**2)**2
    dCr_dm = amp_Cs_AC**2/amp_Cr_AC * (w_AC**2)*M / ((w_AC**2)*(L**2)+R_l**2)

    delta_amp_cr = np.sqrt((d_L*dCr_dl)**2 + (d_M*dCr_dm)
                           ** 2 + (delta_amp_cs*dCr_dCs)**2)'''

    V0 = 7.5
    '''d_v0 = 0.1'''
    beta = (0.2e-6) * N_windings**2 * np.pi * D_B / d_ * lam_
    '''d_beta = beta*np.sqrt((d_d/d_)**2+(d_lam/lam_)**2)'''

    phase_shift = 0  # np.pi
    phase_Cs_Cr_AC = (np.angle(C_s(w_AC))-np.angle(C_r(w_AC))) + phase_shift

    return -beta * ((V0**2)/4) * amp_Cs_AC * amp_Cr_AC * np.cos(phase_Cs_Cr_AC)


def ac_force_estimate_sheet(l, r, k):
    '''Computes an estimate of the A coefficient as a function of the frequency
    in the range [80, 5000] Hz and returns a np.array. D can be obtained dividing
    by -4. Constants have to be manually tuned inside the function definition.
    Notice: to match the sign of our data, this function doesn't output A, but rather
    -A.'''
    f_AC = np.arange(80, 5000, 1)
    w_AC = 2 * np.pi * f_AC
    amp_Cs_AC = np.abs(C_s_sheet(w_AC, l, r, k))
    amp_Cr_AC = np.abs(C_r_sheet(w_AC, l, r, k))
    N_windings_coil = 84
    N_windings_sheet = 75
    D_B = 0.18     # m
    d_ = 0.0156      # m
    V0 = 7.5
    beta = (0.2e-6) * N_windings_coil*N_windings_sheet * np.pi * D_B / d_

    phase_Cs_Cr_AC = (np.angle(C_s_sheet(w_AC, l, r, k))-np.angle(C_r_sheet(w_AC, l, r, k)))

    return -beta * ((V0**2)/4) * amp_Cs_AC * amp_Cr_AC * np.cos(phase_Cs_Cr_AC)

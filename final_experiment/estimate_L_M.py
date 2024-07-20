import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


def Vs_Vgen_open_model(w, L_value, Rl_value, Rgen_value):
    return np.sqrt((Rl_value**2+(w*L_value)**2)/(((Rl_value+Rgen_value)**2)+((w*L_value)**2)))


def Vr_Vs_open_model(w, L_value, M_value, Rl_value):
    return np.sqrt(((w*M_value)**2)/((Rl_value**2)+((w*L_value)**2)))


def Vs_Vgen_short_model_approx(w, L_value, k_value, Rgen_value):
    fact_1 = (w*L_value*(1-k_value**2))
    return np.sqrt(fact_1**2/((fact_1**2)+Rgen_value**2))


def Vs_Vgen_short_model(w, L_value, k_value, Rgen_value, Rl_value):
    fact_1 = 1 - (1j*w*L_value*(k_value**2)) / (1j*w*L_value + Rl_value)
    res = (Rl_value + 1j*w*L_value*fact_1) / \
        (Rgen_value + Rl_value + 1j*w*L_value*fact_1)
    return np.abs(res)


R_gen = 50  # Ohm (nominal)
R_l = 4.51  # Ohm (measured from multimeter)
V_gen = 15
# rigol accuracy is 1% of amplitude value + 1mV
delta_Vgen = V_gen * 0.01 + 0.001


'OPEN CIRCUIT'

# this is not so clean, L and M come from fits below
L = 0.00439  # mH
M = 2.1*10**(-3)  # mH
k = 0.3

f_theo = np.arange(0, 100000, 10)
w_theo = 2 * np.pi * f_theo


'Data'
df = pd.read_csv(
    'final_experiment/data/LM_measurement/V_s_V_r_open.csv', delimiter=';')

f_data = df['Freq'].to_numpy()
w_data = 2*np.pi*f_data
V_s_data_open = df['V_s'].to_numpy()
V_r_data_open = df['V_r'].to_numpy()

# for uncertainties I used dual cursor accuracy on the manual, but I'm not so sure about it
# delta = dc vertical gain acc + 0.5% full scale
# where in our case dc vertical gain acc = 3% full scale
# and full scale = V/div * 8 (looks like there are 8 divisions)

delta_V_s_open = df['V/div-S'].to_numpy() * 8 * 0.035
delta_V_r_open = df['V/div-R'].to_numpy() * 8 * 0.035

ampl_Vs_Vgen_data_open = V_s_data_open/V_gen
delta_Vs_Vgen_open = ampl_Vs_Vgen_data_open * \
    np.sqrt((delta_V_s_open / V_s_data_open)**2 + (delta_Vgen / V_gen)**2)

ampl_Vr_Vs_data_open = V_r_data_open/V_s_data_open
delta_Vr_Vs_open = ampl_Vr_Vs_data_open * \
    np.sqrt((delta_V_s_open / V_s_data_open)**2 +
            (delta_V_r_open / V_r_data_open)**2)


'Non linear fit to find L.'
# A is L
def model(x, A): return Vs_Vgen_open_model(
    w=x, L_value=A, Rl_value=R_l, Rgen_value=R_gen)


popt, pcov = curve_fit(model, w_data, ampl_Vs_Vgen_data_open, p0=[
                       4.3], sigma=delta_Vs_Vgen_open, absolute_sigma=True)

L = popt[0]
delta_L = np.sqrt(pcov[0, 0])

print('Vs/Vgen (open circuit) fit:')
print(f'L={L} +/- {delta_L} H')


'Vs/Vgen model'
Vs_Vgen_open = (R_l+1j*w_theo*L)/(1j*w_theo*L+R_l+R_gen)
ampl_Vs_Vgen_open = np.abs(Vs_Vgen_open)


'Plot Vs/Vgen data and model'
plt.figure()
plt.errorbar(x=f_data, y=ampl_Vs_Vgen_data_open, yerr=delta_Vs_Vgen_open, linestyle='None',
             marker='.', label='Data: $V_S/V_{gen}$ open')
plt.plot(f_theo, ampl_Vs_Vgen_open, linestyle='-', marker='None',
         label=f'Fit result: $L = {1000*L:.3g}\pm{1000*delta_L:.1g}$ mH')
plt.xlabel('f [Hz]')
plt.ylabel('$V_S/V_{GEN}$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
# plt.savefig('final_experiment/pictures/lm_meas/V_s_open_1.pdf')
# plt.show()


'Non linear fit to find M, using L from fit'
# A is M
def model(x, A): return Vr_Vs_open_model(x, L_value=L, M_value=A, Rl_value=R_l)


popt, pcov = curve_fit(model, w_data, ampl_Vr_Vs_data_open, p0=[
                       2], sigma=delta_Vr_Vs_open, absolute_sigma=True)

M = popt[0]
delta_M = np.sqrt(pcov[0, 0])

print('Vr/Vs (open circuit) fit:')
# print("L=%f" %popt[0], '\pm %f' %np.sqrt(pcov[0,0]))
print(f'M = {M} +/- {delta_M} H')
# print("R_l=%f" %popt[2], '\pm %f' %np.sqrt(pcov[2,2]))


'Vr/Vs model'
Vr_Vs_open = -(1j*w_theo*M)/(R_l+1j*w_theo*L)
ampl_Vr_Vs_open = np.abs(Vr_Vs_open)

'Plot Vr/Vs data and model'
plt.figure()
plt.errorbar(x=f_data, y=ampl_Vr_Vs_data_open, yerr=delta_Vr_Vs_open,
             linestyle='None', marker='.', label='Data: $V_R/V_S$ open')
plt.plot(f_theo, ampl_Vr_Vs_open, linestyle='-', marker='None',
         label=f'Fit result: $M = {1000*M:.3g}\pm{1000*delta_M:.1g}$ mH')
plt.xlabel('f [Hz]')
plt.ylabel('$V_R/V_S$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
# plt.savefig('final_experiment/pictures/lm_meas/V_s_V_r_open_1.pdf')
# plt.show()


'SHORT CIRCUIT'

'Data'
df = pd.read_csv(
    'final_experiment/data/LM_measurement/V_s_short.csv', delimiter=';')

f_data = df['Freq'].to_numpy()
w_data = 2*np.pi*f_data
V_s_data_short = df['V_s'].to_numpy()
delta_V_s_short = df['V/div'].to_numpy() * 8 * 0.035

ampl_Vs_Vgen_data_short = V_s_data_short / V_gen
delta_Vs_Vgen_short = ampl_Vs_Vgen_data_short * \
    np.sqrt((delta_V_s_short / V_s_data_short)**2 + (delta_Vgen / V_gen)**2)


'Non linear fit to find k from V_S data and previous estimate of L'


# A is k
def model(x, A): return Vs_Vgen_short_model(
    x, L_value=L, k_value=A, Rgen_value=R_gen, Rl_value=R_l)


popt_short, pcov_short = curve_fit(model, w_data, ampl_Vs_Vgen_data_short, p0=[
                                   0.5], sigma=delta_Vs_Vgen_short, absolute_sigma=True)

k = popt_short[0]
delta_k = np.sqrt(pcov_short[0, 0])

print('Vs/Vgen (short circuit) fit:')
print(f'k = {k} +/- {delta_k}')

'MODEL'

factor_1 = 1j*w_theo*L * (1 - (1j*w_theo*L*(k**2) / (1j*w_theo*L + R_l)))
Vs_Vgen_short = (R_l+factor_1)/(R_gen+R_l+factor_1)
ampl_Vs_Vgen_short = np.abs(Vs_Vgen_short)

'PLOTS'
plt.figure()
plt.errorbar(x=f_data, y=ampl_Vs_Vgen_data_short, yerr=delta_Vs_Vgen_short,
             linestyle='None', marker='.', label='Data: $V_S/V_{gen}$ short')
plt.plot(f_theo, ampl_Vs_Vgen_short, linestyle='-',
         marker='None', label=f'$k = {k:.3g}\pm{delta_k:.1g}$')
plt.xlabel('f [Hz]')
plt.ylabel('$V_S/V_{GEN}$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
# plt.savefig('final_experiment/pictures/lm_meas/V_s_short_1.pdf')
plt.show()

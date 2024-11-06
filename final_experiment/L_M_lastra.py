import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


def Vs_Vgen_sheet_k(w, L_las, R_las, k):
    M_ = k * np.sqrt(L_coil*L_las)
    i_s_vgen = 1 / ((1j * w * L_coil + R_l + R_gen) +
                    ((w*M_)**2/(1j*w*L_las + R_las)))
    i_r_vgen = (-1j * w * M_)/(1j*w*L_las + R_las) * i_s_vgen
    V_s_vgen = (1j * w * L_coil + R_l) * i_s_vgen + 1j * w * M_ * i_r_vgen
    return np.abs(V_s_vgen)


R_gen = 50  # Ohm (nominal)
R_l = 4.516  # Ohm (measured from multimeter)
V_gen = 15
# rigol accuracy is 1% of amplitude value + 1mV
delta_Vgen = V_gen * 0.01 + 0.001
L_coil = 0.00434

f_theo = np.arange(0, 100000, 10)
w_theo = 2 * np.pi * f_theo


'COPPER SHEET'

'Data'
df = pd.read_csv(
    'final_experiment/data/LM_measurement/V_s_lastra.csv', delimiter=';')

df = df.drop(index=range(4)).reset_index(drop=True)

f_data = df['Freq'].to_numpy()
w_data = 2*np.pi*f_data
V_s_data_sheet = df['V_s'].to_numpy()
delta_V_s_sheet = df['V/div'].to_numpy() * 8 * 0.035

ampl_Vs_Vgen_data = V_s_data_sheet / V_gen
delta_Vs_Vgen = ampl_Vs_Vgen_data * \
    np.sqrt((delta_V_s_sheet / V_s_data_sheet)**2 + (delta_Vgen / V_gen)**2)


'''Non linear fit to find first estimates for k and R_las.
Notice, we are fixing L_las to 0.54 mH.'''

L_fix = 0.00054


# A is k, B is R_las
def model_k(x, A, B): return Vs_Vgen_sheet_k(
    x, L_las=L_fix, k=A, R_las=B)


lower_bounds = [0, 0]
upper_bounds = [1, np.inf]
bo_k = (lower_bounds, upper_bounds)

popt, pcov = curve_fit(model_k, w_data, ampl_Vs_Vgen_data, p0=[
    0.5, 5], sigma=delta_Vs_Vgen, absolute_sigma=True, bounds=bo_k)


k_sheet = popt[0]
delta_k_sheet = np.sqrt(pcov[0, 0])
R_sheet = popt[1]
delta_R_sheet = np.sqrt(pcov[1, 1])

print('Vs/Vgen (copper sheet) fit, fixing L_sheet to 0.54 mH:')
print(f'k = {k_sheet} +/- {delta_k_sheet}')
print(f'R_sheet = {R_sheet} +/- {delta_R_sheet} Ohm')

'Model'

ampl_Vs_Vgen_fit = Vs_Vgen_sheet_k(
    w=w_theo, L_las=L_fix, k=k_sheet, R_las=R_sheet)

label_k = f'$k_{{sheet}} = {k_sheet:.3f} \\pm {delta_k_sheet:.3f}$\n'
label_r_las = f'$R_{{sheet}} = {R_sheet:.2f} \\pm {delta_R_sheet:.2f} \Omega$'


'Plots'
plt.figure()
plt.errorbar(x=f_data, y=ampl_Vs_Vgen_data, yerr=delta_Vs_Vgen,
             linestyle='None', marker='.', label='Data: $V_S/V_{gen}$ copper sheet')
plt.plot(f_theo, ampl_Vs_Vgen_fit, linestyle='-',
         marker='None', label=label_k+label_r_las)
plt.title(f'Fixing $L_{{sheet}}={1000*L_fix} mH$')
plt.xlabel('f [Hz]')
plt.ylabel('$V_S/V_{GEN}$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('final_experiment/pictures/cu_sheet/V_s_sheet.pdf')
plt.savefig('final_experiment/pictures/cu_sheet/V_s_sheet.png')

'''Non linear fit to find first estimates for k and L.
Notice, we are fixing R_las to 0.26 Ohm'''
R_fix = 0.26


# A is k, B is R_las
def model_k(x, A, B): return Vs_Vgen_sheet_k(
    x, L_las=B, k=A, R_las=R_fix)


lower_bounds = [0, 0]
upper_bounds = [1, np.inf]
bo_ = (lower_bounds, upper_bounds)

popt, pcov = curve_fit(model_k, w_data, ampl_Vs_Vgen_data, p0=[
    0.5, 1], sigma=delta_Vs_Vgen, absolute_sigma=True, bounds=bo_)


k_sheet = popt[0]
delta_k_sheet = np.sqrt(pcov[0, 0])
L_sheet = popt[1]
delta_L_sheet = np.sqrt(pcov[1, 1])

print('Vs/Vgen (copper sheet) fit, fixing R_sheet:')
print(f'k = {k_sheet} +/- {delta_k_sheet}')
print(f'L = {L_sheet} +/- {delta_L_sheet}')

'Model'

ampl_Vs_Vgen_fit = Vs_Vgen_sheet_k(
    w=w_theo, L_las=L_sheet, k=k_sheet, R_las=R_fix)

label_k = f'$k_{{sheet}} = {k_sheet:.3f} \\pm {delta_k_sheet:.3f}$\n'
label_l = f'$L_{{sheet}} = {1000*L_sheet:.2g} \\pm {1000*delta_L_sheet:.1g}$ mH'


'Plots'
plt.figure()
plt.errorbar(x=f_data, y=ampl_Vs_Vgen_data, yerr=delta_Vs_Vgen,
             linestyle='None', marker='.', label='Data: $V_S/V_{gen}$ copper sheet')
plt.plot(f_theo, ampl_Vs_Vgen_fit, linestyle='-',
         marker='None', label=label_k+label_l)
plt.title(f'Fixing $R_{{sheet}}={R_fix}\Omega$')
plt.xlabel('f [Hz]')
plt.ylabel('$V_S/V_{GEN}$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
# plt.savefig('final_experiment/pictures/cu_sheet/V_s_sheet.pdf')
# plt.savefig('final_experiment/pictures/cu_sheet/V_s_sheet.png')


'''Non linear fit to find a first estimate for k.
Notice, we are fixing R_las to 0.26 Ohm, and L_las to 0.53 mH.'''


# A is k, B is R_las
def model_k(x, A): return Vs_Vgen_sheet_k(
    x, L_las=L_fix, k=A, R_las=R_fix)


lower_bounds = [0]
upper_bounds = [1]
bo_ = (lower_bounds, upper_bounds)

popt, pcov = curve_fit(model_k, w_data, ampl_Vs_Vgen_data, p0=[
    0.5], sigma=delta_Vs_Vgen, absolute_sigma=True, bounds=bo_)


k_sheet = popt[0]
delta_k_sheet = np.sqrt(pcov[0, 0])

print('Vs/Vgen (copper sheet) fit, fixing R_sheet and L_sheet:')
print(f'k = {k_sheet} +/- {delta_k_sheet}')

'Model'

ampl_Vs_Vgen_fit = Vs_Vgen_sheet_k(
    w=w_theo, L_las=L_fix, k=k_sheet, R_las=R_fix)

label_k = f'$k_{{sheet}} = {k_sheet:.3f} \\pm {delta_k_sheet:.3f}$\n'


'Plots'
plt.figure()
plt.errorbar(x=f_data, y=ampl_Vs_Vgen_data, yerr=delta_Vs_Vgen,
             linestyle='None', marker='.', label='Data: $V_S/V_{gen}$ copper sheet')
plt.plot(f_theo, ampl_Vs_Vgen_fit, linestyle='-',
         marker='None', label=label_k)
plt.title(f'Fixing $R_{{sheet}}={R_fix}\Omega$ and $L_{{sheet}}={1000*L_fix}$mH')
plt.xlabel('f [Hz]')
plt.ylabel('$V_S/V_{GEN}$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
# plt.savefig('final_experiment/pictures/cu_sheet/V_s_sheet.pdf')
# plt.savefig('final_experiment/pictures/cu_sheet/V_s_sheet.png')
plt.show()
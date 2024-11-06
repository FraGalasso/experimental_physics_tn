import pandas as pd
import t0_ABCD
import plot_functions as pltfunc
import conversion as conv
import numpy as np
import matplotlib.pyplot as plt


FS = 1e-4  # V (this was 100 microvolts)
df_list = []
pars_list = []  # this will be a list of lists, element i will be a list [A, B, C, D] of the fit parameters for frequency i
stds_list = []  # this will be a list of lists, element i will be a list [sigmaA, sigmaB, sigmaC, sigmaD] of the uncertainties

for i in range(1, 8):
    df_list.append(
        pd.read_csv(f'final_experiment/data/overnight-lastra/lastra{i}.csv'))
freqs = [100, 200, 500, 1000, 2000, 5000, 700] # Cu sheet freq.
# freqs = [100, 200, 500, 1000, 2000, 5000, 700, 3000] # Fe sheet freq.

i = 0
for item in df_list:
    print(f'Elapsed time: {item["Time"].iloc[-1]-item["Time"].iloc[0]} s')

    time = item['Time'].to_numpy()
    ch1 = conv.compvolt_to_force(item['Channel1'].to_numpy(), FS)
    ch3 = item['Channel3'].to_numpy()

    t0, f_MOD = t0_ABCD.determine_t0_fmod_function(time, ch3, 0.05, False, 28)

    t0 += 0.3

    A_mean, A_std, B_mean, B_std, C_mean, C_std, D_mean, D_std = t0_ABCD.determine_A_B_C_D_func_alt(
        freq=freqs[i], time=time, channel=ch1, t_0=t0, f_MOD=f_MOD, interval_length=300, plot=False)
    pars_list.append([A_mean, B_mean, C_mean, D_mean])
    stds_list.append([A_std, B_std, C_std, D_std])
    i = i + 1

pltfunc.plot_forcegraph(freqs, pars_list, stds_list, plot_expected=True, is_sheet=True)

a_values, a_std, b_values, b_std, c_values, c_std, d_values, d_std = t0_ABCD.extract_values(
    df_freq=0, pars_list=pars_list, stds_list=stds_list)

a_array = np.array(a_values)
d_array = np.array(d_values)
a_std_array = np.array(a_std)
d_std_array = np.array(d_std)

ad_1_4 = d_array/a_array
delta_ad_1_4 = np.abs(ad_1_4) * np.sqrt((a_std_array/a_array)**2 + (d_std_array/d_array)**2)

print(t0_ABCD.stats(ad_1_4))

plt.figure()
plt.errorbar(freqs, ad_1_4, yerr=delta_ad_1_4, linestyle='None', marker='.', label='$\\frac{D}{A}$')
plt.axhline(y=-0.25, color='red', linestyle='-', linewidth=1)
plt.xscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('')
plt.legend()
plt.grid()
plt.tight_layout()
# plt.ticklabel_format(axis='y', style='sci', scilimits=(-6, -6))
plt.savefig('final_experiment/pictures/coolplots/check1_4_coils_sheet.pdf')
plt.show()
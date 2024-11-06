import matplotlib.pyplot as plt
import pandas as pd
import t0_AB
import conversion as conv

'SECTION: DC CALIBRATION'

df = pd.read_csv('final_experiment/data/DCmeasurement/totalDC1.csv')
fs = 0.0005

print(f'Elapsed time: {df["Time"].iloc[-1]-df["Time"].iloc[0]} s')

time = df['Time']
force = conv.compvolt_to_force(df['Channel1'], fs)
force_quad = conv.compvolt_to_force(df['Channel2'], fs)
ch3 = df['Channel3']


plt.figure()
plt.plot(time[3000:4000], force[3000:4000], linestyle='None', marker='.', markersize=2,  label='In phase')
plt.plot(time[3000:4000], force_quad[3000:4000], linestyle='None', marker='.', markersize=2, label='Quadrature')
plt.xlabel('Time stamp (s)')
plt.ylabel('Measured Force [N]')
plt.legend()
plt.tight_layout()
plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
plt.grid()
# plt.savefig('final_experiment/pictures/calibration/dc_time.pdf')
plt.show()

t0, f_MOD = t0_AB.determine_t0_fmod_function(time, ch3, 0.1, False, 31)

t0 += 0.3

A_mean, A_std, B_mean, B_std = t0_AB.determine_A_B_func_alt(
    time=time, channel=force, t_0=t0, f_MOD=f_MOD, interval_length=300, plot=True)

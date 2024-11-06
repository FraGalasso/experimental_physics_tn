import numpy as np
import estimate_force as f
import matplotlib.pyplot as plt
from matplotlib import cm

f_AC = np.arange(80, 5000, 1)
w_AC = 2 * np.pi * f_AC
values = [0.1, 0.5, 1, 5, 10, 50, 100]
cmap = cm.get_cmap('cividis')
colors = cmap(np.linspace(0, 1, len(values)))
plt.figure()
for i in range(len(values)):
    A = f.ac_force_estimate_sheet(values[i]/1000, values[i], 0.5)
    plt.plot(f_AC, A, linestyle='-', marker='None',
             color=colors[i], label=f'$L={values[i]}$ mH, $R = {values[i]}\Omega$')
plt.xscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Force amplitude coefficients [N]')
# plt.title(f'$\\frac{{L}}{{R}} = 1ms$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.ticklabel_format(axis='y', style='sci', scilimits=(-6, -6))
plt.savefig('final_experiment/pictures/coolplots/fixed_t.pdf')
plt.savefig('final_experiment/pictures/coolplots/fixed_t.png')

values = [0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
colors = cmap(np.linspace(0, 1, len(values)))
plt.figure()
for i in range(len(values)):
    A = f.ac_force_estimate_sheet(0.5/1000, 0.5/values[i], 0.5)
    plt.plot(f_AC, A, linestyle='-', marker='None',
             color=colors[i], label=f'$\\frac{{L}}{{R}} = {values[i]}ms$')
plt.xscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Force amplitude coefficients [N]')
# plt.title(f'$\\frac{{L}}{{R}} = 1ms$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.ticklabel_format(axis='y', style='sci', scilimits=(-6, -6))
plt.savefig('final_experiment/pictures/coolplots/moving_t.pdf')
plt.savefig('final_experiment/pictures/coolplots/moving_t.png')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from child_data import *

def print_child_data():
      print('filament length: %f $\pm$ ' %
            filament_len, '%f' % delta_filament_len,  'm')
      print('grid length: %f $\pm$ ' % grid_len, '%f' % delta_grid_len, 'm')
      print('grid diameter: %f $\pm$ ' %
            grid_diam, '%f' % delta_grid_diam, 'm\n')

      print('long filament length: %f $\pm$ ' %
            long_filament, '%f' % delta_ruler,  'm')
      print('long filament resistance: %f $\pm$ ' %
            long_resistance, '%f' % delta_long_resistance, 'm\n')

      print('R_0: %f $\pm$ ' % r0, '%f' % delta_r0, 'Ohm')
      print('R of system at T0: %f $\pm' %
            res_system_T0, '%f' % delta_res_system_T0)
      print('Constant resistance of the circuit: %f $\pm$' %
            r_rest, '%f' % delta_r_rest, 'Ohm\n')
      print('3 cases of temperature for the Child law part of the experiment:\n')
      print('T1: %f $\pm$ ' % T[0], '%f ' % delta_T[0], 'Celsius')
      print('T2: %f $\pm$ ' % T[1], '%f ' % delta_T[1], 'Celsius')
      print('T3: %f $\pm$ ' % T[2], '%f ' % delta_T[2], 'Celsius')

def plot_child_data():
      plt.figure(dpi=200)
      
      plt.errorbar(V_g1, I_g1, yerr=delta_I_g1, xerr=delta_V_g1, marker='.',  linestyle='None',
                   label='T$\simeq$' + str(np.ceil(T[0]))+' $\pm$ ' + str(np.ceil(delta_T[0])) + ' Celsius')
      plt.errorbar(V_g2, I_g2, yerr=delta_I_g2, xerr=delta_V_g2, marker='.',  linestyle='None',
                   label='T$\simeq$' + str(np.ceil(T[1]))+' $\pm$ ' + str(np.ceil(delta_T[1])) + ' Celsius')
      plt.errorbar(V_g3, I_g3, yerr=delta_I_g3, xerr=delta_V_g3, marker='.',  linestyle='None',
                   label='T$\simeq$' + str(np.ceil(T[2]))+' $\pm$ ' + str(np.ceil(delta_T[2])) + ' Celsius')
      plt.xlabel('Grid Voltage $V_g$ [V]')
      plt.ylabel('Grid Current $I_g$ [A]')
      plt.legend()
      plt.grid()
      plt.subplots_adjust(left=0.18, bottom=0.18)
      plt.savefig('data_child.png')
      plt.show()
      
      plt.close()

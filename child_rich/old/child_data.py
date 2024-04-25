import numpy as np
import constants as cst

'instrument resolution and measurement resolution (or uncertainty)'
cal_res = 0.005 * 10**(-2)  # m
delta_caliper = cal_res / np.sqrt(12)  # m  (standard resolution uncertainty)

'lengths, average and related uncertainties of the filament eventually heated for production of electrons'
filament_lens = np.array([2.09, 2.12, 2.05, 2.05, 2.02, 2.17]) * 10**(-2)  # m
# delta_filament_lens = np.ones(np.size(filament_lens)) * delta_caliper  # m

filament_len = np.average(filament_lens)  # m
delta_filament_len = (filament_lens.max()-filament_lens.min())/2  # m

'length, diamenter, radius, averages and related uncertainties of the grid that collects the electrons'
grid_lens = np.array([2.64, 2.95, 2.97]) * 10**(-2)  # m
# delta_grid_lens = np.ones(np.size(grid_lens)) * delta_caliper       #m

grid_len = np.average(grid_lens)  # m
delta_grid_len = (grid_lens.max()-grid_lens.min())/2  # m

grid_diams = np.array([1.23, 1.23, 1.2]) * 10**(-2)  # m
# delta_grid_diams = np.ones(np.size(grid_diams)) * delta_caliper  # m

grid_diam = np.average(grid_diams)  # m
delta_grid_diam = (grid_diams.max() - grid_diams.min())/2

grid_radius = grid_diam/2
delta_grid_radius = delta_grid_diam/2

'Length, resistance and related uncertainty of long filament of tungsten measure withe the 4-wire ohm function'
long_filament = 16.60 * 10**(-2)            # m
# standard resolution uncertainty
delta_ruler = (0.1 * 10**(-2))/np.sqrt(12)  # m

# for a 100 Ohm range and the accuracy is %0.01 reading + %0.04 of range.
# the minimum range is 100 Ohm
long_resistance = 0.297  # Ohm
delta_long_resistance = long_resistance * (0.01/100) + 100*(0.004/100)  # Ohm

'resistance and related uncertainty of filament of tungsten used for the emission of electrons'
r0 = long_resistance * filament_len / long_filament  # Ohm
delta_r0 = r0 * np.sqrt((delta_filament_len / filament_len) ** 2 + (delta_ruler /
                        long_filament) ** 2 + (delta_long_resistance / long_resistance) ** 2)  # Ohm

'resistance of all circuit and filament and related uncertainty'
# for a 100 Ohm range and the accuracy is %0.01 reading +  %0.04 of range.
res_system_T0 = 0.235  # Ohm
delta_res_system_T0 = res_system_T0 * (0.01/100) + 100*(0.004/100)  # Ohm

'resisatnce and related uncertainty of all circuit without the filament'
r_rest = res_system_T0 - r0
delta_r_rest = np.sqrt(((delta_res_system_T0)**2) + ((delta_r0)**2))

'Room temperature and related uncertainty'
T_0 = 21  # Celsius
delta_T_0 = 1 / np.sqrt(12)  # Celsius

'Current and related uncertainties programmed on Rigol power supply for 3 cases'
# accuracy on the programmed current is 0.1% of output programmed current +5mA
amp_system = np.array([4.105, 4.005, 3.905])            # A
delta_amp_system = (0.1/100) * amp_system + (10 * 10**(-3))   # A

'Readback tension and related uncertainties read on Rigol power supply for 3 cases'
# accuracy on the read voltage is 0.05% of output programmed voltage +10mV
volt_system = np.array([3.214, 3.078, 2.947])  # V
delta_volt_system = (0.05/100) * volt_system + (10*10**(-3))  # V

'Resistance and related uncertainty of circuit during current flow for 3 cases'
res_system = volt_system/amp_system
delta_res_system = res_system * np.sqrt((delta_volt_system/volt_system) ** 2
                                        + (delta_amp_system/amp_system) ** 2)

'Resistance and related uncertainty of filament during current flow for 3 cases'
r_f = res_system - r_rest
delta_rf = np.sqrt((delta_res_system**2) + (delta_r_rest**2))

'Temperature and related uncertainties of filament in 3 cases'
T = T_0 + (1 / cst.alpha) * ((r_f / r0) - 1)
delta_T = np.sqrt((delta_T_0 ** 2) + ((delta_rf / (cst.alpha * r0)) ** 2)
                  + ((r_f * delta_r0 / (cst.alpha * r0 ** 2)) ** 2))


'First temperature'
# accuracy is equal to 0.05%output voltage + 100mV offset
V_g1 = np.array([50.05, 45.05, 40.04, 35.04, 30.04, 25.04, 20.04, 19.05, 18.03, 17.02, 16.02, 15.02, 14.02, 13.02, 12.02,
                11.02, 10.02, 9.02, 8.02, 7.02, 6.52, 6.02, 5.52, 5.02, 4.52, 4.02, 3.53, 3.03, 2.52, 2.03, 1.52, 1.02, 0.55])  # V
delta_V_g1 = (0.05/100)*V_g1+(100*10**(-3))  # V

# accuracy is equal to 0.05% of reading + 0.02% of range
# in case of our measurement the minimum range is 10 mA
I_g1 = np.array([1.18800, 1.17900, 1.16800, 1.15500, 1.13700, 1.10700, 0.95800, 0.90300, 0.84200, 0.77700, 0.71000, 0.64200, 0.57400, 0.50600, 0.43800, 0.37200, 0.30700,
                0.24400, 0.18700, 0.13700, 0.11400, 0.09400, 0.07500, 0.06000, 0.04600, 0.04352, 0.02590, 0.01820, 0.01200, 0.00760, 0.00450, 0.00240, 0.00110])*10**(-3)  # A
delta_I_g1 = (0.05/100)*I_g1+(0.02/100)*(10*10**(-3))  # A


'Second temperature'
# accuracy is equal to 0.05%output voltage + 100mV offset
V_g2 = np.array([50.05, 45.05, 40.04, 35.04, 30.04, 25.04, 24.03, 23.03, 22.02, 21.03, 20.04, 19.05, 18.03, 17.02, 16.02, 15.02,
                14.02, 13.02, 12.02, 11.02, 10.02, 9.02, 8.02, 7.02, 6.52, 6.02, 5.52, 5.02, 4.51, 4.02, 3.53, 3.03, 2.52, 2.03, 1.52, 1.03, 0.55])
delta_V_g2 = (0.05/100)*V_g2+(100*10**(-3))  # V

# accuracy is equal to 0.05% of reading + 0.02% of range
# in case of our measurement the minimum range is 10 mA
I_g2 = np.array([0.70000, 0.69500, 0.69100, 0.68300, 0.67400, 0.66400, 0.66320, 0.66030, 0.65730, 0.65380, 0.64950, 0.64330, 0.63090, 0.60700, 0.58300, 0.54710, 0.50320, 0.45270,
                0.39800, 0.34290, 0.28610, 0.23050, 0.17780, 0.13070, 0.10920, 0.08990, 0.07250, 0.05760, 0.04470, 0.03390, 0.02480, 0.01740, 0.01150, 0.00710, 0.00420, 0.00220, 0.00100])*10**(-3)
delta_I_g2 = (0.05/100)*I_g2+(0.02/100)*(10*10**(-3))  # A

'third temperature'
# accuracy is equal to 0.05%output voltage + 100mV offset
V_g3 = np.array([50.04, 45.04, 40.04, 35.04, 30.04, 25.04, 24.03, 23.03, 22.03, 21.03, 20.04, 19.05, 18.03, 17.02, 16.02, 15.02,
                14.02, 13.02, 12.01, 11.02, 10.02, 9.02, 8.01, 7.02, 6.52, 6.02, 5.52, 5.02, 4.51, 4.02, 3.53, 3.03, 2.52, 2.03, 1.52, 1.03, 0.55])
delta_V_g3 = (0.05/100)*V_g3+(100*10**(-3))  # V

# accuracy is equal to 0.05% of reading + 0.02% of range
# in case of our measurement the minimum range is 10 mA
I_g3 = np.array([0.37810, 0.37550, 0.37210, 0.36850, 0.36440, 0.35930, 0.35890, 0.35770, 0.35660, 0.35570, 0.35440, 0.35300, 0.35190, 0.35000, 0.34750, 0.34430, 0.33760, 0.32410,
                0.30390, 0.27740, 0.24370, 0.20300, 0.16010, 0.11970, 0.10100, 0.08370, 0.06830, 0.05460, 0.04250, 0.03240, 0.02380, 0.01670, 0.01120, 0.00690, 0.00410, 0.00210, 0.00090])*10**(-3)
delta_I_g3 = (0.05/100)*I_g3+(0.02/100)*(10*10**(-3))  # A
from scipy.optimize import curve_fit
from scipy import odr
import matplotlib.pyplot as plt
import numpy as np


def linear(x, a, b):
    return a + b * x


def lin(pp, x):
    return pp[0] + pp[1] * x


intensities = np.array([26.0, 23.2, 20.9, 18.4, 17.3,
                       15.9, 14.2, 11.7, 6.4, 9.05, 6.8, 4.5, 1.67])
offset = np.array([0.18, 0.12, 0.2, 0.13, 0.13, 0.14, 0.16,
                  0.21, 0.16, 0.34, 0.33, 0.23, 0.18])
delta_intensities = np.array(
    [0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.1, 0.05, 0.05, 0.5, 0.03])
delta_offset = np.array([0.02, 0.01, 0.02, 0.04, 0.02,
                        0.03, 0.02, 0.03, 0.05, 0.02, 0.07, 0.05, 0.01])

intensities_eff = intensities - offset
delta_intensities_eff = np.sqrt(delta_intensities**2 + delta_offset**2)

co2_flux = np.array([3.78, 3.48, 3.18, 2.88, 2.58, 2.28,
                    1.98, 1.68, 1.38, 1.08, 0.68, 0.28, 0])
air_flux = np.array([497, 497, 497, 497, 497, 497,
                    497, 497, 497, 497, 497, 497, 497])
delta_co2_flux = co2_flux * 0.002
delta_air_flux = air_flux * 0.002
delta_co2_flux[-1] = delta_co2_flux[-2]
concentrations = co2_flux/(co2_flux + air_flux)*100
delta_concentrations = np.sqrt((air_flux * delta_co2_flux)**2 +
                               (co2_flux * delta_air_flux)**2) / ((co2_flux + air_flux)**2) * 100

'''popt, pcov = curve_fit(linear, concentrations, intensities_eff,
                       sigma=delta_intensities_eff, absolute_sigma=True)

intercept = popt[0]
delta_intercept = np.sqrt(pcov[0, 0])
slope = popt[1]
delta_slope = np.sqrt(pcov[1, 1])'''

data_odr = odr.RealData(concentrations, intensities_eff,
                        delta_concentrations, delta_intensities_eff)
model = odr.Model(lin)
myodr = odr.ODR(data_odr, model, beta0=[1.54, 30.5])
myodr.set_job(fit_type=0)
output = myodr.run()

intercept = output.beta[0]
delta_intercept = output.sd_beta[0]
slope = output.beta[1]
delta_slope = output.sd_beta[1]

x_fit = np.linspace(-0.05, max(concentrations))
y_fit = intercept + slope * x_fit

print(f'Intercept = {intercept} +/- {delta_intercept}')
print(f'Slope = {slope} +/- {delta_slope}')

conc_co2 = intercept/slope
delta_conc_co2 = np.abs(conc_co2) * np.sqrt((delta_intercept / intercept) ** 2 +
                                            (delta_slope / slope) ** 2)

print(f'Estimated CO2 concentration: {conc_co2} +/- {delta_conc_co2}')

plt.figure(dpi=200)
plt.xlabel('% concentration')
plt.ylabel('Intensities [mV]')
plt.errorbar(x=concentrations, y=intensities_eff, xerr=delta_concentrations, yerr=delta_intensities_eff,
             marker='.', linestyle='None', label='data')
plt.plot(x_fit, y_fit, label='fit')
plt.title(
    f'Estimated CO2 concentration: {conc_co2:.3g} +/- {delta_conc_co2:.3g} %')
plt.legend()
plt.grid()

plt.show()

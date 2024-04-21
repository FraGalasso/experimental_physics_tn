import numpy as np
from scipy.optimize import curve_fit
from scipy import odr

def linear(x, a, b):
    return a + b * x


def linear_odr(pp, x):
    return pp[0] + pp[1] * x


def power_law(x, a, b):
    return (a * (x ** b))


def power_law_odr(pp, x):
    # Python complains at runtime because for low voltages it has
    # a fractional power of a negative number
    # the only fix I can think of is to bring the value of the voltage
    # back above zero when it happens
    if np.any(x < 0):
        x = np.abs(x)
    return (pp[0] * (x ** pp[1]))

# the quality of the fit might get worse due to this issue so we might
# just end up having linear regression with odr as our best option

def odr_child_fitter(x, y, dx, dy, bg, linear_fit=True):
    '''Fits data with scipy.odr: one has to provide data arrays, a guess for the first parameter
    and a boolean determining whether we want a linear or a power law fit'''

    data_to_fit = odr.RealData(x, y, dx, dy)

    # selecting the proper function
    if linear_fit:
        model = odr.Model(linear_odr)
    else:
        model = odr.Model(power_law_odr)

    myodr = odr.ODR(data_to_fit, model, beta0=[bg, 1.5])
    myodr.set_job(fit_type=0)
    output = myodr.run()
    return np.array([output.beta, output.sd_beta])


def curve_fit_child_fitter(x, y, dy, bg, linear_fit=True):
    '''Fits data with scipy.curve_fit: one has to provide data arrays, a guess for the first 
    parameter and a boolean determining whether we want a linear or a power law fit'''

    # selecting the proper function
    if linear_fit:
        func = linear
    else:
        func = power_law

    popt, pcov = curve_fit(func, x, y, sigma=dy, p0=[
                           bg, 1.5], absolute_sigma=True)

    return np.array([popt, np.sqrt(np.diag(pcov))])

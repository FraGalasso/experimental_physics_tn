import numpy as np
from scipy.optimize import curve_fit
from scipy import odr

def linear(pp, x):
    return pp[0] + pp[1] * x

def power_law(pp, x):
    # Python complains at runtime because for low voltages it has
    # a fractional power of a negative number
    # the only fix I can think of is to bring the value of the voltage
    # back above zero when it happens
    if np.any(x < 0):
        x = np.abs(x)
    return (pp[0] * (x ** pp[1]))

# the quality of the fit might get worse due to this issue so we might
# just end up having linear regression with odr as our best option

def linear_fixed_coeff(pp, x):
    return (pp[0] + 1.5 * np.log(x + pp[1]))


def linear_volt_shift(pp, x):
    return (pp[0] + pp[1] * np.log(x + pp[2]))

def power_law_curr_shift(pp, x):
    if np.any(x < 0):
        x = np.abs(x)
    return (pp[0] * (x ** pp[1]) + pp[2])


def child_fitter(x, y, dx, dy, beta, func='V'):
    '''Fits data with scipy.odr: one has to provide data arrays, best guesses
    and a character determining which is the function to use'''

    data_to_fit = odr.RealData(x, y, dx, dy)

    functions = {
        'L' : linear,
        'P' : power_law,
        'F' : linear_fixed_coeff,
        'V' : linear_volt_shift,
        'J' : power_law_curr_shift
    }

    model = odr.Model(functions.get(func))

    myodr = odr.ODR(data_to_fit, model, beta0=beta)
    myodr.set_job(fit_type=0)
    output = myodr.run()
    return np.array([output.beta, output.sd_beta])


'''if (func == 'V') | (func =='J'):
        beta = [bg, 1.5, 0]
    elif (func == 'F'):
        beta = [bg, 0]
    else:
        beta = [bg, 1.5]'''
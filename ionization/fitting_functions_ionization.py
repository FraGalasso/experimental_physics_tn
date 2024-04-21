import numpy as np
from scipy import odr

def linear_odr(pp, x):
    return pp[0] + pp[1] * x

def odr_linear_fitter(x, y, dx, dy, bg=[1, 1]):
    '''Fits data with scipy.odr: one has to provide data arrays, 
    and guesses for the parameters'''

    data_to_fit = odr.RealData(x, y, dx, dy)
    model = odr.Model(linear_odr)
    myodr = odr.ODR(data_to_fit, model, beta0=bg)
    myodr.set_job(fit_type=0)
    output = myodr.run()
    return np.array([output.beta, output.sd_beta])
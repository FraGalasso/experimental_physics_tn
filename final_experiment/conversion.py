import numpy as np

FS = 100 * 10**(-6) # V (this was 100 microvolts)
dvdf = 1.25 * 10**(-3) #V/N (this was 1.25 millivolts/N)

#conversions
def compvolt_to_realvolt(vec, FS):
    f = np.sqrt(2) * FS / 10
    return f * np.array(vec)    #10V/FullScale #returns Volts

def realvolt_to_force(vec):
    f = 1/dvdf
    return f * np.array(vec)  #returns N

def compvolt_to_force(vec, FS):
    return realvolt_to_force(compvolt_to_realvolt(vec, FS))

import numpy as np
import ionization_fit as ion
import ionization_data as id

id.plot_ionization_data()

# ionization fit with threshold at 25 V
ion.ion_fit_N()

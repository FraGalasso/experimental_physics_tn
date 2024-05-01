import ionization_fit as ion
import ionization_data as id

id.plot_ionization_data()

ion.ion_fit(17, 27, 'N')
ion.ion_fit(27, 55, 'N')

ion.ion_fit(17, 23, 'Ar')
ion.ion_fit(23, 55, 'Ar')

ion.ion_fit(18, 27, 'He')
ion.ion_fit(27, 55, 'He')

ion.ion_fit(16, 23, 'CO2')
ion.ion_fit(23, 55, 'CO2')
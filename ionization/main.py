import ionization_fit as ion
import ionization_data as id

id.plot_ionization_data()

# results = []
# 
# for i in range(5):
#    for j in range(i + 1, 9):
#        print(f'\nV_min = {15+5*i} V, V_max = {15+5*j} V')
#        r = ion.ion_fit(15 + 5 * i, 15 + 5 * j, 'CO2')
#        results.append(-r[0, 0] / r[0, 1])
# 
# print(results)


ion.ion_fit(22, 36, 'N')
ion.ion_fit(15, 55, 'N')
ion.ion_fit(20, 38, 'Ar')
ion.ion_fit(18, 40, 'Ar')
ion.ion_fit(15, 55, 'Ar')

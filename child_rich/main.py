import numpy as np
import child_fit as child
import richardson_fit as rich
import print_child_data as cd
import richardson_data as rd

'''cd.plot_child_data()
rd.plot_richardson_data()

# just a few cases
for i in range(3):
    child.child_fit(V_min=0, V_max=20, temp=i+1, func='V')
    child.child_fit(V_min=3, V_max=20, temp=i+1, func='V')
    child.child_fit(V_min=0, V_max=13, temp=i+1, func='V')
    child.child_fit(V_min=3, V_max=13, temp=i+1, func='V')'''

# plotting richardson results
rich.richardson_fit()
'''

child.child_fit(V_min=3, V_max=20, temp=1, func='V')
child.child_fit(V_min=3, V_max=17, temp=2, func='V')
child.child_fit(V_min=3, V_max=13, temp=3, func='V')

# print(cd.V_g1)
# print(cd.delta_V_g1)
# print(cd.delta_V_g1/cd.V_g1)'''
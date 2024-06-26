import numpy as np
import child_fit as child
import richardson_fit as rich
import print_child_data as cd
import richardson_data as rd

cd.plot_child_data()
rd.plot_richardson_data()

# plotting child results, fit with odr
# all 3 different temperatures, picking V_cutoff at 3 and 13 V

for i in range(3):
    child.child_fit(3, i+1)
    child.child_fit(13, i+1)


# plotting richardson results
rich.richardson_fit()
rich.richardson_fit(True)
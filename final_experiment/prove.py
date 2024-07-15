import pandas as pd
from myfunctions import determine_t0_fmod_function

df = pd.read_csv(
    'final_experiment/forcenoise_DC1A_100mHz15V_105_v.dat', delimiter='\t')

time = df['Time'].to_numpy()
channel3 = df['Channel3'].to_numpy()

determine_t0_fmod_function(time, channel3, 0.1, 10, True)

import reading_file as rd

filenames1 = ['noise1_51_v.dat']
filenames2 = ['noise2_53_v.dat']

# for i in range(4, 14):
#     filenames.append(f'AC_measurement_{i}_v.dat')

for item in filenames1:
    rd.first_line(item, 'final_experiment/data/noise_16afternoon/')

for item in filenames2:
    rd.first_line(item, 'final_experiment/data/noise_16afternoon/')

rd.file_segmentation(4726706.538+120, filenames1, 2073.101-120, 1, 'noise1', 'final_experiment/data/noise_16afternoon/')
rd.file_segmentation(4728902.438+120, filenames2, 2261.901-120, 1, 'noise2', 'final_experiment/data/noise_16afternoon/')


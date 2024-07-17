import reading_file as rd

filenames = []

for i in range(6, 8):
    filenames.append(f'noise5_{i}_v.dat')

for item in filenames:
    rd.first_line(item, 'final_experiment/data/noise_17morning/')

rd.file_segmentation(t_start=4734228.339, files=filenames, time_block=8400, blocks=6,
                       filename='ac_second', path='final_experiment/data/noise_17morning/')

rd.file_segmentation_1(t_start=4790716.167, t_finish=4792565.668, files=filenames, delay_start=120, delay_finish=120,
                       filename='noise5', path='final_experiment/data/noise_17morning/')
import reading_file as rd

filenames = []

for i in range(20, 37):
    filenames.append(f'data_{i}_v.dat')

for item in filenames:
    rd.first_line(item, 'final_experiment/data/overnight22/')

rd.file_segmentation(t_start=5243593.652, files=filenames, time_block=7200, blocks=8,
                       filename='overnight22_', path='final_experiment/data/overnight22/')

# rd.file_segmentation_1(t_start=4776228.437, t_finish=4785304.039, files=filenames, delay_start=0, delay_finish=1800,
#                        filename='ac_second6_', path='final_experiment/data/ACmeasurement_secondbatch/')
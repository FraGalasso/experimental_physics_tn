import pandas as pd


def first_line(filename, path='final_experiment/'):
    new_line = 'Time\tChannel1\tChannel2\tChannel3\n'

    with open(path+filename, 'r') as file:
        lines = file.readlines()

    lines[:3] = [new_line]

    with open(path+filename, 'w') as file:
        file.writelines(lines)


def file_segmentation(t_start, files, time_block, blocks, filename='file', path='final_experiment/'):
    '''Reads data from files and rearranges them into time blocks in separate files.
    files is a list with the names of the files to be analysed, time_block is the length of each block,
    block is the number of blocks, filename is the name of the final files, path is where to read and
    save the files.'''

    files = [path + item for item in files]
    df = pd.concat((pd.read_csv(item, delimiter='\t')
                   for item in files), ignore_index=True)

    for i in range(blocks):
        filtered_df = df[(df['Time'] > t_start + i * time_block)
                         & (df['Time'] < t_start + (i+1) * time_block)]
        filtered_df.to_csv(path + filename + str(i+1) + '.csv', index=False)


def file_segmentation_1(t_start, t_finish, files, delay_start=0, delay_finish=0, filename='file', path='final_experiment/'):

    file_segmentation(t_start=t_start+delay_start, files=files,
                      time_block=t_finish-t_start-delay_finish, blocks=1, filename=filename, path=path)

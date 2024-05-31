from utilities import read_data, plot_spectrum

led_data1 = read_data('led_spectrum1.csv')
led_data2 = read_data('led_spectrum2.csv')
plot_spectrum(led_data1)
plot_spectrum(led_data2, 'Calibrated plot', 'data')

'''I noticed we have a small tail of the plot with data with the potentiometer
going backwards. For now I left it in the plot, in case we want to remove this
part, the magic numbers are 15709 for led 1 and 15866 for led 2.'''


final_data = read_data('backwards_highnoise.csv')
plot_spectrum(final_data, 'Ar - Kr spectrum', filename='backwards_highnoise.png')
plot_spectrum(final_data, 'Ar - Kr spectrum', filename='backwards_highnoise.pdf')

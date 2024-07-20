import t0_AB
import t0_ABCD
import matplotlib.pyplot as plt
import conversion as conv


def plot_parsgraph(df_freq, pars_list, stds_list, FS):
    # Extract lists from lists of lists
    a_values, a_std, b_values, b_std, c_values, c_std, d_values, d_std = t0_ABCD.extract_values(
        df_freq, pars_list, stds_list)

    # Convert into real volts using compvolt_to_realvolt function
    a_values = conv.compvolt_to_realvolt(a_values, FS)
    a_std = conv.compvolt_to_realvolt(a_std, FS)
    b_values = conv.compvolt_to_realvolt(b_values, FS)
    b_std = conv.compvolt_to_realvolt(b_std, FS)
    c_values = conv.compvolt_to_realvolt(c_values, FS)
    c_std = conv.compvolt_to_realvolt(c_std, FS)
    d_values = conv.compvolt_to_realvolt(d_values, FS)
    d_std = conv.compvolt_to_realvolt(d_std, FS)

    # Now plot
    plt.figure(dpi=200)
    plt.errorbar(df_freq, a_values, yerr=a_std, linestyle='None',
                 marker='.', label='$A_{sin f}$')
    plt.errorbar(df_freq, b_values, yerr=b_std, linestyle='None',
                 marker='.', label='$B_{cos f}$')
    plt.errorbar(df_freq, c_values, yerr=c_std, linestyle='None',
                 marker='.', label='$C_{sin 2f}$')
    plt.errorbar(df_freq, d_values, yerr=d_std, linestyle='None',
                 marker='.', label='$D_{cos 2f}$')
    plt.xscale('log')
    plt.title('Voltage amplitudes as a function of frequency')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Voltage amplitude coefficients [V]')
    plt.legend()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-6, -6))
    plt.grid()
    plt.tight_layout()
    plt.savefig('graphs/ABCDplot_volts.png')
    plt.close()


def plot_forcegraph(df_freq, pars_list, stds_list, FS):
    # Extract lists from lists of lists
    a_values, a_std, b_values, b_std, c_values, c_std, d_values, d_std = t0_ABCD.extract_values(
        df_freq, pars_list, stds_list)

    # Convert to force using conv.compvolt_to_force function
    a_values = conv.compvolt_to_force(a_values, FS)
    a_std = conv.compvolt_to_force(a_std, FS)
    b_values = conv.compvolt_to_force(b_values, FS)
    b_std = conv.compvolt_to_force(b_std, FS)
    c_values = conv.compvolt_to_force(c_values, FS)
    c_std = conv.compvolt_to_force(c_std, FS)
    d_values = conv.compvolt_to_force(d_values, FS)
    d_std = conv.compvolt_to_force(d_std, FS)

    # Now plot
    plt.figure(dpi=200)
    plt.errorbar(df_freq, a_values, yerr=a_std, linestyle='None',
                 marker='.', label='$A_{sin f}$')
    plt.errorbar(df_freq, b_values, yerr=b_std, linestyle='None',
                 marker='.', label='$B_{cos f}$')
    plt.errorbar(df_freq, c_values, yerr=c_std, linestyle='None',
                 marker='.', label='$C_{sin 2f}$')
    plt.errorbar(df_freq, d_values, yerr=d_std, linestyle='None',
                 marker='.', label='$D_{cos 2f}$')
    plt.xscale('log')
    plt.title('Force amplitudes as a function of frequency')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Force amplitude coefficients [N]')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-6, -6))
    plt.savefig('graphs/ABCDplot_force.png')
    plt.close()


def generate_plot_A_B(xmean_vec, A_vec, B_vec, is_force=True):
    '''Generates plot of A B, assumes data to be already in force units,
    if not use is_force=False.'''
    plt.figure()
    plt.plot(xmean_vec, A_vec, linestyle='None', marker='.', label='$A_{sin}$')
    plt.plot(xmean_vec, B_vec, linestyle='None', marker='.', label='$B_{cos}$')
    # plt.suptitle('Values of force amplitudes A and B of sinusoidal model')
    # A_mean, A_err = stats(A_vec)
    # B_mean, B_err = stats(B_vec)
    # plt.title('A: %f +/- %f, B: %f +/- %f' % (A_mean, A_err, B_mean, B_err))
    plt.xlabel('Interval index [s]')
    if is_force:
        plt.ylabel('Force amplitude coefficients [N]')
        plt.axhline(y=t0_AB.dc_force_estimate(), color='r', linestyle='--',
                    linewidth=2, label='Estimated force')
    else:
        plt.ylabel('Voltage amplitude coefficients [V]')
    plt.tight_layout()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    plt.legend()
    plt.grid()
    # plt.savefig('final_experiment/pictures/calibration/ab_force.pdf')
    plt.show()


def generate_plot_A_B_C_D(freq, xmean_vec, A_vec, B_vec, C_vec, D_vec, is_force=True):
    '''Generates plot of A B C D, assumes data to be already in force units,
    if not use is_force=False.'''
    plt.figure()
    plt.plot(xmean_vec, A_vec, linestyle='None', marker='.', label='$A_{sin}$')
    plt.plot(xmean_vec, B_vec, linestyle='None', marker='.', label='$B_{cos}$')
    plt.plot(xmean_vec, C_vec, linestyle='None',
             marker='.', label='$C_{sin2}$')
    plt.plot(xmean_vec, D_vec, linestyle='None',
             marker='.', label='$D_{cos2}$')
    # A_mean, A_err = t0_ABCD.stats(A_vec)
    # B_mean, B_err = t0_ABCD.stats(B_vec)
    # C_mean, C_err = t0_ABCD.stats(C_vec)
    # D_mean, D_err = t0_ABCD.stats(D_vec)
    # stats_str = ('A: {:.2e} +/- {:.2e}, B: {:.2e} +/- {:.2e}, \n'
    #              'C: {:.2e} +/- {:.2e}, D: {:.2e} +/- {:.2e}').format(
    #     A_mean, A_err, B_mean, B_err, C_mean, C_err, D_mean, D_err)
    # plt.suptitle(stats_str)
    plt.xlabel('Time stamp (s)')
    if is_force:
        plt.ylabel('Force amplitude coefficients [N]')
        plt.title('Force amplitudes A, B, C, D: ' + str(freq) + ' Hz')
    else:
        plt.ylabel('Voltage amplitude coefficients [V]')
        plt.title('Voltage amplitudes A, B, C, D: ' + str(freq) + ' Hz')
    plt.legend()
    plt.tight_layout()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-6, -6))
    plt.grid()
    # plt.savefig('final_experiment/pictures/ABCD_' + str(freq) + '.pdf')
    plt.show()

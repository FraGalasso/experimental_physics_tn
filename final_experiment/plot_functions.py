import t0_AB
import t0_ABCD
import matplotlib.pyplot as plt
import numpy as np
import conversion as conv
from estimate_force import ac_force_estimate, ac_force_estimate_sheet
from brokenaxes import brokenaxes


def plot_forcegraph(df_freq, pars_list, stds_list, is_force=True, plot_expected=True, is_sheet=False):
    '''Generates plot of A and D forces as function of frequencies, assumes data to be already in force units,
    if not use is_force=False. Compares data to expected force values, to deactivate use plot_expected=False.'''
    a_values, a_std, b_values, b_std, c_values, c_std, d_values, d_std = t0_ABCD.extract_values(
        df_freq, pars_list, stds_list)

    # Now plot
    plt.figure()
    plt.errorbar(df_freq, a_values, yerr=a_std, linestyle='None',
                 marker='.', label='$A_{sin f}$')
    plt.errorbar(df_freq, b_values, yerr=b_std, linestyle='None',
                 marker='.', label='$B_{cos f}$')
    plt.errorbar(df_freq, c_values, yerr=c_std, linestyle='None',
                 marker='.', label='$C_{sin 2f}$')
    plt.errorbar(df_freq, d_values, yerr=d_std, linestyle='None',
                 marker='.', label='$D_{cos 2f}$')
    if plot_expected:
        f_AC = np.arange(80, 5000, 1)
        if is_sheet:
            '''exp_A_1 = ac_force_estimate_sheet(0.00026, 0.26, 0.617)
            exp_D_1 = exp_A_1/(-4)
            exp_A_2 = ac_force_estimate_sheet(0.00053, 0.53, 0.617)
            exp_D_2 = exp_A_2/(-4)
            exp_A_3 = ac_force_estimate_sheet(0.00026, 0.53, 0.603)
            exp_D_3 = exp_A_3/(-4)'''
            '''exp_A_4 = ac_force_estimate_sheet(0.00054, 0.54, 0.617)
            exp_D_4 = exp_A_4/(-4)#questo è quello buono'''
            exp_A_5 = ac_force_estimate_sheet(0.00011, 1.479, 0.41)
            exp_D_5 = exp_A_5/(-4)
            
            '''plt.plot(f_AC, exp_A_1, linestyle='-',
                     marker='None', label='A, L=0.26mH, R=0.26$\Omega$')
            plt.plot(f_AC, exp_D_1, linestyle='-',
                     marker='None', label='D, L=0.26mH, R=0.26$\Omega$')
            plt.plot(f_AC, exp_A_2, linestyle='-',
                     marker='None', label='A, L=0.53mH, R=0.53$\Omega$')
            plt.plot(f_AC, exp_D_2, linestyle='-',
                     marker='None', label='D, L=0.53mH, R=0.53$\Omega$')
            plt.plot(f_AC, exp_A_3, linestyle='-',
                     marker='None', label='A, L=0.53mH, R=0.26$\Omega$')
            plt.plot(f_AC, exp_D_3, linestyle='-',
                     marker='None', label='D, L=0.53mH, R=0.26$\Omega$')'''
            '''plt.plot(f_AC, exp_A_4, linestyle='-',
                     marker='None', label='A, L=0.54mH, R=0.54$\Omega$')
            plt.plot(f_AC, exp_D_4, linestyle='-',
                     marker='None', label='D, L=0.54mH, R=0.54$\Omega$')'''
            plt.plot(f_AC, exp_A_5, linestyle='-',
                     marker='None', label='A')
            plt.plot(f_AC, exp_D_5, linestyle='-',
                     marker='None', label='D')
        else:
            exp_A = ac_force_estimate()
            exp_D = exp_A/(-4)
            plt.plot(f_AC, exp_A, linestyle='-',
                     marker='None', label='$A_{sin f}$')
            plt.plot(f_AC, exp_D, linestyle='-',
                     marker='None', label='$D_{sin 2f}$')
    plt.xscale('log')
    plt.xlabel('Frequency [Hz]')
    if is_force:
        plt.ylabel('Force amplitude coefficients [N]')
    else:
        plt.ylabel('Voltage amplitude coefficients [V]')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-6, -6))
    # plt.savefig('final_experiment/pictures/force/force_freq.pdf')
    plt.show()


def plot_exp_force():
    '''Plots just expected values of the force.'''
    '''A_ampl = ac_force_estimate()
    D_ampl = -A_ampl/4'''
    A_ampl = ac_force_estimate_sheet(0.00011, 1.479, 0.41)
    D_ampl = A_ampl/(-4)
    f_AC = np.arange(80, 5000, 1)

    plt.figure()
    plt.plot(f_AC, A_ampl, linestyle='-', marker='None', label='$A_{sin}$')
    plt.plot(f_AC, D_ampl, linestyle='-', marker='None', label='$D_{cos}$')
    plt.xlabel('Frequency')
    plt.ylabel('Force amplitude coefficients [N]')
    plt.xscale('log')
    plt.tight_layout()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-6, -6))
    plt.legend()
    plt.grid()
    plt.show()


def generate_plot_A_B(xmean_vec, A_vec, B_vec, is_force=True):
    '''Generates plot of A B, assumes data to be already in force units,
    if not use is_force=False.'''
    plt.figure()
    '''plt.plot(xmean_vec, A_vec, linestyle='None', marker='.', markersize=8, label='$A_{sin}$')
    plt.plot(xmean_vec, B_vec, linestyle='None', marker='.', markersize=8, label='$B_{cos}$')'''
    bax = brokenaxes(ylims=((-0.25e-3, 0.75e-3), (6e-3, 7.5e-3)), hspace=0.1)
    bax.plot(xmean_vec, A_vec, linestyle='None',
             marker='.', markersize=8, label='$A_{sin}$')
    bax.plot(xmean_vec, B_vec, linestyle='None',
             marker='.', markersize=8, label='$B_{cos}$')
    # plt.suptitle('Values of force amplitudes A and B of sinusoidal model')
    # A_mean, A_err = stats(A_vec)
    # B_mean, B_err = stats(B_vec)
    # plt.title('A: %f +/- %f, B: %f +/- %f' % (A_mean, A_err, B_mean, B_err))
    bax.set_xlabel('Interval timestamp [s]')
    if is_force:
        bax.set_ylabel('Force amplitude coefficients [N]')
        f, df = t0_AB.dc_force_estimate()
        bax.axhline(y=f, color='black', linestyle='-',
                    linewidth=1, label='Estimated force')
        bax.axhspan(f - df, f + df, color='green', alpha=0.3)

    else:
        bax.set_ylabel('Voltage amplitude coefficients [V]')
    bax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    bax.legend()
    bax.grid()
    # plt.savefig('final_experiment/pictures/calibration/ab_force_2.pdf')
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
    plt.xlabel('Interval timestamp (s)')
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
    # plt.savefig('final_experiment/pictures/force/ABCD_' + str(freq) + '.pdf')
    plt.show()

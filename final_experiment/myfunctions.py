import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sin_cos_func(x, A, B, f, x0):
    
    return A*np.sin(2*np.pi*f*(x-x0))+B*np.cos(2*np.pi*f*(x-x0))

def lin_func(x, m, q):
    return m*x+q

def determine_t0_fmod_function(time, channel3, f_MOD, plot, n_divisions):
    n_samples = len(time)
    n_samples_per_division = int(n_samples/n_divisions)

    xmean_vec = np.zeros(n_divisions)
    phi_vec = np.zeros(n_divisions)

    delta = 0

    for i in range(2):
        f_MOD = f_MOD-delta
        def model(x, A, B): return sin_cos_func(x, A, B, f=f_MOD, x0=0)

        for i in range(n_divisions):
            start_point = int(i*n_samples_per_division)
            end_point = int((i+1)*n_samples_per_division)

            x_new = time[start_point:end_point]
            y_new = channel3[start_point:end_point]

            popt, pcov = curve_fit(model, x_new, y_new)

            xmean_vec[i] = np.mean(x_new)
            phi_vec[i] = np.arctan2(-popt[1], popt[0])  # -B/A

        popt, pcov = curve_fit(lin_func, xmean_vec, phi_vec)
        m = popt[0]
        q = popt[1]
        delta = m/(2*np.pi)

        t0 = np.mean(phi_vec)/(2*np.pi*f_MOD)

        print(f'dphi_dt = {m} +/- {np.sqrt(pcov[0,0])}')
        print(f'phi = {np.mean(phi_vec)}')
        print(f't0 = {t0}')
        print(f'f_mod = {f_MOD}')
        print(f'delta f = {delta} +/- {np.sqrt(pcov[0,0])/(2*np.pi)}\n')

        if plot:
            x_fit = np.linspace(time[0], time[-1], 1000)
            y_fit = m * x_fit + q
            plt.figure(dpi=120)
            plt.plot(xmean_vec, phi_vec, linestyle='None',
                     marker='+', label='data of intervals')
            plt.plot(x_fit, y_fit, label='fit')
            plt.xlabel('t [s]')
            plt.ylabel('phi [radians]')
            plt.legend()
            plt.grid()
            plt.show()

    return t0, f_MOD
    
# Determine A and B for a given time interval
def determine_A_B_in_interval(time, channel, t_0, f_MOD):
    model = lambda time, A, B: sin_cos_func(time, A, B, f=f_MOD, x0=t_0)
    popt, _ = curve_fit(model, time, channel)
    A = popt[0]
    B = popt[1]
    return A, B

# Generate plot of A and B
def generate_plot_A_B(xmean_vec, A_vec, B_vec, A_mean, A_std, B_mean, B_std):
    plt.figure(dpi=120)
    plt.plot(xmean_vec, A_vec, linestyle='None', marker='+', label='$A_{sin}$')
    plt.plot(xmean_vec, B_vec, linestyle='None', marker='+', label='$B_{cos}$')
    plt.title('Values of force amplitudes A and B of sinusoidal model')
    plt.suptitle('A: %f +/- %f, B: %f +/- %f' % (A_mean, A_std, B_mean, B_std))
    plt.xlabel('Interval index')
    plt.ylabel('Force amplitude coefficients')
    plt.legend()
    plt.grid()
    plt.show()

def determine_A_B_func(time, channel, f_MOD, t_0, plot, n_divisions,):
    # Generate intervals
    n_samples = len(time)
    n_samples_per_division = int(n_samples / n_divisions)  # Here we are always rounding down, so we will cut out some data points at the end

    # Initialize lists to store A and B values
    A_vec = []
    B_vec = []
    xmean_vec = []

    # Find A, B for each interval and append to vectors A_vec and B_vec
    for i in range(0, n_divisions):
        start_point = int(i * n_samples_per_division)
        end_point = int((i + 1) * n_samples_per_division)

        time_int = time[start_point:end_point]
        channel_int = channel[start_point:end_point]

        A, B = determine_A_B_in_interval(time_int, channel_int, t_0, f_MOD)

        A_vec.append(A)
        B_vec.append(B)
        xmean_vec.append(np.mean(time_int))

    # Calculate mean and std devs and print
    A_mean = np.mean(A_vec)
    A_std = np.std(A_vec)
    B_mean = np.mean(B_vec)
    B_std = np.std(B_vec)

    print('A: %f +/- %f' % (A_mean, A_std))
    print('B: %f +/- %f' % (B_mean, B_std))

    # Plot the values
    if plot:
        generate_plot_A_B(xmean_vec, A_vec, B_vec, A_mean, A_std, B_mean, B_std)

    return A_mean, A_std, B_mean, B_std

def fft_coeff_and_PSD_funct(x, Fs, window):
    
    # dt=1/Fs
    n=len(x)
    
    # freq=(1/(n*dt))*np.arange(n)
    
    if window=='boxcar':
        f_hat=np.fft.fft(x, n)
        PSD=2*np.real(f_hat*np.conj(f_hat))/(Fs*n)
        PSD[0]=PSD[0]/2
        
    elif window=='blackmanharris':
        w_n=Blackman_Harris_wind_func(n)
        x=x*w_n
        f_hat=np.fft.fft(x, n)
        
        norm=sum(w_n**2)
        PSD=2*np.real(f_hat*np.conj(f_hat))/(Fs*norm)
        PSD[0]=PSD[0]/2
        
    elif window=='hann':
        w_n=Hann_wind_func(n)
        x=x*w_n
        f_hat=np.fft.fft(x, n)
        
        norm=sum(w_n**2)
        PSD=2*np.real(f_hat*np.conj(f_hat))/(Fs*norm)
        PSD[0]=PSD[0]/2
    
    return  f_hat[:n//2], PSD[:n//2]
    
def Blackman_Harris_wind_func(N_samples):
    
    a0=0.35875
    a1=0.48829
    a2=0.14128
    a3=0.01168

    n=np.arange(N_samples)
    phi=2*np.pi*n/N_samples
    w_n=a0-a1*np.cos(phi)+a2*np.cos(2*phi)-a3*np.cos(3*phi)
    
    return w_n

def Hann_wind_func(N_samples):
    
    n=np.arange(N_samples)
    phi=2*np.pi*n/(N_samples-1)
    w_n=0.5*(1-np.cos(phi))
    
    return w_n

def average_PSD(x, Fs, window, nperseg):
    
    n_samples=len(x)
    n_samples_per_division=nperseg
    n_divisions=int(n_samples/n_samples_per_division)
    
    freq=(Fs/n_samples_per_division)*np.arange(n_samples_per_division)
    freq=freq[:n_samples_per_division//2]
    print('Number samples: %d' %n_samples)
    print('Number divisions: %d' %n_divisions)
    print('Number samples per divisions: %d\n' %n_samples_per_division)
    j=0
    PSD_sum=np.zeros(n_samples_per_division//2)
    for i in range(0, int((2*n_divisions)-1)):
        start_point=int((i/2)*n_samples_per_division)
        end_point=int((1+(i/2))*n_samples_per_division)
        
        x_new=x[start_point:end_point]
        f_hat, PSD=fft_coeff_and_PSD_funct(x_new, Fs, window=window)
        PSD_sum=PSD_sum+PSD
        j=j+1
    
    PSD_ave=PSD_sum/(j+1)
    
    return freq, PSD_ave





def do_graphs(nrows, ncols, comb, X, Y, plot_params, graph_settings, ngraph=None):
    
    fig, ax = plt.subplots(nrows,ncols, num=ngraph, dpi=120)
    
    if nrows==1 and ncols==1:
        n=0
        if comb[0]==1:
            ax.plot(X, Y, linestyle=plot_params[0], marker=plot_params[1], color=plot_params[2], label=plot_params[3])
        else:
            for l in range(comb[0]):
                ax.plot(X, Y[n], linestyle=plot_params[n][0], marker=plot_params[n][1], color=plot_params[n][2], label=plot_params[n][3])
                n=n+1
        ax.set_xlabel(graph_settings[0][0])
        ax.set_ylabel(graph_settings[0][1])
        if graph_settings[0][2]==None:
            ax.legend()
        else:
            ax.legend(loc=graph_settings[0][2])
        
        ax.set_xscale(graph_settings[0][3])
        ax.set_yscale(graph_settings[0][4])
        ax.grid()
        
        fig.suptitle(graph_settings[1][0], fontsize=graph_settings[1][1])
    
    elif (nrows>1 and ncols==1) or (nrows==1 and ncols>1):
        n=0
        m=0
        for i in range(max(nrows, ncols)):
            for l in range(comb[m]):
                ax[i].plot(X[m], Y[n], linestyle=plot_params[n][0], marker=plot_params[n][1], color=plot_params[n][2], label=plot_params[n][3])
                n=n+1
            ax[i].set_xlabel(graph_settings[m][0])
            ax[i].set_ylabel(graph_settings[m][1])
            if graph_settings[m][2]==None:
                ax[i].legend()
            else:
                ax[i].legend(loc=graph_settings[m][2])
            
            ax[i].set_xscale(graph_settings[m][3])
            ax[i].set_yscale(graph_settings[m][4])
            ax[i].grid()
            m=m+1
        
        fig.suptitle(graph_settings[m][0], fontsize=graph_settings[m][1])
        
    else: 
        m=0
        n=0
        for i in range(nrows):
            for k in range(ncols):
                for l in range(comb[m]):
                    ax[i,k].plot(X[m], Y[n], linestyle=plot_params[n][0], marker=plot_params[n][1], color=plot_params[n][2], label=plot_params[n][3])
                    n=n+1
                ax[i,k].set_xlabel(graph_settings[m][0])
                ax[i,k].set_ylabel(graph_settings[m][1])
                # if graph_setting[m][2]=!None:
                    # ax[i,k].set_xlimit
                if graph_settings[m][2]==None:
                    ax[i,k].legend()
                else:
                    ax[i,k].legend(loc=graph_settings[m][2])
                
                ax[i,k].set_xscale(graph_settings[m][3])
                ax[i,k].set_yscale(graph_settings[m][4])
                ax[i,k].grid()
                m=m+1
        
        fig.suptitle(graph_settings[m][0], fontsize=graph_settings[m][1])




class Circuit:
    def __init__(self, V_PP_str_gau_value, f_str_gau_value, V_PP_source_value, f_source_value, i_0_rec_value, 
                 time_constant_value, FS_value, t_values, V_CH1_values, V_CH2_values, V_CH3_values):
        self.load_cell_constants={}
        self.load_cell_constants["R_SG [Ohm]"]=1*10**3                      #Ohm strain gauge resistance
        self.load_cell_constants["r_SG [Ohm]"]=25.0                           #Ohm input resistance
        self.load_cell_constants["F_FS [N]"]=6.0                              #N nominal full scale (balance range)
        self.load_cell_constants["alpha [1mv/V(per F_FS)]"]=1*10**(-3)      #1mv/V(per F_FS) nominal sensitivity
        self.load_cell_constants["V_BIAS [V]"]=10.0                           #V recommended bias
        
        self.coils={}
        self.coils["N"]=84                                                  #windings of both source and receiver coils 
        self.coils["phi_Cu [m]"]= 560*10**(-6)                              #m metallic wire diameter 
        self.coils["phi_TOT  [m]"]= 650*10**(-6)                            #m metallic wire diameter
        self.coils["D_B [m]"]=235*10**(-3)                                  #m coild internal diameter
        self.coils["n_z"]=7                                                 #windings per layer
        self.coils["n_R"]=12                                                #number of winding layers
        self.coils["R_L [Ohm]"]=4.5                                         #Ohm resistance
        self.coils["L [H]"]=4.3*10**(-3)                                    #Henry  inductance
        self.coils["d [m]"]=1.1*10**(-2)                                    #m distance between coils
        self.coils["lambda_fact"]=0.95
        
        self.strain_gauge_parameters={}
        self.strain_gauge_parameters["V_PP_str_gau [V]"]=V_PP_str_gau_value
        self.strain_gauge_parameters["V_0P_str_gau [V]"]=self.strain_gauge_parameters["V_PP_str_gau [V]"]/2
        self.strain_gauge_parameters["f_str_gau [Hz]"]=f_str_gau_value
        
        self.source_parameters={}
        self.source_parameters["V_PP_source [V]"]=V_PP_source_value
        self.source_parameters["V_0P_source [V]"]=self.source_parameters["V_PP_source [V]"]/2
        self.source_parameters["f_source [Hz]"]=f_source_value
        self.source_parameters["R_source [Ohm]"]=50
        self.source_parameters["Z_source [Ohm]"]=self.source_parameters["R_source [Ohm]"]+self.coils["R_L [Ohm]"]+1j*(2*np.pi*self.source_parameters["f_source [Hz]"])
        self.source_parameters["i_0_source [A]"]=np.real(self.source_parameters["V_0P_source [V]"]/self.source_parameters["Z_source [Ohm]"])
        
        self.receiver_gauge_parameters={}
        self.receiver_gauge_parameters["i_0_rec [A]"]=i_0_rec_value
        
        
        k1=self.source_parameters["i_0_source [A]"]*self.receiver_gauge_parameters["i_0_rec [A]"]
        k2=((self.coils["N"])**2)*np.pi*self.coils["D_B [m]"]/self.coils["d [m]"]
        
        self.output={
            "F_sampling [Hz]": 10,
            "F_L_estimate [N]": 0.2*10**(-6)*k1*k2*self.coils["lambda_fact"],
            "lock_in_time_constant [s]": time_constant_value,
            "lock_in_FS [V]": FS_value,
            "t [s]": t_values,
            "V_CH1 [V]": V_CH1_values,
            "V_CH1_lock_in [V]": 0,
            "V_CH2 [V]": V_CH2_values,
            "V_CH2_lock_in [V]": 0,
            "V_CH3 [V]": V_CH3_values
            }
        self.output["N_samples"]=len(self.output["t [s]"])
        self.output["V_CH1_lock_in [V]"]=np.sqrt(2)*self.output["V_CH1 [V]"]*self.output["lock_in_FS [V]"]/10.0
        self.output["V_CH2_lock_in [V]"]=np.sqrt(2)*self.output["V_CH2 [V]"]*self.output["lock_in_FS [V]"]/10.0
        self.output["dV_dF [V/N]"]=self.load_cell_constants["alpha [1mv/V(per F_FS)]"]*self.strain_gauge_parameters["V_0P_str_gau [V]"]/self.load_cell_constants["F_FS [N]"]
        self.output["F_t [N]"]=self.output["V_CH1_lock_in [V]"]/self.output["dV_dF [V/N]"]
        self.output["F_QUAD_t [N]"]=self.output["V_CH2_lock_in [V]"]/self.output["dV_dF [V/N]"]
        
        T_s=1/self.output["F_sampling [Hz]"]
        self.output["frequencies [Hz]"]=(1/(self.output["N_samples"]*T_s))*np.arange(self.output["N_samples"])[0:self.output["N_samples"]//2]
        # self.output["L"]=np.arange(1, np.floor(self.output["N_samples"]/2), dtype='int')
        
        self.output["fhat_F [N/Hz]"], self.output["S_F [N^2/Hz]"]=fft_coeff_and_PSD_funct(x=self.output["F_t [N]"], Fs=self.output["F_sampling [Hz]"], window='boxcar')
        self.output["fhat_F_QUAD [N/Hz]"], self.output["S_F_QUAD [N^2/Hz]"]=fft_coeff_and_PSD_funct(x=self.output["F_QUAD_t [N]"], Fs=self.output["F_sampling [Hz]"], window='boxcar')
        self.output["fhat_V_CH1_lock_in [V/Hz]"], self.output["S_V_CH1_lock_in [V^2/Hz]"]=fft_coeff_and_PSD_funct(x=self.output["V_CH1_lock_in [V]"], Fs=self.output["F_sampling [Hz]"], window='boxcar')
        self.output["fhat_V_CH2_lock_in [V/Hz]"], self.output["S_V_CH2_lock_in [V^2/Hz]"]=fft_coeff_and_PSD_funct(x=self.output["V_CH2_lock_in [V]"], Fs=self.output["F_sampling [Hz]"], window='boxcar')
        
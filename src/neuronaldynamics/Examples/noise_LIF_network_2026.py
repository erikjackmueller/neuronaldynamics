import numpy as np
import matplotlib
from Model.LIF import Conductance_LIF, LIF_basic
from Model.Nykamp_Model import Nykamp_Model_1, FPE_Population
from Utils import compare_firing_rate
matplotlib.use('TkAgg')

########################################################################################################################
# LIF model #
########################################################################################################################

# set-up time and input model
def v0(t):
    v0_bar = 700  # 700 spikes / second / 1000 to convert to ms
    f = 10
    return (v0_bar/1000) * (1 + np.sin(2*np.pi*f/1000*t))

T = 20
# dt_LIF = 0.001
dt_LIF = 0.001
t_LIF = np.arange(0.0, T, dt_LIF)
dim = 1000

def sine(t):
    f = 10
    i_ext_0 = 1e-1  # 200µA / 10 mS  =  20mV input
    return i_ext_0 * (1 + np.sin(2*np.pi*f/1000*t))

def two_step(t):
    t1 = 0
    t2 = 15
    t3 = 30
    t4 = 45
    i_0 = 1.0e-9 # 1nA
    res = np.zeros_like(t)
    res[t > t1] = i_0
    res[t > t2] = 0
    res[t > t3] = i_0
    res[t > t4] = 0
    return res


def step(t):
    t1 = 0
    t2 = 15
    i_0 = 1.3e-9 # 1nA
    res = np.zeros_like(t)
    res[t > t1] = i_0
    res[t > t2] = 0
    return res

i_ext_vals = step(t_LIF)
# i_ext_vals = step(t)
i_ext_vals_n_neuron = i_ext_vals.repeat(dim).reshape(t_LIF.shape[0], dim).T
sigma = 0.5
neuron_parameters = {'T': T, 'dt':dt_LIF, 'tau_m': 12, 't_ref': 0, 'n_neurons': dim, 'Iinj': i_ext_vals_n_neuron, 'init_pdf_sigma':1,
                     'noise_sigma': sigma, 'V_reset': -70, 'E_rest':-70, 'g_r': 9.167e-9}
lif = LIF_basic(parameters=neuron_parameters)

lif.run()

# # visualize
# lif.plot_volt_trace(idx=3, population_idx=2)
# # lif.raster_plot()
# lif.plot_populations(bins=1000, smoothing=True, sigma=100, hide_refractory=True, cutoff=10, size=1.0, ylims=(-70, -55))

########################################################################################################################
# FPE model #
########################################################################################################################

dt_nmm = 0.01
t_NMM = np.arange(0, T, dt_nmm)
# input_current_A = two_step(t_NMM)
input_current_A = step(t_NMM)
# good fit: 'current_factor': 0.605e-6, 'voltage_sigma': 4
nmm_parameters = {'T': T, 'dt': dt_nmm, 'dv': 0.01, 'tau_mem': [12], 'tau_ref': [0], 'voltage_sigma':  sigma,
                  'input_function': input_current_A, 'input_type': 'stochastic-current', 'solver': 'hu-2021',
                  'current_factor': 1, 'static_noise': True, 'connectivity_matrix': np.array([[0]]),
                  'init_pdf_weight': 0, 'init_pdf_sigma': 1, 'init_pdf_offset': 0, 'flux_scheme': 'bernoulli',
                  'time_scheme': 'bdf2'}

NMM = FPE_Population(parameters=nmm_parameters, name='FPE')
NMM.simulate()
# NMM.plot(heat_map=True, z_limit=0.3, animate=False, savefig=False)
r_LIF_time = np.interp(t_LIF, t_NMM, NMM.r[0])
NMM.r = [r_LIF_time]
NMM.clean()
NMM.t = t_LIF
NMM.save_sim_results(r=NMM.r, rho=np.array([0]), rho_plot=np.array([0]))

compare_firing_rate('FPE', 'basic_LIF', dt=dt_LIF, sigma=100)
# Conductance_LIF.clean()
NMM.clean()





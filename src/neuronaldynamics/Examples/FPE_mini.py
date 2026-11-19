"""
Example of the minimal Fokker-Planck model FPE_mini (Model/Nykamp_Model.py), the original structure preserving scheme
of Hu, Liu, Xie and Zhou (J. Comput. Phys. 433:110195, 2021) in the formulation of the first Hu-2021 solver
(Maxwellian and harmonic mean as values, sparse solver), for coupled LIF populations. First order in dt and dv.
FPE_mini takes the parameter dict of FPE_Population (the current and more complex solver), so tests written for
FPE_mini run in FPE_Population without changes: FPE_Population(parameters, name).simulate().

Test cases (switched on with SINGLE_POPULATION):
1. FPE_mini: one population with a step input current
2. FPE_mini_3pop: three coupled populations (exc 1 -> exc 2, exc 1 -> inh, inh -> exc 1 and exc 2), each with its own
   mean input current and noise
For every case: FPE_mini simulation (results in <name>.hdf5, format of FPE_Population.save_sim_results), the density
p(v, t) and the rates plotted with the plot of FPE_Population, and optionally (COMPARE_LIF) a LIF network with the same
neurons, inputs, noise and coupling (LIF_basic.run_low_mem, N_LIF neurons per population): the nrmse (Utils.nrmse) of
its rates against FPE_mini per population is printed, the rates of both are plotted in <name>_rates.png (LIF rates
smoothed with a Gaussian of SMOOTH ms).

Usage (from src/neuronaldynamics):  python Examples/FPE_mini.py
"""
import sys
from pathlib import Path
from tabnanny import verbose

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tables import parameters

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/neuronaldynamics
from Model.Nykamp_Model import FPE_mini, FPE_Population
from Model.LIF import LIF_basic
from Utils import nrmse
matplotlib.use('TkAgg')

SINGLE_POPULATION = False # option to run either single population or 3 population example
COMPARE_LIF = True  # also simulate the LIF network and compare the firing rates
N_LIF = 100000  # neurons per population of the LIF network
SMOOTH = 5  # width (ms) of the Gaussian smoothing of the LIF rates

T, DT = 100., 0.1  # simulated time (ms), time step (ms)
t = np.arange(0, T, DT)
step = np.where((t >= 10.) & (t < 60.), 0.12e-9, 0.)  # step input current (A) from 10 to 60 ms

# parameter dict of FPE_Population, used unchanged by FPE_mini
PARAMS_1POP = dict(T=T, dt=DT, dv=0.1, population_type=['exc'],
                   tau_mem=20., c_mem=1.1e-10,  # membrane time constant (ms) and capacitance (F)
                   u_rest=-70., u_reset=-70., u_thr=-55., u_inh=-90.,  # rest, reset, threshold, lower grid end (mV)
                   voltage_sigma=5.,  # noise of the input (mV), can minimally go to 3mV with this solver
                   input_function=step,  # input current (A), (Nt,): into the populations input_function_idx
                   input_function_idx=[0],
                   connectivity_matrix=[[0.]],  # self-feedback weight (mV ms), 0: no feedback
                   p_save_every=10,
                   verbose=True)

# three populations: every population has its own mean current (rows of input_function, A) and noise (mV); the
# connectivity matrix W[k, j] (mV ms) is the coupling from population k to j, < 0 inhibitory
PARAMS_3POP = dict(PARAMS_1POP, dv=0.2, population_type=['exc', 'exc', 'inh'],
                   input_function=np.array([step, np.zeros_like(t), np.zeros_like(t)]),
                   voltage_sigma=[5., 3., 4.],
                   connectivity_matrix=np.array([[0., 500., 500.],  # exc 1 -> exc 2, inh
                                                 [0., 0., 0.],
                                                 [-1e3, -1e3, 0.]])) # inh -> exc 1, exc 2
if SINGLE_POPULATION:
    params = PARAMS_1POP
    name = 'FPE_1pop'
else:
    params = PARAMS_3POP
    name = 'FPE_3pop'


fpe = FPE_mini(params, name=name)
fpe.simulate()
FPE_Population(params, name=name).plot(heat_map=True)  # plots <name>.hdf5 written by FPE_mini through parent class FPE_population plotting tool
rates = {'FPE_mini': fpe.r * 1000}  # conversion from 1/ms to 1/s

if COMPARE_LIF:
    # LIF network with the neurons (g_r = c_mem / tau), inputs, noise and coupling of FPE_mini
    lif = LIF_basic(dict(T=T, dt=DT, tau_m=params['tau_mem'], g_r=params['c_mem'] / (params['tau_mem'] * 1e-3),
                         E_rest=params['u_rest'], V_reset=params['u_reset'], V_th=params['u_thr'],
                         V_init=params['u_rest'], init_pdf_sigma=1, t_ref=0., n_neurons=N_LIF,
                         n_populations=fpe.n_populations, population_type=params['population_type'],
                         Iinj=fpe.i_ext, noise_sigma=fpe.sigma[:, :1],
                         weights=np.asarray(params['connectivity_matrix']), fname=name + '_LIF', verbose=True))
    lif.run_low_mem(save=False)
    r_lif = lif.pop_rate
    r_lif_smooth = gaussian_filter1d(r_lif, sigma=SMOOTH / DT, axis=1)
    rates[f'LIF network ({N_LIF} per population)'] = r_lif
    errors = [nrmse(rates['FPE_mini'][j], r_lif[j]) for j in range(fpe.n_populations)]
    print(f'{name}: nrmse of the LIF network rates against FPE_mini per population: {np.round(errors, 3)}')

# PLOT COMPARISON OF RATES
fig, axes = plt.subplots(fpe.n_populations, 1, figsize=(8, 2.6 * fpe.n_populations), sharex=True, squeeze=False)
for j, ax in enumerate(axes[:, 0]):
    for (label, r), style in zip(rates.items(), ['-', ':']):
        ax.plot(t, r[j], style, label=label)
    ax.set_title(f'population {j} ({params["population_type"][j]})')
    ax.set_ylabel('rate (Hz)')
    ax.grid()
axes[0, 0].legend()
axes[-1, 0].set_xlabel('t (ms)')
plt.tight_layout()
plt.savefig(name + '_rates.png')
plt.show()

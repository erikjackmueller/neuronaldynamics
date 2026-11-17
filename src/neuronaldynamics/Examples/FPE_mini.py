"""
Example of the minimal Fokker-Planck model FPE_mini (Model/Nykamp_Model.py), the original structure preserving scheme
of Hu, Liu, Xie and Zhou (J. Comput. Phys. 433:110195, 2021) in the formulation of the first Hu-2021 solver
(Maxwellian and harmonic mean as values, sparse solver), for one LIF population with optional self-feedback.
First order in dt and dv; FPE_Population is the current and more complex solver.

1. one FPE_mini simulation of T = 100 ms with a step input current, plotted with the plot of FPE_Population (density
   p(v, t) as heat map and firing rate); the results are written to FPE_mini.hdf5 in the format of
   FPE_Population.save_sim_results for that
2. optional (COMPARE_LIF): a LIF network with the same neurons, input and noise (LIF_basic.run_low_mem), and the
   comparison of both firing rates with Utils.compare_firing_rate, as in Examples/noise_LIF_network_2026.py

Usage (from src/neuronaldynamics):  python Examples/solver_tests/FPE_mini.py
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/neuronaldynamics
from Model.Nykamp_Model import FPE_mini, FPE_Population
from Model.LIF import LIF_basic
from Utils import compare_firing_rate
matplotlib.use('TkAgg')

TAU, U_REST, U_THR, U_MIN = 20., -70., -55., -90. # membrane time constant, resting potnetial, threshold voltage and minimal voltage of grid (mV)
C_MEM = 1.1e-10 # membrane capacitance (F)
SIGMA = 5.0 # noise of input voltage (mV)
T, DT, DV = 100., 0.01, 0.01 # simulated time (ms), time step (ms) and voltage step (mV)
AMP, STEP_ON, STEP_OFF = 0.12e-9, 10., 60. # step input current (A), on from STEP_ON to STEP_OFF (ms)
W = 0. # self-feedback weight (mV ms), 0: no feedback
COMPARE_LIF = True # also simulate the LIF network and compare the firing rates
N_LIF = 10000 # neurons of the LIF network


def step(t):
    """
    Create a step as input function
    """
    return np.where((t >= STEP_ON) & (t < STEP_OFF), AMP, 0.)


# FPE simulation
fpe = FPE_mini(T=T, dt=DT, dv=DV, tau=TAU, u_rest=U_REST, u_reset=U_REST, u_thr=U_THR, u_min=U_MIN, sigma=SIGMA,
               c_mem=C_MEM, weights=W, save_every=10)
fpe.simulate(step)

# results in the format of FPE_Population.save_sim_results, plotted with FPE_Population.plot
with h5py.File('FPE_mini.hdf5', 'w') as h5file:
    h5file.create_dataset('t', data=fpe.t)
    h5file.create_dataset('v', data=fpe.v)
    h5file.create_dataset('r', data=fpe.r[np.newaxis])
    h5file.create_dataset('in', data=step(fpe.t)[np.newaxis])
    h5file.create_dataset('rho', data=fpe.p.T[np.newaxis])
    h5file.create_dataset('rho_plot', data=fpe.p.T[np.newaxis])
    h5file.create_dataset('t_rho', data=fpe.t_p)
    h5file.create_dataset('p_types', data=['exc'])
FPE_Population(dict(T=T, dt=DT, dv=DV), name='FPE_mini').plot(heat_map=True)

if COMPARE_LIF:
    # LIF network with the same neurons (g_r = c_mem / tau), input and noise
    lif = LIF_basic(dict(T=T, dt=DT, tau_m=TAU, g_r=C_MEM / (TAU * 1e-3), E_rest=U_REST, V_reset=U_REST, V_th=U_THR,
                         V_init=U_REST, init_pdf_sigma=1, t_ref=0., n_neurons=N_LIF, n_populations=1,
                         population_type=['exc'], Iinj=step(np.arange(0, T, DT))[np.newaxis], noise_sigma=SIGMA,
                         fname='LIF_mini'))
    lif.run_low_mem(save=False)
    # spike counts of the network per time step (format of LIF_basic.run for compare_firing_rate)
    with h5py.File('LIF_mini.hdf5', 'w') as h5file:
        h5file.create_dataset('t', data=lif.t)
        h5file.create_dataset('r', data=lif.pop_rate / 1000 * DT * N_LIF)
    compare_firing_rate('FPE_mini', 'LIF_mini', n_neurons=N_LIF, dt=DT, sigma=10)

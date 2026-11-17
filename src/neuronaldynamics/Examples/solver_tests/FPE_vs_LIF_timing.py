"""
Computation time of the Fokker-Planck population model (FPE_Population, standard solver: Bernoulli flux + BDF2)
vs. a LIF network with 30k, 50k and 100k neurons per population, for

    1 population:   one excitatory population with a step input current
    3 populations:  exc 1 (gets the input), exc 2, inh, coupled with the connectivity matrix W

and five noise levels between 1e-2 and 10 mV. The LIF network with 100k neurons per population is the reference for the
error of the FPE and of the smaller LIF networks (relative L2 error of the smoothed rates of all populations).
With --repeat [N] (default N = 10) every run is repeated N times (LIF with a new seed each time, the error is computed
against the reference of the same repetition); table, csv and figure show the mean (and std) of error and time.

Coupling (as in the Hu-2021 / FPE_Population solver): population j gets the additional drive
sum_k W[k, j] * (alpha * r_k)(t) in mV, with r_k the population rate in 1/ms and W in mV*ms. The LIF network uses the
same mean-field coupling (all-to-all, weights scaled by 1/n, as the alpha-kernel input of LIF.py but from the
population spike count), so both models simulate the same system. The LIF network is simulated with the vectorised
LIF_basic.run_low_mem from Model/LIF.py.

Outputs in Examples/solver_tests/FPE_vs_LIF_timing_results:
    fpe_<setup>_sigma_<sigma>_plot.png      FPE_Population.plot() (rho(v, t) and rates)
    lif_<setup>_sigma_<sigma>_plot.png      Neuron_population.plot_populations() of the 100k reference (voltage
                                            histogram of the first N_REC neurons and rates)
    error_and_time_vs_sigma.png/.pdf        error vs. the 100k reference and computation time against sigma
    timing_table.csv                        all numbers of the printed table

Usage (from src/neuronaldynamics):  python Examples/solver_tests/FPE_vs_LIF_timing.py [--repeat [N]]
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/neuronaldynamics, for direct calls
from Model.Nykamp_Model import FPE_Population
from Model.LIF import LIF_basic
from Utils import nrmse

# after the imports: Utils (imported by Model.LIF) switches to TkAgg, with Agg no windows pop up and plt.show() in the
# plot functions does nothing, so the figures can be saved afterwards
matplotlib.use('Agg')
warnings.filterwarnings('ignore', message='.*non-interactive.*')  # plt.show() in the plot functions with Agg

OUT_DIR = Path(__file__).resolve().parent / 'FPE_vs_LIF_laptop_feedback'
OUT_DIR.mkdir(exist_ok=True)

# neuron model and simulation (identical for FPE and LIF)
TAU, E_REST, V_RESET, V_TH, U_MIN = 12.0, -70.0, -70.0, -55.0, -100.0  # [ms], [mV]
V_INIT, INIT_SIGMA = -70.0, 1.0                 # initial distribution [mV]
G_LEAK = 1.1e-10 / (TAU * 1e-3)                 # [S], as in FPE_Population
T, DT, DV = 50.0, 0.01, 0.01                    # [ms], [ms], [mV]
AMP, STEP_ON, STEP_OFF = 0.5e-9, 5.0, 30.0      # step input current [A] into population 0
SIGMAS = np.logspace(-2, 1, 5)                  # noise [mV]
N_LIST = [30_000, 50_000, 100_000]              # neurons per population, the last one is the reference
RATE_KERNEL_MS = 0.2                            # gaussian kernel [ms] applied to all rates before the comparison
N_REC, REC_EVERY = 1000, 10                     # LIF voltages saved for plot_populations: neurons, time steps

parser = argparse.ArgumentParser()
parser.add_argument('--repeat', type=int, nargs='?', const=10, default=1, metavar='N',
                    help='repeat every run N times (default 10) and average error and time, without: a single run')
N_REPEATS = parser.parse_args().repeat          # repetitions of every run, error and time are averaged

# W[k, j]: from population k to population j [mV*ms]. exc 1 fires at ~240 Hz during the step, so the feed-forward drive
# 120 mV*ms * 0.24 / ms ~ 29 mV is clearly above the 15 mV threshold distance: exc 2 and inh spike also without noise
W_3POP = np.array([[150, 120.0, 120.0],         # exc 1 (input) -> exc 2, inh
                   [120, 50, 120],           # exc 2 -> exc 1, inh
                   [-200, -200, -20]])        # inh -> exc 1, exc 2
SETUPS = {'1pop': (['exc'], np.zeros((1, 1))),
          '3pop': (['exc', 'exc', 'inh'], W_3POP)}

t = np.arange(0, T, DT)
i_input = np.where((t > STEP_ON) & (t <= STEP_OFF), AMP, 0.0)


def make_fpe(pop_types, W, sigma, name, save_p='none'):
    return FPE_Population(parameters=dict(
        T=T, dt=DT, dv=DV, tau_mem=[TAU] * len(pop_types), population_type=pop_types, connectivity_matrix=W,
        voltage_sigma=sigma, static_noise=True, input_function=i_input, input_function_idx=[0],
        init_pdf_sigma=INIT_SIGMA, init_pdf_offset=0, init_pdf_weight=0, u_inh=U_MIN, u_rest=E_REST,
        u_reset=V_RESET, u_thr=V_TH, save_p=save_p, p_save_every=REC_EVERY, tqdm_disable=True),
        name=name)


def make_lif(n, pop_types, W, sigma, kernel, name):
    """LIF network with n neurons in each population, simulated with LIF_basic.run_low_mem"""
    i_inj = np.zeros((len(pop_types), len(t)))
    i_inj[0] = i_input
    return LIF_basic(dict(
        T=T, dt=DT, tau_m=TAU, g_r=G_LEAK, E_rest=E_REST, V_reset=V_RESET, V_th=V_TH, V_init=V_INIT,
        init_pdf_sigma=INIT_SIGMA, t_ref=0.0, n_neurons=n, n_populations=len(pop_types), population_type=pop_types,
        Iinj=i_inj, noise_sigma=sigma, weights=W, delay_kernel=kernel, fname=name))


def v_lims(sigma):
    """voltage axis of the plots: at least -75 mV to threshold, wider for strong noise (stationary std ~ sigma)"""
    return max(U_MIN, min(-75.0, E_REST - 4 * sigma - 2)), V_TH


def plot_lif(lif, sigma):
    """voltage histogram and rates of the recorded neurons with Neuron_population.plot_populations"""
    lif.save_populations()
    lif.plot_populations(ylims=v_lims(sigma))
    plt.savefig(lif.fname + '_plot.png')
    plt.close('all')
    Path(lif.fname + '.hdf5').unlink()


def smooth(r_hz):
    return gaussian_filter1d(r_hz, sigma=RATE_KERNEL_MS / DT, axis=-1, mode='nearest')


def rel_l2(r_hz, ref_hz):
    return float(np.linalg.norm(smooth(r_hz) - smooth(ref_hz)) / np.linalg.norm(smooth(ref_hz)))


rows = []
for setup, (pop_types, W) in SETUPS.items():
    for sigma in SIGMAS:
        tag = f'{setup}_sigma_{sigma:.3g}'
        print(f'{setup}, sigma = {sigma:.3g} mV')

        # per repetition: time, error vs. the reference of the same repetition and mean rates of every model
        res = {key: dict(time=[], err=[], rates=[]) for key in [('FPE', None)] + [('LIF', n) for n in N_LIST]}
        for rep in range(N_REPEATS):
            # timed FPE run (only the rate is saved), in the first repetition also a run with rho for FPE.plot()
            fpe = make_fpe(pop_types, W, sigma, str(OUT_DIR / f'fpe_{tag}'))
            t0 = time.perf_counter()
            fpe.simulate()
            t_fpe = time.perf_counter() - t0
            r_fpe = fpe.r * 1000
            r_fpe[:, -1] = r_fpe[:, -2]                     # last time point is not computed by the time loop
            fpe.clean()
            if rep == 0:
                fpe_plot = make_fpe(pop_types, W, sigma, str(OUT_DIR / f'fpe_{tag}'), save_p='all')
                fpe_plot.simulate()
                fpe_plot.plot(savefig=True, heat_map=True, ylims=v_lims(sigma))
                fpe_plot.clean()

            # LIF networks with a new seed in every repetition, the largest one is the reference
            runs = {('FPE', None): (t_fpe, r_fpe)}
            for n in N_LIST:
                lif = make_lif(n, pop_types, W, sigma, fpe.delay_kernel, str(OUT_DIR / f'lif_{tag}'))
                t0 = time.perf_counter()
                lif.run_low_mem(n_rec=N_REC, rec_every=REC_EVERY, seed=n + rep, save=False)
                runs[('LIF', n)] = (time.perf_counter() - t0, lif.pop_rate)
                if n == N_LIST[-1] and rep == 0:
                    plot_lif(lif, sigma)
            r_ref = runs[('LIF', N_LIST[-1])][1]

            for key, (cpu, r) in runs.items():
                res[key]['time'].append(cpu)
                res[key]['err'].append(nrmse(r, r_ref) if key != ('LIF', N_LIST[-1]) else np.nan)
                res[key]['rates'].append(r.mean(axis=1))
            print(f'  repetition {rep + 1}/{N_REPEATS}: ' + ', '.join(
                f'{m}{"" if n is None else f" {n:,}"} {runs[(m, n)][0]:.2f} s' for m, n in runs))

        for (model, n), v in res.items():
            rows.append(dict(setup=setup, sigma=sigma, model=model, n=n,
                             time=np.mean(v['time']), time_std=np.std(v['time']),
                             err=np.mean(v['err']), err_std=np.std(v['err']), rates=np.mean(v['rates'], axis=0)))

########################################################################################################################
# TABLE
########################################################################################################################
header = (f'{"setup":<6} {"sigma (mV)":>10} {"model":<5} {"n / pop.":>9} {"time (s)":>9} {"+- std":>7} {"x FPE":>7} '
          f'{"rel. L2 err":>11} {"+- std":>9}   mean rates (Hz)')
print('\n' + '=' * len(header) + f'\nFPE (dv = {DV} mV, dt = {DT} ms, Bernoulli + BDF2) vs. LIF network (dt = {DT} ms), '
      f'T = {T:g} ms\nerror: nrmse of the rates (kernel {RATE_KERNEL_MS} ms) vs. LIF n = {N_LIST[-1]:,}\n'
      + (f'mean and std over {N_REPEATS} repetitions\n' if N_REPEATS > 1 else '')
      + '=' * len(header) + '\n' + header + '\n' + '-' * len(header))
for k, row in enumerate(rows):
    if k > 0 and row['model'] == 'FPE':
        print('-' * len(header) if row['setup'] == rows[k - 1]['setup'] else '=' * len(header))
    t_fpe = next(r['time'] for r in rows if r['model'] == 'FPE' and r['setup'] == row['setup']
                 and r['sigma'] == row['sigma'])
    n_str = '-' if row['n'] is None else f'{row["n"]:,}'
    err_str, err_std_str = ('reference', '') if np.isnan(row['err']) else (f'{row["err"]:.3e}', f'{row["err_std"]:.2e}')
    print(f'{row["setup"]:<6} {row["sigma"]:>10.3g} {row["model"]:<5} {n_str:>9} {row["time"]:>9.2f} '
          f'{row["time_std"]:>7.2f} {row["time"] / t_fpe:>7.1f} {err_str:>11} {err_std_str:>9}   '
          + ', '.join(f'{x:6.1f}' for x in row['rates']))
print('=' * len(header))

with open(OUT_DIR / 'timing_table.csv', 'w') as f:
    f.write('setup,sigma_mV,model,n_per_pop,time_s,time_std_s,rel_l2_err,rel_l2_err_std,mean_rates_Hz\n')
    for row in rows:
        f.write(f'{row["setup"]},{row["sigma"]},{row["model"]},{row["n"] or ""},{row["time"]},{row["time_std"]},'
                f'{row["err"]},{row["err_std"]},'
                + ' '.join(f'{x:.3f}' for x in row['rates']) + '\n')

########################################################################################################################
# FIGURE: ERROR AND COMPUTATION TIME VS. NOISE
########################################################################################################################
styles = {('FPE', None): dict(color='#2a78d6', marker='o', label='FPE'),
          ('LIF', N_LIST[0]): dict(color='#b4b3ad', marker='s', label=f'LIF n={N_LIST[0]:,}'),
          ('LIF', N_LIST[1]): dict(color='#8a8984', marker='^', label=f'LIF n={N_LIST[1]:,}'),
          ('LIF', N_LIST[2]): dict(color='#3d3c39', marker='D', label=f'LIF n={N_LIST[2]:,} (reference)')}
fig, axs = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, constrained_layout=True)
for i, setup in enumerate(SETUPS):
    for (model, n), st in styles.items():
        sel = [r for r in rows if r['setup'] == setup and r['model'] == model and r['n'] == n]
        sig = [r['sigma'] for r in sel]
        if n != N_LIST[-1]:
            axs[i, 0].errorbar(sig, [r['err'] for r in sel], yerr=[r['err_std'] for r in sel], mfc='white',
                               capsize=2, **st)
        axs[i, 1].errorbar(sig, [r['time'] for r in sel], yerr=[r['time_std'] for r in sel], mfc='white',
                           capsize=2, **st)
    axs[i, 0].set_ylabel(f'{setup}\nrel. L2 error vs. LIF n={N_LIST[-1]:,}')
    axs[i, 1].set_ylabel('computation time (s)')
for ax in axs.ravel():
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(which='major', lw=0.4, color='#e0dfdb')
    ax.spines[['top', 'right']].set_visible(False)
for ax in axs[-1]:
    ax.set_xlabel(r'noise $\sigma$ (mV)')
axs[0, 0].set_title('error')
axs[0, 1].set_title('computation time')
handles, labels = axs[0, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside upper center', ncol=4, frameon=False)
for ext in ['png', 'pdf']:
    fig.savefig(OUT_DIR / f'error_and_time_vs_sigma.{ext}', dpi=300)
plt.close(fig)
print(f'figures and table saved in {OUT_DIR}')

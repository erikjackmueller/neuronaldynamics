"""
Firing patterns of the Izhikevich (IK) cell types: the recommended fast FPE against a network of 100k neurons.

Cell types of the simple model (Izhikevich 2007, Dynamical Systems in Neuroscience, ch. 8) with the linear adaptation
du/dt = a (b (v - v_r) - u) of the FPE solver:

    RS   regular spiking (L5 pyramidal cell): adapting tonic spiking
    IB   intrinsically bursting (L5 pyramidal cell): high-frequency onset, then tonic spiking
    CH   chattering (L2/3 pyramidal cell): repetitive bursts of 2-4 spikes
    LTS  low-threshold spiking interneuron: adapting tonic spiking (v_peak and c fixed; in the book they depend on u)
    FS   fast spiking interneuron: tonic spiking without adaptation (b = 0; in the book u depends nonlinearly on v)

CH needs one corrector pass of the FPE step (FPE_EXTRA) to be stable at dt = 0.05 ms, see below.
Every type gets a step current (T_ON to T_OFF) with voltage noise SIGMA, single population, no coupling.

    FPE:      FPE_Population(soma_model='IK', adaptation_closure='conditional_quadrature', flux_limiter='fct', dv=0.1,
              dt=0.05), the recommended model (two-node closure with flux-corrected transport)
    network:  N_NET Izhikevich neurons, stochastic Heun with interpolated spike times, dt = NET_DT (converged, as
              FPE_vs_IK_network.ik_network)

Error: relative L2 error of the rate in 1 ms bins and after a 2 ms gaussian kernel (single populations: the 1 ms bins
are the convention of the notes). Runs are done in parallel (N_PROCESSES) and skipped if their rate file exists.

Outputs in Examples/solver_tests/FPE_IK_dynamics_single_pop:
    rates_<type>_fpe.npz, rates_<type>_network.npz     binned rates (Hz), <u>, computation time
    fpe_<type>_plot.png                                FPE_Population.plot(): p(v, t) and rate
    ik_dynamics_single_pop.png/.pdf                    all cell types: single neuron and population rates
    ik_dynamics_single_pop.csv                         errors and computation times

Usage (from src/neuronaldynamics):
    python Examples/solver_tests/FPE_IK_dynamics_single_pop.py
"""
import os
for _var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_var, '1')
import sys
import csv
import time
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/neuronaldynamics, for direct calls
from Model.Nykamp_Model import FPE_Population, IZHIKEVICH_DEFAULTS
matplotlib.use('Agg')  # after the imports (Utils switches the backend)
warnings.filterwarnings('ignore', message='.*non-interactive.*')

########################################################################################################################
# SETTINGS
########################################################################################################################
OUT_DIR = Path(__file__).resolve().parent / 'FPE_IK_dynamics_single_pop'
OUT_DIR.mkdir(exist_ok=True)
T, T_ON, T_OFF = 300., 20., 270.            # ms: simulation time, step input on / off
SIGMA = 2.                                   # mV, voltage noise (tau_noise = 10 ms)
FPE_SETTINGS = dict(soma_model='IK', adaptation_closure='conditional_quadrature', flux_limiter='fct', dv=0.1,
                    dt=0.05, v_min=-90., init_pdf_sigma=1.0)
N_NET, NET_DT, SEED = 100_000, 0.01, 1       # network: neurons, time step (ms), seed
RATE_BIN_MS, SMOOTH_MS = 1.0, 2.0            # ms, bins of the rates, gaussian kernel of the smoothed error
N_PROCESSES = 4

# name: (description, parameters (C pF, k nS/mV, v in mV, a 1/ms, b nS, d pA), step current I0 in pA)
# CH: one corrector pass (n_corrector=1). Without it the two-node closure diverges at dt >= 0.025 ms in the fast
# bursts (intra-burst intervals of 2-3 ms, reset at c = v_t); with it the error equals the converged one (dt 0.0025,
# dv 0.05: 26 %), i.e. the remaining error is closure error, not resolution.
FPE_EXTRA = {'CH': dict(n_corrector=1)}
CELL_TYPES = {
    'RS':  ('regular spiking', dict(C=100., k=0.7, v_r=-60., v_t=-40., v_peak=35., a=0.03, b=-2., c=-50., d=100.),
            150.),
    'IB':  ('intrinsically bursting', dict(C=150., k=1.2, v_r=-75., v_t=-45., v_peak=50., a=0.01, b=5., c=-56.,
                                           d=130.), 800.),
    'CH':  ('chattering', dict(C=50., k=1.5, v_r=-60., v_t=-40., v_peak=25., a=0.03, b=1., c=-40., d=150.), 500.),
    'LTS': ('low-threshold spiking', dict(C=100., k=1., v_r=-56., v_t=-42., v_peak=40., a=0.03, b=8., c=-53., d=20.),
            250.),
    'FS':  ('fast spiking', dict(C=20., k=1., v_r=-55., v_t=-40., v_peak=25., a=0.2, b=0., c=-45., d=0.), 80.),
}

# style of the LIF figures (FPE_solver_comparison.py): IOP / J. Neural Eng. column width, 7-8 pt fonts
MM = 1 / 25.4
WIDTH_DOUBLE = 180 * MM
C_FPE, C_REF, C_REF_BAND, C_INPUT = '#2a78d6', '#52514e', '#d9d8d4', '#f2efe8'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'lines.linewidth': 1.2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': C_REF,
    'xtick.color': C_REF, 'ytick.color': C_REF, 'axes.labelcolor': '#0b0b0b', 'legend.frameon': False,
    'legend.handlelength': 2.4, 'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'mathtext.fontset': 'dejavusans',
})


def step(t, i0):
    return np.where((t >= T_ON) & (t < T_OFF), i0, 0.)


def binned_rate(t, r, dt):
    """rate (Hz) in RATE_BIN_MS bins from the cumulative spike count of r (1/ms) on the time axis t (step dt)"""
    edges = np.arange(0, T + 1e-9, RATE_BIN_MS)
    cum = np.concatenate([[0.], np.cumsum(r * dt)])
    return np.diff(np.interp(edges, np.concatenate([t, [t[-1] + dt]]), cum)) / RATE_BIN_MS * 1e3


def ik_network(par, i0, n, dt, sigma, seed):
    """
    n uncoupled Izhikevich neurons, stochastic Heun (predictor-corrector in the drift, the same noise increment) with
    the spike time interpolated within the step: a neuron that crosses v_peak at t + theta dt is reset to (c, u + d)
    and integrated (drift only) over the rest of the step. n = 1, sigma = 0: a single noise-free neuron.
    :return: t, rate (1/ms), <u> (pA), v of the first neuron (mV, spikes drawn at v_peak)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, dt)
    I = step(t, i0)
    noise = np.sqrt(2 * sigma ** 2 / IZHIKEVICH_DEFAULTS['tau_noise'] * dt)
    C, k, v_r, v_t, v_peak, a, b, c, d = [par[x] for x in ['C', 'k', 'v_r', 'v_t', 'v_peak', 'a', 'b', 'c', 'd']]
    f_v = lambda v, u, i_in: (k * (v - v_r) * (v - v_t) - u + i_in) / C
    f_u = lambda v, u: a * (b * (v - v_r) - u)
    v = v_r + (rng.normal(0., 1.0, n) if n > 1 else np.zeros(1))      # as init_pdf_sigma = 1 of the FPE
    u = b * (v.mean() - v_r) * np.ones(n)
    r, u_mean, v_trace = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    u_mean[0], v_trace[0] = u.mean(), v[0]
    for i in range(1, len(t)):
        dvdt, dudt = f_v(v, u, I[i]), f_u(v, u)
        xi = noise * rng.standard_normal(n)
        v_p = np.minimum(v + dt * dvdt + xi, 10 * v_peak)                 # predictor (bounded near the blow-up)
        u_p = u + dt * dudt
        v_new = v + 0.5 * dt * (dvdt + f_v(v_p, u_p, I[i])) + xi
        u_new = u + 0.5 * dt * (dudt + f_u(v_p, u_p))
        spikes = v_new >= v_peak
        theta = np.clip((v_peak - v) / np.maximum(v_new - v, 1e-12), 0., 1.)
        u_s = u + theta * (u_new - u) + d
        rest = (1. - theta) * dt
        v_s = c + rest * f_v(c, u_s, I[i])
        u_s = u_s + rest * f_u(c, u_s)
        v = np.where(spikes, v_s, v_new)
        u = np.where(spikes, u_s, u_new)
        r[i] = spikes.sum() / (n * dt)
        u_mean[i] = u.mean()
        v_trace[i] = v_peak if spikes[0] else v[0]
    return t, r, u_mean, v_trace


def run(task):
    """task = (cell type, 'fpe' or 'network'): simulate, save the binned rate (and for the FPE its plot)"""
    name, model = task
    out = OUT_DIR / f'rates_{name}_{model}.npz'
    if out.exists():
        return task
    _, par, i0 = CELL_TYPES[name]
    t0 = time.perf_counter()
    if model == 'network':
        t, r, u_mean, _ = ik_network(par, i0, N_NET, NET_DT, SIGMA, SEED)
        np.savez(out, rb=binned_rate(t, r, NET_DT), u_mean=u_mean[::int(round(RATE_BIN_MS / NET_DT))],
                 time=time.perf_counter() - t0)
    else:
        fpe = FPE_Population(dict(FPE_SETTINGS, **FPE_EXTRA.get(name, {}), **par, T=T, voltage_sigma=SIGMA,
                                  input_function=lambda t: step(t, i0), input_function_idx=[0],
                                  save_p='all', p_save_every=int(round(0.5 / FPE_SETTINGS['dt'])), tqdm_disable=True),
                             name=str(OUT_DIR / f'fpe_{name}'))
        fpe.simulate()
        cpu = time.perf_counter() - t0
        fpe.plot(savefig=True, heat_map=True, ylims=(par['v_r'] - 20., par['v_t'] + 20.))
        plt.close('all')
        np.savez(out, rb=binned_rate(fpe.t, fpe.r[0], FPE_SETTINGS['dt']),
                 u_mean=fpe.u_mean[0][::int(round(RATE_BIN_MS / FPE_SETTINGS['dt']))], time=cpu)
        fpe.clean()
    print(f'{name} {model}: {time.perf_counter() - t0:.1f} s', flush=True)
    return task


if __name__ == '__main__':
    # all runs, largest first (the networks take ~1-2 min each, the FPE runs a few seconds)
    tasks = [(name, 'network') for name in CELL_TYPES] + [(name, 'fpe') for name in CELL_TYPES]
    with Pool(N_PROCESSES) as pool:
        pool.map(run, tasks, chunksize=1)

    # errors and table
    smooth = lambda x: gaussian_filter1d(x, SMOOTH_MS / RATE_BIN_MS, mode='nearest')
    rel_l2 = lambda x, ref: float(np.linalg.norm(x - ref) / np.linalg.norm(ref))
    res, rows = {}, []
    for name in CELL_TYPES:
        res[name] = {m: dict(np.load(OUT_DIR / f'rates_{name}_{m}.npz')) for m in ['fpe', 'network']}
        fpe, ref = res[name]['fpe'], res[name]['network']
        rows.append(dict(cell_type=name, I0_pA=CELL_TYPES[name][2], err_1ms=rel_l2(fpe['rb'], ref['rb']),
                         err_smoothed=rel_l2(smooth(fpe['rb']), smooth(ref['rb'])),
                         rate_fpe_Hz=float(fpe['rb'].mean()), rate_network_Hz=float(ref['rb'].mean()),
                         time_fpe_s=float(fpe['time']), time_network_s=float(ref['time'])))
    with open(OUT_DIR / 'ik_dynamics_single_pop.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nFPE two-node + FCT (dv 0.1, dt 0.05) against {N_NET:,} Heun neurons (dt {NET_DT}), sigma = {SIGMA} mV')
    print(f'{"type":5s} {"I0 (pA)":>8s} {"err 1 ms":>9s} {"smoothed":>9s} {"mean rate FPE / net (Hz)":>25s} '
          f'{"time FPE / net (s)":>20s}')
    for r in rows:
        print(f'{r["cell_type"]:5s} {r["I0_pA"]:8.0f} {r["err_1ms"]:9.3f} {r["err_smoothed"]:9.3f} '
              f'{r["rate_fpe_Hz"]:12.1f} / {r["rate_network_Hz"]:<10.1f} {r["time_fpe_s"]:9.1f} / '
              f'{r["time_network_s"]:<8.0f}')

    # figure: per cell type a noise-free single neuron (left) and the population rates (right)
    tb = np.arange(0, T, RATE_BIN_MS) + 0.5 * RATE_BIN_MS
    fig, axs = plt.subplots(len(CELL_TYPES), 2, figsize=(WIDTH_DOUBLE, 30 * MM * len(CELL_TYPES)), sharex=True,
                            constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.6]))
    for row, (name, (desc, par, i0)) in enumerate(CELL_TYPES.items()):
        t, _, _, v_trace = ik_network(par, i0, 1, NET_DT, 0., SEED)
        ax = axs[row, 0]
        ax.axvspan(T_ON, T_OFF, color=C_INPUT, lw=0, zorder=0)
        ax.plot(t, v_trace, color=C_REF, lw=0.6)
        ax.set_ylabel('$v$ (mV)')
        ax.set_title(f'{name}, {desc}, $I_0$ = {i0:g} pA', loc='left')
        ax.text(-0.2, 1.04, f'({"abcdefghijklmnop"[2 * row]})', transform=ax.transAxes, fontweight='bold',
                fontsize=9, va='bottom', ha='left')

        ax = axs[row, 1]
        fpe, ref = res[name]['fpe'], res[name]['network']
        r_ = rows[row]
        ax.axvspan(T_ON, T_OFF, color=C_INPUT, lw=0, zorder=0)
        ax.plot(tb, ref['rb'], color=C_REF_BAND, lw=3.0, label=f'IK network ($n$ = {N_NET:,}, Heun)', zorder=1)
        ax.plot(tb, fpe['rb'], color=C_FPE, lw=0.9, label='FPE, two-node closure + FCT', zorder=3)
        ax.set_ylabel('rate (Hz)')
        ax.set_ylim(0, 1.4 * max(ref['rb'].max(), fpe['rb'].max()))     # room for the annotation
        ax.text(0.99, 0.95, f'error {100 * r_["err_1ms"]:.1f} % (smoothed {100 * r_["err_smoothed"]:.1f} %)\n'
                f'FPE {r_["time_fpe_s"]:.1f} s, network {r_["time_network_s"]:.0f} s', transform=ax.transAxes,
                ha='right', va='top', fontsize=7, color=C_REF)
        ax.text(-0.12, 1.04, f'({"abcdefghijklmnop"[2 * row + 1]})', transform=ax.transAxes, fontweight='bold',
                fontsize=9, va='bottom', ha='left')
    for ax in axs[-1]:
        ax.set_xlabel('time (ms)')
        ax.set_xlim(0, T)
    handles, labels = axs[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=2)
    for ext in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'ik_dynamics_single_pop.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'figures and table saved in {OUT_DIR}')

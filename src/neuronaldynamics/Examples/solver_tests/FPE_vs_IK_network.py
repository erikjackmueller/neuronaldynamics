"""
Computation time and error of the Fokker-Planck population model of Izhikevich neurons (FPE_Population with
soma_model='izhikevich', three adaptation closures) vs. a network of Izhikevich (IK) neurons with 20k, 50k and 100k
neurons per population, in the style of FPE_vs_LIF_timing.py, for

    1 population:   one excitatory (RS) population with a step input current
    3 populations:  exc 1 (RS, gets the input), exc 2 (RS), inh (FS-like), coupled with the connectivity matrix W.
                    The coupling is recurrent: exc 1 -> exc 2, inh and the feedback exc 2 -> exc 1 and inh -> exc 1
                    reaches the population with the input again (loops exc 1 -> exc 2 -> exc 1,
                    exc 1 -> inh -> exc 1 and exc 1 -> exc 2 -> inh -> exc 1)

and five noise levels voltage_sigma between 0.1 and 10 mV. The IK network with 100k neurons per population is the
reference for the error of the FPE and of the smaller networks (relative L2 error of the rates of all populations in
RATE_BIN_MS bins). With --repeat [N] (default N = 10) every run is repeated N times (IK networks with a new seed each
time, the error is computed against the reference of the same repetition); table, csv and figure show mean and std.

Coupling (as in FPE_Population): population j gets the additional input current sum_k W[k, j] * (alpha * r_k)(t) in pA,
with r_k the population rate in 1/ms and W in pA*ms (alpha kernel of FPE_Population, n_alpha = 9, tau_alpha = 1/3 ms,
mean delay 3 ms). The IK network uses the same mean-field coupling (all-to-all, weights scaled by 1/n, i.e. from the
population spike count of the network), so both models simulate the same system.

Input into exc 1 (INPUT), T = 200 ms, on for the first 100 ms, then off (as in FPE_Izhikevich_closure_test.py):
    'step': I0,    'sine': I0 + I0 sin(2 pi SINE_FREQUENCY t)    (FPE_vs_IK_network_sine.py runs the sine input)
FPE grid: DV, DT, chosen from the grid convergence study FPE_Izhikevich_resolution.py; the IK network is simulated with
MC_METHOD and MC_DT. Default: stochastic Heun with interpolated spike times, converged at MC_DT = 0.01 ms
(FPE_IK_reference_dt.py); 'euler' (Euler-Maruyama, spikes on the grid) is not converged there for synchronised
populations, its results are in the folders without the method suffix.

Inhibitory population: FS-like parameters of Izhikevich (2007) (C = 20 pF, k = 1, v_r = -55, v_t = -40, a = 0.2,
c = -45 mV) without adaptation (b = d = 0, Izhikevich's FS cell has a nonlinear u-nullcline that is not part of the
model), v_peak is shared by all populations (35 mV).

Outputs in Examples/solver_tests/FPE_vs_IK_network<method_suffix()> (FPE_vs_IK_network_heun for MC_METHOD = 'heun'):
    rates_<setup>_sigma_<sigma>.png         rates and <u> of all populations: IK reference, IK 20k and the FPE closures
    error_and_time_vs_sigma.png/.pdf        error vs. the 100k reference and computation time against sigma
    timing_table.csv                        all numbers of the printed table
    timing_table.tex / .pdf                 LaTeX table of time and error

Usage (from src/neuronaldynamics):  python Examples/solver_tests/FPE_vs_IK_network.py [--repeat [N]]
"""
import os
for _var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_var, '1')
import sys
import time
import shutil
import argparse
import subprocess
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/neuronaldynamics, for direct calls
from Model.Nykamp_Model import FPE_Population, IZHIKEVICH_DEFAULTS
matplotlib.use('Agg')
warnings.filterwarnings('ignore', message='.*non-interactive.*')

########################################################################################################################
# SETTINGS
########################################################################################################################
T, T_ON, I0 = 200., 100., 150.               # ms, ms, pA (input into exc 1)
INPUT = 'step'                               # 'step' or 'sine', see set_input()
SINE_FREQUENCY = 0.1                         # 1/ms (100 Hz), as in FPE_Izhikevich_closure_test.py
DV, DT = 0.1, 0.05                           # FPE grid (mV, ms), see FPE_Izhikevich_resolution.py
MC_DT = 0.01                                 # ms, IK network
MC_METHOD = 'heun'                           # IK network integrator: 'heun' or 'euler' (Euler-Maruyama), see ik_network
CLOSURES = ['mean', 'conditional', 'conditional_variance']
SIGMAS = np.logspace(-1, 1, 5)               # voltage_sigma (mV)
N_LIST = [20_000, 50_000, 100_000]           # IK neurons per population, the last one is the reference
RATE_BIN_MS = 1.0                            # bins of the rate for the error
N_PROCESSES = 4

# neuron parameters per population (scalars: the same for all populations of a setup)
RS = dict(C=100., k=0.7, v_r=-60., v_t=-40., a=0.03, b=-2., c=-50., d=100.)
FS = dict(C=20., k=1.0, v_r=-55., v_t=-40., a=0.2, b=0., c=-45., d=0.)

# W[k, j]: from population k to population j (pA*ms), feedback from exc 2 and inh back to exc 1 (first column)
# (exc 1 ~ 20-25 Hz, exc 2 ~ 20 Hz, inh ~ 130 Hz during the input)
W_3POP = np.array([[0.0, 6000.0, 1500.0],       # exc 1 (input) -> exc 2, inh
                   [2000.0, 0.0, 2000.0],       # exc 2 -> exc 1 (feedback), inh
                   [-400.0, -300.0, 0.0]])      # inh -> exc 1 (feedback), exc 2
SETUPS = {'1pop': (['exc'], [RS], np.zeros((1, 1))),
          '3pop': (['exc', 'exc', 'inh'], [RS, RS, FS], W_3POP)}
POP_NAMES = {'1pop': ['exc'], '3pop': ['exc 1 (input)', 'exc 2', 'inh']}
METHOD_NAMES = {'euler': 'Euler-Maruyama', 'heun': 'stochastic Heun'}


def method_suffix():
    """suffix of the result folders: '' for Euler-Maruyama (the old results), '_heun' for Heun"""
    return '' if MC_METHOD == 'euler' else f'_{MC_METHOD}'


OUT_DIR = Path(__file__).resolve().parent / f'FPE_vs_IK_network{method_suffix()}'
OUT_DIR.mkdir(exist_ok=True)

CLOSURE_LABELS = {'mean': r'FPE mean $\langle u\rangle$', 'conditional': r'FPE cond., $\sigma_u^2=0$',
                  'conditional_variance': r'FPE cond., $\sigma_u^2(t)$'}
CLOSURE_SHORT = {'mean': 'mean', 'conditional': 'cond.', 'conditional_variance': 'cond.+var.'}


def neuron_parameters(setup):
    """parameters as one array per name (one value per population)"""
    pars = SETUPS[setup][1]
    return {name: np.array([p[name] for p in pars]) for name in pars[0]}


def set_input(kind, out_dir=None, frequency=None):
    """input type 'step' or 'sine', sine frequency in 1/ms (None: keep SINE_FREQUENCY) and the result folder
    (default: FPE_vs_IK_network_<kind><method_suffix()>, 'step': FPE_vs_IK_network<method_suffix()>)"""
    global INPUT, OUT_DIR, SINE_FREQUENCY
    if kind not in ['step', 'sine']:
        raise ValueError(f'input {kind} not implemented, chose step or sine')
    INPUT = kind
    if frequency is not None:
        SINE_FREQUENCY = frequency
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / ('FPE_vs_IK_network' + ('' if kind == 'step' else f'_{kind}')
                                                     + method_suffix())
    OUT_DIR = Path(out_dir)
    OUT_DIR.mkdir(exist_ok=True)


def input_current(t):
    """input current (pA) into exc 1: step I0 or sine I0 + I0 sin(2 pi f t) for t < T_ON, then 0"""
    if INPUT == 'sine':
        return np.where(t < T_ON, I0 + I0 * np.sin(2 * np.pi * SINE_FREQUENCY * t), 0.)
    return step_input(t)


def input_label():
    if INPUT == 'sine':
        return f'sine {I0:g} + {I0:g} sin(2$\\pi$ {SINE_FREQUENCY * 1e3:g} Hz t) pA'
    return f'step {I0:g} pA'


def step_input(t):
    return np.where(t < T_ON, I0, 0.)


def alpha_kernel(dt):
    """the delay kernel of FPE_Population on the time step dt"""
    return FPE_Population(dict(T=T, dt=dt)).delay_kernel


def binned_rate(r, dt):
    """rate (Hz) in RATE_BIN_MS bins from the cumulative spike count, r: (n_pop, Nt) in 1/ms"""
    t = np.arange(r.shape[1]) * dt
    edges = np.arange(0, T + 1e-9, RATE_BIN_MS)
    out = []
    for rk in r:
        cum = np.concatenate([[0.], np.cumsum(rk * dt)])
        out.append(np.diff(np.interp(edges, np.concatenate([t, [t[-1] + dt]]), cum)) / RATE_BIN_MS * 1e3)
    return np.array(out)


def ik_network(setup, sigma, n, seed, method=None, snapshot_times=None):
    """
    network of n Izhikevich neurons per population with mean-field coupling, time step MC_DT
    method (default MC_METHOD):
        'euler': Euler-Maruyama, spikes detected on the time grid (v >= v_peak at the end of the step)
        'heun': stochastic Heun (predictor-corrector in the drift, same noise increment; for additive noise strong order
            1, second order in the drift) with the spike time interpolated linearly within the step: a neuron that
            crosses v_peak at t + theta dt is reset to (c, u + d) and integrated (drift only) over the rest of the step
    snapshot_times (ms, optional): v and u of all neurons at the first time step >= each time
    :return: rate (n_pop, Nt) in 1/ms, <u> (n_pop, Nt) in pA (and, with snapshot_times, the times of the snapshots and
        the arrays v, u of shape (n_snap, n_pop, n))
    """
    method = MC_METHOD if method is None else method
    if method not in ('euler', 'heun'):
        raise ValueError(f"method has to be 'euler' or 'heun', not {method!r}")
    p = neuron_parameters(setup)
    W = SETUPS[setup][2]
    n_pop = len(SETUPS[setup][0])
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, MC_DT)
    i_ext = np.zeros((n_pop, len(t)))
    i_ext[0] = input_current(t)
    kernel = alpha_kernel(MC_DT)
    n_k_max = len(kernel)
    # (n_pop, 1) columns for broadcasting over the neurons
    C, k, v_r, v_t, a, b, c, d = [p[x][:, None] for x in ['C', 'k', 'v_r', 'v_t', 'a', 'b', 'c', 'd']]
    v_peak = IZHIKEVICH_DEFAULTS['v_peak']
    D = sigma ** 2 / IZHIKEVICH_DEFAULTS['tau_noise']
    noise = np.sqrt(2 * D * MC_DT)
    v = v_r + rng.normal(0., 1.0, (n_pop, n))
    u = b * (v.mean(axis=1, keepdims=True) - v_r) * np.ones((1, n))
    r = np.zeros((n_pop, len(t)))
    u_mean = np.zeros((n_pop, len(t)))
    u_mean[:, 0] = u.mean(axis=1)
    snap_idx = [] if snapshot_times is None else sorted({int(np.searchsorted(t, ts - 1e-9)) for ts in snapshot_times
                                                         if ts - 1e-9 <= t[-1]})
    v_snap = np.zeros((len(snap_idx), n_pop, n))
    u_snap = np.zeros((len(snap_idx), n_pop, n))
    for i in range(1, len(t)):
        n_k = min(i + 1, n_k_max)
        r_conv = r[:, i - n_k + 1:i + 1][:, ::-1] @ kernel[:n_k] * MC_DT  # as in FPE_Population (r[:, i] = 0 here)
        i_tot = (i_ext[:, i] + W.T @ r_conv)[:, None]
        dvdt = (k * (v - v_r) * (v - v_t) - u + i_tot) / C
        dudt = a * (b * (v - v_r) - u)
        if method == 'euler':
            v = v + MC_DT * dvdt + noise * rng.standard_normal((n_pop, n))
            u = u + MC_DT * dudt
            spikes = v >= v_peak
            v = np.where(spikes, c, v)
            u = u + spikes * d
        else:
            xi = noise * rng.standard_normal((n_pop, n))
            v_p = np.minimum(v + MC_DT * dvdt + xi, 10 * v_peak)   # predictor (bounded: no overflow near blow-up)
            u_p = u + MC_DT * dudt
            v_new = v + 0.5 * MC_DT * (dvdt + (k * (v_p - v_r) * (v_p - v_t) - u_p + i_tot) / C) + xi
            u_new = u + 0.5 * MC_DT * (dudt + a * (b * (v_p - v_r) - u_p))
            spikes = v_new >= v_peak
            # crossing time t + theta dt (linear in the step), reset and drift over the rest (1 - theta) dt
            theta = np.clip((v_peak - v) / np.maximum(v_new - v, 1e-12), 0., 1.)
            u_s = u + theta * (u_new - u) + d
            rest = (1. - theta) * MC_DT
            v_s = c + rest * ((k * (c - v_r) * (c - v_t) - u_s + i_tot) / C)
            u_s = u_s + rest * a * (b * (c - v_r) - u_s)
            v = np.where(spikes, v_s, v_new)
            u = np.where(spikes, u_s, u_new)
        r[:, i] = spikes.sum(axis=1) / (n * MC_DT)
        u_mean[:, i] = u.mean(axis=1)
        if i in snap_idx:
            v_snap[snap_idx.index(i)], u_snap[snap_idx.index(i)] = v, u
    if snapshot_times is not None:
        return r, u_mean, t[snap_idx], v_snap, u_snap
    return r, u_mean


def fpe(setup, sigma, closure):
    """FPE_Population of Izhikevich neurons, returns the simulated model"""
    pop_types, _, W = SETUPS[setup]
    model = FPE_Population(dict(
        soma_model='IK', adaptation_closure=closure, T=T, dt=DT, dv=DV, population_type=pop_types,
        connectivity_matrix=W, voltage_sigma=sigma, input_function=input_current, input_function_idx=[0],
        init_pdf_sigma=1.0, save_p='none', tqdm_disable=True,
        **{name: list(x) for name, x in neuron_parameters(setup).items()}),
        name=str(OUT_DIR / f'fpe_{setup}_{closure}_sigma_{sigma:.3g}'))
    model.simulate()
    return model


def run(task):
    """one timed simulation, task = (setup, sigma, model, n or None, repetition)"""
    setup, sigma, kind, n, rep = task
    if kind == 'IK':
        t0 = time.perf_counter()
        r, u = ik_network(setup, sigma, n, seed=n + 1000 * rep)
        cpu = time.perf_counter() - t0
        return dict(task=task, time=cpu, rb=binned_rate(r, MC_DT), u=u[:, ::int(round(DT / MC_DT))])
    t0 = time.perf_counter()
    model = fpe(setup, sigma, kind)
    cpu = time.perf_counter() - t0
    model.clean()
    return dict(task=task, time=cpu, rb=binned_rate(model.r, DT), u=model.u_mean.copy())


def rel_l2(x, ref):
    """relative L2 error of the binned rates of all populations"""
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


########################################################################################################################
# PLOTS (style of FPE_vs_LIF_timing.py)
########################################################################################################################
STYLES = {('FPE', 'mean'): dict(color='#2a78d6', marker='o', label=CLOSURE_LABELS['mean']),
          ('FPE', 'conditional'): dict(color='#d6612a', marker='s', label=CLOSURE_LABELS['conditional']),
          ('FPE', 'conditional_variance'): dict(color='#2a9d6b', marker='D',
                                                label=CLOSURE_LABELS['conditional_variance']),
          ('IK', N_LIST[0]): dict(color='#b4b3ad', marker='v', label=f'IK n={N_LIST[0]:,}'),
          ('IK', N_LIST[1]): dict(color='#8a8984', marker='^', label=f'IK n={N_LIST[1]:,}'),
          ('IK', N_LIST[2]): dict(color='#3d3c39', marker='P', label=f'IK n={N_LIST[2]:,} (reference)')}
MODEL_KEYS = [('FPE', c) for c in CLOSURES] + [('IK', n) for n in N_LIST]


def style(ax):
    ax.grid(which='major', lw=0.4, color='#e0dfdb')
    ax.spines[['top', 'right']].set_visible(False)


def plot_rates(setup, sigma, res, times):
    """rates and <u> of all populations (first repetition)"""
    names = POP_NAMES[setup]
    edges = np.arange(0, T + 1e-9, RATE_BIN_MS)
    tb = 0.5 * (edges[1:] + edges[:-1])
    t_u = np.arange(0, T, DT)
    fig, axs = plt.subplots(len(names), 2, figsize=(11, 2.3 * len(names) + 0.9), sharex=True, squeeze=False,
                            constrained_layout=True, gridspec_kw=dict(width_ratios=[1.6, 1]))
    for key in [('IK', N_LIST[-1]), ('IK', N_LIST[0])] + [('FPE', c) for c in CLOSURES]:
        st = STYLES[key]
        is_ref = key == ('IK', N_LIST[-1])
        kw = dict(color=st['color'], lw=1.0 if key[0] == 'IK' else 1.4,
                  ls='--' if key == ('FPE', 'conditional_variance') else '-', alpha=0.9 if is_ref else 0.85,
                  label=f"{st['label']} ({times[key]:.1f} s)")
        for j in range(len(names)):
            axs[j, 0].plot(tb, res[key]['rb'][j], **kw)
            axs[j, 1].plot(t_u, res[key]['u'][j], **kw)
    for j, name in enumerate(names):
        axs[j, 0].set_ylabel(f'{name}\nrate (Hz)')
        axs[j, 1].set_ylabel(r'$\langle u\rangle$ (pA)')
        for ax in axs[j]:
            ax.axvline(T_ON, color='#b4b2a9', lw=0.6, ls=':')
            style(ax)
    for ax in axs[-1]:
        ax.set_xlabel('t (ms)')
    axs[0, 0].set_title(f'population rates ({RATE_BIN_MS:g} ms bins)', fontsize=9)
    axs[0, 1].set_title('mean adaptation', fontsize=9)
    time_str = '   '.join(f"{STYLES[k]['label'].replace('FPE ', '')} {times[k]:.1f} s" for k in MODEL_KEYS)
    fig.suptitle(f'{setup}, $\\sigma$ = {sigma:.3g} mV, {input_label()} into exc 1 for t < {T_ON:g} ms\n'
                 f'computation time: {time_str}', fontsize=9)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False, fontsize=8)
    fig.savefig(OUT_DIR / f'rates_{setup}_sigma_{sigma:.3g}.png', dpi=200)
    plt.close(fig)


def plot_error_and_time(rows):
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 6.8), sharex=True, constrained_layout=True)
    for i, setup in enumerate(SETUPS):
        for key in MODEL_KEYS:
            sel = [r for r in rows if r['setup'] == setup and (r['model'], r['variant']) == key]
            sig = [r['sigma'] for r in sel]
            kw = dict(STYLES[key], mfc='white', capsize=2)
            if key != ('IK', N_LIST[-1]):
                axs[i, 0].errorbar(sig, [r['err'] for r in sel], yerr=[r['err_std'] for r in sel], **kw)
            axs[i, 1].errorbar(sig, [r['time'] for r in sel], yerr=[r['time_std'] for r in sel], **kw)
        axs[i, 0].set_ylabel(f'{setup}\nrel. L2 error vs. IK n={N_LIST[-1]:,}')
        axs[i, 1].set_ylabel('computation time (s)')
    for ax in axs.ravel():
        ax.set_xscale('log')
        ax.set_yscale('log')
        style(ax)
    for ax in axs[-1]:
        ax.set_xlabel(r'noise $\sigma$ (mV)')
    axs[0, 0].set_title('error')
    axs[0, 1].set_title('computation time')
    fig.suptitle(f'FPE: $\\Delta v$ = {DV} mV, $\\Delta t$ = {DT} ms;   IK network: {METHOD_NAMES[MC_METHOD]}, '
                 f'$\\Delta t$ = {MC_DT} ms;   '
                 f'{N_PROCESSES} parallel processes', fontsize=9)
    handles, labels = axs[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False, fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT_DIR / f'error_and_time_vs_sigma.{ext}', dpi=300)
    plt.close(fig)


########################################################################################################################
# TABLE
########################################################################################################################
def latex_table(rows, n_repeats):
    keys = MODEL_KEYS
    head = ' & '.join(CLOSURE_SHORT[v] if m == 'FPE' else f'IK {v // 1000}k' for m, v in keys)
    nk = len(keys)
    lines = [r'\begin{tabular}{l r ' + 'r' * nk + ' ' + 'r' * (nk - 1) + '}', r'\toprule',
             r' & & \multicolumn{' + str(nk) + r'}{c}{computation time (s)} & \multicolumn{' + str(nk - 1) +
             r'}{c}{rel.\ L2 error vs.\ IK ' + f'{N_LIST[-1] // 1000}k' + r'} \\',
             r'\cmidrule(lr){3-' + str(nk + 2) + r'} \cmidrule(lr){' + str(nk + 3) + '-' + str(2 * nk + 1) + '}',
             r'setup & $\sigma$ (mV) & ' + head + ' & ' + ' & '.join(head.split(' & ')[:-1]) + r' \\']
    for setup in SETUPS:
        lines.append(r'\midrule')
        for sigma in SIGMAS:
            sel = {(r['model'], r['variant']): r for r in rows if r['setup'] == setup and r['sigma'] == sigma}
            errs = [sel[k]['err'] for k in keys[:-1]]
            best = int(np.nanargmin(errs))
            err_str = ' & '.join((r'\textbf{' + f'{e:.3f}' + '}') if q == best else f'{e:.3f}'
                                 for q, e in enumerate(errs))
            lines.append(f'{setup} & {sigma:.3g} & ' + ' & '.join(f"{sel[k]['time']:.1f}" for k in keys) +
                         f' & {err_str} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    caption = (f'FPE of Izhikevich populations (three closures, $\\Delta v$ = {DV} mV, $\\Delta t$ = {DT} ms) vs.\\ '
               f'IK networks ({METHOD_NAMES[MC_METHOD]}, $\\Delta t$ = {MC_DT} ms) with $n$ neurons per population. Input: '
               f'{input_label()} into exc 1, T = {T:g} ms, on for {T_ON:g} ms; 3pop: exc 1, exc 2, inh with feedback to '
               f'exc 1. Error: rel.\\ L2 error of the rates of all populations ({RATE_BIN_MS:g} ms bins). '
               f'{N_PROCESSES} parallel processes' + (f', mean over {n_repeats} repetitions' if n_repeats > 1 else '')
               + '. Bold: smallest error.')
    return '\n'.join(lines), caption


def save_table_pdf(tabular, caption, fname):
    doc = (r'\documentclass[border=8pt]{standalone}' + '\n' + r'\usepackage{booktabs}' + '\n' +
           r'\begin{document}' + '\n' + r'\begin{minipage}{22cm}\small' + '\n' + tabular + '\n\n' + r'\vspace{6pt}' +
           '\n' + r'\footnotesize ' + caption + '\n' + r'\end{minipage}' + '\n' + r'\end{document}' + '\n')
    (OUT_DIR / f'{fname}.tex').write_text(doc)
    for engine in [['latexmk', '-pdf', '-quiet'], ['pdflatex', '-interaction=nonstopmode'], ['tectonic']]:
        exe = shutil.which(engine[0]) or (str(Path.home() / '.local/bin' / engine[0])
                                          if (Path.home() / '.local/bin' / engine[0]).exists() else None)
        if exe:
            proc = subprocess.run([exe] + engine[1:] + [f'{fname}.tex'], cwd=OUT_DIR, capture_output=True)
            if proc.returncode == 0:
                print(f'compiled {fname}.pdf with {engine[0]}')
                return
    print('no LaTeX engine found, only the .tex file was written')


def main(n_repeats=1):
    """run all simulations, print and save table and figures"""

    tasks = [(s, sig, 'IK', n, rep) for rep in range(n_repeats) for s in SETUPS for sig in SIGMAS
             for n in N_LIST[::-1]]
    tasks += [(s, sig, c, None, rep) for rep in range(n_repeats) for s in SETUPS for sig in SIGMAS for c in CLOSURES]
    tasks.sort(key=lambda x: -(len(SETUPS[x[0]][0]) * (x[3] or 0)))  # largest networks first (load balancing)
    t_start = time.perf_counter()
    results = {}
    with Pool(N_PROCESSES) as pool:
        for q, res in enumerate(pool.imap_unordered(run, tasks), start=1):
            results[res['task']] = res
            print(f'[{q}/{len(tasks)}] {res["task"]}: {res["time"]:.1f} s', flush=True)
    print(f'total wall time {time.perf_counter() - t_start:.0f} s')

    def get(setup, sigma, key, rep):
        model, variant = key
        return results[(setup, sigma, 'IK', variant, rep)] if model == 'IK' else \
            results[(setup, sigma, variant, None, rep)]

    rows = []
    for setup in SETUPS:
        for sigma in SIGMAS:
            for key in MODEL_KEYS:
                times, errs, rates = [], [], []
                for rep in range(n_repeats):
                    res = get(setup, sigma, key, rep)
                    ref = get(setup, sigma, ('IK', N_LIST[-1]), rep)
                    times.append(res['time'])
                    errs.append(rel_l2(res['rb'], ref['rb']) if key != ('IK', N_LIST[-1]) else np.nan)
                    rates.append(res['rb'].mean(axis=1))
                rows.append(dict(setup=setup, sigma=sigma, model=key[0], variant=key[1], time=np.mean(times),
                                 time_std=np.std(times), err=np.mean(errs), err_std=np.std(errs),
                                 rates=np.mean(rates, axis=0)))
            plot_rates(setup, sigma, {k: get(setup, sigma, k, 0) for k in MODEL_KEYS},
                       {k: get(setup, sigma, k, 0)['time'] for k in MODEL_KEYS})

    # table (as in FPE_vs_LIF_timing.py)
    header = (f'{"setup":<6} {"sigma (mV)":>10} {"model":<26} {"time (s)":>9} {"+- std":>7} {"x FPE cond.":>11} '
              f'{"rel. L2 err":>11} {"+- std":>9}   mean rates (Hz)')
    print('\n' + '=' * len(header) + f'\nFPE Izhikevich (dv = {DV} mV, dt = {DT} ms) vs. IK network '
          f'({METHOD_NAMES[MC_METHOD]}, dt = {MC_DT} ms), '
          f'T = {T:g} ms\nerror: rel. L2 error of the rates ({RATE_BIN_MS:g} ms bins) vs. IK n = {N_LIST[-1]:,}\n'
          + (f'mean and std over {n_repeats} repetitions\n' if n_repeats > 1 else '') + '=' * len(header) + '\n'
          + header + '\n' + '-' * len(header))
    for q, row in enumerate(rows):
        if q > 0 and row['model'] == 'FPE' and row['variant'] == CLOSURES[0]:
            print('-' * len(header) if row['setup'] == rows[q - 1]['setup'] else '=' * len(header))
        t_cond = next(r['time'] for r in rows if r['setup'] == row['setup'] and r['sigma'] == row['sigma']
                      and r['variant'] == 'conditional')
        name = f'FPE {row["variant"]}' if row['model'] == 'FPE' else f'IK n={row["variant"]:,}'
        err_str, err_std_str = ('reference', '') if np.isnan(row['err']) else (f'{row["err"]:.3e}',
                                                                              f'{row["err_std"]:.2e}')
        print(f'{row["setup"]:<6} {row["sigma"]:>10.3g} {name:<26} {row["time"]:>9.2f} {row["time_std"]:>7.2f} '
              f'{row["time"] / t_cond:>11.1f} {err_str:>11} {err_std_str:>9}   '
              + ', '.join(f'{x:6.1f}' for x in row['rates']))
    print('=' * len(header))

    with open(OUT_DIR / 'timing_table.csv', 'w') as f:
        f.write('setup,sigma_mV,model,variant,time_s,time_std_s,rel_l2_err,rel_l2_err_std,mean_rates_Hz\n')
        for row in rows:
            f.write(f'{row["setup"]},{row["sigma"]},{row["model"]},{row["variant"]},{row["time"]},'
                    f'{row["time_std"]},{row["err"]},{row["err_std"]},' + ' '.join(f'{x:.3f}' for x in row['rates'])
                    + '\n')
    plot_error_and_time(rows)
    save_table_pdf(*latex_table(rows, n_repeats), 'timing_table')
    print(f'figures and table saved in {OUT_DIR}')


def parse_repeat():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, nargs='?', const=10, default=1, metavar='N',
                        help='repeat every run N times (default 10) and average error and time')
    return parser.parse_args().repeat


if __name__ == '__main__':
    main(parse_repeat())

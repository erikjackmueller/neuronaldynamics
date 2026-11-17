"""
Comparison of Fokker-Planck (NNLIF) solvers against each other and against a LIF network
==========================================================================================

Model: leaky integrate-and-fire neurons with white-noise current input (no coupling, no refractory period)

    tau dV = (E_rest + mu(t) - V) dt + sigma sqrt(2 tau) dW,   V >= V_th -> spike, V -> V_reset,
    mu(t) = I(t) / g_leak  (in mV),

and the corresponding Fokker-Planck equation (see NNLIF_solver_update.pdf for the numerics).

Experiments (each produces one or more figures in OUT_DIR, all results are cached in OUT_DIR/cache):

    lif_filter      Why does the LIF rate look different for dt = 0.1 / 0.01 / 0.001 ms?
                    Separates the gaussian filter (sigma in samples -> width in ms changes with dt) from the
                    discretisation of the LIF network (spike detection on the time grid, O(dt) bias).
    lif_dt          Stationary LIF rate vs. dt against the exact Siegert rate: spike detection on the grid (as
                    LIF_basic) vs. linear interpolation of the threshold crossing (+ Brownian-bridge correction),
                    and example traces for different dt -> which dt is necessary.
    lif_n           LIF rate vs. number of neurons n against a fine FPE solution -> which n gives the smooth result.
    solvers         Original vs. new solver: flux 'harmonic' vs. 'bernoulli' x time stepping 'backward-euler' vs.
                    'bdf2', for several sigma, input amplitudes and the input shapes step / two-step / sine, compared
                    with a LIF reference network.
    convergence     Time convergence (BE vs. BDF2) against a fine-dt reference and space convergence (harmonic vs.
                    Bernoulli) against the exact Siegert rate.
    resolution      New FPE solver against the LIF reference for a grid of dv x dt: error, computation time and an
                    error vs. computation time comparison (FPE vs. LIF network).

Usage (from src/neuronaldynamics, like the other example scripts):

    python Examples/solver_tests/FPE_solver_comparison.py                         # all experiments, full resolution
    python Examples/solver_tests/FPE_solver_comparison.py --exp solvers lif_n     # selected experiments
    python Examples/solver_tests/FPE_solver_comparison.py --quick                 # reduced n / grids (minutes instead of hours)
    python Examples/solver_tests/FPE_solver_comparison.py --smoke                 # tiny settings, only checks that everything runs
    python Examples/solver_tests/FPE_solver_comparison.py --check-lif             # streaming LIF == LIF_basic (spike by spike)

Notes:
    - The LIF network is simulated with a streaming, vectorised re-implementation of LIF_basic.run (identical update
      rule and random numbers in 'legacy' mode, see check_lif_equivalence()). LIF_basic stores four (n x Nt) arrays,
      which does not fit in memory for n = 1e5 and dt = 1e-3 ms.
    - The original flux ('harmonic') evaluates M = exp(-U) on the nodes and underflows for small sigma and strong
      inputs (e.g. sigma = 2 mV at >= 1 nA); such runs are skipped (NaN in the tables, no curve in the figures).
    - All rates are compared after the same gaussian kernel (RATE_KERNEL_MS, in ms, NOT in samples) on a common
      evaluation grid (EVAL_DT), for the FPE solutions as well as for the LIF network.
"""

import os
import sys
import json
import time
import hashlib
import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.integrate import quad
from scipy.special import erfcx
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/neuronaldynamics, for direct calls
from Model.Nykamp_Model import Nykamp_Model_1, FPE_Population
from Model.LIF import LIF_basic

########################################################################################################################
# CONFIGURATION #
########################################################################################################################

OUT_DIR = Path(__file__).resolve().parent / 'FPE_solver_comparison_results'
CACHE_DIR = OUT_DIR / 'cache'
TMP_DIR = OUT_DIR / 'tmp'
FORCE_RECOMPUTE = False

# neuron model (identical for LIF network and FPE)
TAU = 12.0                                # membrane time constant [ms]
E_REST = -70.0                            # leak / resting potential [mV]
V_RESET = -70.0                           # reset potential [mV]
V_TH = -55.0                              # threshold [mV]
V_INIT, INIT_SIGMA = -70.0, 1.0           # initial distribution N(V_INIT, INIT_SIGMA) [mV]
U_MIN = -100.0                            # lower end of the FPE voltage grid [mV] (far below V_reset for sigma 4.5)
G_LEAK = 1.1e-10 / (TAU * 1e-3)           # leak conductance [S], same as FPE_Population / Nykamp_Model_1

# inputs [ms], [A]
T_SIM = 50.0
STEP_ON, STEP_OFF = 5.0, 30.0
TWO_STEP = [(5.0, 18.0), (28.0, 41.0)]
SINE_ON, SINE_FREQ_HZ = 5.0, 50.0         # sine: mean = amp/2, amplitude = amp/2 -> between 0 and amp
INPUT_KINDS = ['step', 'two-step', 'sine']
AMPLITUDES = [0.5e-9, 1.0e-9, 1.5e-9]

# noise levels
SIGMAS_ALL = [2.0, 3.0, 4.5]              # all four solver combinations
SIGMAS_BERNOULLI_ONLY = [0.1, 0.5, 1.0]   # original harmonic flux fails below ~2 mV

# rate estimation
EVAL_DT = 0.01                            # common evaluation grid [ms]
RATE_KERNEL_MS = 0.2                      # gaussian kernel std [ms], applied to every rate before comparison

# solver settings
FPE_DT, FPE_DV = 0.01, 0.01               # default FPE resolution [ms], [mV]
SOLVER_COMBOS = [('bernoulli', 'bdf2'), ('bernoulli', 'backward-euler'),
                 ('harmonic', 'bdf2'), ('harmonic', 'backward-euler')]

# LIF reference network ("good LIF result")
LIF_REF_N, LIF_REF_DT = 100_000, 0.001
LIF_REF_TIMING = 'interpolated'           # 'grid' (as LIF_basic) or 'interpolated' (crossing time interpolated)

# studies
LIF_DT_LIST = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
LIF_DT_N = 20_000
LIF_DT_T_STAT, LIF_DT_T_SKIP = 200.0, 50.0
LIF_N_LIST = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
LIF_N_SEEDS = [0, 1, 2]
LIF_N_DT = 0.005
RES_DV_LIST = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
RES_DT_LIST = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
CONV_DT_LIST = [0.08, 0.04, 0.02, 0.01, 0.005]
CONV_DT_REF = 0.000625
CONV_DV_LIST = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]


def apply_quick_settings(level):
    """reduce the cost: 'quick' for a first look (minutes to ~1 h), 'smoke' only to check that everything runs"""
    g = globals()
    if level == 'quick':
        g.update(LIF_REF_N=20_000, LIF_REF_DT=0.005, LIF_DT_N=5_000, LIF_DT_T_STAT=100.0, LIF_DT_T_SKIP=30.0,
                 LIF_N_LIST=[100, 1_000, 10_000], LIF_N_SEEDS=[0], RES_DV_LIST=[0.1, 0.05, 0.02, 0.01],
                 RES_DT_LIST=[0.1, 0.05, 0.02, 0.01], CONV_DT_LIST=[0.08, 0.04, 0.02, 0.01], CONV_DT_REF=0.0025,
                 CONV_DV_LIST=[0.1, 0.05, 0.02, 0.01], LIF_DT_LIST=[0.1, 0.05, 0.02, 0.01, 0.005])
    elif level == 'smoke':
        g.update(T_SIM=12.0, STEP_ON=2.0, STEP_OFF=8.0, TWO_STEP=[(2.0, 5.0), (7.0, 10.0)], SINE_ON=2.0,
                 SINE_FREQ_HZ=200.0, AMPLITUDES=[1.0e-9], SIGMAS_ALL=[3.0], SIGMAS_BERNOULLI_ONLY=[1.0],
                 FPE_DT=0.02, FPE_DV=0.05, LIF_REF_N=500, LIF_REF_DT=0.01, LIF_DT_LIST=[0.1, 0.02],
                 LIF_DT_N=300, LIF_DT_T_STAT=20.0, LIF_DT_T_SKIP=5.0, LIF_N_LIST=[100, 300], LIF_N_SEEDS=[0],
                 LIF_N_DT=0.02, RES_DV_LIST=[0.1, 0.05], RES_DT_LIST=[0.05, 0.02], CONV_DT_LIST=[0.04, 0.02],
                 CONV_DT_REF=0.01, CONV_DV_LIST=[0.1, 0.05], U_MIN=-90.0)
    g['CACHE_DIR'] = OUT_DIR / f'cache_{level}'
    g['FIG_SUFFIX'] = f'_{level}'


FIG_SUFFIX = ''

########################################################################################################################
# FIGURE STYLE #
########################################################################################################################

# categorical slots (validated: colour-blind separation, used together with line styles as secondary encoding)
C_BERNOULLI = '#2a78d6'
C_HARMONIC = '#eb6834'
C_REF = '#52514e'                          # neutral ink for the LIF reference
C_REF_BAND = '#d9d8d4'
LS_TIME = {'bdf2': '-', 'backward-euler': (0, (4, 2))}
C_FLUX = {'bernoulli': C_BERNOULLI, 'harmonic': C_HARMONIC}
LABEL_FLUX = {'bernoulli': 'Bernoulli (SG)', 'harmonic': 'harmonic mean'}
LABEL_TIME = {'bdf2': 'BDF2', 'backward-euler': 'backward Euler'}
# ordinal blue ramp (light -> dark) for dt / n sweeps, lightest step still >= 2:1 on white
BLUE_RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#184f95', '#104281', '#0d366b']
CMAP_SEQ = LinearSegmentedColormap.from_list('seq_blue', ['#f4f8fd', '#cde2fb', '#86b6ef', '#2a78d6', '#184f95',
                                                          '#0d366b'])



def sty(flux, ts):
    """colour = flux, line style = time stepping"""
    return dict(color=C_FLUX[flux], ls=LS_TIME[ts], label=f'{LABEL_FLUX[flux]}, {LABEL_TIME[ts]}')


MM = 1 / 25.4
WIDTH_SINGLE, WIDTH_DOUBLE = 88 * MM, 180 * MM      # Elsevier / IOP column widths


def set_style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5, 'lines.linewidth': 1.2, 'lines.markersize': 4,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e',
        'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b',
        'legend.frameon': False, 'legend.handlelength': 2.4, 'savefig.dpi': 300, 'figure.dpi': 120,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'dejavusans',
    })


def panel_label(ax, label, x=-0.16, y=1.04):
    ax.text(x, y, f'({label})', transform=ax.transAxes, fontweight='bold', fontsize=9, va='bottom', ha='left')


def save_fig(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'{name}{FIG_SUFFIX}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {name}{FIG_SUFFIX}.pdf/.png')


def slope_guide(ax, x, order, y0, label, color='#8a8984'):
    """reference line ~ x^order through (x[0], y0)"""
    x = np.asarray(x)
    if x.size < 2:
        return
    ax.plot(x, y0 * (x / x[0]) ** order, ls=':', lw=0.9, color=color)
    ax.text(x[-1], y0 * (x[-1] / x[0]) ** order, f' {label}', color=color, fontsize=6.5, va='center', ha='left')


########################################################################################################################
# INPUTS, THEORY, RATE ESTIMATION #
########################################################################################################################

def input_current(kind, amp, t):
    """input current [A] on the time grid t [ms]"""
    t = np.asarray(t, dtype=float)
    if kind == 'const':
        return np.full_like(t, amp)
    if kind == 'step':
        return np.where((t > STEP_ON) & (t <= STEP_OFF), amp, 0.0)
    if kind == 'two-step':
        res = np.zeros_like(t)
        for on, off in TWO_STEP:
            res[(t > on) & (t <= off)] = amp
        return res
    if kind == 'sine':
        return np.where(t > SINE_ON, 0.5 * amp * (1 + np.sin(2 * np.pi * SINE_FREQ_HZ * 1e-3 * (t - SINE_ON))), 0.0)
    raise ValueError(kind)


def mu_of(amp):
    """input current [A] -> mean drive [mV]"""
    return amp / G_LEAK * 1e3


def siegert_rate(amp, sigma):
    """exact stationary rate [Hz] of the LIF with white noise (no refractory period), Siegert / Brunel-Hakim"""
    mu = mu_of(amp)
    if sigma == 0:
        return 1000 / (TAU * np.log(mu / (mu - (V_TH - E_REST)))) if mu > V_TH - E_REST else 0.0
    lo = (V_RESET - E_REST - mu) / (np.sqrt(2) * sigma)
    hi = (V_TH - E_REST - mu) / (np.sqrt(2) * sigma)
    integral = quad(lambda u: erfcx(-u), lo, hi, limit=500)[0]
    return 1000 / (TAU * np.sqrt(np.pi) * integral)


def eval_grid():
    return np.arange(0, T_SIM, EVAL_DT)


def smooth(rate_eval):
    """gaussian kernel with a fixed width in ms (independent of any simulation time step)"""
    return gaussian_filter1d(rate_eval, sigma=RATE_KERNEL_MS / EVAL_DT, mode='nearest')


def fpe_rate_on_eval(t, r_hz):
    return smooth(np.interp(eval_grid(), t, r_hz))


def lif_rate_on_eval(res, n):
    """rate [Hz] on the evaluation grid from a LIF result. 'grid' timing: the spikes counted at step k crossed the
    threshold during (t_{k-1}, t_k] and are spread uniformly over that interval (cumulative count, linear
    interpolation); 'interpolated' timing: spikes were already histogrammed on the evaluation grid."""
    te = eval_grid()
    if 'eval_counts' in res:
        counts = res['eval_counts']
    else:
        t, c = res['t'], res['step_counts']
        dt = t[1] - t[0]
        cum = np.concatenate([[0.0], np.cumsum(c)])
        t_cum = np.concatenate([[t[0] - dt], t])
        edges = np.append(te, te[-1] + EVAL_DT)
        counts = np.diff(np.interp(edges, t_cum, cum))
    return smooth(counts / (n * EVAL_DT * 1e-3))


def rel_l2(x, ref, window=None):
    sel = slice(None) if window is None else window
    return float(np.sqrt(np.mean((x[sel] - ref[sel]) ** 2)) / np.sqrt(np.mean(ref[sel] ** 2)))


########################################################################################################################
# CACHE #
########################################################################################################################

def _key(tag, kw):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)
    glob = dict(T=T_SIM, steps=(STEP_ON, STEP_OFF, TWO_STEP, SINE_ON, SINE_FREQ_HZ), u_min=U_MIN, tau=TAU)
    s = json.dumps({'tag': tag, 'glob': glob, **kw}, sort_keys=True, default=conv)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def cached(tag, fn, **kw):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f'{tag}_{_key(tag, kw)}.npz'
    if path.exists() and not FORCE_RECOMPUTE:
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    res = fn(**kw)
    np.savez_compressed(path, **res)
    return res


########################################################################################################################
# SIMULATORS #
########################################################################################################################

def _fpe_common(kind, amp, sigma, dt, dv, T):
    t = np.arange(0, T, dt)
    return t, dict(T=T, dt=dt, dv=dv, tau_mem=[TAU], voltage_sigma=sigma, input_function=input_current(kind, amp, t),
                   static_noise=True, connectivity_matrix=np.zeros((1, 1)), init_pdf_sigma=INIT_SIGMA,
                   init_pdf_offset=0, init_pdf_weight=0, u_inh=U_MIN, u_rest=E_REST, u_reset=V_RESET, u_thr=V_TH,
                   save_p='end', tqdm_disable=True, verbose=0)


def run_fpe_scheme(kind, amp, sigma, flux, time_scheme, dt, dv, T=None):
    """original / new solver combinations of Nykamp_Model_1 (solver 'hu-2021'):
    flux 'harmonic' | 'bernoulli', time_scheme 'backward-euler' | 'bdf2'.
    backward Euler = original time stepping (rate from the previous step, reinjection after the solve with
    renormalisation of the mass), BDF2 = new time stepping (implicit reinjection, Sherman-Morrison)."""
    T = T_SIM if T is None else T
    t, pars = _fpe_common(kind, amp, sigma, dt, dv, T)
    pars.update(input_type='stochastic-current', solver='hu-2021', current_factor=1, tau_ref=[0],
                flux_scheme=flux, time_scheme=time_scheme, linear_solver='banded')
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    name = str(TMP_DIR / f'nyk_{os.getpid()}')
    ok, mass = True, np.nan
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = Nykamp_Model_1(parameters=pars, name=name)
            model.simulate()
        r = np.asarray(model.r[0], dtype=float) * 1000
        r[-1] = r[-2]                       # last time point is not computed by the time loop
        if model.p is not None:
            mass = float(model.p[0, :, -1].sum() * dv)
        model.clean()
        if not np.all(np.isfinite(r)) or r.max() > 1e6:
            ok = False
    except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError) as e:
        ok, r = False, np.full(t.shape, np.nan)
        print(f'    {flux}/{time_scheme} failed for sigma={sigma}, amp={amp * 1e9:.1f} nA: {e}')
    cpu = time.perf_counter() - t0
    if not ok:
        r = np.full(t.shape, np.nan)
    return dict(t=t, r=r, cpu=np.array(cpu), ok=np.array(ok), mass=np.array(mass))


def run_fpe_new(kind, amp, sigma, dt, dv, T=None):
    """production solver FPE_Population (Bernoulli flux, BDF2, implicit reinjection, mass-preserving cutoff)"""
    T = T_SIM if T is None else T
    t, pars = _fpe_common(kind, amp, sigma, dt, dv, T)
    pars.update(flux_scheme='bernoulli')
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    name = str(TMP_DIR / f'fpe_{os.getpid()}')
    t0 = time.perf_counter()
    model = FPE_Population(parameters=pars, name=name)
    model.simulate()
    cpu = time.perf_counter() - t0
    r = np.asarray(model.r[0], dtype=float) * 1000
    mass = float(model.p[0, :, -1].sum() * dv)
    model.clean()
    return dict(t=t, r=r, cpu=np.array(cpu), ok=np.array(True), mass=np.array(mass))


def run_lif(kind, amp, sigma, n, dt, T=None, seed=0, timing='grid', bridge=False, t_count_from=None):
    """
    Streaming, vectorised LIF network with the update rule of LIF_basic.run (t_ref = 0):
        spike check -> reset -> Euler-Maruyama step  v += (-(v - E_rest) + mu) dt/tau + sigma sqrt(2 dt/tau) xi.
    timing 'grid':          spike is detected at the first grid point with v >= V_th (as LIF_basic), O(dt) bias
    timing 'interpolated':  threshold crossing time linearly interpolated within the step, the neuron is reset at
                            the crossing time and continues for the rest of the step (Hansel et al. 1998)
    bridge:                 additionally count crossings between two grid points that stay below threshold at both
                            ends, with the Brownian-bridge crossing probability exp(-2 (V_th-v0)(V_th-v1)/s^2)
                            (removes the O(sqrt(dt)) bias of discretely monitored thresholds in the noise-driven
                            regime, cf. Gobet 2000)
    t_count_from:           if given, also return the total number of spikes after this time (stationary rates)
    returns step counts (grid) or counts on the evaluation grid (interpolated) and the computation time
    """
    T = T_SIM if T is None else T
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, dt)
    Nt = len(t)
    mu = mu_of(input_current(kind, amp, t))                  # [mV]
    v = rng.normal(V_INIT, INIT_SIGMA, n)
    v[v > V_TH - 0.1] = V_TH - 0.1
    s = sigma * np.sqrt(2 * dt / TAU)
    step_counts = np.zeros(Nt)
    n_eval = len(np.arange(0, T, EVAL_DT))
    eval_counts = np.zeros(n_eval + 1)
    n_after = 0
    t0 = time.perf_counter()
    for it in range(Nt - 1):
        if timing == 'grid':
            fired = v >= V_TH
            step_counts[it] = np.count_nonzero(fired)
            v[fired] = V_RESET
            v += (-(v - E_REST) + mu[it]) * (dt / TAU) + s * rng.standard_normal(n)
        else:
            v_new = v + (-(v - E_REST) + mu[it]) * (dt / TAU) + s * rng.standard_normal(n)
            fired = v_new >= V_TH
            theta = np.empty(0)
            idx = np.flatnonzero(fired)
            if idx.size:
                theta = np.clip((V_TH - v[idx]) / (v_new[idx] - v[idx]), 0.0, 1.0)
            if bridge and s > 0:
                cand = np.flatnonzero((~fired) & (v_new > V_TH - 6 * s))
                if cand.size:
                    p = np.exp(-2 * (V_TH - v[cand]) * (V_TH - v_new[cand]) / s ** 2)
                    hit = cand[rng.random(cand.size) < p]
                    idx = np.concatenate([idx, hit])
                    theta = np.concatenate([theta, np.full(hit.size, 0.5)])
            if idx.size:
                rest = 1.0 - theta
                v_new[idx] = (V_RESET + (-(V_RESET - E_REST) + mu[it]) * rest * (dt / TAU)
                              + s * np.sqrt(rest) * rng.standard_normal(idx.size))
                t_spk = t[it] + theta * dt
                np.add.at(eval_counts, np.minimum((t_spk / EVAL_DT).astype(int), n_eval), 1)
                if t_count_from is not None:
                    n_after += np.count_nonzero(t_spk >= t_count_from)
            v = v_new
    cpu = time.perf_counter() - t0
    if timing == 'grid':
        if t_count_from is not None:
            n_after = step_counts[t >= t_count_from].sum()
        return dict(t=t, step_counts=step_counts, cpu=np.array(cpu), n_after=np.array(n_after))
    return dict(t=t, eval_counts=eval_counts[:n_eval], cpu=np.array(cpu), n_after=np.array(n_after))


def lif_reference(kind, amp, sigma, seed=0):
    return cached('lif', run_lif, kind=kind, amp=amp, sigma=sigma, n=LIF_REF_N, dt=LIF_REF_DT, seed=seed,
                  timing=LIF_REF_TIMING, bridge=LIF_REF_TIMING == 'interpolated')


def check_lif_equivalence(n=40, dt=0.01, T=15.0, seed=3):
    """run LIF_basic and the streaming implementation with the same random numbers (legacy numpy order:
    noise array in __init__, initial voltages in run()) and compare the spike counts step by step"""
    t = np.arange(0, T, dt)
    I = input_current('step', 1.0e-9, t)
    np.random.seed(seed)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    lif = LIF_basic(parameters={'T': T, 'dt': dt, 'tau_m': TAU, 't_ref': 0, 'n_neurons': n,
                                'Iinj': np.tile(I, (n, 1)), 'init_pdf_sigma': INIT_SIGMA, 'noise_sigma': 2.0,
                                'V_reset': V_RESET, 'E_rest': E_REST, 'V_th': V_TH, 'V_init': V_INIT,
                                'g_r': G_LEAK, 'fname': str(TMP_DIR / 'lif_basic_check')})
    lif.run()
    counts_basic = lif.t_spikes.sum(axis=0)
    os.remove(str(TMP_DIR / 'lif_basic_check') + '.hdf5')
    # same update with the legacy random numbers
    np.random.seed(seed)
    noise = np.random.normal(0, 1, (n, len(t)))
    v = np.random.normal(V_INIT, INIT_SIGMA, n)
    v[v > V_TH - 0.1] = V_TH - 0.1
    vinj = 1e3 * I / G_LEAK
    counts = np.zeros(len(t))
    for it in range(len(t) - 1):
        fired = v >= V_TH
        counts[it] = fired.sum()
        v[fired] = V_RESET
        v = v + (-(v - E_REST) + vinj[it]) * (dt / TAU) + 2.0 * np.sqrt(2 * dt / TAU) * noise[:, it]
    same = np.array_equal(counts, counts_basic)
    print(f'LIF_basic vs. streaming update: {"identical" if same else "DIFFERENT"} spike counts '
          f'({int(counts_basic.sum())} spikes)')
    return same


########################################################################################################################
# EXPERIMENT 1: LIF RATE AND THE GAUSSIAN FILTER #
########################################################################################################################

def exp_lif_filter():
    """
    Diagnosis of the LIF rate for different dt:
    (a) single deterministic neuron: ISI error of spike detection on the grid (O(dt)) vs. interpolated crossing
    (b) LIF rate filtered as in Utils.compare_firing_rate (sigma = 10 samples -> width 10 dt in ms)
    (c) the same simulations, filtered with a fixed kernel in ms -> differences that remain are discretisation
    """
    print('LIF rate vs. filter')
    amp, sigma, n = 1.3e-9, 0.1, 1000           # case of noise_LIF_network_2026.py
    dts = [0.1, 0.01, 0.001]

    # (a) deterministic single neuron, exact ISI
    mu = mu_of(amp)
    isi_exact = TAU * np.log(mu / (mu - (V_TH - E_REST)))
    dt_list = np.logspace(-3.5, -0.7, 14)
    err_grid, err_interp = [], []
    for dt in dt_list:
        # grid detection: first grid point with v >= V_th, exact solution between grid points
        k = np.ceil(isi_exact / dt - 1e-12)
        err_grid.append(abs(k * dt - isi_exact) / isi_exact)
        # explicit Euler + interpolated crossing
        v, tt = V_RESET, 0.0
        while True:
            v_new = v + (-(v - E_REST) + mu) * dt / TAU
            if v_new >= V_TH:
                tt += dt * (V_TH - v) / (v_new - v)
                break
            v, tt = v_new, tt + dt
        err_interp.append(abs(tt - isi_exact) / isi_exact)

    sims = {dt: cached('lif', run_lif, kind='step', amp=amp, sigma=sigma, n=n, dt=dt, seed=1, timing='grid',
                       bridge=False, T=20.0) for dt in dts}

    fig, axs = plt.subplots(1, 3, figsize=(WIDTH_DOUBLE, 52 * MM), constrained_layout=True)
    ax = axs[0]
    ax.loglog(dt_list, err_grid, 'o-', color=C_REF, mfc='white', label='spike on grid (LIF_basic)')
    ax.loglog(dt_list, err_interp, 's-', color=C_BERNOULLI, mfc='white', label='interpolated crossing')
    slope_guide(ax, dt_list[[2, -3]], 1, err_grid[2] * 2.2, r'$\propto\Delta t$')
    slope_guide(ax, dt_list[[2, -3]], 2, max(err_interp[2], 1e-12) * 0.35, r'$\propto\Delta t^2$')
    ax.set(xlabel=r'time step $\Delta t$ (ms)', ylabel='relative ISI error', title='single neuron, no noise')
    ax.legend(loc='lower right')
    panel_label(ax, 'a')

    for ax, label, mode in [(axs[1], 'b', 'samples'), (axs[2], 'c', 'ms')]:
        for c, dt in zip([BLUE_RAMP[1], BLUE_RAMP[3], BLUE_RAMP[6]], dts):
            res = sims[dt]
            r_step = res['step_counts'] / (n * dt * 1e-3)
            if mode == 'samples':        # exactly what compare_firing_rate does
                ax.plot(res['t'], gaussian_filter1d(r_step, 10), color=c, lw=1.0,
                        label=fr'$\Delta t$={dt} ms (kernel {10 * dt:g} ms)')
            else:
                ax.plot(eval_grid(), lif_rate_on_eval(res, n), color=c, lw=1.0, label=fr'$\Delta t$={dt} ms')
        ax.axhline(siegert_rate(amp, sigma), color='#8a8984', lw=0.8, ls=':')
        ax.set(xlabel='time (ms)', ylabel='rate (Hz)', xlim=(0, 20))
        ax.set_title('kernel 10 samples (compare_firing_rate)' if mode == 'samples'
                     else f'kernel {RATE_KERNEL_MS} ms for all ' + r'$\Delta t$')
        ax.legend(loc='upper right')
        panel_label(ax, label)
    save_fig(fig, 'fig1_lif_rate_filter_vs_dt')

    # text summary
    print(f'  exact ISI {isi_exact:.4f} ms ({1000 / isi_exact:.1f} Hz)')
    for dt in dts:
        res = sims[dt]
        r = lif_rate_on_eval(res, n)
        t_a, t_b = STEP_ON + 3.0, min(STEP_OFF, 20.0) - 1.0
        sel = (eval_grid() > t_a) & (eval_grid() < t_b)
        print(f'  dt={dt:<6}: mean rate {t_a:g}-{t_b:g} ms {r[sel].mean():7.1f} Hz, kernel of compare_firing_rate '
              f'{10 * dt:.3f} ms, cpu {float(res["cpu"]):.1f} s')


########################################################################################################################
# EXPERIMENT 2: LIF TIME STEP #
########################################################################################################################

def exp_lif_dt(sigma=2.0):
    """stationary rate vs. dt against the Siegert rate, and step responses for different dt"""
    print('LIF time step study')
    variants = [('grid', False, 'spike on grid (LIF_basic)', C_REF, 'o'),
                ('interpolated', False, 'interpolated crossing', C_HARMONIC, 's'),
                ('interpolated', True, 'interpolated + bridge', C_BERNOULLI, 'D')]
    fig, axs = plt.subplots(1, len(AMPLITUDES) + 1, figsize=(WIDTH_DOUBLE, 55 * MM), constrained_layout=True)
    rows = []
    for k, amp in enumerate(AMPLITUDES):
        ax = axs[k]
        r_exact = siegert_rate(amp, sigma)
        for timing, bridge, label, c, m in variants:
            errs = []
            for dt in LIF_DT_LIST:
                res = cached('lif_stat', run_lif, kind='const', amp=amp, sigma=sigma, n=LIF_DT_N, dt=dt,
                             T=LIF_DT_T_STAT, seed=0, timing=timing, bridge=bridge, t_count_from=LIF_DT_T_SKIP)
                rate = float(res['n_after']) / (LIF_DT_N * (LIF_DT_T_STAT - LIF_DT_T_SKIP) * 1e-3)
                errs.append(abs(rate - r_exact) / r_exact)
                rows.append(dict(amp_nA=amp * 1e9, sigma=sigma, dt=dt, timing=timing, bridge=bridge, rate=rate,
                                 siegert=r_exact, cpu=float(res['cpu'])))
            ax.loglog(LIF_DT_LIST, errs, marker=m, color=c, mfc='white', label=label)
        slope_guide(ax, np.array(LIF_DT_LIST[1:-1]), 0.5, 0.25, r'$\Delta t^{1/2}$')
        slope_guide(ax, np.array(LIF_DT_LIST[1:-1]), 1, 0.05, r'$\Delta t$')
        stat_err = 1 / np.sqrt(r_exact * LIF_DT_N * (LIF_DT_T_STAT - LIF_DT_T_SKIP) * 1e-3)
        ax.axhline(stat_err, color='#b4b3ad', lw=0.8, ls='--')
        ax.text(LIF_DT_LIST[-1], stat_err * 1.15, 'counting noise', fontsize=6, color='#8a8984')
        ax.set(xlabel=r'$\Delta t$ (ms)', title=f'{amp * 1e9:.1f} nA, ' + r'$\sigma$' + f'={sigma:g} mV '
               f'(Siegert {r_exact:.0f} Hz)')
        if k == 0:
            ax.set_ylabel('relative error of stationary rate')
            ax.legend(loc='lower right')
        panel_label(ax, 'abcd'[k])
    # step response traces for different dt, grid timing (as LIF_basic)
    ax = axs[-1]
    amp = AMPLITUDES[len(AMPLITUDES) // 2]
    ref = lif_reference('step', amp, sigma)
    ax.plot(eval_grid(), lif_rate_on_eval(ref, LIF_REF_N), color=C_REF, lw=2.2, alpha=0.35, label='reference')
    for c, dt in zip(BLUE_RAMP[1::2], [0.1, 0.05, 0.01, 0.001]):
        res = cached('lif', run_lif, kind='step', amp=amp, sigma=sigma, n=LIF_DT_N, dt=dt, seed=0, timing='grid',
                     bridge=False)
        ax.plot(eval_grid(), lif_rate_on_eval(res, LIF_DT_N), color=c, lw=0.9, label=fr'$\Delta t$={dt}')
    ax.set(xlabel='time (ms)', ylabel='rate (Hz)', xlim=(STEP_ON - 1, STEP_ON + 8),
           title=f'step onset, {amp * 1e9:.1f} nA, grid timing')
    ax.legend(loc='upper right')
    panel_label(ax, 'abcd'[len(AMPLITUDES)])
    save_fig(fig, 'fig2_lif_timestep')
    _write_table(rows, 'table_lif_timestep')


########################################################################################################################
# EXPERIMENT 3: NUMBER OF NEURONS #
########################################################################################################################

def fpe_fine(kind, amp, sigma):
    """fine solution of the new solver, used as the n -> infinity limit of the LIF network"""
    return cached('fpe_new', run_fpe_new, kind=kind, amp=amp, sigma=sigma, dt=0.002, dv=0.005)


def exp_lif_n(sigma=2.0, amp=1.0e-9):
    print('LIF population size study')
    kinds = ['step', 'sine']
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 95 * MM), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    rows = []
    for k, kind in enumerate(kinds):
        fine = fpe_fine(kind, amp, sigma)
        r_fpe = fpe_rate_on_eval(fine['t'], fine['r'])
        ax = fig.add_subplot(gs[k, 0:2])
        ax.plot(eval_grid(), r_fpe, color=C_BERNOULLI, lw=1.6, label='FPE (new solver, fine)', zorder=5)
        shown = LIF_N_LIST[::2] if len(LIF_N_LIST) > 3 else LIF_N_LIST
        for n in shown:
            res = cached('lif', run_lif, kind=kind, amp=amp, sigma=sigma, n=n, dt=LIF_N_DT, seed=LIF_N_SEEDS[0],
                         timing='interpolated', bridge=True)
            ax.plot(eval_grid(), lif_rate_on_eval(res, n), color=_gray_for(n), lw=0.7, label=f'LIF n={n:,}')
        ax.set(xlabel='time (ms)', ylabel='rate (Hz)', xlim=(0, T_SIM), title=f'{kind} input, {amp * 1e9:.1f} nA, '
               + r'$\sigma$' + f'={sigma:g} mV, ' + r'$\Delta t$' + f'={LIF_N_DT} ms')
        ax.legend(loc='upper right', ncol=2)
        panel_label(ax, 'ac'[k], x=-0.08)

        ax = fig.add_subplot(gs[k, 2])
        errs = np.full((len(LIF_N_SEEDS), len(LIF_N_LIST)), np.nan)
        for i, seed in enumerate(LIF_N_SEEDS):
            for j, n in enumerate(LIF_N_LIST):
                res = cached('lif', run_lif, kind=kind, amp=amp, sigma=sigma, n=n, dt=LIF_N_DT, seed=seed,
                             timing='interpolated', bridge=True)
                errs[i, j] = rel_l2(lif_rate_on_eval(res, n), r_fpe)
                rows.append(dict(kind=kind, n=n, seed=seed, rel_l2=errs[i, j], cpu=float(res['cpu'])))
        ax.loglog(LIF_N_LIST, errs.mean(axis=0), 'o-', color=C_REF, mfc='white', label='mean over seeds')
        if len(LIF_N_SEEDS) > 1:
            ax.fill_between(LIF_N_LIST, errs.min(axis=0), errs.max(axis=0), color=C_REF_BAND, lw=0)
        slope_guide(ax, np.array(LIF_N_LIST[:-1]), -0.5, errs.mean(axis=0)[0] * 1.6, r'$n^{-1/2}$')
        ax.set(xlabel='number of neurons n', ylabel='relative L2 error vs. FPE', title=f'kernel {RATE_KERNEL_MS} ms')
        panel_label(ax, 'bd'[k])
    save_fig(fig, 'fig3_lif_population_size')
    _write_table(rows, 'table_lif_population_size')


def _gray_for(n):
    """light gray for small n, dark for large n (ordinal, neutral, so the FPE line keeps the colour)"""
    lo, hi = np.log10(min(LIF_N_LIST)), np.log10(max(LIF_N_LIST))
    f = 0 if hi == lo else (np.log10(n) - lo) / (hi - lo)
    g = 0.78 - 0.55 * f
    return (g, g, g * 0.98)


########################################################################################################################
# EXPERIMENT 4: ORIGINAL VS. NEW SOLVER #
########################################################################################################################

def scheme_result(kind, amp, sigma, flux, ts, dt=None, dv=None):
    return cached('fpe_scheme', run_fpe_scheme, kind=kind, amp=amp, sigma=sigma, flux=flux, time_scheme=ts,
                  dt=FPE_DT if dt is None else dt, dv=FPE_DV if dv is None else dv)


def exp_solvers():
    print('original vs. new solver')
    rows = []
    sigmas = SIGMAS_BERNOULLI_ONLY + SIGMAS_ALL
    # traces: one figure per input shape, rows = amplitude, columns = sigma (all four combinations)
    for kind in INPUT_KINDS:
        fig, axs = plt.subplots(len(AMPLITUDES), len(SIGMAS_ALL), figsize=(WIDTH_DOUBLE, 40 * MM * len(AMPLITUDES)),
                                sharex=True, constrained_layout=True, squeeze=False)
        for i, amp in enumerate(AMPLITUDES):
            for j, sigma in enumerate(SIGMAS_ALL):
                ax = axs[i, j]
                ref = lif_reference(kind, amp, sigma)
                r_ref = lif_rate_on_eval(ref, LIF_REF_N)
                ax.plot(eval_grid(), r_ref, color=C_REF_BAND, lw=3.0, label=f'LIF (n={LIF_REF_N:,})', zorder=1)
                for flux, ts in SOLVER_COMBOS:
                    res = scheme_result(kind, amp, sigma, flux, ts)
                    if not bool(res['ok']):      # original flux under-/overflows: skipped
                        continue
                    ax.plot(eval_grid(), fpe_rate_on_eval(res['t'], res['r']), lw=0.9, zorder=3, **sty(flux, ts))
                if i == 0:
                    ax.set_title(r'$\sigma$' + f' = {sigma:g} mV')
                if j == 0:
                    ax.set_ylabel(f'{amp * 1e9:.1f} nA\nrate (Hz)')
                if i == len(AMPLITUDES) - 1:
                    ax.set_xlabel('time (ms)')
                panel_label(ax, 'abcdefghijklmnop'[i * len(SIGMAS_ALL) + j], x=-0.2)
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside upper center', ncol=5)
        save_fig(fig, f'fig4_solver_traces_{kind}')

    # error summary: relative L2 error vs. LIF reference against sigma
    fig, axs = plt.subplots(1, len(INPUT_KINDS), figsize=(WIDTH_DOUBLE, 58 * MM), sharey=True,
                            constrained_layout=True)
    markers = ['o', 's', '^', 'D', 'v']
    for k, kind in enumerate(INPUT_KINDS):
        ax = axs[k]
        for flux, ts in SOLVER_COMBOS:
            for a, amp in enumerate(AMPLITUDES):
                s_used = SIGMAS_ALL if flux == 'harmonic' else sigmas
                errs = []
                for sigma in s_used:
                    res = scheme_result(kind, amp, sigma, flux, ts)
                    r_ref = lif_rate_on_eval(lif_reference(kind, amp, sigma), LIF_REF_N)
                    e = rel_l2(fpe_rate_on_eval(res['t'], res['r']), r_ref) if bool(res['ok']) else np.nan
                    errs.append(e)
                    rows.append(dict(kind=kind, amp_nA=amp * 1e9, sigma=sigma, flux=flux, time=ts, rel_l2=e,
                                     cpu=float(res['cpu']), mass=float(res['mass']), ok=bool(res['ok'])))
                st = sty(flux, ts)
                if a > 0:
                    st['label'] = None
                ax.semilogy(s_used, errs, marker=markers[a], mfc='white', lw=0.9, **st)
        # LIF noise floor: two reference networks with different seeds
        floor = [rel_l2(lif_rate_on_eval(lif_reference(kind, AMPLITUDES[0], s, seed=1), LIF_REF_N),
                        lif_rate_on_eval(lif_reference(kind, AMPLITUDES[0], s), LIF_REF_N)) / np.sqrt(2)
                 for s in sigmas]
        ax.plot(sigmas, floor, color='#b4b3ad', ls='--', lw=0.8, label='LIF noise floor')
        ax.axvspan(min(sigmas) * 0.9, 2.0, color='#f0efec', lw=0, zorder=0)
        ax.set_xscale('log')
        ax.set(xlabel=r'noise $\sigma$ (mV)', title=f'{kind} input')
        if k == 0:
            ax.set_ylabel('relative L2 error vs. LIF network')
            ax.legend(loc='upper right', fontsize=6)
            ax.text(0.03, 0.03, 'markers: ' + ', '.join(f'{m} {a * 1e9:.1f} nA' for m, a in
                                                       zip(markers, AMPLITUDES)),
                    transform=ax.transAxes, fontsize=6, color='#52514e')
        panel_label(ax, 'abc'[k])
    save_fig(fig, 'fig5_solver_error_summary')
    _write_table(rows, 'table_solver_comparison')


########################################################################################################################
# EXPERIMENT 5: CONVERGENCE IN TIME AND SPACE #
########################################################################################################################

def exp_convergence(sigma=2.0, amp=1.0e-9):
    print('convergence in time and space')
    rows = []
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH_DOUBLE, 55 * MM), constrained_layout=True)
    dv_time = 0.02
    for k, kind in enumerate(['step', 'sine']):
        ax = axs[k]
        for flux, ts in SOLVER_COMBOS:
            ref = scheme_result(kind, amp, sigma, flux, 'bdf2', dt=CONV_DT_REF, dv=dv_time)
            if not bool(ref['ok']):
                continue
            r_ref = fpe_rate_on_eval(ref['t'], ref['r'])
            errs = []
            for dt in CONV_DT_LIST:
                res = scheme_result(kind, amp, sigma, flux, ts, dt=dt, dv=dv_time)
                e = rel_l2(fpe_rate_on_eval(res['t'], res['r']), r_ref) if bool(res['ok']) else np.nan
                errs.append(e)
                rows.append(dict(study='time', kind=kind, flux=flux, time=ts, dt=dt, dv=dv_time, rel_l2=e))
            ax.loglog(CONV_DT_LIST, errs, marker='o' if ts == 'bdf2' else 's', mfc='white', lw=0.9, **sty(flux, ts))
        x = np.array(CONV_DT_LIST[1:])
        slope_guide(ax, x, 1, 0.3, r'$\Delta t$')
        slope_guide(ax, x, 2, 0.02, r'$\Delta t^2$')
        ax.set(xlabel=r'$\Delta t$ (ms)', ylabel='relative L2 error' if k == 0 else None,
               title=f'time convergence, {kind} input\n(reference: same flux, BDF2, ' + r'$\Delta t$' +
               f'={CONV_DT_REF} ms, ' + r'$\Delta v$' + f'={dv_time} mV)')
        if k == 0:
            ax.legend(loc='lower right', fontsize=6)
        panel_label(ax, 'ab'[k])

    # space: stationary rate vs. Siegert
    ax = axs[2]
    for c_i, sig in enumerate([2.0, 4.5]):
        r_exact = siegert_rate(amp, sig)
        for flux in ['bernoulli', 'harmonic']:
            errs = []
            for dv in CONV_DV_LIST:
                res = cached('fpe_scheme_stat', run_fpe_scheme, kind='const', amp=amp, sigma=sig, flux=flux,
                             time_scheme='bdf2', dt=0.005, dv=dv, T=100.0)
                rate = float(np.mean(res['r'][res['t'] > 60.0])) if bool(res['ok']) else np.nan
                errs.append(abs(rate - r_exact) / r_exact)
                rows.append(dict(study='space', sigma=sig, flux=flux, dv=dv, rate=rate, siegert=r_exact,
                                 rel_err=errs[-1]))
            ax.loglog(CONV_DV_LIST, errs, marker='o' if c_i == 0 else '^', color=C_FLUX[flux], mfc='white', lw=0.9,
                      ls='-' if c_i == 0 else (0, (3, 1.5)), label=f'{LABEL_FLUX[flux]}, ' + r'$\sigma$' + f'={sig:g}')
    x = np.array(CONV_DV_LIST[1:])
    slope_guide(ax, x, 1, 0.1, r'$\Delta v$')
    slope_guide(ax, x, 2, 0.01, r'$\Delta v^2$')
    ax.set(xlabel=r'$\Delta v$ (mV)', ylabel='relative error of stationary rate',
           title=f'space convergence vs. Siegert rate\n({amp * 1e9:.1f} nA, BDF2, ' + r'$\Delta t$=0.005 ms)')
    ax.legend(loc='lower right', fontsize=6)
    panel_label(ax, 'c')
    save_fig(fig, 'fig6_convergence_time_space')
    _write_table(rows, 'table_convergence')


########################################################################################################################
# EXPERIMENT 6: FPE RESOLUTION VS. LIF, COMPUTATION TIME #
########################################################################################################################

def exp_resolution(kind='step', sigma=2.0, amp=1.0e-9):
    print('FPE resolution vs. LIF reference')
    ref = lif_reference(kind, amp, sigma)
    r_ref = lif_rate_on_eval(ref, LIF_REF_N)
    err = np.full((len(RES_DV_LIST), len(RES_DT_LIST)), np.nan)
    cpu = np.full_like(err, np.nan)
    rows = []
    for i, dv in enumerate(RES_DV_LIST):
        for j, dt in enumerate(RES_DT_LIST):
            res = cached('fpe_new', run_fpe_new, kind=kind, amp=amp, sigma=sigma, dt=dt, dv=dv)
            err[i, j] = rel_l2(fpe_rate_on_eval(res['t'], res['r']), r_ref)
            cpu[i, j] = float(res['cpu'])
            rows.append(dict(kind=kind, sigma=sigma, amp_nA=amp * 1e9, dv=dv, dt=dt, rel_l2=err[i, j], cpu=cpu[i, j],
                             mass=float(res['mass'])))
    floor = rel_l2(lif_rate_on_eval(lif_reference(kind, amp, sigma, seed=1), LIF_REF_N), r_ref) / np.sqrt(2)

    # LIF configurations (n, dt) against the same reference
    lif_pts = []
    for n in [n_ for n_ in LIF_N_LIST if n_ < LIF_REF_N]:
        for dt in [0.05, 0.01, LIF_N_DT]:
            res = cached('lif', run_lif, kind=kind, amp=amp, sigma=sigma, n=n, dt=dt, seed=LIF_N_SEEDS[0],
                         timing='interpolated', bridge=True)
            lif_pts.append((n, dt, rel_l2(lif_rate_on_eval(res, n), r_ref), float(res['cpu'])))
            rows.append(dict(kind=kind, model='LIF', n=n, dt=dt, rel_l2=lif_pts[-1][2], cpu=lif_pts[-1][3]))

    fig, axs = plt.subplots(1, 3, figsize=(WIDTH_DOUBLE, 60 * MM), constrained_layout=True)
    for ax, data, title, lab, norm in [(axs[0], err, 'relative L2 error vs. LIF', 'a', LogNorm()),
                                       (axs[1], cpu, 'computation time (s)', 'b', LogNorm())]:
        im = ax.imshow(data, cmap=CMAP_SEQ, norm=norm, origin='upper', aspect='auto')
        ax.set_xticks(range(len(RES_DT_LIST)), [f'{d:g}' for d in RES_DT_LIST], rotation=45)
        ax.set_yticks(range(len(RES_DV_LIST)), [f'{d:g}' for d in RES_DV_LIST])
        ax.set(xlabel=r'$\Delta t$ (ms)', ylabel=r'$\Delta v$ (mV)', title=title)
        for (i, j), val in np.ndenumerate(data):
            if np.isfinite(val):
                light = im.norm(val) > 0.6
                ax.text(j, i, f'{val:.2g}', ha='center', va='center', fontsize=5.5,
                        color='white' if light else '#0b0b0b')
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.spines[:].set_visible(False)
        panel_label(ax, lab, x=-0.25)

    ax = axs[2]
    ax.loglog(cpu.ravel(), err.ravel(), 'o', color=C_BERNOULLI, mfc='white', label=r'FPE ($\Delta v$, $\Delta t$ grid)')
    lp = np.array([(p[3], p[2]) for p in lif_pts])
    ax.loglog(lp[:, 0], lp[:, 1], 's', color=C_REF, mfc='white', label=r'LIF network (n, $\Delta t$)')
    for n, dt, e, c in lif_pts:
        if dt == LIF_N_DT:
            ax.annotate(f'n={n:,}', (c, e), fontsize=5.5, color='#52514e', xytext=(3, 2), textcoords='offset points')
    ax.axhline(floor, color='#b4b3ad', ls='--', lw=0.8)
    ax.text(ax.get_xlim()[0] if np.isfinite(floor) else 1, floor * 1.1, ' reference noise floor', fontsize=6,
            color='#8a8984')
    ax.set(xlabel='computation time (s)', ylabel='relative L2 error vs. LIF reference',
           title=f'accuracy vs. cost ({kind}, {amp * 1e9:.1f} nA, ' + r'$\sigma$' + f'={sigma:g} mV)')
    ax.legend(loc='upper right', fontsize=6)
    panel_label(ax, 'c')
    save_fig(fig, f'fig7_fpe_resolution_{kind}')
    _write_table(rows, f'table_fpe_resolution_{kind}')


########################################################################################################################
# HELPERS #
########################################################################################################################

def _write_table(rows, name):
    """csv table of all numbers behind a figure"""
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path = OUT_DIR / f'{name}{FIG_SUFFIX}.csv'
    with open(path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join('' if r.get(k) is None else str(r.get(k)) for k in keys) + '\n')
    print(f'  saved {path.name}')


EXPERIMENTS = {'lif_filter': exp_lif_filter, 'lif_dt': exp_lif_dt, 'lif_n': exp_lif_n, 'solvers': exp_solvers,
               'convergence': exp_convergence, 'resolution': lambda: [exp_resolution(k) for k in INPUT_KINDS]}


def main():
    global FORCE_RECOMPUTE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exp', nargs='+', default=['all'], choices=['all'] + list(EXPERIMENTS))
    parser.add_argument('--quick', action='store_true', help='reduced resolution for a first look')
    parser.add_argument('--smoke', action='store_true', help='tiny settings, only checks that everything runs')
    parser.add_argument('--force', action='store_true', help='ignore cached results')
    parser.add_argument('--check-lif', action='store_true', help='check streaming LIF against LIF_basic and exit')
    args = parser.parse_args()
    if args.smoke:
        apply_quick_settings('smoke')
    elif args.quick:
        apply_quick_settings('quick')
    FORCE_RECOMPUTE = args.force
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.check_lif:
        check_lif_equivalence()
        return
    names = list(EXPERIMENTS) if 'all' in args.exp else args.exp
    for name in names:
        t0 = time.perf_counter()
        EXPERIMENTS[name]()
        print(f'  [{name}] done in {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()

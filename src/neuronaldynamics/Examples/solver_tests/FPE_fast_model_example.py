"""
Example of the recommended fast model against the converged network: three populations, step input, sigma = 3.16 mV
(FPE_2D_reference_test.CASES['3pop_step_3.16']).

    FPE: FPE_Population(soma_model='IK', adaptation_closure='conditional_quadrature', flux_limiter='fct', dv=0.1,
         dt=0.05), p(v, t) saved every SNAP_EVERY ms
    network: IK network, stochastic Heun, MC_DT = 0.01 ms, N_NET neurons per population, histograms of v every
         SNAP_EVERY ms (reference for the voltage densities)
    reference rates: FPE_reference_rebuild.py (Heun, dt 0.005 ms, 100k neurons per population)

The one-dimensional closure evolves p(v, t) and the conditional moments of u; the heat maps compare its p(v, t) with the
voltage histogram of the network (the marginal of p(v, u, t) over u).

Outputs in Examples/solver_tests/FPE_fast_model_example: fpe.npz, network.npz, fast_model_example.png/.pdf

Usage (from src/neuronaldynamics):
    python Examples/solver_tests/FPE_fast_model_example.py run       (FPE and network in parallel, ~6 min)
    python Examples/solver_tests/FPE_fast_model_example.py analyze
"""
import os
for _var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_var, '1')
import sys
import time
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Model.Nykamp_Model import FPE_Population
import FPE_vs_IK_network as net
import FPE_Izhikevich_3pop_improvements as imp
import FPE_2D_reference_test as t2d
import FPE_reference_rebuild as rebuild
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'FPE_fast_model_example'
OUT_DIR.mkdir(exist_ok=True)
CASE = '3pop_step_3.16'
DV, DT = 0.1, 0.05
SNAP_EVERY = 0.5                          # ms
N_NET, NET_SEED, NET_DT = 50_000, 700_000, 0.01
V_BINS = np.arange(-90., 35.01, 0.5)      # mV, bins of the voltage histograms (whole grid, v_min to v_peak)
POP = ['exc 1 (input)', 'exc 2', 'inh']


def snapshot_times():
    return np.arange(0., net.T + 1e-9, SNAP_EVERY)


def binned_density(v, p):
    """p on the (possibly non-uniform) FPE grid v -> density per mV in V_BINS (mass of the control volumes)"""
    w = np.gradient(v)
    idx = np.digitize(v, V_BINS) - 1
    ok = (idx >= 0) & (idx < len(V_BINS) - 1)
    return np.bincount(idx[ok], weights=(p * w)[ok], minlength=len(V_BINS) - 1) / np.diff(V_BINS)


def run(task):
    t0 = time.perf_counter()
    pars = t2d.setup_case(CASE)
    if task == 'fpe':
        # timed run without saving p (as in the cost table), then a run that saves p for the heat maps
        m = FPE_Population(dict(pars, soma_model='IK', adaptation_closure='conditional_quadrature', flux_limiter='fct',
                                dt=DT, dv=DV, init_pdf_sigma=1.0, save_p='none', tqdm_disable=True),
                           name=str(OUT_DIR / 'fpe_timed'))
        m.simulate()
        cpu = time.perf_counter() - t0
        m.clean()
        m = FPE_Population(dict(pars, soma_model='IK', adaptation_closure='conditional_quadrature', flux_limiter='fct',
                                dt=DT, dv=DV, init_pdf_sigma=1.0, save_p='all',
                                p_save_every=int(round(SNAP_EVERY / DT)), tqdm_disable=True),
                           name=str(OUT_DIR / 'fpe_p'))
        m.simulate()
        dens = np.array([[binned_density(m.v, m.p[j, :, k]) for k in range(len(m.t_p))] for j in range(3)])
        np.savez(OUT_DIR / 'fpe.npz', t_snap=m.t_p, dens=dens, rb=net.binned_rate(m.r, DT), time=cpu)
        m.clean()
    else:
        imp.set_input(t2d.CASES[CASE][2])
        net.MC_DT, net.MC_METHOD = NET_DT, 'heun'
        r, _, t_s, v_s, _ = net.ik_network('3pop', t2d.CASES[CASE][1], N_NET, seed=NET_SEED,
                                           snapshot_times=snapshot_times())
        dens = np.array([[np.histogram(v_s[k, j], V_BINS)[0] / (N_NET * np.diff(V_BINS)) for k in range(len(t_s))]
                         for j in range(3)])
        np.savez(OUT_DIR / 'network.npz', t_snap=t_s, dens=dens, rb=net.binned_rate(r, NET_DT),
                 time=time.perf_counter() - t0)
    print(f'{task}: {time.perf_counter() - t0:.0f} s', flush=True)


def analyze():
    fpe = dict(np.load(OUT_DIR / 'fpe.npz'))
    nw = dict(np.load(OUT_DIR / 'network.npz'))
    # ik_network starts its loop at the first step, so a snapshot at t = 0 is never filled (v = 0): drop it
    keep = nw['t_snap'] > 0
    nw['t_snap'], nw['dens'] = nw['t_snap'][keep], nw['dens'][:, keep]
    fpe_k = np.isin(np.round(fpe['t_snap'], 6), np.round(nw['t_snap'], 6))
    ref = dict(np.load(rebuild.ref_file(CASE, rebuild.REF_DT, rebuild.N_REF)))
    e_fpe = imp.measures(fpe['rb'], ref['rb'])
    e_nw = imp.measures(nw['rb'], ref['rb'])
    print(f'FPE two-node + FCT, dv {DV}, dt {DT}: {float(fpe["time"]):.1f} s, smoothed rel. L2 '
          f'{e_fpe["rel_l2_smooth"]:.4f} (per population '
          + ', '.join(f'{e_fpe[f"rel_l2_smooth_pop{j}"]:.4f}' for j in range(3)) + f'), unsmoothed {e_fpe["rel_l2"]:.4f}')
    print(f'network Heun {N_NET // 1000}k: {float(nw["time"]):.0f} s, smoothed rel. L2 {e_nw["rel_l2_smooth"]:.4f}'
          f', unsmoothed {e_nw["rel_l2"]:.4f}')
    vc = 0.5 * (V_BINS[1:] + V_BINS[:-1])
    for j in range(3):
        d = np.abs(fpe['dens'][j][fpe_k] - nw['dens'][j]).sum(axis=1) * np.diff(V_BINS)[0]
        print(f'{POP[j]}: L1 distance of the voltage densities (FPE vs network {N_NET // 1000}k), '
              f'median {np.median(d):.3f}, max {d.max():.3f} at t = {nw["t_snap"][np.argmax(d)]:.1f} ms')

    tb = np.arange(ref['rb'].shape[1]) * net.RATE_BIN_MS + 0.5 * net.RATE_BIN_MS
    fig, axs = plt.subplots(3, 3, figsize=(13, 8.6), constrained_layout=True, sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1.25, 1.25]))
    ylim = [(-75, -35), (-75, -35), (-65, -35)]
    for j in range(3):
        ax = axs[0, j]
        ax.plot(tb, imp.smooth(ref['rb'][j]), color='#3d3c39', lw=2.2, label='network, $10^5$ (reference)')
        ax.plot(tb, imp.smooth(fpe['rb'][j]), color='#d6612a', lw=1.3, ls='--', label='FPE two-node + FCT')
        ax.plot(tb, ref['rb'][j], color='#3d3c39', lw=0.6, alpha=0.35)
        ax.set_title(f'{POP[j]}: smoothed error {e_fpe[f"rel_l2_smooth_pop{j}"] * 100:.1f}\\,%'.replace('\\,', ' '),
                     fontsize=10)
        ax.set_ylabel('rate (Hz)')
        ax.axvspan(0, net.T_ON, color='#f2efe8', zorder=0)
        ax.grid(lw=0.4, color='#e0dfdb')
        ax.spines[['top', 'right']].set_visible(False)
        vmax = max(fpe['dens'][j].max(), nw['dens'][j].max())
        norm = LogNorm(vmin=vmax * 1e-3, vmax=vmax)
        for row, (d, name) in enumerate([(fpe['dens'][j], 'FPE'), (nw['dens'][j], f'network {N_NET // 1000}k')], 1):
            ax = axs[row, j]
            ts = fpe['t_snap'] if name == 'FPE' else nw['t_snap']
            im = ax.pcolormesh(ts, vc, np.maximum(d, vmax * 1e-3).T, norm=norm, cmap='magma', shading='nearest',
                               rasterized=True)
            ax.set_ylim(*ylim[j])
            ax.set_ylabel(f'{name}\n$v$ (mV)')
            if row == 2:
                ax.set_xlabel('time (ms)')
        fig.colorbar(im, ax=axs[1:, j], shrink=0.8, label='$p(v,t)$ (1/mV)')
    axs[0, 0].legend(fontsize=8, frameon=False, loc='upper right')
    for ext in ['png', 'pdf']:
        fig.savefig(OUT_DIR / f'fast_model_example.{ext}', dpi=200)
    print('saved', OUT_DIR / 'fast_model_example.png')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'analyze'
    if mode == 'run':
        with Pool(2) as pool:
            pool.map(run, ['network', 'fpe'])
    analyze()

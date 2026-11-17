"""
Grid convergence and computation time of FPE_Izhikevich (FPE_Population with soma_model='izhikevich') for the three
adaptation closures, compared with Monte Carlo simulations of Izhikevich neuron populations of different size, in the
style of FPE_vs_LIF_timing.py.

Sweep: voltage step DVS x time step DTS for every closure in CLOSURES, step input (T = 200 ms, input on for the first
100 ms, then off) in the regimes of REGIMES. Monte Carlo (MC_METHOD, dt = MC_DT) with N_LIST neurons, the largest
one (N_LIST[-1]) is the reference.

Errors (rate binned in RATE_BIN_MS bins, so runs with different dt are compared on the same time axis; the bins are
computed from the cumulative spike count, i.e. without interpolation errors):
- err_ref:   relative L2 error of the binned rate against the Monte Carlo reference (closure error + discretization
             error, for the smaller Monte Carlo runs: finite size error)
- err_grid:  relative L2 error against the FPE with the same closure on the finest grid (DVS[0], DTS[0]), i.e. the pure
             discretization error of the grid
- err_u:     RMS error of <u> (pA) against the Monte Carlo reference

Computation time: time.perf_counter() around the simulation only, N_PROCESSES simulations in parallel (numpy limited to
one thread each, N_PROCESSES = 1 gives the cleanest timings, but takes much longer).

Outputs in Examples/solver_tests/FPE_Izhikevich_resolution_heun (MC_METHOD = 'heun', the default; Euler-Maruyama:
FPE_Izhikevich_resolution):
    error_and_time_vs_dv.png/.pdf           per regime: err_ref, err_grid and computation time against dv (one line per
                                            dt, one column block per closure), Monte Carlo as horizontal lines
    work_precision.png/.pdf                 err_ref against computation time of all FPE grids and Monte Carlo sizes
    rates_<regime>.png                      rates of the finest and the recommended grid against the reference
    resolution_table.csv                    all numbers
    resolution_table.tex / .pdf             recommended (fastest) grid per closure and regime with err_grid < GRID_TOL

Usage (from src/neuronaldynamics):  python Examples/solver_tests/FPE_Izhikevich_resolution.py
"""
import os
for _var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_var, '1')
import sys
import csv
import time
import shutil
import subprocess
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/neuronaldynamics, for direct calls
from Model.Nykamp_Model import FPE_Izhikevich, IZHIKEVICH_DEFAULTS
matplotlib.use('Agg')  # after the imports (Utils switches the backend)
warnings.filterwarnings('ignore', message='.*non-interactive.*')


########################################################################################################################
# SETTINGS
########################################################################################################################
CLOSURES = ['mean', 'conditional', 'conditional_variance']
DVS = [0.02, 0.05, 0.1, 0.25, 0.5]          # mV, every dv has to put c (and v_peak) on the grid (v_min = -90 mV)
DTS = [0.01, 0.025, 0.05, 0.1]              # ms
MC_DT = 0.01                                 # ms, Monte Carlo time step
MC_METHOD = 'heun'                           # 'heun' (FPE_vs_IK_network.ik_network) or 'euler' (Euler-Maruyama)
OUT_DIR = Path(__file__).resolve().parent / ('FPE_Izhikevich_resolution' + ('' if MC_METHOD == 'euler' else f'_{MC_METHOD}'))
OUT_DIR.mkdir(exist_ok=True)
N_LIST = [20_000, 50_000, 100_000]           # Monte Carlo neurons, the last one is the reference
N_PROCESSES = 4
T, T_ON = 200., 100.                         # ms
RATE_BIN_MS = 1.0                            # bins of the rate for all errors
GRID_TOL = 0.02                              # recommended grid: fastest grid with err_grid < GRID_TOL

# name: (title, parameters differing from IZHIKEVICH_DEFAULTS (RS neuron), step amplitude I0 in pA)
REGIMES = {
    'rs_default':   ('RS default (a=0.03, d=100, I0=100)', dict(), 100.),
    'rs_high_rate': ('RS, high rate (I0=300)', dict(), 300.),
    'fast':         ('fast adaptation (a=0.1, I0=200)', dict(a=0.1), 200.),
}
CLOSURE_LABELS = {'mean': r'mean $\langle u\rangle$', 'conditional': r'cond., $\sigma_u^2=0$',
                  'conditional_variance': r'cond., $\sigma_u^2(t)$'}
CLOSURE_SHORT = {'mean': 'mean', 'conditional': 'cond.', 'conditional_variance': 'cond.+var.'}


def step_input(t, i0):
    return np.where(t < T_ON, i0, 0.)


def binned_rate(t, r, dt):
    """
    rate (Hz) in RATE_BIN_MS bins from the cumulative spike count of a rate r (1/ms) on the time axis t (step dt)
    """
    edges = np.arange(0, T + 1e-9, RATE_BIN_MS)
    cum = np.concatenate([[0.], np.cumsum(r * dt)])
    t_cum = np.concatenate([t, [t[-1] + dt]])
    return np.diff(np.interp(edges, t_cum, cum)) / RATE_BIN_MS * 1e3


def monte_carlo(par, i0, n, seed):
    """
    simulation of n Izhikevich neurons (as in FPE_Izhikevich_closure_test.py), integrator MC_METHOD: Euler-Maruyama
    ('euler') or stochastic Heun with interpolated spike times ('heun', as FPE_vs_IK_network.ik_network)
    :return: t, rate (1/ms), <u> (pA)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, MC_DT)
    I = step_input(t, i0)
    D = par['voltage_sigma'] ** 2 / par['tau_noise']
    v = rng.normal(par['v_r'], 1.0, n)
    u = par['b'] * (v.mean() - par['v_r']) * np.ones(n)
    r = np.zeros(len(t))
    u_mean = np.zeros(len(t))
    u_mean[0] = u.mean()
    noise = np.sqrt(2 * D * MC_DT)
    f_v = lambda v, u, i_in: (par['k'] * (v - par['v_r']) * (v - par['v_t']) - u + i_in) / par['C']
    f_u = lambda v, u: par['a'] * (par['b'] * (v - par['v_r']) - u)
    for i in range(1, len(t)):
        dvdt = f_v(v, u, I[i])
        dudt = f_u(v, u)
        if MC_METHOD == 'euler':
            v = v + MC_DT * dvdt + noise * rng.standard_normal(n)
            u = u + MC_DT * dudt
            spikes = v >= par['v_peak']
            v[spikes] = par['c']
            u[spikes] += par['d']
        else:
            xi = noise * rng.standard_normal(n)
            v_p = np.minimum(v + MC_DT * dvdt + xi, 10 * par['v_peak'])
            u_p = u + MC_DT * dudt
            v_new = v + 0.5 * MC_DT * (dvdt + f_v(v_p, u_p, I[i])) + xi
            u_new = u + 0.5 * MC_DT * (dudt + f_u(v_p, u_p))
            spikes = v_new >= par['v_peak']
            theta = np.clip((par['v_peak'] - v) / np.maximum(v_new - v, 1e-12), 0., 1.)
            u_s = u + theta * (u_new - u) + par['d']
            rest = (1. - theta) * MC_DT
            v_s = par['c'] + rest * f_v(par['c'], u_s, I[i])
            u_s = u_s + rest * f_u(par['c'], u_s)
            v = np.where(spikes, v_s, v_new)
            u = np.where(spikes, u_s, u_new)
        r[i] = spikes.sum() / (n * MC_DT)
        u_mean[i] = u.mean()
    return t, r, u_mean


def run(task):
    """
    one timed simulation, task = (regime, 'mc', n, None) or (regime, closure, dv, dt)
    """
    regime, kind, x, dt = task
    par_regime, i0 = REGIMES[regime][1], REGIMES[regime][2]
    if kind == 'mc':
        par = dict(IZHIKEVICH_DEFAULTS, **par_regime)
        t0 = time.perf_counter()
        t, r, u = monte_carlo(par, i0, x, seed=x)
        cpu = time.perf_counter() - t0
        return dict(task=task, time=cpu, t=t, r=r, u=u, rb=binned_rate(t, r, MC_DT))
    fpe = FPE_Izhikevich(dict(par_regime, T=T, dt=dt, dv=x, input_function=lambda t: step_input(t, i0),
                              input_function_idx=[0], save_p='none', init_pdf_sigma=1.0, tqdm_disable=True,
                              adaptation_closure=kind),
                         name=str(OUT_DIR / f'{regime}_{kind}_dv{x}_dt{dt}'))
    t0 = time.perf_counter()
    fpe.simulate()
    cpu = time.perf_counter() - t0
    fpe.clean()
    r = fpe.r[0].copy()
    return dict(task=task, time=cpu, t=fpe.t, r=r, u=fpe.u_mean[0].copy(), rb=binned_rate(fpe.t, r, dt))


def rel_l2(x, ref):
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


def rms_u(res, ref):
    """RMS error of <u> on the Monte Carlo time axis"""
    u = np.interp(ref['t'], res['t'], res['u'])
    return float(np.sqrt(np.mean((u - ref['u']) ** 2)))


########################################################################################################################
# PLOTS (style of FPE_vs_LIF_timing.py)
########################################################################################################################
DT_COLORS = dict(zip(DTS, ['#1b4f8f', '#2a78d6', '#6fa8e6', '#b3d1f2']))
MC_COLORS = dict(zip(N_LIST, ['#b4b3ad', '#8a8984', '#3d3c39']))
CLOSURE_MARKERS = {'mean': 'o', 'conditional': 's', 'conditional_variance': 'D'}
CLOSURE_COLORS = {'mean': '#2a78d6', 'conditional': '#d6612a', 'conditional_variance': '#2a9d6b'}


def style(ax):
    ax.grid(which='major', lw=0.4, color='#e0dfdb')
    ax.spines[['top', 'right']].set_visible(False)


def plot_error_and_time(rows):
    """
    rows: regimes, columns: per closure err_ref | err_grid | time against dv, one line per dt
    """
    n_reg, n_cl = len(REGIMES), len(CLOSURES)
    fig, axs = plt.subplots(n_reg, 3 * n_cl, figsize=(4.0 * n_cl * 1.55, 2.6 * n_reg + 0.9), sharex=True,
                            constrained_layout=True)
    axs = np.atleast_2d(axs)
    for i, regime in enumerate(REGIMES):
        mc = {r['n']: r for r in rows if r['regime'] == regime and r['model'] == 'MC'}
        for c_idx, closure in enumerate(CLOSURES):
            a_ref, a_grid, a_time = axs[i, 3 * c_idx:3 * c_idx + 3]
            for dt in DTS:
                sel = sorted([r for r in rows if r['regime'] == regime and r['model'] == closure and r['dt'] == dt],
                             key=lambda r: r['dv'])
                dvs = [r['dv'] for r in sel]
                kw = dict(color=DT_COLORS[dt], marker=CLOSURE_MARKERS[closure], ms=4, mfc='white', lw=1.2,
                          label=f'FPE dt={dt} ms')
                a_ref.plot(dvs, [r['err_ref'] for r in sel], **kw)
                g = [(d, r['err_grid']) for d, r in zip(dvs, sel) if r['err_grid'] > 0]
                if g:
                    a_grid.plot(*zip(*g), **kw)
                a_time.plot(dvs, [r['time'] for r in sel], **kw)
            for n, r in mc.items():
                kw = dict(color=MC_COLORS[n], lw=1.0, ls='--' if n != N_LIST[-1] else '-',
                          label=f'MC n={n:,}' + (' (reference)' if n == N_LIST[-1] else ''))
                if n != N_LIST[-1]:
                    a_ref.axhline(r['err_ref'], **kw)
                a_time.axhline(r['time'], **kw)
            a_grid.axhline(GRID_TOL, color='#c44', lw=0.8, ls=':', label=f'tolerance {GRID_TOL}')
            for ax in (a_ref, a_grid, a_time):
                ax.set_xscale('log')
                ax.set_yscale('log')
                style(ax)
            if i == 0:
                # closure on the first line, quantity on the second (short titles, no overlap of neighbouring panels)
                a_ref.set_title(f'{CLOSURE_LABELS[closure]}\nerror vs. MC ref.', fontsize=8)
                a_grid.set_title(f'{CLOSURE_LABELS[closure]}\nerror vs. finest grid', fontsize=8)
                a_time.set_title(f'{CLOSURE_LABELS[closure]}\ncomputation time (s)', fontsize=8)
            if i == n_reg - 1:
                for ax in (a_ref, a_grid, a_time):
                    ax.set_xlabel(r'$\Delta v$ (mV)')
        axs[i, 0].set_ylabel(f'{REGIMES[regime][0]}\nrel. L2 error of the rate', fontsize=9)
    handles, labels = [], []
    for ax in (axs[0, 2], axs[0, 1]):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc='outside upper center', ncol=len(labels), frameon=False, fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT_DIR / f'error_and_time_vs_dv.{ext}', dpi=200)
    plt.close(fig)


def plot_work_precision(rows):
    """
    error against the reference vs. computation time: all FPE grids (marker: closure, colour: dt, size: dv) and MC
    """
    fig, axs = plt.subplots(1, len(REGIMES), figsize=(4.6 * len(REGIMES), 4.2), constrained_layout=True)
    sizes = dict(zip(DVS, np.linspace(60, 10, len(DVS))))
    for ax, regime in zip(np.atleast_1d(axs), REGIMES):
        for closure in CLOSURES:
            sel = [r for r in rows if r['regime'] == regime and r['model'] == closure]
            ax.scatter([r['time'] for r in sel], [r['err_ref'] for r in sel], s=[sizes[r['dv']] for r in sel],
                       c=[CLOSURE_COLORS[closure]] * len(sel), marker=CLOSURE_MARKERS[closure], alpha=0.75,
                       edgecolors='white', linewidths=0.4, label=f'FPE {CLOSURE_LABELS[closure]}')
        mc = sorted([r for r in rows if r['regime'] == regime and r['model'] == 'MC' and r['n'] != N_LIST[-1]],
                    key=lambda r: r['n'])
        ax.plot([r['time'] for r in mc], [r['err_ref'] for r in mc], color='#3d3c39', marker='^', lw=1,
                label='Monte Carlo (finite size)')
        for r in mc:
            ax.annotate(f"n={r['n'] // 1000}k", (r['time'], r['err_ref']), fontsize=7, xytext=(4, 4),
                        textcoords='offset points', color='#3d3c39')
        ref_time = next(r['time'] for r in rows if r['regime'] == regime and r['model'] == 'MC'
                        and r['n'] == N_LIST[-1])
        ax.axvline(ref_time, color='#3d3c39', lw=0.8, ls=':', label=f'MC n={N_LIST[-1]:,} (reference)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('computation time (s)')
        ax.set_title(REGIMES[regime][0], fontsize=9)
        style(ax)
    np.atleast_1d(axs)[0].set_ylabel(f'rel. L2 error of the rate vs. MC n={N_LIST[-1]:,}')
    handles, labels = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=len(labels), frameon=False, fontsize=8)
    fig.suptitle(r'work-precision: marker size $\propto$ grid resolution (largest: $\Delta v$=' + f'{DVS[0]} mV)',
                 fontsize=9)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT_DIR / f'work_precision.{ext}', dpi=200)
    plt.close(fig)


def plot_rates(results, recommended):
    """
    binned rates of the reference, the finest and the recommended grid of every closure
    """
    edges = np.arange(0, T + 1e-9, RATE_BIN_MS)
    tb = 0.5 * (edges[1:] + edges[:-1])
    for regime in REGIMES:
        fig, axs = plt.subplots(len(CLOSURES), 1, figsize=(8, 2.2 * len(CLOSURES) + 0.6), sharex=True,
                                constrained_layout=True)
        ref = results[(regime, 'mc', N_LIST[-1], None)]
        for ax, closure in zip(axs, CLOSURES):
            ax.plot(tb, ref['rb'], color='#3d3c39', lw=0.9, label=f'MC n={N_LIST[-1]:,} ({ref["time"]:.0f} s)')
            fine = results[(regime, closure, DVS[0], DTS[0])]
            ax.plot(tb, fine['rb'], color=CLOSURE_COLORS[closure], lw=1.5,
                    label=f'FPE dv={DVS[0]}, dt={DTS[0]} ({fine["time"]:.1f} s)')
            dv_r, dt_r = recommended[(regime, closure)]
            if (dv_r, dt_r) != (DVS[0], DTS[0]):
                rec = results[(regime, closure, dv_r, dt_r)]
                ax.plot(tb, rec['rb'], color=CLOSURE_COLORS[closure], lw=1.2, ls='--',
                        label=f'FPE dv={dv_r}, dt={dt_r} ({rec["time"]:.1f} s)')
            coarse = results[(regime, closure, DVS[-1], DTS[-1])]
            ax.plot(tb, coarse['rb'], color='#c9a227', lw=0.9, ls=':',
                    label=f'FPE dv={DVS[-1]}, dt={DTS[-1]} ({coarse["time"]:.1f} s)')
            ax.set_ylabel('rate (Hz)')
            ax.set_title(f'{CLOSURE_LABELS[closure]}', fontsize=9, loc='left')
            ax.axvline(T_ON, color='#b4b2a9', lw=0.6, ls=':')
            ax.legend(fontsize=7, frameon=False, loc='upper right')
            style(ax)
        axs[-1].set_xlabel('t (ms)')
        fig.suptitle(REGIMES[regime][0], fontsize=10)
        fig.savefig(OUT_DIR / f'rates_{regime}.png', dpi=200)
        plt.close(fig)


########################################################################################################################
# TABLE
########################################################################################################################
def latex_table(rows, recommended):
    lines = [r'\begin{tabular}{l l rr r r r r r}', r'\toprule',
             r'regime & closure & $\Delta v$ & $\Delta t$ & time (s) & speed-up & err.\ grid & err.\ MC ref. & '
             r'err.\ MC ref.\ finest \\', r'\midrule']
    for regime in REGIMES:
        mc = {r['n']: r for r in rows if r['regime'] == regime and r['model'] == 'MC'}
        for closure in CLOSURES:
            dv_r, dt_r = recommended[(regime, closure)]
            rec = next(r for r in rows if r['regime'] == regime and r['model'] == closure and r['dv'] == dv_r
                       and r['dt'] == dt_r)
            fine = next(r for r in rows if r['regime'] == regime and r['model'] == closure and r['dv'] == DVS[0]
                        and r['dt'] == DTS[0])
            lines.append(f"{REGIMES[regime][0].split(' (')[0]} & {CLOSURE_SHORT[closure]} & {dv_r} & {dt_r} & "
                         f"{rec['time']:.1f} & {fine['time'] / rec['time']:.0f}$\\times$ & {rec['err_grid']:.3f} & "
                         f"{rec['err_ref']:.3f} & {fine['err_ref']:.3f} \\\\")
        for n in N_LIST[:-1]:
            lines.append(f" & MC $n$={n:,} & -- & {MC_DT} & {mc[n]['time']:.1f} & & & {mc[n]['err_ref']:.3f} & \\\\")
        lines.append(f" & MC $n$={N_LIST[-1]:,} (ref.) & -- & {MC_DT} & {mc[N_LIST[-1]]['time']:.1f} & & & & \\\\")
        lines.append(r'\midrule' if regime != list(REGIMES)[-1] else r'\bottomrule')
    lines.append(r'\end{tabular}')
    return '\n'.join(lines)


def save_table_pdf(tabular, fname, caption):
    doc = (r'\documentclass[border=8pt]{standalone}' + '\n' + r'\usepackage{booktabs}' + '\n' +
           r'\begin{document}' + '\n' + r'\begin{minipage}{20cm}\small' + '\n' + tabular + '\n\n' + r'\vspace{6pt}' +
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


if __name__ == '__main__':
    tasks = [(reg, 'mc', n, None) for reg in REGIMES for n in N_LIST[::-1]]
    # expensive runs first (better load balancing)
    tasks += sorted([(reg, c, dv, dt) for reg in REGIMES for c in CLOSURES for dv in DVS for dt in DTS],
                    key=lambda x: x[2] * x[3])
    t_start = time.perf_counter()
    results = {}
    with Pool(N_PROCESSES) as pool:
        for k, res in enumerate(pool.imap_unordered(run, tasks), start=1):
            results[res['task']] = res
            print(f'[{k}/{len(tasks)}] {res["task"]}: {res["time"]:.1f} s', flush=True)
    print(f'total wall time {time.perf_counter() - t_start:.0f} s')

    rows = []
    for regime in REGIMES:
        ref = results[(regime, 'mc', N_LIST[-1], None)]
        for n in N_LIST:
            res = results[(regime, 'mc', n, None)]
            rows.append(dict(regime=regime, model='MC', n=n, dv=None, dt=MC_DT, time=res['time'],
                             err_ref=rel_l2(res['rb'], ref['rb']) if n != N_LIST[-1] else np.nan,
                             err_grid=np.nan, err_u=rms_u(res, ref) if n != N_LIST[-1] else np.nan,
                             spikes=res['rb'].sum() * RATE_BIN_MS * 1e-3))
        for closure in CLOSURES:
            fine = results[(regime, closure, DVS[0], DTS[0])]
            for dv in DVS:
                for dt in DTS:
                    res = results[(regime, closure, dv, dt)]
                    rows.append(dict(regime=regime, model=closure, n=None, dv=dv, dt=dt, time=res['time'],
                                     err_ref=rel_l2(res['rb'], ref['rb']), err_grid=rel_l2(res['rb'], fine['rb']),
                                     err_u=rms_u(res, ref), spikes=res['rb'].sum() * RATE_BIN_MS * 1e-3))

    # recommended grid: fastest one with err_grid < GRID_TOL (the finest grid always qualifies)
    recommended = {}
    for regime in REGIMES:
        for closure in CLOSURES:
            ok = [r for r in rows if r['regime'] == regime and r['model'] == closure and r['err_grid'] < GRID_TOL]
            best = min(ok, key=lambda r: r['time'])
            recommended[(regime, closure)] = (best['dv'], best['dt'])

    fields = ['regime', 'model', 'n', 'dv', 'dt', 'time', 'err_ref', 'err_grid', 'err_u', 'spikes']
    with open(OUT_DIR / 'resolution_table.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    np.savez(OUT_DIR / 'resolution_rates.npz',
             **{'_'.join(str(x) for x in task): res['rb'] for task, res in results.items()})

    header = f"{'regime':13s} {'model':21s} {'n':>7s} {'dv':>5s} {'dt':>6s} {'time (s)':>8s} {'err ref':>8s} " \
             f"{'err grid':>8s} {'RMS <u>':>8s}"
    print('\n' + header + '\n' + '-' * len(header))
    for r in rows:
        mark = ' <- recommended' if r['model'] in CLOSURES and recommended[(r['regime'], r['model'])] == \
            (r['dv'], r['dt']) else ''
        print(f"{r['regime']:13s} {r['model']:21s} {r['n'] or '':>7} {r['dv'] or '':>5} {r['dt']:>6} "
              f"{r['time']:8.2f} {r['err_ref']:8.4f} {r['err_grid']:8.4f} {r['err_u']:8.2f}{mark}")

    plot_error_and_time(rows)
    plot_work_precision(rows)
    plot_rates(results, recommended)
    caption = (f'Recommended grid per closure: fastest $(\\Delta v, \\Delta t)$ with a rel.\\ L2 error of the rate '
               f'against the finest grid ($\\Delta v$={DVS[0]} mV, $\\Delta t$={DTS[0]} ms) below {GRID_TOL}. '
               f'Speed-up relative to the finest grid. err.\\ MC ref.: rel.\\ L2 error of the rate '
               f'({RATE_BIN_MS:g} ms bins) against Monte Carlo with $n$={N_LIST[-1]:,} neurons. Step input, '
               f'T = {T:.0f} ms, on for {T_ON:.0f} ms, {N_PROCESSES} parallel processes.')
    save_table_pdf(latex_table(rows, recommended), 'resolution_table', caption)
    print(f'results saved in {OUT_DIR}')

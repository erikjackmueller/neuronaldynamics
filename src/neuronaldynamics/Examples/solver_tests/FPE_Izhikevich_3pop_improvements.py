"""
Improvements of the Izhikevich FPE for the 3 population network with feedback (exc 1 with input, exc 2, inh) at low
noise, where the FPE on the grid dv = 0.1 mV, dt = 0.05 ms of FPE_vs_IK_network.py fails (sigma = 1 mV). The goal is a
better fit than before while staying faster than the IK network with 20k neurons per population.

Configurations (CONFIGS), each adds one of the proposed changes:
    1. closure 'conditional_variance' instead of 'conditional' (best closure for 1 population at sigma = 1 mV)
    2. finer grid dv = 0.05 mV, dt = 0.01 ms (converged grid of FPE_Izhikevich_low_noise.py for 3 populations)
    3. non-uniform grid v_grid='refined': spacing dv in (min(v_r) - 20, max(v_t) + 10) mV, growing by 5 % per cell to
       0.5 mV towards v_min and v_peak (about 2.2x fewer points, see izhikevich_grid in Model/Nykamp_Model.py)
    4. time step kernels compiled with numba (use_numba=True, default now, all configurations): the old numpy / scipy
       solver gives the same results (max relative difference of the rate ~1e-9) and is 2-4x slower
Reference: IK network with 100k neurons per population; IK 20k for the finite size error and as timing target.

Inputs: step and 20 Hz sine (I0 + I0 sin(2 pi 20 Hz t)) into exc 1, I0 = 150 pA, on for t < 100 ms, T = 200 ms
(FPE_vs_IK_network.py). Noise levels SIGMAS.

Error measures (rates in 1 ms bins, all populations together and per population):
    rel_l2          ||x - ref|| / ||ref||                                     (as in FPE_vs_IK_network.py)
    rel_l2_smooth   the same after a gaussian kernel of SMOOTH_MS on both rates: tolerates timing jitter < ~SMOOTH_MS
    nrmse           Utils.nrmse(x, ref) = RMS(x - ref) / (max(ref) - min(ref))  (as in FPE_vs_LIF_timing.py: normalised
                    by the range of the reference instead of its norm, the same sensitivity to timing jitter)
    nrmse_smooth    nrmse after the gaussian kernel
    count_err       relative error of the spike count per neuron of every population in the windows 0-100 ms (input
                    on) and 100-200 ms (off), maximum over populations and windows

Outputs in Examples/solver_tests/FPE_Izhikevich_3pop_improvements:
    work_precision.png/.pdf              error (rel_l2_smooth and rel_l2) against computation time per sigma and input
    rates_<input>_sigma_<sigma>.png      rates of the reference, IK 20k, the baseline and the best configuration
    improvements_table.csv               all numbers
    improvements_table.tex / .pdf        main numbers

Usage (from src/neuronaldynamics):  python Examples/solver_tests/FPE_Izhikevich_3pop_improvements.py
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
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/neuronaldynamics, for direct calls
sys.path.insert(0, str(Path(__file__).resolve().parent))      # FPE_vs_IK_network (same folder)
from Model.Nykamp_Model import FPE_Population
from Utils import nrmse
import FPE_vs_IK_network as net
matplotlib.use('Agg')
warnings.filterwarnings('ignore', message='.*non-interactive.*')
warnings.filterwarnings('ignore', category=RuntimeWarning)

OUT_DIR = Path(__file__).resolve().parent / f'FPE_Izhikevich_3pop_improvements{net.method_suffix()}'
OUT_DIR.mkdir(exist_ok=True)

########################################################################################################################
# SETTINGS
########################################################################################################################
SETUP = '3pop'
SIGMAS = [1.0, float(np.sqrt(10))]           # mV
INPUTS = {'step': None, 'sine20Hz': 0.02}    # input type: sine frequency in 1/ms (None: step)
N_LIST = [20_000, 100_000]                   # IK neurons per population, the last one is the reference
SMOOTH_MS = 2.0                              # gaussian kernel (std) for the smoothed errors
N_PROCESSES = 4

# name: (label, FPE parameters)
CONFIGS = {
    'base':          ('baseline: cond., uniform dv=0.1, dt=0.05',
                      dict(adaptation_closure='conditional', dv=0.1, dt=0.05)),
    'var':           ('1: cond.+var., uniform dv=0.1, dt=0.05',
                      dict(adaptation_closure='conditional_variance', dv=0.1, dt=0.05)),
    'var_fine':      ('1+2: cond.+var., uniform dv=0.05, dt=0.01',
                      dict(adaptation_closure='conditional_variance', dv=0.05, dt=0.01)),
    'var_ref05':     ('1+2+3: cond.+var., refined dv=0.05, dt=0.01',
                      dict(adaptation_closure='conditional_variance', dv=0.05, dt=0.01, v_grid='refined')),
    'var_ref05_dt':  ('1+3: cond.+var., refined dv=0.05, dt=0.025',
                      dict(adaptation_closure='conditional_variance', dv=0.05, dt=0.025, v_grid='refined')),
    'var_ref02':     ('1+3+4: cond.+var., refined dv=0.02, dt=0.01',
                      dict(adaptation_closure='conditional_variance', dv=0.02, dt=0.01, v_grid='refined')),
    'cond_ref02':    ('cond., refined dv=0.02, dt=0.01',
                      dict(adaptation_closure='conditional', dv=0.02, dt=0.01, v_grid='refined')),
    'var_uni02':     ('cond.+var., uniform dv=0.02, dt=0.01 (finest)',
                      dict(adaptation_closure='conditional_variance', dv=0.02, dt=0.01)),
}
CONFIG_COLORS = dict(zip(CONFIGS, ['#8a8984', '#2a78d6', '#1b4f8f', '#d6612a', '#e8a33b', '#7a1f5c', '#c2185b',
                                   '#2a9d6b']))


def set_input(kind):
    """input of FPE_vs_IK_network for the current process"""
    freq = INPUTS[kind]
    net.set_input('step' if freq is None else 'sine', out_dir=net.OUT_DIR, frequency=freq)


def run(task):
    """one timed simulation: (input, sigma, 'IK', n) or (input, sigma, config, None)"""
    kind, sigma, model, n = task
    set_input(kind)
    if model == 'IK':
        t0 = time.perf_counter()
        r, _ = net.ik_network(SETUP, sigma, n, seed=n)
        cpu = time.perf_counter() - t0
        return dict(task=task, time=cpu, rb=net.binned_rate(r, net.MC_DT))
    pars = dict(soma_model='IK', T=net.T, population_type=net.SETUPS[SETUP][0],
                connectivity_matrix=net.SETUPS[SETUP][2], voltage_sigma=sigma, input_function=net.input_current,
                input_function_idx=[0], init_pdf_sigma=1.0, save_p='none', tqdm_disable=True,
                **{name: list(v) for name, v in net.neuron_parameters(SETUP).items()})
    pars.update(CONFIGS[model][1])
    fpe = FPE_Population(pars, name=str(OUT_DIR / f'fpe_{kind}_{model}_sigma_{sigma:.3g}'))
    t0 = time.perf_counter()
    fpe.simulate()
    cpu = time.perf_counter() - t0
    fpe.clean()
    return dict(task=task, time=cpu, rb=net.binned_rate(fpe.r, pars['dt']), n_points=len(fpe.v),
                n_fallback=fpe.n_fallback)


def smooth(x):
    return gaussian_filter1d(x, SMOOTH_MS / net.RATE_BIN_MS, axis=-1, mode='nearest')


def measures(x, ref):
    """all error measures of the binned rates x against ref (n_pop, n_bins)"""
    out = dict(rel_l2=float(np.linalg.norm(x - ref) / np.linalg.norm(ref)),
               rel_l2_smooth=float(np.linalg.norm(smooth(x) - smooth(ref)) / np.linalg.norm(smooth(ref))),
               nrmse=float(nrmse(x, ref)), nrmse_smooth=float(nrmse(smooth(x), smooth(ref))))
    for j in range(ref.shape[0]):
        out[f'rel_l2_pop{j}'] = float(np.linalg.norm(x[j] - ref[j]) / np.linalg.norm(ref[j]))
        out[f'rel_l2_smooth_pop{j}'] = float(np.linalg.norm(smooth(x[j]) - smooth(ref[j])) /
                                             np.linalg.norm(smooth(ref[j])))
    half = int(net.T_ON / net.RATE_BIN_MS)
    counts = lambda y: np.stack([y[:, :half].sum(1), y[:, half:].sum(1)]) * net.RATE_BIN_MS * 1e-3
    c_x, c_ref = counts(x), counts(ref)
    out['count_err'] = float(np.max(np.abs(c_x - c_ref) / np.maximum(c_ref, 1e-3)))
    return out


########################################################################################################################
# PLOTS
########################################################################################################################
def style(ax):
    ax.grid(which='major', lw=0.4, color='#e0dfdb')
    ax.spines[['top', 'right']].set_visible(False)


def plot_work_precision(rows):
    combos = [(k, s) for k in INPUTS for s in SIGMAS]
    fig, axs = plt.subplots(2, len(combos), figsize=(4.4 * len(combos), 7.4), constrained_layout=True,
                            squeeze=False)
    for col, (kind, sigma) in enumerate(combos):
        sel = [r for r in rows if r['input'] == kind and r['sigma'] == sigma]
        for row_idx, measure in enumerate(['rel_l2_smooth', 'rel_l2']):
            ax = axs[row_idx, col]
            for r in sel:
                if r['model'] == 'IK':
                    if r['n'] == N_LIST[-1]:
                        ax.axvline(r['time'], color='#3d3c39', lw=0.8, ls=':',
                                   label=f'IK n={N_LIST[-1]:,} (reference, time)')
                    else:
                        ax.scatter(r['time'], r[measure], marker='^', s=60, color='#3d3c39', zorder=5,
                                   label=f'IK n={r["n"]:,}')
                        ax.axvline(r['time'], color='#3d3c39', lw=0.6, ls='--')
                else:
                    ax.scatter(r['time'], r[measure], s=55, color=CONFIG_COLORS[r['model']], edgecolors='white',
                               linewidths=0.5, zorder=4, label=CONFIGS[r['model']][0])
            ax.set_xscale('log')
            ax.set_yscale('log')
            style(ax)
            if row_idx == 0:
                ax.set_title(f'{kind}, $\\sigma$ = {sigma:.3g} mV', fontsize=9)
            else:
                ax.set_xlabel('computation time (s)')
        axs[0, 0].set_ylabel(f'rel. L2 error, smoothed ({SMOOTH_MS:g} ms)')
        axs[1, 0].set_ylabel(f'rel. L2 error, {net.RATE_BIN_MS:g} ms bins')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc='outside lower center', ncol=3, frameon=False, fontsize=8)
    fig.suptitle('3 populations with feedback: FPE configurations vs. IK networks (dashed: IK 20k time = budget)',
                 fontsize=10)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT_DIR / f'work_precision.{ext}', dpi=200)
    plt.close(fig)


def plot_rates(results, best):
    edges = np.arange(0, net.T + 1e-9, net.RATE_BIN_MS)
    tb = 0.5 * (edges[1:] + edges[:-1])
    for kind in INPUTS:
        for sigma in SIGMAS:
            fig, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, constrained_layout=True)
            ref = results[(kind, sigma, 'IK', N_LIST[-1])]
            small = results[(kind, sigma, 'IK', N_LIST[0])]
            b = best[(kind, sigma)]
            curves = [(ref, '#3d3c39', 1.6, f'IK n={N_LIST[-1]:,} ({ref["time"]:.0f} s)'),
                      (small, '#b4b3ad', 1.0, f'IK n={N_LIST[0]:,} ({small["time"]:.0f} s)'),
                      (results[(kind, sigma, 'base', None)], CONFIG_COLORS['base'], 1.0,
                       f"{CONFIGS['base'][0]} ({results[(kind, sigma, 'base', None)]['time']:.1f} s)"),
                      (results[(kind, sigma, b, None)], CONFIG_COLORS[b], 1.3,
                       f"{CONFIGS[b][0]} ({results[(kind, sigma, b, None)]['time']:.1f} s)")]
            for j, ax in enumerate(axs):
                for res, color, lw, label in curves:
                    ax.plot(tb, res['rb'][j], color=color, lw=lw, label=label)
                ax.set_ylabel(f'{net.POP_NAMES[SETUP][j]}\nrate (Hz)')
                ax.axvline(net.T_ON, color='#b4b2a9', lw=0.6, ls=':')
                style(ax)
            axs[0].legend(fontsize=7, frameon=False, loc='upper right')
            axs[-1].set_xlabel('t (ms)')
            fig.suptitle(f'{kind}, $\\sigma$ = {sigma:.3g} mV: best configuration below the IK 20k time', fontsize=10)
            fig.savefig(OUT_DIR / f'rates_{kind}_sigma_{sigma:.3g}.png', dpi=200)
            plt.close(fig)


########################################################################################################################
# TABLE
########################################################################################################################
def latex_table(rows):
    lines = [r'\begin{tabular}{l l r r r r r r r}', r'\toprule',
             r'input, $\sigma$ & model & points & time (s) & rel.\ L2 & rel.\ L2 smooth & nrmse & nrmse smooth & '
             r'count err.\ \\', r'\midrule']
    for kind in INPUTS:
        for sigma in SIGMAS:
            sel = [r for r in rows if r['input'] == kind and r['sigma'] == sigma and
                   not (r['model'] == 'IK' and r['n'] == N_LIST[-1])]
            budget = next(r['time'] for r in sel if r['model'] == 'IK')
            best = min([r for r in sel if r['model'] != 'IK' and r['time'] < budget],
                       key=lambda r: r['rel_l2_smooth'], default=None)
            for r in sel:
                name = f"IK {r['n'] // 1000}k" if r['model'] == 'IK' else CONFIGS[r['model']][0].replace('_', ' ')
                if best is not None and r is best:
                    name = r'\textbf{' + name + '}'
                pts = '' if r['model'] == 'IK' else str(r['n_points'])
                lines.append(f"{kind}, {sigma:.3g} & {name} & {pts} & {r['time']:.1f} & {r['rel_l2']:.3f} & "
                             f"{r['rel_l2_smooth']:.3f} & {r['nrmse']:.4f} & {r['nrmse_smooth']:.4f} & "
                             f"{r['count_err']:.3f} \\\\")
            lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    caption = (f'3 populations with feedback, errors against the IK network with {N_LIST[-1]:,} neurons per '
               f'population (rates in {net.RATE_BIN_MS:g} ms bins, smoothed: gaussian kernel {SMOOTH_MS:g} ms; '
               f'nrmse: RMS error / range of the reference; count err.: maximal relative error of the spike counts '
               f'per population while the input is on / off). Bold: smallest smoothed error of the FPE runs faster '
               f'than IK {N_LIST[0] // 1000}k. {N_PROCESSES} parallel processes.')
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
    print('no LaTeX engine found (or compilation failed), only the .tex file was written')


def main():
    # compile the numba kernels once before the timed runs (cache=True: the workers load the compiled code)
    run((list(INPUTS)[0], SIGMAS[0], list(CONFIGS)[0], None))
    tasks = [(k, s, 'IK', n) for k in INPUTS for s in SIGMAS for n in N_LIST[::-1]]
    tasks += [(k, s, c, None) for k in INPUTS for s in SIGMAS for c in CONFIGS]
    t_start = time.perf_counter()
    results = {}
    with Pool(N_PROCESSES) as pool:
        for q, res in enumerate(pool.imap_unordered(run, tasks), start=1):
            results[res['task']] = res
            print(f'[{q}/{len(tasks)}] {res["task"]}: {res["time"]:.1f} s', flush=True)
    print(f'total wall time {time.perf_counter() - t_start:.0f} s')

    rows = []
    for (kind, sigma, model, n), res in results.items():
        ref = results[(kind, sigma, 'IK', N_LIST[-1])]
        row = dict(input=kind, sigma=sigma, model=model, n=n, time=res['time'], n_points=res.get('n_points'),
                   n_fallback=res.get('n_fallback'))
        row.update(measures(res['rb'], ref['rb']))
        rows.append(row)
    order = ['IK'] + list(CONFIGS)
    rows.sort(key=lambda r: (list(INPUTS).index(r['input']), r['sigma'], order.index(r['model']), r['n'] or 0))

    fields = list(rows[0].keys())
    with open(OUT_DIR / 'improvements_table.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    np.savez(OUT_DIR / 'improvements_rates.npz',
             **{'_'.join(str(v) for v in task): res['rb'] for task, res in results.items()})

    print(f"\n{'input':9s} {'sigma':>5s} {'model':48s} {'pts':>5s} {'time':>7s} {'L2':>6s} {'L2 sm':>6s} "
          f"{'nrmse':>7s} {'nrmse sm':>8s} {'count':>6s}   L2 sm per pop")
    for r in rows:
        name = f"IK n={r['n']:,}" if r['model'] == 'IK' else CONFIGS[r['model']][0]
        print(f"{r['input']:9s} {r['sigma']:5.3g} {name:48s} {r['n_points'] or '':>5} {r['time']:7.1f} "
              f"{r['rel_l2']:6.3f} {r['rel_l2_smooth']:6.3f} {r['nrmse']:7.4f} {r['nrmse_smooth']:8.4f} "
              f"{r['count_err']:6.3f}   " +
              ' '.join(f"{r[f'rel_l2_smooth_pop{j}']:.3f}" for j in range(3)))

    # best configuration below the IK 20k time (smoothed error)
    best = {}
    for kind in INPUTS:
        for sigma in SIGMAS:
            budget = results[(kind, sigma, 'IK', N_LIST[0])]['time']
            cands = [r for r in rows if r['input'] == kind and r['sigma'] == sigma and r['model'] != 'IK'
                     and r['time'] < budget]
            best[(kind, sigma)] = min(cands, key=lambda r: r['rel_l2_smooth'])['model'] if cands else 'base'
    plot_work_precision(rows)
    plot_rates(results, best)
    save_table_pdf(*latex_table(rows), 'improvements_table')
    print(f'results saved in {OUT_DIR}')


if __name__ == '__main__':
    main()

"""
Time step convergence of the IK network reference of the hardest case (3 populations with feedback, sigma = 1 mV,
step input, FPE_dt_reference_check.py). The reference networks there use Euler-Maruyama with MC_DT = 0.01 ms and spike
detection on the time grid; the post-stimulus event (exc 2 volley at ~122 ms, inhibitory bursts at 127-135 ms in which
almost all inhibitory neurons fire within 1 ms) is sensitive to spike timing, so the networks are repeated with
MC_DT = 0.005 and 0.0025 ms (100k neurons per population).

Comparison: smoothed rel. L2 error and event measures (FPE_2D_fct_3pop.event_measures) of every network against the
averaged MC_DT = 0.01 reference; the spread of the MC_DT = 0.01 networks (seeds) is the scale against which the dt
effect is judged.

Outputs in Examples/solver_tests/FPE_IK_reference_dt: reference_dt_rates.npz, reference_dt_table.csv, run_log.txt,
rates_zoom.png

Usage (from src/neuronaldynamics):
    python Examples/solver_tests/FPE_IK_reference_dt.py run [dt,n,seed[,input,sigma] ...] [--proc P] [--method heun]
        (default: TASKS; --method heun: stochastic Heun with interpolated spike times, files ik_heun_...)
    (input 'step' or 'sine20Hz'; other cases than step / sigma = 1 mV are saved with a suffix, not in the analysis)
    python Examples/solver_tests/FPE_IK_reference_dt.py analyze
    python Examples/solver_tests/FPE_IK_reference_dt.py figure    (Euler vs Heun, models vs converged reference)
"""
import os
for _var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_var, '1')
import sys
import csv
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import FPE_vs_IK_network as net
import FPE_Izhikevich_3pop_improvements as imp
import FPE_2D_fct_3pop as ev

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'FPE_IK_reference_dt'
OUT_DIR.mkdir(exist_ok=True)
SIGMA = 1.0
METHOD = 'euler'           # integrator of the networks (FPE_vs_IK_network.ik_network), command line: --method heun
# (MC_DT, neurons per population, seed); MC_DT = 0.01, 100k runs are loaded from FPE_dt_reference_check /
# FPE_Izhikevich_3pop_improvements. Default: the first study (100k); further tasks on the command line as dt,n,seed
TASKS = [(0.0025, 100_000, 300_000, 'step', SIGMA), (0.005, 100_000, 300_001, 'step', SIGMA),
         (0.005, 100_000, 300_002, 'step', SIGMA)]


def run(task):
    dt, n, seed, kind, sigma = task
    imp.set_input(kind)
    net.MC_DT = dt                 # ik_network reads the module constants
    net.MC_METHOD = METHOD
    t0 = time.perf_counter()
    r, _ = net.ik_network('3pop', sigma, n, seed=seed)
    return dict(task=task, time=time.perf_counter() - t0, rb=net.binned_rate(r, dt))


def file_name(dt, n, seed, kind='step', sigma=SIGMA):
    """the hard case (step, sigma = 1 mV) without suffix, other cases with _<input>_sigma<sigma>"""
    extra = '' if (kind, sigma) == ('step', SIGMA) else f'_{kind}_sigma{sigma:g}'
    method = '' if METHOD == 'euler' else f'_{METHOD}'
    return OUT_DIR / f'ik{method}_dt{dt:g}_{n // 1000}k_s{seed}{extra}.npz'


def old_references():
    d = np.load(HERE / 'FPE_dt_reference_check' / 'dt_reference_rates.npz')
    e = np.load(HERE / 'FPE_Izhikevich_3pop_improvements' / 'improvements_rates.npz')
    runs = {'dt0.01_100k_s100000': e['step_1.0_IK_100000'],
            'dt0.01_100k_s100001': d['IK_step_100000_100001'],
            'dt0.01_100k_s100002': d['IK_step_100000_100002'],
            'dt0.01_200k_s200000': d['IK_step_200000_200000']}
    return runs, d['avg_step']


def main_run(tasks, n_proc):
    t_start = time.perf_counter()
    tasks = sorted(tasks, key=lambda x: x[1] / x[0], reverse=True)   # expensive first
    with Pool(min(n_proc, len(tasks))) as pool:
        for res in pool.imap_unordered(run, tasks):
            np.savez(file_name(*res['task']), rb=res['rb'], time=res['time'])
            print(f'{res["task"]}: {res["time"]:.0f} s', flush=True)
    print(f'total wall time {time.perf_counter() - t_start:.0f} s', flush=True)


def main_analyze():
    runs, avg = old_references()
    for f in sorted([f for f in OUT_DIR.glob('ik_dt*.npz') if 'sigma' not in f.stem], key=lambda f: (-int(f.stem.split('_')[2][:-1]),
                                                                -float(f.stem.split('_')[1][2:]))):
        runs[f.stem[3:]] = np.load(f)['rb']
    ev_ref = ev.event_measures(avg)
    rows = []
    for name, rb in runs.items():
        m = imp.measures(rb, avg)
        e = ev.event_measures(rb)
        rows.append(dict(run=name, err_smooth=m['rel_l2_smooth'], count_err=m['count_err'],
                         **{f'pop{j}': m[f'rel_l2_smooth_pop{j}'] for j in range(3)},
                         **ev.event_errors(e, ev_ref)))
    with open(OUT_DIR / 'reference_dt_table.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print('event measures of the averaged dt = 0.01 reference:')
    ev.print_events(ev_ref)
    ev.print_table(rows, 'run')

    t = np.arange(avg.shape[1]) * net.RATE_BIN_MS + 0.5 * net.RATE_BIN_MS
    fig, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, constrained_layout=True)
    for j, ax in enumerate(axs):
        for name, rb in runs.items():
            new = not name.startswith('dt0.01')
            ax.plot(t, rb[j], lw=1.2 if new else 0.8, color=None if new else '#b4b3ad', label=name)
        ax.set_xlim(100, 150)
        ax.set_ylabel(f'{net.POP_NAMES["3pop"][j]}\nrate')
        imp.style(ax)
    axs[0].legend(fontsize=7, frameon=False, ncol=2)
    axs[-1].set_xlabel('t (ms)')
    fig.suptitle('IK networks, 3 pop., step, $\\sigma$ = 1 mV: Euler-Maruyama time step', fontsize=10)
    fig.savefig(OUT_DIR / 'rates_zoom.png', dpi=150)
    plt.close(fig)


EULER_20K = [(0.01, 400005), (0.005, 400004), (0.0025, 400003), (0.00125, 400002), (0.000625, 400001),
             (0.0003125, 400000)]
HEUN_20K = [(0.01, 700004), (0.005, 700003), (0.0025, 700002), (0.00125, 700001), (0.000625, 700000)]


def centroid(rb, j, t0, t1):
    t = (np.arange(rb.shape[1]) + 0.5) * net.RATE_BIN_MS
    sel = (t > t0) & (t < t1)
    return float(np.dot(rb[j, sel], t[sel]) / rb[j, sel].sum())


def main_figure():
    """time step convergence of Euler-Maruyama and Heun (20k networks) and the models against the converged reference"""
    euler = {dt: np.load(OUT_DIR / f'ik_dt{dt:g}_20k_s{s}.npz')['rb'] for dt, s in EULER_20K}
    heun = {dt: np.load(OUT_DIR / f'ik_heun_dt{dt:g}_20k_s{s}.npz')['rb'] for dt, s in HEUN_20K}
    ref = heun[min(heun)]
    sols, others, old_ref = ev.collect()
    models = {'1D two-node closure, FCT': (others['1D quadrature FCT dt0.005'], '#d6612a', '--'),
              '2D, FCT, $\\Delta u$ = 1 pA': (sols['dv0.1_du1_dt0.01_fct'], '#1b4f8f', '-')}
    fig = plt.figure(figsize=(11, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    for runs, label, color in [(euler, 'Euler-Maruyama', '#b4b3ad'), (heun, 'Heun, interpolated spikes', '#3d3c39')]:
        dts = sorted(runs)
        ax.plot(dts, [centroid(runs[dt], 2, 66, 90) for dt in dts], marker='o', mfc='white', color=color, label=label)
    for name, (rb, color, ls) in models.items():
        ax.axhline(centroid(rb, 2, 66, 90), color=color, ls=ls, lw=1, label=name)
    ax.set_xscale('log')
    ax.set_xlabel('network time step (ms)')
    ax.set_ylabel('centroid of the inhibitory\nbursts 66-90 ms (ms)')
    ax.legend(fontsize=7, frameon=False)
    imp.style(ax)
    ax = fig.add_subplot(gs[0, 1])
    for runs, label, color in [(euler, 'Euler-Maruyama', '#b4b3ad'), (heun, 'Heun', '#3d3c39')]:
        dts = [dt for dt in sorted(runs) if runs[dt] is not ref]
        ax.plot(dts, [imp.measures(runs[dt], ref)['rel_l2_smooth'] for dt in dts], marker='o', mfc='white',
                color=color, label=label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('network time step (ms)')
    ax.set_ylabel(f'smoothed rel. L2 against\nHeun, dt = {min(heun):g} ms')
    ax.legend(fontsize=7, frameon=False)
    imp.style(ax)
    t = (np.arange(ref.shape[1]) + 0.5) * net.RATE_BIN_MS
    sub = gs[1, :].subgridspec(1, 2)
    for col, (j, (t0, t1)) in enumerate([(1, (60, 140)), (2, (60, 140))]):
        ax = fig.add_subplot(sub[0, col])
        ax.plot(t, old_ref[j], color='#b4b3ad', lw=1.4, label='old reference (Euler, dt = 0.01 ms)')
        ax.plot(t, ref[j], color='#3d3c39', lw=1.6, label=f'Heun, dt = {min(heun):g} ms')
        for name, (rb, color, ls) in models.items():
            ax.plot(t, rb[j], color=color, ls=ls, lw=1.1, label=name)
        ax.set_xlim(t0, t1)
        if j == 2:
            ax.set_yscale('symlog', linthresh=10)
            ax.set_ylim(bottom=0)
        ax.set_xlabel('t (ms)')
        ax.set_ylabel(f'{net.POP_NAMES["3pop"][j]} rate (Hz)')
        imp.style(ax)
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle('3 populations, step, $\\sigma$ = 1 mV: time step of the IK network reference (20k neurons)',
                 fontsize=10)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT_DIR / f'reference_dt_convergence.{ext}', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    # run [dt,n,seed ...] [--proc P] | analyze
    mode = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if mode == 'run':
        args = sys.argv[2:]
        n_proc = 4
        if '--method' in args:
            METHOD = args[args.index('--method') + 1]
            del args[args.index('--method'):args.index('--method') + 2]
        if '--proc' in args:
            n_proc = int(args[args.index('--proc') + 1])
            del args[args.index('--proc'):args.index('--proc') + 2]
        tasks = []
        for a in args:   # dt,n,seed[,input,sigma]
            x = a.split(',')
            tasks.append((float(x[0]), int(x[1]), int(x[2]), x[3] if len(x) > 3 else 'step',
                          float(x[4]) if len(x) > 4 else SIGMA))
        main_run(tasks or TASKS, n_proc)
    if mode == 'figure':
        main_figure()
    else:
        main_analyze()

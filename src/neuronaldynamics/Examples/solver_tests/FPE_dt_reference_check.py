"""
Two checks for the troublesome case of FPE_fct_closure_test.py (3 populations with feedback, sigma = 1 mV):

1. Time step convergence of the FPE: dt in DTS on the grid dv = 0.1 mV for
       'conditional_variance' (SG flux) and 'conditional_second_moment' with FCT,
   each without and with one corrector pass (n_corrector=1: the step is repeated with the adaptation fields of the
   predictor at the new time instead of the extrapolated ones). If the error keeps falling with dt, the closure limit
   is not reached yet.
2. Convergence of the reference: the IK network with 100k neurons per population (seed 100000, from
   FPE_Izhikevich_3pop_improvements.py) is repeated with two more seeds, and one network with 200k neurons is run.
   The spread between these runs is the uncertainty of the reference; their average (weighted by the number of
   neurons) is used as an improved estimate of the N -> infinity limit that the FPE approximates.

Inputs: step and 20 Hz sine (FPE_vs_IK_network.py, I0 = 150 pA into exc 1 for t < 100 ms, T = 200 ms).
Error: relative L2 error of the rates of all populations (1 ms bins) after a 2 ms gaussian kernel (as in
FPE_Izhikevich_3pop_improvements.py), unsmoothed error and spike count error in the table.

Outputs in Examples/solver_tests/FPE_dt_reference_check:
    dt_convergence.png/.pdf         smoothed error against dt (vs. the 100k reference and vs. the averaged reference),
                                    grey band: spread of the large reference networks
    rates_<input>.png               rates of the reference networks and of the finest FPE runs
    dt_reference_table.csv, dt_reference_table.tex / .pdf

Heun (net.MC_METHOD = 'heun', default): the FPE runs are compared with the converged references of
FPE_reference_rebuild.py (stochastic Heun, dt = 0.005 ms, 100k neurons per population) instead of the Euler-Maruyama
networks, check 2 is not repeated (the Heun references are converged in N and dt). The FPE rates of the Euler run
(FPE_dt_reference_check/dt_reference_rates.npz) do not depend on the reference and are reused; configurations that are
missing there (the two-node closure) are simulated. Outputs in Examples/solver_tests/FPE_dt_reference_check_heun:
dt_reference_table.csv, dt_reference_rates.npz, dt_convergence.png/.pdf

Usage (from src/neuronaldynamics):  python Examples/solver_tests/FPE_dt_reference_check.py
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/neuronaldynamics, for direct calls
sys.path.insert(0, str(Path(__file__).resolve().parent))      # other solver tests (same folder)
from Model.Nykamp_Model import FPE_Population
import FPE_vs_IK_network as net
import FPE_Izhikevich_3pop_improvements as imp
matplotlib.use('Agg')
warnings.filterwarnings('ignore', message='.*non-interactive.*')
warnings.filterwarnings('ignore', category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / f'FPE_dt_reference_check{net.method_suffix()}'
OUT_DIR.mkdir(exist_ok=True)
IMP_DIR = HERE / f'FPE_Izhikevich_3pop_improvements{net.method_suffix()}'

########################################################################################################################
# SETTINGS
########################################################################################################################
N_PROCESSES = 4
SIGMA, DV = 1.0, 0.1
INPUTS = ['step', 'sine20Hz']
DTS = [0.05, 0.025, 0.01, 0.005, 0.0025]
CONFIGS = {'var_sg': ('cond.+var., SG', 'conditional_variance', 'none'),
           'm2_fct': ('2nd moment, FCT', 'conditional_second_moment', 'fct')}
CONFIGS_HEUN = dict(CONFIGS, quad_fct=('two-node, FCT', 'conditional_quadrature', 'fct'))
EULER_DIR = HERE / 'FPE_dt_reference_check'  # FPE rates of the Euler run, reused in main_heun
HEUN_CASES = {'step': '3pop_step_1', 'sine20Hz': '3pop_sine20Hz_1'}  # cases of FPE_reference_rebuild.py
CORRECTORS = [0, 1]
# reference networks: (neurons per population, seed); (100_000, 100_000) is loaded from the improvements test
NEW_REFS = [(100_000, 100_001), (100_000, 100_002), (200_000, 200_000)]
BASE_REF = (100_000, 100_000)
COLORS = {'var_sg': '#2a9d6b', 'm2_fct': '#d6612a', 'quad_fct': '#2a78d6'}


def run(task):
    if task[0] == 'IK':
        _, kind, n, seed = task
        imp.set_input(kind)
        t0 = time.perf_counter()
        r, _ = net.ik_network('3pop', SIGMA, n, seed=seed)
        return dict(task=task, time=time.perf_counter() - t0, rb=net.binned_rate(r, net.MC_DT))
    _, kind, config, dt, n_corr = task
    imp.set_input(kind)
    _, closure, limiter = CONFIGS_HEUN[config]
    pars = dict(soma_model='IK', T=net.T, dt=dt, dv=DV, population_type=net.SETUPS['3pop'][0],
                connectivity_matrix=net.SETUPS['3pop'][2], voltage_sigma=SIGMA, input_function=net.input_current,
                input_function_idx=[0], init_pdf_sigma=1.0, save_p='none', tqdm_disable=True,
                adaptation_closure=closure, flux_limiter=limiter, n_corrector=n_corr,
                **{name: list(v) for name, v in net.neuron_parameters('3pop').items()})
    model = FPE_Population(pars, name=str(OUT_DIR / ('fpe_' + '_'.join(str(x) for x in task[1:]))))
    t0 = time.perf_counter()
    model.simulate()
    cpu = time.perf_counter() - t0
    model.clean()
    return dict(task=task, time=cpu, rb=net.binned_rate(model.r, dt), n_fallback=model.n_fallback)


def style(ax):
    ax.grid(which='major', lw=0.4, color='#e0dfdb')
    ax.spines[['top', 'right']].set_visible(False)


def main():
    # base reference and IK 20k from the improvements test (same network, inputs and seeds)
    d = np.load(IMP_DIR / 'improvements_rates.npz')
    refs = {kind: {BASE_REF: d[f'{kind}_{SIGMA}_IK_{BASE_REF[0]}'], (20_000, 20_000): d[f'{kind}_{SIGMA}_IK_20000']}
            for kind in INPUTS}

    run(('FPE', INPUTS[0], 'm2_fct', 0.05, 0))  # compile the numba kernels
    tasks = [('IK', kind, n, seed) for kind in INPUTS for n, seed in sorted(NEW_REFS, reverse=True)]
    fpe_tasks = [('FPE', kind, c, dt, nc) for kind in INPUTS for c in CONFIGS for dt in DTS for nc in CORRECTORS]
    tasks += sorted(fpe_tasks, key=lambda x: x[3] / (1 + x[4]))  # expensive first
    t_start = time.perf_counter()
    results = {}
    with Pool(N_PROCESSES) as pool:
        for q, res in enumerate(pool.imap_unordered(run, tasks), start=1):
            results[res['task']] = res
            print(f'[{q}/{len(tasks)}] {res["task"]}: {res["time"]:.1f} s', flush=True)
    print(f'total wall time {time.perf_counter() - t_start:.0f} s')

    rows = []
    for kind in INPUTS:
        for n, seed in NEW_REFS:
            refs[kind][(n, seed)] = results[('IK', kind, n, seed)]['rb']
        large = [key for key in refs[kind] if key[0] >= 100_000]
        weights = np.array([key[0] for key in large], dtype=float)
        avg = np.tensordot(weights / weights.sum(), np.array([refs[kind][key] for key in large]), axes=1)
        refs[kind]['avg'] = avg
        base = refs[kind][BASE_REF]
        # spread of the references
        for key, rb in refs[kind].items():
            if key == 'avg':
                continue
            m_base = imp.measures(rb, base)
            m_avg = imp.measures(rb, avg)
            t_ref = results[('IK', kind) + key]['time'] if ('IK', kind) + key in results else np.nan
            rows.append(dict(input=kind, model=f'IK n={key[0]:,} seed={key[1]}', config='IK', dt=0.01, n_corrector='',
                             time=t_ref, err_base=m_base['rel_l2_smooth'], err_avg=m_avg['rel_l2_smooth'],
                             err_base_raw=m_base['rel_l2'], count_err_avg=m_avg['count_err'],
                             **{f'pop{j}_avg': m_avg[f'rel_l2_smooth_pop{j}'] for j in range(3)}))
        for c in CONFIGS:
            for dt in DTS:
                for nc in CORRECTORS:
                    res = results[('FPE', kind, c, dt, nc)]
                    m_base = imp.measures(res['rb'], base)
                    m_avg = imp.measures(res['rb'], avg)
                    rows.append(dict(input=kind, model=CONFIGS[c][0], config=c, dt=dt, n_corrector=nc,
                                     time=res['time'], err_base=m_base['rel_l2_smooth'],
                                     err_avg=m_avg['rel_l2_smooth'], err_base_raw=m_base['rel_l2'],
                                     count_err_avg=m_avg['count_err'], n_fallback=res['n_fallback'],
                                     **{f'pop{j}_avg': m_avg[f'rel_l2_smooth_pop{j}'] for j in range(3)}))

    fields = ['input', 'model', 'config', 'dt', 'n_corrector', 'time', 'err_base', 'err_avg', 'err_base_raw',
              'count_err_avg', 'n_fallback', 'pop0_avg', 'pop1_avg', 'pop2_avg']
    with open(OUT_DIR / 'dt_reference_table.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    np.savez(OUT_DIR / 'dt_reference_rates.npz',
             **{'_'.join(str(v) for v in task): res['rb'] for task, res in results.items()},
             **{f'avg_{kind}': refs[kind]['avg'] for kind in INPUTS})

    print(f"\n{'input':9s} {'model':28s} {'dt':>7s} {'corr':>4s} {'time':>7s} {'err base':>8s} {'err avg':>8s} "
          f"{'raw':>6s} {'count':>6s}   per pop (avg)")
    for r in rows:
        print(f"{r['input']:9s} {r['model']:28s} {r['dt']:7} {r['n_corrector']!s:>4s} {r['time']:7.1f} "
              f"{r['err_base']:8.3f} {r['err_avg']:8.3f} {r['err_base_raw']:6.3f} {r['count_err_avg']:6.3f}   "
              + ' '.join(f"{r[f'pop{j}_avg']:.3f}" for j in range(3)))

    # figure: error against dt
    fig, axs = plt.subplots(2, len(INPUTS), figsize=(6 * len(INPUTS), 7.5), sharex=True, constrained_layout=True)
    for col, kind in enumerate(INPUTS):
        for row, (key, label) in enumerate([('err_base', 'IK 100k (seed 100000)'), ('err_avg', 'averaged reference')]):
            ax = axs[row, col]
            ik = [r for r in rows if r['input'] == kind and r['config'] == 'IK' and r[key] > 0]
            if ik:
                lo, hi = min(r[key] for r in ik if '20,000' not in r['model']), \
                    max(r[key] for r in ik if '20,000' not in r['model'])
                ax.axhspan(lo, hi, color='#d9d8d4', label='IK $\\geq$100k networks (spread)')
                small = [r for r in ik if '20,000' in r['model']]
                if small:
                    ax.axhline(small[0][key], color='#3d3c39', ls='--', lw=1, label='IK 20k')
            for c in CONFIGS:
                for nc, ls in zip(CORRECTORS, ['-', '--']):
                    sel = sorted([r for r in rows if r['input'] == kind and r['config'] == c and r['n_corrector'] == nc],
                                 key=lambda r: r['dt'])
                    ax.plot([r['dt'] for r in sel], [r[key] for r in sel], color=COLORS[c], ls=ls, marker='o',
                            mfc='white', ms=4, label=f'{CONFIGS[c][0]}, corrector {nc}')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel(f'smoothed rel. L2 error\nvs. {label}')
            style(ax)
            if row == 0:
                ax.set_title(f'3 populations, {kind}, $\\sigma$ = {SIGMA:g} mV, $\\Delta v$ = {DV} mV', fontsize=9)
            else:
                ax.set_xlabel(r'$\Delta t$ (ms)')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False, fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT_DIR / f'dt_convergence.{ext}', dpi=200)
    plt.close(fig)

    # rates
    edges = np.arange(0, net.T + 1e-9, net.RATE_BIN_MS)
    tb = 0.5 * (edges[1:] + edges[:-1])
    for kind in INPUTS:
        fig, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, constrained_layout=True)
        for j, ax in enumerate(axs):
            for key, rb in refs[kind].items():
                if key == 'avg' or key[0] < 100_000:
                    continue
                ax.plot(tb, rb[j], color='#b4b3ad', lw=0.8)
            ax.plot(tb, refs[kind]['avg'][j], color='#3d3c39', lw=1.6, label='averaged reference')
            for c in CONFIGS:
                res = results[('FPE', kind, c, DTS[-1], 1)]
                ax.plot(tb, res['rb'][j], color=COLORS[c], lw=1.2,
                        label=f'{CONFIGS[c][0]}, dt={DTS[-1]}, corrector 1 ({res["time"]:.0f} s)')
            ax.set_ylabel(f'{net.POP_NAMES["3pop"][j]}\nrate (Hz)')
            ax.axvline(net.T_ON, color='#b4b2a9', lw=0.6, ls=':')
            style(ax)
        axs[0].legend(fontsize=7, frameon=False)
        axs[-1].set_xlabel('t (ms)')
        fig.suptitle(f'3 populations, {kind}, $\\sigma$ = {SIGMA:g} mV (grey: single reference networks)', fontsize=10)
        fig.savefig(OUT_DIR / f'rates_{kind}.png', dpi=200)
        plt.close(fig)

    lines = [r'\begin{tabular}{l l r r r r r r}', r'\toprule',
             r'input & model & $\Delta t$ & corr. & time (s) & err.\ 100k & err.\ avg. & count err. \\', r'\midrule']
    for r in rows:
        lines.append(f"{r['input']} & {r['model'].replace('_', ' ')} & {r['dt']} & {r['n_corrector']} & "
                     f"{r['time']:.1f} & {r['err_base']:.3f} & {r['err_avg']:.3f} & {r['count_err_avg']:.3f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    caption = (f'3 populations, $\\sigma$ = {SIGMA:g} mV, FPE on $\\Delta v$ = {DV} mV: smoothed relative L2 error '
               f'(2 ms) against the IK network with 100k neurons (seed 100000) and against the average of all networks '
               f'with $\\geq$100k neurons per population. {N_PROCESSES} parallel processes.')
    imp.OUT_DIR = OUT_DIR
    imp.save_table_pdf('\n'.join(lines), caption, 'dt_reference_table')
    print(f'results saved in {OUT_DIR}')


def main_heun():
    """
    FPE runs against the converged Heun references (FPE_reference_rebuild.py), see the docstring of the module
    """
    import FPE_reference_rebuild as rebuild
    refs = {kind: np.load(rebuild.ref_file(case, rebuild.REF_DT, rebuild.N_REF))['rb'] for kind, case in HEUN_CASES.items()}
    tasks = [('FPE', kind, c, dt, nc) for kind in INPUTS for c in CONFIGS_HEUN for dt in DTS for nc in CORRECTORS]
    key = lambda task: '_'.join(str(v) for v in task)
    results = {}
    if (EULER_DIR / 'dt_reference_rates.npz').exists():
        old = np.load(EULER_DIR / 'dt_reference_rates.npz')
        with open(EULER_DIR / 'dt_reference_table.csv') as f:
            old_time = {('FPE', r['input'], r['config'], float(r['dt']), int(r['n_corrector'])): float(r['time'])
                        for r in csv.DictReader(f) if r['config'] != 'IK'}
        results = {task: dict(task=task, rb=old[key(task)], time=old_time.get(task, np.nan), n_fallback=np.nan)
                   for task in tasks if key(task) in old.files}
    todo = sorted([task for task in tasks if task not in results], key=lambda x: x[3] / (1 + x[4]))  # expensive first
    print(f'{len(results)} FPE runs from {EULER_DIR.name}, {len(todo)} to simulate')
    if todo:
        run(todo[-1][:3] + (0.05, 0))  # compile the numba kernels
        t_start = time.perf_counter()
        with Pool(N_PROCESSES) as pool:
            for q, res in enumerate(pool.imap_unordered(run, todo), start=1):
                results[res['task']] = res
                print(f'[{q}/{len(todo)}] {res["task"]}: {res["time"]:.1f} s', flush=True)
        print(f'total wall time {time.perf_counter() - t_start:.0f} s')

    rows = []
    for task in tasks:
        _, kind, c, dt, nc = task
        res = results[task]
        m = imp.measures(res['rb'], refs[kind])
        rows.append(dict(input=kind, model=CONFIGS_HEUN[c][0], config=c, dt=dt, n_corrector=nc, time=res['time'],
                         err=m['rel_l2_smooth'], err_raw=m['rel_l2'], count_err=m['count_err'],
                         n_fallback=res['n_fallback'], **{f'pop{j}': m[f'rel_l2_smooth_pop{j}'] for j in range(3)}))
    fields = ['input', 'model', 'config', 'dt', 'n_corrector', 'time', 'err', 'err_raw', 'count_err', 'n_fallback',
              'pop0', 'pop1', 'pop2']
    with open(OUT_DIR / 'dt_reference_table.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    np.savez(OUT_DIR / 'dt_reference_rates.npz', **{key(task): res['rb'] for task, res in results.items()})

    print(f"\nsmoothed rel. L2 error against the Heun reference (dt = {rebuild.REF_DT} ms, {rebuild.N_REF:,} neurons)")
    print(f"{'input':9s} {'model':16s} {'corr':>4s} " + ' '.join(f'{dt:>7}' for dt in DTS))
    for kind in INPUTS:
        for c in CONFIGS_HEUN:
            for nc in CORRECTORS:
                errs = [next(r['err'] for r in rows if (r['input'], r['config'], r['dt'], r['n_corrector']) ==
                             (kind, c, dt, nc)) for dt in DTS]
                print(f"{kind:9s} {CONFIGS_HEUN[c][0]:16s} {nc:4d} " + ' '.join(f'{e:7.3f}' for e in errs))

    fig, axs = plt.subplots(1, len(INPUTS), figsize=(5.5 * len(INPUTS), 3.8), sharey=True, constrained_layout=True)
    for ax, kind in zip(axs, INPUTS):
        for c in CONFIGS_HEUN:
            for nc, ls in zip(CORRECTORS, ['-', '--']):
                sel = sorted([r for r in rows if r['input'] == kind and r['config'] == c and r['n_corrector'] == nc],
                             key=lambda r: r['dt'])
                ax.plot([r['dt'] for r in sel], [r['err'] for r in sel], color=COLORS[c], ls=ls, marker='o',
                        mfc='white', ms=4, label=f'{CONFIGS_HEUN[c][0]}, corrector {nc}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\Delta t$ (ms)')
        ax.set_title(f'3 populations, {kind}, $\\sigma$ = {SIGMA:g} mV, $\\Delta v$ = {DV} mV', fontsize=9)
        style(ax)
    axs[0].set_ylabel('smoothed rel. L2 error\nvs. Heun reference')
    axs[0].legend(fontsize=7, frameon=False)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT_DIR / f'dt_convergence.{ext}', dpi=200)
    plt.close(fig)
    print(f'results saved in {OUT_DIR}')


if __name__ == '__main__':
    main_heun() if net.MC_METHOD == 'heun' else main()

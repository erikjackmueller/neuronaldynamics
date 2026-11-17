"""
Cost comparison of the Izhikevich FPE solvers against converged (stochastic Heun) IK networks.

The earlier cost statements (FPE_Izhikevich_resolution.py, FPE_vs_IK_network.py and the sections built on them) used
networks with Euler-Maruyama, MC_DT = 0.01 ms, which is not converged for synchronised populations
(FPE_IK_reference_dt.py). Heun with interpolated spike times is converged at MC_DT = 0.01 ms and costs about twice as
much per step. This script times, for the six cases of FPE_2D_reference_test.CASES,

    IK networks, Heun, MC_DT = 0.01 ms, N_LIST neurons per population (finite-size error)
    IK network, Euler-Maruyama, MC_DT = 0.01 ms, 20k neurons (the old setting: time step + finite-size error)
    1D closures, dv = 0.1 mV: two-node and second moment with FCT on several dt, the old variance closure
    2D solver with FCT, dv = 0.1 mV, du = 2 pA, dt = 0.01 ms

and measures their error against the converged references of FPE_reference_rebuild.py (Heun, dt 0.005 ms, 100k
neurons per population): rel. L2 in 1 ms bins (1 population), rel. L2 after the 2 ms gaussian kernel (3 populations).
N_PROCESSES runs in parallel (as in the earlier timing studies), numpy with one thread each.

Outputs in Examples/solver_tests/FPE_IK_cost_heun: runs/<case>__<model>.npz (binned rates and time, the runs are
skipped if the file exists), cost_heun_table.csv, cost_heun_work_precision.png/.pdf

Usage (from src/neuronaldynamics):
    python Examples/solver_tests/FPE_IK_cost_heun.py run [--proc P] [case ...]
    python Examples/solver_tests/FPE_IK_cost_heun.py analyze
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Model.Izhikevich_2D import FPE_Izhikevich2D
from Model.Nykamp_Model import IZHIKEVICH_DEFAULTS
import FPE_vs_IK_network as net
import FPE_Izhikevich_3pop_improvements as imp
import FPE_Izhikevich_resolution as res
import FPE_2D_reference_test as t2d
import FPE_reference_rebuild as rebuild
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'FPE_IK_cost_heun'
RUN_DIR = OUT_DIR / 'runs'
RUN_DIR.mkdir(parents=True, exist_ok=True)
N_PROCESSES = 4
MC_DT = 0.01
N_LIST = [5_000, 10_000, 20_000, 50_000]
SEED = 800_000
DV = 0.1
# model: (kind, label, settings)
MODELS = {f'ik_heun_{n // 1000}k': ('IK', f'IK Heun {n // 1000}k', dict(method='heun', n=n)) for n in N_LIST}
MODELS['ik_euler_20k'] = ('IK', 'IK Euler 20k', dict(method='euler', n=20_000))
for dt in [0.05, 0.01, 0.005]:
    MODELS[f'1d_twonode_fct_dt{dt:g}'] = ('1D', f'1D two-node FCT dt {dt:g}',
                                          dict(closure='conditional_quadrature', limiter='fct', dt=dt))
for dt in [0.05, 0.005]:
    MODELS[f'1d_2ndmom_fct_dt{dt:g}'] = ('1D', f'1D 2nd moment FCT dt {dt:g}',
                                         dict(closure='conditional_second_moment', limiter='fct', dt=dt))
MODELS['1d_variance_dt0.05'] = ('1D', '1D variance (old) dt 0.05',
                                dict(closure='conditional_variance', limiter='none', dt=0.05))
MODELS['2d_fct_du2_dt0.01'] = ('2D', '2D FCT du 2 dt 0.01', dict(du=2., dt=0.01))


def run_file(case, model):
    return RUN_DIR / f'{case}__{model}.npz'


def run(task):
    case, model = task
    kind, _, s = MODELS[model]
    setup, sigma, regime, _ = t2d.CASES[case]
    if kind == 'IK':
        t0 = time.perf_counter()
        if setup == '1pop':
            _, par_regime, i0 = res.REGIMES[regime]
            res.MC_DT, res.MC_METHOD = MC_DT, s['method']
            t, r, _ = res.monte_carlo(dict(IZHIKEVICH_DEFAULTS, **par_regime, voltage_sigma=sigma), i0, s['n'],
                                      seed=SEED + s['n'])
            rb = np.atleast_2d(res.binned_rate(t, r, MC_DT))
        else:
            imp.set_input(regime)
            net.MC_DT, net.MC_METHOD = MC_DT, s['method']
            r, _ = net.ik_network('3pop', sigma, s['n'], seed=SEED + s['n'])
            rb = net.binned_rate(r, MC_DT)
        cpu = time.perf_counter() - t0
    elif kind == '1D':
        out = t2d.run((case, s['closure'], s['limiter'], s['dt']))
        rb, cpu = out['rb'], out['time']
    else:
        m = FPE_Izhikevich2D(dict(t2d.setup_case(case), dt=s['dt'], dv=DV, du=s['du'], init_pdf_sigma=1.0,
                                  flux_limiter='fct'))
        m.simulate()
        rb, cpu = net.binned_rate(m.r, s['dt']), m.cpu
    np.savez(run_file(case, model), rb=rb, time=cpu)
    print(f'{case} {model}: {cpu:.1f} s', flush=True)
    return task


def expected_cost(task):
    """rough cost for the load balancing (largest first)"""
    case, model = task
    kind, _, s = MODELS[model]
    n_pop = 1 if t2d.CASES[case][0] == '1pop' else 3
    if kind == 'IK':
        return n_pop * s['n'] * (2 if s['method'] == 'heun' else 1) / 500
    if kind == '2D':
        return 600 * n_pop
    return n_pop * 0.5 / s['dt']


def references():
    out = {}
    for case, (setup, *_rest) in t2d.CASES.items():
        f = rebuild.ref_file(case, rebuild.REF_DT, rebuild.N_REF)
        out[case] = (np.load(f), 'rel_l2' if setup == '1pop' else 'rel_l2_smooth')
    return out


def analyze():
    refs = references()
    rows = []
    for case in t2d.CASES:
        ref, measure = refs[case]
        rows.append(dict(case=case, model='ik_heun_100k_ref', label='IK Heun 100k dt 0.005 (reference)',
                         time=float(ref['time']), err=np.nan))
        for model, (kind, label, _) in MODELS.items():
            f = run_file(case, model)
            if not f.exists():
                continue
            d = np.load(f)
            rows.append(dict(case=case, model=model, label=label, time=float(d['time']),
                             err=imp.measures(d['rb'], ref['rb'])[measure]))
    with open(OUT_DIR / 'cost_heun_table.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cases = list(t2d.CASES)
    print(f'\ncomputation time (s) / error against the converged reference (Heun, dt 0.005, 100k), '
          f'{N_PROCESSES} runs in parallel')
    print(f'{"model":36s} ' + ' '.join(f'{c:>17s}' for c in cases))
    for model in ['ik_heun_100k_ref'] + list(MODELS):
        sel = {r['case']: r for r in rows if r['model'] == model}
        if not sel:
            continue
        label = next(iter(sel.values()))['label']
        print(f'{label:36s} ' + ' '.join(f'{sel[c]["time"]:8.1f}/{sel[c]["err"]:<8.3f}' if c in sel else f'{"":17s}'
                                         for c in cases))

    styles = {'IK': dict(color='#3d3c39', marker='o'), 'ik_euler_20k': dict(color='#b4b3ad', marker='x', ms=7),
              'twonode': dict(color='#d6612a', marker='s'), '2ndmom': dict(color='#2a9d6b', marker='D'),
              'variance': dict(color='#8a5cc4', marker='^'), '2d': dict(color='#1b4f8f', marker='*', ms=10)}
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for ax, case in zip(axs.ravel(), cases):
        sel = [r for r in rows if r['case'] == case and np.isfinite(r['err'])]
        ik = [r for r in sel if r['model'].startswith('ik_heun')]
        ax.plot([r['time'] for r in ik], [r['err'] for r in ik], '-', lw=1, **styles['IK'], mfc='white',
                label='IK Heun 5k-50k')
        for r in ik:
            ax.annotate(r['model'].split('_')[-1], (r['time'], r['err']), fontsize=6, xytext=(3, 3),
                        textcoords='offset points')
        for key, name in [('twonode', '1D two-node FCT'), ('2ndmom', '1D 2nd moment FCT')]:
            m = [r for r in sel if key in r['model']]
            m.sort(key=lambda r: r['time'])
            ax.plot([r['time'] for r in m], [r['err'] for r in m], '--', lw=0.8, **styles[key], label=name)
            for r in m:
                ax.annotate(r['model'].split('_')[-1].replace('dt', ''), (r['time'], r['err']), fontsize=6,
                            xytext=(3, -8), textcoords='offset points', color=styles[key]['color'])
        for key, name, model in [('variance', '1D variance (old), dt 0.05', '1d_variance_dt0.05'),
                                 ('2d', '2D FCT du 2, dt 0.01', '2d_fct_du2_dt0.01'),
                                 ('ik_euler_20k', 'IK Euler 20k, dt 0.01 (old)', 'ik_euler_20k')]:
            m = [r for r in sel if r['model'] == model]
            ax.plot([r['time'] for r in m], [r['err'] for r in m], ls='none', **styles[key], label=name)
        t_ref = next(r['time'] for r in rows if r['case'] == case and r['model'] == 'ik_heun_100k_ref')
        ax.axvline(t_ref, color='#3d3c39', lw=0.7, ls=':', label='reference (Heun 100k, dt 0.005)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(case, fontsize=9)
        ax.grid(which='major', lw=0.4, color='#e0dfdb')
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel('computation time (s)')
    for ax in axs[:, 0]:
        ax.set_ylabel('error against the converged reference')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False, fontsize=8)
    fig.suptitle('Work-precision of the Izhikevich FPE solvers against stochastic Heun networks '
                 f'($\\Delta t$ = {MC_DT} ms), {N_PROCESSES} runs in parallel', fontsize=10)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT_DIR / f'cost_heun_work_precision.{ext}', dpi=200)
    plt.close(fig)
    print(f'results saved in {OUT_DIR}')


def option(args, name, default, typ):
    if name in args:
        i = args.index(name)
        value = typ(args[i + 1])
        del args[i:i + 2]
        return value
    return default


if __name__ == '__main__':
    args = sys.argv[1:]
    mode = args.pop(0) if args else 'analyze'
    n_proc = option(args, '--proc', N_PROCESSES, int)
    if mode == 'run':
        N_PROCESSES = n_proc
        cases = args or list(t2d.CASES)
        tasks = [(c, m) for c in cases for m in MODELS if not run_file(c, m).exists()]
        tasks.sort(key=lambda x: -expected_cost(x))
        # compile the numba kernels once before the timed runs
        t2d.run(('1pop_rs_default', 'conditional_quadrature', 'fct', 0.05))
        FPE_Izhikevich2D(dict(t2d.setup_case('1pop_rs_default'), T=1., dt=0.05, dv=DV, du=4.,
                              flux_limiter='fct')).simulate()
        t_start = time.perf_counter()
        with Pool(min(n_proc, len(tasks))) as pool:
            for q, _ in enumerate(pool.imap_unordered(run, tasks), start=1):
                print(f'[{q}/{len(tasks)}]', flush=True)
        print(f'total wall time {time.perf_counter() - t_start:.0f} s', flush=True)
    analyze()

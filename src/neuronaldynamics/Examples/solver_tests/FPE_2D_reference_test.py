"""
Test of the two-dimensional (v, u) Fokker-Planck solver (Model/Izhikevich_2D.py, no closure of the adaptation) against
the IK network references, on the cases where the one-dimensional closures fail or stall:

    1 population, sigma = 2 mV, step (regimes of FPE_Izhikevich_resolution.py), reference: Monte Carlo 100k
    3 populations with feedback, sigma = 1 mV, step and 20 Hz sine, reference: average of the networks with >= 100k
      neurons per population of FPE_dt_reference_check.py
    3 populations, sigma = 3.16 mV, step, reference: IK network 100k (FPE_Izhikevich_3pop_improvements.py)

Grids: dv = 0.1 mV, du in DUS (pA), dt in DTS (ms). The best one-dimensional closures (converged in dt) are run for
comparison: 'conditional_second_moment' and 'conditional_quadrature' with FCT, dt = 0.005 ms.

Error: relative L2 error of the rates in 1 ms bins (1 population, as in FPE_Izhikevich_resolution.py) or after a 2 ms
gaussian kernel (3 populations, as in FPE_Izhikevich_3pop_improvements.py).

Outputs in Examples/solver_tests/FPE_2D_reference_test: fpe2d_table.csv, rates_<case>.png, run_log.txt

Usage (from src/neuronaldynamics):  python Examples/solver_tests/FPE_2D_reference_test.py
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
from Model.Nykamp_Model import FPE_Population, IZHIKEVICH_DEFAULTS
import FPE_vs_IK_network as net
import FPE_Izhikevich_3pop_improvements as imp
import FPE_Izhikevich_resolution as res
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'FPE_2D_reference_test'
OUT_DIR.mkdir(exist_ok=True)
N_PROCESSES = 4
DUS = [4., 2.]
DTS = [0.05, 0.01]
DV = 0.1
CLOSURES_1D = [('conditional_second_moment', 'fct'), ('conditional_quadrature', 'fct')]
DT_1D = 0.005
# case: (setup, sigma, input, reference key)
CASES = {'1pop_rs_default': ('1pop', 2.0, 'rs_default', None), '1pop_rs_high_rate': ('1pop', 2.0, 'rs_high_rate', None),
         '1pop_fast': ('1pop', 2.0, 'fast', None),
         '3pop_step_1': ('3pop', 1.0, 'step', None), '3pop_sine20Hz_1': ('3pop', 1.0, 'sine20Hz', None),
         '3pop_step_3.16': ('3pop', float(np.sqrt(10)), 'step', None)}


def setup_case(case):
    """parameters (dict for both solvers), time step independent input function"""
    setup, sigma, kind, _ = CASES[case]
    if setup == '1pop':
        title, par, i0 = res.REGIMES[kind]
        pars = dict(par, voltage_sigma=sigma, input_function=lambda t, i0=i0: np.where(t < res.T_ON, i0, 0.),
                    input_function_idx=[0], T=res.T)
        return pars
    imp.set_input(kind)
    pars = dict(population_type=net.SETUPS['3pop'][0], connectivity_matrix=net.SETUPS['3pop'][2], voltage_sigma=sigma,
                input_function=net.input_current, input_function_idx=[0], T=net.T,
                **{name: list(v) for name, v in net.neuron_parameters('3pop').items()})
    return pars


def run(task):
    case, model, x, dt = task
    pars = setup_case(case)
    if model == '2D':
        m = FPE_Izhikevich2D(dict(pars, dt=dt, dv=DV, du=x, init_pdf_sigma=1.0))
        m.simulate()
        return dict(task=task, time=m.cpu, rb=net.binned_rate(m.r, dt), mass=min(m.mass), n_fallback=m.n_fallback)
    closure, limiter = model, x
    m = FPE_Population(dict(pars, soma_model='IK', adaptation_closure=closure, flux_limiter=limiter, dt=dt, dv=DV,
                            init_pdf_sigma=1.0, save_p='none', tqdm_disable=True),
                       name=str(OUT_DIR / ('fpe_' + '_'.join(str(v) for v in task))))
    t0 = time.perf_counter()
    m.simulate()
    cpu = time.perf_counter() - t0
    m.clean()
    return dict(task=task, time=cpu, rb=net.binned_rate(m.r, dt), mass=np.nan, n_fallback=m.n_fallback)


def references():
    refs = {}
    d = np.load(HERE / 'FPE_Izhikevich_resolution' / 'resolution_rates.npz')
    d_avg = np.load(HERE / 'FPE_dt_reference_check' / 'dt_reference_rates.npz')
    d_imp = np.load(HERE / 'FPE_Izhikevich_3pop_improvements' / 'improvements_rates.npz')
    for case, (setup, sigma, kind, _) in CASES.items():
        if setup == '1pop':
            refs[case] = (np.atleast_2d(d[f'{kind}_mc_100000_None']), np.atleast_2d(d[f'{kind}_mc_20000_None']),
                          'rel_l2')
        elif sigma == 1.0:
            refs[case] = (d_avg[f'avg_{kind}'], d_imp[f'{kind}_1.0_IK_20000'], 'rel_l2_smooth')
        else:
            refs[case] = (d_imp[f'{kind}_{sigma}_IK_100000'], d_imp[f'{kind}_{sigma}_IK_20000'], 'rel_l2_smooth')
    return refs


def main():
    refs = references()
    run(('1pop_rs_default', '2D', 4., 0.05))   # compile
    tasks = [(case, '2D', du, dt) for case in CASES for du in DUS for dt in DTS]
    tasks += [(case, c, lim, DT_1D) for case in CASES for c, lim in CLOSURES_1D]
    tasks.sort(key=lambda t: (t[1] != '2D', t[3], -t[2] if t[1] == '2D' else 0))   # expensive 2D runs first
    t_start = time.perf_counter()
    results = {}
    with Pool(N_PROCESSES) as pool:
        for q, r in enumerate(pool.imap_unordered(run, tasks), start=1):
            results[r['task']] = r
            print(f'[{q}/{len(tasks)}] {r["task"]}: {r["time"]:.1f} s', flush=True)
    print(f'total wall time {time.perf_counter() - t_start:.0f} s')
    # raw results first (the analysis below must not lose them)
    np.savez(OUT_DIR / 'fpe2d_rates.npz', **{'_'.join(str(v) for v in t): r['rb'] for t, r in results.items()},
             **{'_'.join(str(v) for v in t) + '_time': r['time'] for t, r in results.items()})

    rows = []
    for case in CASES:
        ref, small, measure = refs[case]
        m_small = imp.measures(small, ref)
        rows.append(dict(case=case, model='IK 20k', grid='', time=np.nan, err=m_small[measure],
                         err_smooth=m_small['rel_l2_smooth'], count_err=m_small['count_err'], mass=np.nan))
        for task, r in results.items():
            if task[0] != case:
                continue
            m = imp.measures(r['rb'], ref)
            grid = f'du={task[2]}, dt={task[3]}' if task[1] == '2D' else f'{task[2]}, dt={task[3]}'
            rows.append(dict(case=case, model='2D' if task[1] == '2D' else task[1], grid=grid, time=r['time'],
                             err=m[measure], err_smooth=m['rel_l2_smooth'], count_err=m['count_err'], mass=r['mass']))
    with open(OUT_DIR / 'fpe2d_table.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{'case':18s} {'model':26s} {'grid':22s} {'time':>7s} {'err':>7s} {'err sm':>7s} {'count':>6s} {'mass':>12s}")
    for r in rows:
        print(f"{r['case']:18s} {r['model']:26s} {r['grid']:22s} {r['time']:7.1f} {r['err']:7.3f} "
              f"{r['err_smooth']:7.3f} {r['count_err']:6.3f} {r['mass']:12.9f}")

    edges = np.arange(0, net.T + 1e-9, net.RATE_BIN_MS)
    tb = 0.5 * (edges[1:] + edges[:-1])
    for case in CASES:
        ref = refs[case][0]
        n_pop = ref.shape[0]
        fig, axs = plt.subplots(n_pop, 1, figsize=(10, 2.4 * n_pop + 0.8), sharex=True, squeeze=False,
                                constrained_layout=True)
        best2d = (case, '2D', min(DUS), min(DTS))
        for j in range(n_pop):
            ax = axs[j, 0]
            ax.plot(tb, ref[j], color='#3d3c39', lw=1.6, label='reference')
            ax.plot(tb, results[best2d]['rb'][j], color='#1b4f8f', lw=1.2,
                    label=f'2D FPE du={min(DUS)}, dt={min(DTS)} ({results[best2d]["time"]:.0f} s)')
            c, lim = CLOSURES_1D[-1]
            r1 = results[(case, c, lim, DT_1D)]
            ax.plot(tb, r1['rb'][j], color='#d6612a', lw=1.0, ls='--', label=f'1D {c}, {lim} ({r1["time"]:.0f} s)')
            ax.set_ylabel('rate (Hz)' if n_pop == 1 else f'{net.POP_NAMES["3pop"][j]}\nrate (Hz)')
            ax.grid(lw=0.4, color='#e0dfdb')
            ax.spines[['top', 'right']].set_visible(False)
        axs[0, 0].legend(fontsize=7, frameon=False)
        axs[-1, 0].set_xlabel('t (ms)')
        fig.suptitle(case, fontsize=10)
        fig.savefig(OUT_DIR / f'rates_{case}.png', dpi=200)
        plt.close(fig)
    print(f'results saved in {OUT_DIR}')


if __name__ == '__main__':
    main()

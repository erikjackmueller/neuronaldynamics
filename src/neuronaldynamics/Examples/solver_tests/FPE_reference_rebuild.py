"""
Rebuild of the IK network references with a converged Euler-Maruyama time step.

All references of the solver tests (FPE_Izhikevich_resolution.py: 1 population, FPE_vs_IK_network.py,
FPE_Izhikevich_3pop_improvements.py, FPE_dt_reference_check.py: 3 populations) used MC_DT = 0.01 ms. That is not
converged (FPE_IK_reference_dt.py): in the 3 pop sigma = 1 mV step case it creates a post-stimulus volley of exc 2 and
an inhibitory burst that disappear for MC_DT -> 0, and the 20 Hz case moves by ~15 %. Euler-Maruyama with spikes on the
time grid converges very slowly (sub-ms drift of the burst times down to MC_DT = 0.0003 ms); the stochastic Heun scheme
with interpolated spike times (METHOD = 'heun') agrees with it there and is converged already at MC_DT = 0.01 ms.
This script reruns the reference of every case of FPE_2D_reference_test.CASES with METHOD, MC_DT = REF_DT and recomputes the errors of the stored model results
(FPE_2D_reference_test/fpe2d_rates.npz: 1D closures and 2D solver without FCT; FPE_2D_fct_3pop/runs: 2D solver of the
hard case, with FCT) against the old and the new reference.

Error measure per case as in FPE_2D_reference_test.references: rel. L2 in 1 ms bins (1 population), rel. L2 after the
2 ms gaussian kernel (3 populations).

Outputs in Examples/solver_tests/FPE_reference_rebuild: ref_<case>_dt<REF_DT>_<n>k.npz, rebuild_table.csv, run_log.txt

Usage (from src/neuronaldynamics):
    python Examples/solver_tests/FPE_reference_rebuild.py run [--dt REF_DT] [--n N] [--proc P] [case ...]
    python Examples/solver_tests/FPE_reference_rebuild.py analyze [--dt REF_DT] [--n N]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Model.Nykamp_Model import IZHIKEVICH_DEFAULTS
import FPE_vs_IK_network as net
import FPE_Izhikevich_3pop_improvements as imp
import FPE_Izhikevich_resolution as res
import FPE_2D_reference_test as t2d
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'FPE_reference_rebuild'
OUT_DIR.mkdir(exist_ok=True)
REF_DT = 0.005
METHOD = 'heun'            # converged at MC_DT <= 0.01 ms (FPE_IK_reference_dt.py), Euler-Maruyama needs <= 0.0003 ms
N_REF = 100_000
SEED = 600_000


def ref_file(case, dt, n):
    return OUT_DIR / f'ref_{case}_{METHOD}_dt{dt:g}_{n // 1000}k.npz'


def run(task):
    case, dt, n = task
    setup, sigma, kind, _ = t2d.CASES[case]
    t0 = time.perf_counter()
    if setup == '1pop':
        _, par_regime, i0 = res.REGIMES[kind]
        res.MC_DT, res.MC_METHOD = dt, METHOD     # monte_carlo reads the module constants
        t, r, _ = res.monte_carlo(dict(IZHIKEVICH_DEFAULTS, **par_regime, voltage_sigma=sigma), i0, n, seed=SEED)
        rb = np.atleast_2d(res.binned_rate(t, r, dt))
    else:
        imp.set_input(kind)
        net.MC_DT, net.MC_METHOD = dt, METHOD     # ik_network reads the module constants
        r, _ = net.ik_network('3pop', sigma, n, seed=SEED)
        rb = net.binned_rate(r, dt)
    cpu = time.perf_counter() - t0
    np.savez(ref_file(case, dt, n), rb=rb, time=cpu)
    print(f'{case}: {cpu:.0f} s', flush=True)
    return task


def models(case):
    """stored model results of the case: name -> binned rates"""
    old = np.load(HERE / 'FPE_2D_reference_test' / 'fpe2d_rates.npz')
    out = {'1D 2nd moment, FCT': old[f'{case}_conditional_second_moment_fct_0.005'],
           '1D two-node, FCT': old[f'{case}_conditional_quadrature_fct_0.005']}
    for du, dt in [(4., 0.05), (2., 0.05), (2., 0.01)]:
        out[f'2D du={du:g} dt={dt:g}'] = old[f'{case}_2D_{du}_{dt}']
    if case == '3pop_step_1':
        for f in sorted((HERE / 'FPE_2D_fct_3pop' / 'runs').glob('*.npz')):
            out[f.stem if f.stem.startswith('1d_') else f'2D {f.stem}'] = np.load(f)['rb']
    return out


def analyze(dt, n):
    refs_old = t2d.references()
    rows = []
    for case in t2d.CASES:
        f = ref_file(case, dt, n)
        if not f.exists():
            print(f'{case}: no new reference yet')
            continue
        new = np.load(f)['rb']
        old_ref, small, measure = refs_old[case]
        entries = [('old reference (MC_DT 0.01)', old_ref), ('IK 20k, MC_DT 0.01', small)] + list(models(case).items())
        for name, rb in entries:
            rows.append(dict(case=case, model=name, measure=measure,
                             err_old=imp.measures(rb, old_ref)[measure] if name != 'old reference (MC_DT 0.01)' else 0.,
                             err_new=imp.measures(rb, new)[measure]))
    if not rows:
        return
    with open(OUT_DIR / f'rebuild_table_dt{dt:g}_{n // 1000}k.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nerrors against the old (MC_DT 0.01) and the new reference (MC_DT {dt:g}, {n:,} neurons per population)')
    print(f'{"case":18s} {"model":34s} {"measure":>13s} {"old ref":>8s} {"new ref":>8s}')
    for r in rows:
        print(f'{r["case"]:18s} {r["model"]:34s} {r["measure"]:>13s} {r["err_old"]:8.3f} {r["err_new"]:8.3f}')


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
    dt = option(args, '--dt', REF_DT, float)
    n = option(args, '--n', N_REF, int)
    n_proc = option(args, '--proc', 4, int)
    if mode == 'run':
        cases = args or list(t2d.CASES)
        # expensive (3 populations) first
        tasks = sorted([(c, dt, n) for c in cases], key=lambda x: t2d.CASES[x[0]][0] == '1pop')
        t_start = time.perf_counter()
        with Pool(min(n_proc, len(tasks))) as pool:
            for _ in pool.imap_unordered(run, tasks):
                pass
        print(f'total wall time {time.perf_counter() - t_start:.0f} s', flush=True)
    analyze(dt, n)

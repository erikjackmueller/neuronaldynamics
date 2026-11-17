"""
Convergence of the two-dimensional solver (Model/Izhikevich_2D.py) in du and dt for the hardest case: 3 populations
with feedback, sigma = 1 mV, step input (FPE_2D_reference_test.py: du = 2 pA, dt = 0.01 ms was not converged).
Reference: average of the IK networks with >= 100k neurons per population (FPE_dt_reference_check.py).
Results appended to Examples/solver_tests/FPE_2D_reference_test/convergence_3pop_step_1.csv / .npz

Usage (from src/neuronaldynamics):  python Examples/solver_tests/FPE_2D_convergence_3pop.py
"""
import os
for _var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_var, '1')
import sys
import csv
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import FPE_2D_reference_test as t

GRIDS = [(1., 0.01), (2., 0.005), (1., 0.005)]


def run(grid):
    return t.run(('3pop_step_1', '2D', grid[0], grid[1]))


if __name__ == '__main__':
    ref = t.references()['3pop_step_1'][0]
    with Pool(len(GRIDS)) as pool:
        res = pool.map(run, GRIDS)
    np.savez(t.OUT_DIR / 'convergence_3pop_step_1.npz', **{f'du{g[0]}_dt{g[1]}': r['rb'] for g, r in zip(GRIDS, res)})
    with open(t.OUT_DIR / 'convergence_3pop_step_1.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['du', 'dt', 'time', 'err_smooth', 'count_err', 'pop0', 'pop1', 'pop2'])
        for g, r in zip(GRIDS, res):
            m = t.imp.measures(r['rb'], ref)
            row = [g[0], g[1], r['time'], m['rel_l2_smooth'], m['count_err']] + [m[f'rel_l2_smooth_pop{j}'] for j in range(3)]
            w.writerow(row)
            print(row, flush=True)

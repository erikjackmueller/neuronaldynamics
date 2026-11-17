"""
Two-dimensional solver (Model/Izhikevich_2D.py) with and without flux-corrected transport in v, hardest case:
3 populations with feedback, sigma = 1 mV, step input (FPE_2D_reference_test.py, FPE_2D_convergence_3pop.py).
Question: is the saturation of the 2D error at ~0.55 (dv = 0.1 mV, no FCT) the numerical diffusion of the
Scharfetter-Gummel flux in v? Grids (dv, du, dt, flux_limiter) are given on the command line; every run is saved on
its own (runs/<tag>.npz), the analysis uses all saved runs plus the earlier runs without FCT on dv = 0.1 mV.

Error measures against the averaged IK reference (>= 100k neurons per population, FPE_dt_reference_check.py):
- smoothed rel. L2 (2 ms kernel) and count error as before (FPE_Izhikevich_3pop_improvements.measures)
- event measures (timing tolerant), see event_measures: the post-stimulus volley of exc 2 (window 105-150 ms) and the
  inhibitory bursts it triggers (115-150 ms): spikes per neuron in the window, onset (10 % of the window count),
  centroid and peak time, peak rate
- cum_dist_ms (per population, t >= T_ON): integral |N_x(t) - N_ref(t)| dt / max(N_ref(T), N_x(T)), N cumulative
  spike count from T_ON: equals the shift in ms for a pure time shift of the response, and ~ f * (T - t) when a fraction
  f of the spikes is missing from time t on.
- self convergence: smoothed rel. L2 distance between the 2D solutions on different grids (independent of the
  reference).

Usage (from src/neuronaldynamics):
    python Examples/solver_tests/FPE_2D_fct_3pop.py run 0.1,2,0.01,fct 0.1,1,0.01,fct     (runs in parallel)
    python Examples/solver_tests/FPE_2D_fct_3pop.py analyze
    python Examples/solver_tests/FPE_2D_fct_3pop.py run1d 0.05,0.005,conditional_quadrature   (1D closure with FCT)
    python Examples/solver_tests/FPE_2D_fct_3pop.py easy 1pop_rs_default,2,0.01 3pop_step_3.16,2,0.01   (no regression
        on the cases of FPE_2D_reference_test.py that already agreed, with FCT; analyze_easy: analysis only)
"""
import os
for _var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_var, '1')
import sys
import csv
import warnings
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
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'FPE_2D_fct_3pop'
RUN_DIR = OUT_DIR / 'runs'
RUN_DIR.mkdir(parents=True, exist_ok=True)
EASY_DIR = OUT_DIR / 'runs_easy'
EASY_DIR.mkdir(exist_ok=True)
CASE = '3pop_step_1'
# event windows (ms) per population: exc 2 volley and inhibitory bursts after the end of the step (T_ON = 100 ms)
EVENT_WINDOWS = {1: (105., 150.), 2: (115., 150.)}
ONSET_FRACTION = 0.1


########################################################################################################################
# EVENT MEASURES
########################################################################################################################
def event_measures(rb):
    """timing tolerant measures of the binned rates rb (n_pop, n_bins), rate in Hz in RATE_BIN_MS bins"""
    dtb = net.RATE_BIN_MS
    t = (np.arange(rb.shape[1]) + 0.5) * dtb
    out = {}
    for j, (t0, t1) in EVENT_WINDOWS.items():
        sel = (t >= t0) & (t < t1)
        x, tw = rb[j, sel], t[sel]
        count = x.sum() * dtb * 1e-3                  # spikes per neuron
        out[f'count{j}'] = count
        out[f'peak{j}'] = x.max()
        if count < 1e-4:
            out[f'onset{j}'] = out[f'centroid{j}'] = out[f'tpeak{j}'] = np.nan
            continue
        cum = np.concatenate([[0.], np.cumsum(x)]) / x.sum()
        edges = np.concatenate([tw - 0.5 * dtb, [tw[-1] + 0.5 * dtb]])
        out[f'onset{j}'] = float(np.interp(ONSET_FRACTION, cum, edges))
        out[f'centroid{j}'] = float(np.dot(x, tw) / x.sum())
        out[f'tpeak{j}'] = float(tw[np.argmax(x)])
    return out


def cum_dist_ms(rb, ref):
    """per population: integral |N_x - N_ref| dt / N_ref(T) over t >= T_ON, N = spikes per neuron since T_ON"""
    dtb = net.RATE_BIN_MS
    i0 = int(net.T_ON / dtb)
    out = []
    for j in range(ref.shape[0]):
        nx = np.cumsum(rb[j, i0:]) * dtb * 1e-3
        nr = np.cumsum(ref[j, i0:]) * dtb * 1e-3
        # normalized by the larger count (finite when one of the two has no spikes after T_ON)
        out.append(float(np.sum(np.abs(nx - nr)) * dtb / max(nr[-1], nx[-1], 1e-12)))
    return out


def event_errors(e, e_ref):
    """differences of the event measures to the reference: count and peak relative, times in ms"""
    out = {}
    for j in EVENT_WINDOWS:
        out[f'count{j}_rel'] = e[f'count{j}'] / e_ref[f'count{j}'] - 1
        out[f'peak{j}_rel'] = e[f'peak{j}'] / e_ref[f'peak{j}'] - 1
        for key in ['onset', 'centroid', 'tpeak']:
            out[f'{key}{j}_ms'] = e[f'{key}{j}'] - e_ref[f'{key}{j}']
    return out


def print_events(e):
    for j in EVENT_WINDOWS:
        print(f'  pop {j}: count {e[f"count{j}"]:.3f}, peak {e[f"peak{j}"]:.0f}, onset {e[f"onset{j}"]:.2f} ms, '
              f'centroid {e[f"centroid{j}"]:.2f} ms, peak at {e[f"tpeak{j}"]:.1f} ms')


def print_table(rows, key):
    cols = [c for c in rows[0] if c != key]
    print(f'{key:34s} ' + ' '.join(f'{c[:11]:>11s}' for c in cols))
    for r in rows:
        print(f'{r[key]:34s} ' + ' '.join(f'{r[c]:11.4g}' if isinstance(r[c], (float, int, np.floating)) else
                                          f'{str(r[c]):>11s}' for c in cols))


########################################################################################################################
# RUNS
########################################################################################################################
def tag(dv, du, dt, limiter):
    return f'dv{dv:g}_du{du:g}_dt{dt:g}_{limiter}'


def run(task):
    import FPE_2D_reference_test as t2d
    from Model.Izhikevich_2D import FPE_Izhikevich2D
    dv, du, dt, limiter = task
    pars = t2d.setup_case(CASE)
    m = FPE_Izhikevich2D(dict(pars, dt=dt, dv=dv, du=du, init_pdf_sigma=1.0, flux_limiter=limiter))
    m.simulate()
    rb = net.binned_rate(m.r, dt)
    np.savez(RUN_DIR / f'{tag(*task)}.npz', rb=rb, time=m.cpu, mass=np.array(m.mass), n_fallback=m.n_fallback)
    print(f'{tag(*task)}: {m.cpu:.0f} s, mass {min(m.mass):.9f}, fallbacks {m.n_fallback}', flush=True)
    return task


def run_easy(task):
    """FCT on the cases where the 2D solver without FCT already agreed with the networks (no regression)"""
    import FPE_2D_reference_test as t2d
    from Model.Izhikevich_2D import FPE_Izhikevich2D
    case, du, dt = task
    m = FPE_Izhikevich2D(dict(t2d.setup_case(case), dt=dt, dv=0.1, du=du, init_pdf_sigma=1.0, flux_limiter='fct'))
    m.simulate()
    rb = net.binned_rate(m.r, dt)
    np.savez(EASY_DIR / f'{case}_du{du:g}_dt{dt:g}_fct.npz', rb=rb, time=m.cpu, mass=np.array(m.mass),
             n_fallback=m.n_fallback)
    print(f'{case} du{du:g} dt{dt:g} fct: {m.cpu:.0f} s, fallbacks {m.n_fallback}', flush=True)
    return task


def analyze_easy():
    import FPE_2D_reference_test as t2d
    refs = t2d.references()
    old = np.load(HERE / 'FPE_2D_reference_test' / 'fpe2d_rates.npz')
    print(f'{"case":20s} {"grid":14s} {"err none":>9s} {"err fct":>9s} {"IK 20k":>8s}')
    for f in sorted(EASY_DIR.glob('*.npz')):
        case, rest = f.stem.split('_du')
        du, dt = rest.split('_')[0], rest.split('_')[1][2:]
        ref, small, measure = refs[case]
        e_fct = imp.measures(np.load(f)['rb'], ref)[measure]
        key = f'{case}_2D_{float(du)}_{float(dt)}'
        e_none = imp.measures(old[key], ref)[measure] if key in old.files else np.nan
        print(f'{case:20s} du{du} dt{dt:8s} {e_none:9.3f} {e_fct:9.3f} {imp.measures(small, ref)[measure]:8.3f}')


def run_1d(task):
    """one-dimensional closure with FCT on the hardest case on a finer voltage grid (is its error grid or closure?)"""
    import FPE_2D_reference_test as t2d
    dv, dt, closure = task
    t2d.DV = dv
    r = t2d.run((CASE, closure, 'fct', dt))
    np.savez(RUN_DIR / f'1d_{closure}_dv{dv:g}_dt{dt:g}_fct.npz', rb=r['rb'], time=r['time'])
    print(f'1D {closure} dv{dv:g} dt{dt:g} fct: {r["time"]:.0f} s', flush=True)
    return task


def parse(arg):
    dv, du, dt, limiter = arg.split(',')
    return float(dv), float(du), float(dt), limiter


########################################################################################################################
# ANALYSIS
########################################################################################################################
def collect():
    """all 2D runs (new and earlier, dv = 0.1 mV without FCT), 1D closures and IK 20k for comparison"""
    sols = {}
    old = np.load(HERE / 'FPE_2D_reference_test' / 'fpe2d_rates.npz')
    for du, dt in [(4., 0.05), (4., 0.01), (2., 0.05), (2., 0.01)]:
        sols[tag(0.1, du, dt, 'none')] = old[f'{CASE}_2D_{du}_{dt}']
    conv = np.load(HERE / 'FPE_2D_reference_test' / 'convergence_3pop_step_1.npz')
    for du, dt in [(1., 0.01), (2., 0.005), (1., 0.005)]:
        sols[tag(0.1, du, dt, 'none')] = conv[f'du{du}_dt{dt}']
    for f in sorted(RUN_DIR.glob('*.npz')):
        sols[f.stem] = np.load(f)['rb']
    others = {'1D quadrature FCT dt0.005': old[f'{CASE}_conditional_quadrature_fct_0.005'],
              '1D 2nd moment FCT dt0.005': old[f'{CASE}_conditional_second_moment_fct_0.005'],
              'IK 20k': np.load(HERE / 'FPE_Izhikevich_3pop_improvements' / 'improvements_rates.npz')[
                  'step_1.0_IK_20000']}
    ref = np.load(HERE / 'FPE_dt_reference_check' / 'dt_reference_rates.npz')['avg_step']
    return sols, others, ref


def analyze():
    sols, others, ref = collect()
    e_ref = event_measures(ref)
    print('reference (average of the IK networks >= 100k):')
    print_events(e_ref)
    rows = []
    for name, rb in list(others.items()) + list(sols.items()):
        m = imp.measures(rb, ref)
        rows.append(dict(model=name, err_smooth=m['rel_l2_smooth'], count_err=m['count_err'],
                         **{f'l2_pop{j}': m[f'rel_l2_smooth_pop{j}'] for j in range(3)},
                         **{f'cum_ms{j}': c for j, c in enumerate(cum_dist_ms(rb, ref))},
                         **event_errors(event_measures(rb), e_ref)))
    with open(OUT_DIR / 'fct2d_table.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print_table(rows, 'model')

    # self convergence: smoothed rel. L2 distance between 2D solutions
    names = sorted(sols)
    dist = np.array([[imp.measures(sols[a], sols[b])['rel_l2_smooth'] for b in names] for a in names])
    with open(OUT_DIR / 'self_convergence.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + names)
        for a, row in zip(names, dist):
            writer.writerow([a] + [f'{x:.4f}' for x in row])
    print('\nself convergence (smoothed rel. L2 distance, row as reference):')
    print(f'{"":30s}' + ''.join(f'{n[:22]:>23s}' for n in names))
    for a, row in zip(names, dist):
        print(f'{a:30s}' + ''.join(f'{x:23.3f}' for x in row))

    # figure: zoom on the post-stimulus event
    t = (np.arange(ref.shape[1]) + 0.5) * net.RATE_BIN_MS
    fig, axs = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, constrained_layout=True)
    show = [n for n in names if n.endswith('_fct')] + [tag(0.1, 1., 0.01, 'none'), tag(0.1, 2., 0.01, 'none')]
    show += [n for n in names if n.startswith('dv0.05') and n.endswith('none')]
    for j, ax in enumerate(axs):
        ax.plot(t, ref[j], color='#3d3c39', lw=2.0, label='IK reference (avg.)')
        ax.plot(t, others['1D quadrature FCT dt0.005'][j], color='#b4b3ad', lw=1.0, ls='--', label='1D quadrature FCT')
        for n in dict.fromkeys(show):
            if n in sols:
                ax.plot(t, sols[n][j], lw=1.1, ls='-' if n.endswith('fct') else ':', label=n)
        ax.set_xlim(95, 160)
        ax.set_ylabel(f'{net.POP_NAMES["3pop"][j]}\nrate (Hz)')
        imp.style(ax)
    axs[2].set_yscale('symlog', linthresh=1e3)
    axs[0].legend(fontsize=7, frameon=False, ncol=2)
    axs[-1].set_xlabel('t (ms)')
    fig.suptitle('3 populations, step, $\\sigma$ = 1 mV: 2D solver with and without FCT (solid: FCT)', fontsize=10)
    fig.savefig(OUT_DIR / 'rates_zoom.png', dpi=150)
    plt.close(fig)
    print(f'results saved in {OUT_DIR}')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'analyze'
    if mode == 'run':
        tasks = [parse(a) for a in sys.argv[2:]]
        with Pool(len(tasks)) as pool:
            for _ in pool.imap_unordered(run, tasks):
                pass
    if mode == 'easy':
        # e.g. easy 1pop_rs_default,2,0.01 3pop_step_3.16,2,0.01
        tasks = [(a.split(',')[0], float(a.split(',')[1]), float(a.split(',')[2])) for a in sys.argv[2:]]
        with Pool(len(tasks)) as pool:
            for _ in pool.imap_unordered(run_easy, tasks):
                pass
    if mode == 'run1d':
        # e.g. run1d 0.05,0.005,conditional_quadrature
        tasks = [(float(a.split(',')[0]), float(a.split(',')[1]), a.split(',')[2]) for a in sys.argv[2:]]
        with Pool(len(tasks)) as pool:
            for _ in pool.imap_unordered(run_1d, tasks):
                pass
    if mode in ('easy', 'analyze_easy'):
        analyze_easy()
    else:
        analyze()

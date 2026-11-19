"""
Test of two extensions of the FPE solvers (Model/Nykamp_Model.py):

- flux_limiter='fct': flux-corrected transport for the Scharfetter-Gummel flux (LIF and Izhikevich): the antidiffusive
  flux F_central - F_SG = (D / h) (P coth P - 1) (p_{i+1} - p_i) is added to the implicit low order step with the
  Zalesak limiter (no new extrema, p >= 0, mass conserving; for the Izhikevich closures the same limiter for p, q and
  q2 plus a failsafe on the ratios E[u^k | v]). Removes the numerical diffusion ~|drift| dv / 2 of the SG flux.
- adaptation_closure='conditional_second_moment' (Izhikevich): q2 = p E[u^2 | v] as third 1D field, Var(u | v) =
  q2 / p - E[u | v]^2 as function of v (instead of 0 or one value for all v), Gaussian closure of the third moment.

Parts:
  A  3 populations with feedback (FPE_vs_IK_network.py), sigma = 1 mV, dv = 0.1 mV, dt = 0.05 ms (the grid on which the
     FPE failed), step and 20 Hz sine input, all closures with and without FCT. Reference: IK networks with 100k and
     20k neurons per population of FPE_Izhikevich_3pop_improvements.py (same network, inputs and seeds, loaded from
     its improvements_rates.npz / improvements_table.csv).
  B  1 population, step, sigma = 0.316 and 1 mV, dv = 0.1 mV, dt = 0.01 ms: numerical diffusion with / without FCT for
     'conditional_variance' and 'conditional_second_moment', against IK 100k (FPE_Izhikevich_low_noise.py) and the FPE
     on dv = 0.01 mV.
  C  LIF population (FPE_vs_LIF_timing.py settings: step 0.5 nA, T = 50 ms), sigma in LIF_SIGMAS, dv in LIF_DVS, with
     and without FCT, against the FPE with dv = 0.005 mV.

Error measures as in FPE_Izhikevich_3pop_improvements.py (rel. L2 of the rates in 1 ms bins, smoothed with a 2 ms
gaussian, nrmse, spike count error).

Outputs in Examples/solver_tests/FPE_fct_closure_test:
    partA_work_precision.png/.pdf, partA_rates_<input>.png, partB_1pop.png, partC_lif.png/.pdf,
    fct_closure_table.csv, fct_closure_table.tex / .pdf

Usage (from src/neuronaldynamics):  python Examples/solver_tests/FPE_fct_closure_test.py
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
OUT_DIR = HERE / f'FPE_fct_closure_test{net.method_suffix()}'
OUT_DIR.mkdir(exist_ok=True)
IMP_DIR = HERE / f'FPE_Izhikevich_3pop_improvements{net.method_suffix()}'
LOW_NOISE_DIR = HERE / f'FPE_Izhikevich_low_noise{net.method_suffix()}'

########################################################################################################################
# SETTINGS
########################################################################################################################
N_PROCESSES = 4
CLOSURES = ['mean', 'conditional', 'conditional_variance', 'conditional_second_moment']
LIMITERS = ['none', 'fct']
CLOSURE_SHORT = {'mean': 'mean', 'conditional': 'cond.', 'conditional_variance': 'cond.+var.',
                 'conditional_second_moment': 'cond. 2nd mom.'}
# part A
A_SIGMA, A_DV, A_DT = 1.0, 0.1, 0.05
A_INPUTS = ['step', 'sine20Hz']
# part B
B_SIGMAS = [float(np.sqrt(0.1)), 1.0]
B_DV, B_DT, B_DV_FINE = 0.1, 0.01, 0.01
B_CLOSURES = ['conditional_variance', 'conditional_second_moment']
# part C (LIF, settings of FPE_vs_LIF_timing.py)
LIF_SIGMAS = [0.1, 0.5, 2.0]
LIF_DVS = [0.2, 0.1, 0.05]
LIF_DV_REF = 0.005
LIF = dict(T=50., dt=0.01, tau=12., u_rest=-70., u_reset=-70., u_thr=-55., u_inh=-100., amp=0.5e-9, on=5., off=30.)

STYLE = {('mean', 'none'): '#8a8984', ('conditional', 'none'): '#2a78d6', ('conditional_variance', 'none'): '#2a9d6b',
         ('conditional_second_moment', 'none'): '#d6612a'}


def run(task):
    part = task[0]
    if part == 'A':
        _, kind, closure, limiter = task
        imp.set_input(kind)
        pars = dict(soma_model='IK', T=net.T, dt=A_DT, dv=A_DV, population_type=net.SETUPS['3pop'][0],
                    connectivity_matrix=net.SETUPS['3pop'][2], voltage_sigma=A_SIGMA, input_function=net.input_current,
                    input_function_idx=[0], init_pdf_sigma=1.0, save_p='none', tqdm_disable=True,
                    adaptation_closure=closure, flux_limiter=limiter,
                    **{name: list(v) for name, v in net.neuron_parameters('3pop').items()})
        dt = A_DT
    elif part == 'B':
        _, sigma, closure, limiter, dv = task
        imp.set_input('step')
        pars = dict(soma_model='IK', T=net.T, dt=B_DT, dv=dv, voltage_sigma=sigma, input_function=net.input_current,
                    input_function_idx=[0], init_pdf_sigma=1.0, save_p='none', tqdm_disable=True,
                    adaptation_closure=closure, flux_limiter=limiter, **{k: v for k, v in net.RS.items()})
        dt = B_DT
    else:
        _, sigma, dv, limiter = task
        t = np.arange(0, LIF['T'], LIF['dt'])
        pars = dict(T=LIF['T'], dt=LIF['dt'], dv=dv, tau_mem=[LIF['tau']], population_type=['exc'],
                    voltage_sigma=sigma, static_noise=True,
                    input_function=np.where((t > LIF['on']) & (t <= LIF['off']), LIF['amp'], 0.),
                    input_function_idx=[0], init_pdf_sigma=1.0, u_inh=LIF['u_inh'], u_rest=LIF['u_rest'],
                    u_reset=LIF['u_reset'], u_thr=LIF['u_thr'], save_p='none', tqdm_disable=True,
                    flux_limiter=limiter)
        dt = LIF['dt']
    model = FPE_Population(pars, name=str(OUT_DIR / ('fpe_' + '_'.join(str(x) for x in task))))
    t0 = time.perf_counter()
    model.simulate()
    cpu = time.perf_counter() - t0
    model.clean()
    out = dict(task=task, time=cpu, r=model.r.copy(), n_fallback=getattr(model, 'n_fallback', 0))
    if part in 'AB':
        out['rb'] = net.binned_rate(model.r, dt)
    return out


def load_references():
    """IK references of part A (improvements test) and B (low noise test)"""
    ref = {}
    d = np.load(IMP_DIR / 'improvements_rates.npz')
    times = {}
    with open(IMP_DIR / 'improvements_table.csv') as f:
        for row in csv.DictReader(f):
            if row['model'] == 'IK' and abs(float(row['sigma']) - A_SIGMA) < 1e-9:
                times[(row['input'], int(row['n']))] = float(row['time'])
    for kind in A_INPUTS:
        for n in imp.N_LIST:
            ref[('A', kind, n)] = dict(rb=d[f'{kind}_{A_SIGMA}_IK_{n}'], time=times[(kind, n)])
    d = np.load(LOW_NOISE_DIR / 'low_noise_rates.npz')
    for sigma in B_SIGMAS:
        for n in [20_000, 100_000]:
            key = next(k for k in d.files if k.startswith('1pop_') and k.endswith(f'_IK_{n}_None')
                       and abs(float(k.split('_')[1]) - sigma) < 1e-9)
            ref[('B', sigma, n)] = dict(rb=d[key])
    return ref


def style(ax):
    ax.grid(which='major', lw=0.4, color='#e0dfdb')
    ax.spines[['top', 'right']].set_visible(False)


def plot_part_a(rows, results, ref):
    fig, axs = plt.subplots(1, len(A_INPUTS), figsize=(6 * len(A_INPUTS), 4.6), constrained_layout=True)
    for ax, kind in zip(np.atleast_1d(axs), A_INPUTS):
        for r in [x for x in rows if x['part'] == 'A' and x['case'] == kind and x['closure'] != 'IK 20k']:
            color = STYLE[(r['closure'], 'none')]
            ax.scatter(r['time'], r['rel_l2_smooth'], s=70, color=color if r['limiter'] == 'none' else 'white',
                       edgecolors=color, linewidths=1.6, marker='o' if r['limiter'] == 'none' else 's', zorder=4,
                       label=f"{CLOSURE_SHORT[r['closure']]}, {'SG' if r['limiter'] == 'none' else 'FCT'}")
        small = [x for x in rows if x['part'] == 'A' and x['case'] == kind and x['closure'] == 'IK 20k'][0]
        ax.scatter(small['time'], small['rel_l2_smooth'], marker='^', s=80, color='#3d3c39', label='IK 20k')
        ax.axvline(small['time'], color='#3d3c39', lw=0.6, ls='--')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('computation time (s)')
        ax.set_title(f'3 populations, {kind}, $\\sigma$ = {A_SIGMA:g} mV, $\\Delta v$ = {A_DV}, $\\Delta t$ = {A_DT}',
                     fontsize=9)
        style(ax)
    np.atleast_1d(axs)[0].set_ylabel('rel. L2 error vs. IK 100k (2 ms smoothing)')
    handles, labels = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=5, frameon=False, fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT_DIR / f'partA_work_precision.{ext}', dpi=200)
    plt.close(fig)
    edges = np.arange(0, net.T + 1e-9, net.RATE_BIN_MS)
    tb = 0.5 * (edges[1:] + edges[:-1])
    for kind in A_INPUTS:
        fig, axs = plt.subplots(3, 2, figsize=(14, 7.5), sharex=True, sharey='row', constrained_layout=True)
        for col, closure in enumerate(['conditional_variance', 'conditional_second_moment']):
            for j in range(3):
                ax = axs[j, col]
                ax.plot(tb, ref[('A', kind, 100_000)]['rb'][j], color='#3d3c39', lw=1.6, label='IK 100k')
                for limiter, ls in [('none', '-'), ('fct', '--')]:
                    res = results[('A', kind, closure, limiter)]
                    ax.plot(tb, res['rb'][j], color=STYLE[(closure, 'none')], lw=1.2, ls=ls,
                            label=f"{'SG' if limiter == 'none' else 'FCT'} ({res['time']:.1f} s)")
                ax.axvline(net.T_ON, color='#b4b2a9', lw=0.6, ls=':')
                style(ax)
                if col == 0:
                    ax.set_ylabel(f'{net.POP_NAMES["3pop"][j]}\nrate (Hz)')
            axs[0, col].set_title(CLOSURE_SHORT[closure], fontsize=9)
            axs[0, col].legend(fontsize=7, frameon=False, loc='upper right')
            axs[-1, col].set_xlabel('t (ms)')
        fig.suptitle(f'3 populations, {kind}, $\\sigma$ = {A_SIGMA:g} mV, $\\Delta v$ = {A_DV} mV, '
                     f'$\\Delta t$ = {A_DT} ms', fontsize=10)
        fig.savefig(OUT_DIR / f'partA_rates_{kind}.png', dpi=200)
        plt.close(fig)


def plot_part_b(results, ref):
    edges = np.arange(0, net.T + 1e-9, net.RATE_BIN_MS)
    tb = 0.5 * (edges[1:] + edges[:-1])
    fig, axs = plt.subplots(len(B_SIGMAS), len(B_CLOSURES), figsize=(13, 3.2 * len(B_SIGMAS) + 0.6), sharex=True,
                            constrained_layout=True, squeeze=False)
    for row, sigma in enumerate(B_SIGMAS):
        for col, closure in enumerate(B_CLOSURES):
            ax = axs[row, col]
            ax.plot(tb, ref[('B', sigma, 100_000)]['rb'][0], color='#3d3c39', lw=1.6, label='IK 100k')
            fine = results[('B', sigma, closure, 'none', B_DV_FINE)]
            ax.plot(tb, fine['rb'][0], color='#b4b3ad', lw=1.2, label=f'SG $\\Delta v$={B_DV_FINE} ({fine["time"]:.0f} s)')
            for limiter, ls in [('none', '-'), ('fct', '--')]:
                res = results[('B', sigma, closure, limiter, B_DV)]
                ax.plot(tb, res['rb'][0], color=STYLE[(closure, 'none')], ls=ls, lw=1.2,
                        label=f"{'SG' if limiter == 'none' else 'FCT'} $\\Delta v$={B_DV} ({res['time']:.1f} s)")
            ax.set_xlim(0, 130)
            ax.set_title(f'{CLOSURE_SHORT[closure]}, $\\sigma$ = {sigma:.3g} mV', fontsize=9, loc='left')
            ax.legend(fontsize=7, frameon=False)
            style(ax)
        axs[row, 0].set_ylabel('rate (Hz)')
    for ax in axs[-1]:
        ax.set_xlabel('t (ms)')
    fig.savefig(OUT_DIR / 'partB_1pop.png', dpi=200)
    plt.close(fig)


def plot_part_c(rows):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = dict(zip(LIF_SIGMAS, ['#7a1f5c', '#e8743b', '#1b4f8f']))
    for sigma in LIF_SIGMAS:
        for limiter, ls, mk in [('none', '-', 'o'), ('fct', '--', 's')]:
            sel = sorted([r for r in rows if r['part'] == 'C' and r['case'] == sigma and r['limiter'] == limiter],
                         key=lambda r: r['dv'])
            kw = dict(color=colors[sigma], ls=ls, marker=mk, mfc='white', ms=5,
                      label=f"$\\sigma$={sigma:g}, {'SG' if limiter == 'none' else 'FCT'}")
            axs[0].plot([r['dv'] for r in sel], [r['rel_l2'] for r in sel], **kw)
            axs[1].plot([r['time'] for r in sel], [r['rel_l2'] for r in sel], **kw)
    axs[0].set_xlabel(r'$\Delta v$ (mV)')
    axs[1].set_xlabel('computation time (s)')
    for ax in axs:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(f'rel. L2 error of the rate vs. $\\Delta v$ = {LIF_DV_REF} mV')
        style(ax)
    axs[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(f'LIF FPE, step {LIF["amp"] * 1e9:g} nA, $\\Delta t$ = {LIF["dt"]} ms: SG flux vs. FCT', fontsize=10)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT_DIR / f'partC_lif.{ext}', dpi=200)
    plt.close(fig)


def latex_table(rows):
    lines = [r'\begin{tabular}{l l l r r r r r}', r'\toprule',
             r'part & case & model & time (s) & rel.\ L2 & rel.\ L2 smooth & nrmse & count err.\ \\', r'\midrule']
    for r in rows:
        if r['part'] == 'C':
            continue
        model = r['closure'] if r['closure'].startswith('IK') else \
            f"{CLOSURE_SHORT[r['closure']]}, {'SG' if r['limiter'] == 'none' else 'FCT'}" + \
            (f", $\\Delta v$={r['dv']}" if r['part'] == 'B' else '')
        case = r['case'] if r['part'] == 'A' else f"$\\sigma$={r['case']:.3g}"
        lines.append(f"{r['part']} & {case} & {model} & {r['time']:.1f} & {r['rel_l2']:.3f} & "
                     f"{r['rel_l2_smooth']:.3f} & {r['nrmse']:.4f} & {r['count_err']:.3f} \\\\")
    lines += [r'\midrule', r'part & $\sigma$ (mV) & flux, $\Delta v$ (mV) & time (s) & rel.\ L2 vs.\ $\Delta v$=' +
              f'{LIF_DV_REF}' + r' & & & \\', r'\midrule']
    for r in rows:
        if r['part'] == 'C':
            lines.append(f"C (LIF) & {r['case']:g} & {'SG' if r['limiter'] == 'none' else 'FCT'}, {r['dv']} & "
                         f"{r['time']:.2f} & {r['rel_l2']:.3f} & & & \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    caption = (f'A: 3 populations with feedback, $\\sigma$ = {A_SIGMA:g} mV, $\\Delta v$ = {A_DV} mV, $\\Delta t$ = '
               f'{A_DT} ms, against IK 100k. B: 1 population (step), $\\Delta v$ = {B_DV} mV, $\\Delta t$ = {B_DT} ms, '
               f'against IK 100k. C: LIF FPE (step {LIF["amp"] * 1e9:g} nA) against the FPE with $\\Delta v$ = '
               f'{LIF_DV_REF} mV. SG: Scharfetter-Gummel flux, FCT: flux-corrected transport. {N_PROCESSES} '
               f'parallel processes.')
    return '\n'.join(lines), caption


def main():
    ref = load_references()
    run(('A', A_INPUTS[0], 'conditional_second_moment', 'fct'))  # compile the numba kernels
    tasks = [('A', k, c, lim) for k in A_INPUTS for c in CLOSURES for lim in LIMITERS]
    tasks += [('B', s, c, lim, B_DV) for s in B_SIGMAS for c in B_CLOSURES for lim in LIMITERS]
    tasks += [('B', s, c, 'none', B_DV_FINE) for s in B_SIGMAS for c in B_CLOSURES]
    tasks += [('C', s, dv, lim) for s in LIF_SIGMAS for dv in LIF_DVS + [LIF_DV_REF] for lim in LIMITERS
              if not (dv == LIF_DV_REF and lim == 'fct')]
    t_start = time.perf_counter()
    results = {}
    with Pool(N_PROCESSES) as pool:
        for q, res in enumerate(pool.imap_unordered(run, tasks), start=1):
            results[res['task']] = res
            print(f'[{q}/{len(tasks)}] {res["task"]}: {res["time"]:.1f} s', flush=True)
    print(f'total wall time {time.perf_counter() - t_start:.0f} s')

    rows = []
    for kind in A_INPUTS:
        ref_rb = ref[('A', kind, 100_000)]['rb']
        small = ref[('A', kind, 20_000)]
        rows.append(dict(part='A', case=kind, closure='IK 20k', limiter='', dv=None, time=small['time'],
                         n_fallback='', **imp.measures(small['rb'], ref_rb)))
        for c in CLOSURES:
            for lim in LIMITERS:
                res = results[('A', kind, c, lim)]
                rows.append(dict(part='A', case=kind, closure=c, limiter=lim, dv=A_DV, time=res['time'],
                                 n_fallback=res['n_fallback'], **imp.measures(res['rb'], ref_rb)))
    for sigma in B_SIGMAS:
        ref_rb = ref[('B', sigma, 100_000)]['rb']
        rows.append(dict(part='B', case=sigma, closure='IK 20k', limiter='', dv=None, time=np.nan, n_fallback='',
                         **imp.measures(ref[('B', sigma, 20_000)]['rb'], ref_rb)))
        for c in B_CLOSURES:
            for lim, dv in [('none', B_DV), ('fct', B_DV), ('none', B_DV_FINE)]:
                res = results[('B', sigma, c, lim, dv)]
                rows.append(dict(part='B', case=sigma, closure=c, limiter=lim, dv=dv, time=res['time'],
                                 n_fallback=res['n_fallback'], **imp.measures(res['rb'], ref_rb)))
    for sigma in LIF_SIGMAS:
        ref_r = results[('C', sigma, LIF_DV_REF, 'none')]['r'][0]
        for dv in LIF_DVS:
            for lim in LIMITERS:
                res = results[('C', sigma, dv, lim)]
                rows.append(dict(part='C', case=sigma, closure='LIF', limiter=lim, dv=dv, time=res['time'],
                                 n_fallback='', rel_l2=float(np.linalg.norm(res['r'][0] - ref_r) /
                                                             np.linalg.norm(ref_r))))

    fields = ['part', 'case', 'closure', 'limiter', 'dv', 'time', 'n_fallback', 'rel_l2', 'rel_l2_smooth', 'nrmse',
              'nrmse_smooth', 'count_err'] + [f'rel_l2_smooth_pop{j}' for j in range(3)]
    with open(OUT_DIR / 'fct_closure_table.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    np.savez(OUT_DIR / 'fct_closure_rates.npz', **{'_'.join(str(v) for v in task): res.get('rb', res['r'])
                                                  for task, res in results.items()})

    print(f"\n{'part':4s} {'case':>8s} {'model':28s} {'flux':5s} {'dv':>6s} {'time':>7s} {'L2':>6s} {'L2 sm':>6s} "
          f"{'nrmse':>7s} {'count':>6s} {'BE':>5s}   L2 sm per pop")
    for r in rows:
        case = r['case'] if isinstance(r['case'], str) else f"{r['case']:.3g}"
        extra = '   ' + ' '.join(f"{r[f'rel_l2_smooth_pop{j}']:.3f}" for j in range(3)
                                 if f'rel_l2_smooth_pop{j}' in r) if 'rel_l2_smooth' in r else ''
        print(f"{r['part']:4s} {case:>8s} {r['closure']:28s} {r['limiter']:5s} {r['dv'] or '':>6} "
              f"{r['time']:7.2f} {r['rel_l2']:6.3f} {r.get('rel_l2_smooth', np.nan):6.3f} "
              f"{r.get('nrmse', np.nan):7.4f} {r.get('count_err', np.nan):6.3f} {r['n_fallback']!s:>5s}{extra}")

    plot_part_a(rows, results, ref)
    plot_part_b(results, ref)
    plot_part_c(rows)
    imp.OUT_DIR = OUT_DIR
    imp.save_table_pdf(*latex_table(rows), 'fct_closure_table')
    print(f'results saved in {OUT_DIR}')


if __name__ == '__main__':
    main()

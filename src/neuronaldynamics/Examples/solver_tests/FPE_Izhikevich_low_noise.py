"""
Grid convergence of the Izhikevich FPE (FPE_Population with soma_model='izhikevich') at low noise: how fine does the
voltage grid have to be when voltage_sigma decreases? Follow-up of FPE_vs_IK_network.py, where the FPE on the grid
dv = 0.1 mV, dt = 0.05 ms (FPE_Izhikevich_resolution.py, sigma = 2 mV) fails for sigma <= 1 mV.

Expectation (numerical diffusion of the Scharfetter-Gummel flux): for a large cell Peclet number
Pe = |h| dv / (2 D), D = sigma^2 / tau_noise, the SG flux becomes upwind and adds the numerical diffusion |h| dv / 2,
i.e. the FPE simulates the noise sigma_eff^2 = sigma^2 + tau_noise |h| dv / 2 instead of sigma^2. With the typical drift
h ~ I0 / C of the firing population the grid is resolved for Pe <~ 1, so the necessary dv shrinks like sigma^2:
dv <~ 2 sigma^2 / (tau_noise h). If this is the reason for the error, the errors of all sigma collapse onto one curve
when plotted against Pe.

Input (set_input): step I0 = 150 pA or sine I0 + I0 sin(2 pi 100 Hz t) into exc 1 for t < 100 ms, T = 200 ms (the
input of FPE_vs_IK_network.py, FPE_Izhikevich_low_noise_sine.py runs the sine input)
Part 1 (1 population, as '1pop' of FPE_vs_IK_network.py):
    SIGMAS x DVS (dt = DT) for all closures; dt convergence (conditional closure, dv = DV_DT_STUDY, DTS)
Part 2 (3 populations with feedback, '3pop' of FPE_vs_IK_network.py):
    SIGMAS_3POP x DVS_3POP (dt = DT), conditional closure
Reference: IK network with N_LIST[-1] neurons per population (net.MC_METHOD, dt = 0.01 ms; results of Euler-Maruyama in the folders
without the suffix _heun), N_LIST[0] for the finite
size error. Errors: relative L2 error of the rates of all populations in 1 ms bins (as in FPE_vs_IK_network.py).

Outputs in Examples/solver_tests/FPE_Izhikevich_low_noise:
    low_noise_1pop.png/.pdf          per closure: error vs. IK reference, error vs. finest grid against dv, error
                                     against the cell Peclet number (collapse) and computation time
    low_noise_dt.png                 dt convergence at dv = DV_DT_STUDY
    rates_1pop.png                   rates of the conditional closure on all grids against the IK reference per sigma
    rates_3pop_sigma_<sigma>.png     rates of the 3 populations on all grids against the IK reference
    low_noise_table.csv              all numbers
    low_noise_table.tex / .pdf       error of every closure and grid per sigma, computation times

Usage (from src/neuronaldynamics):  python Examples/solver_tests/FPE_Izhikevich_low_noise.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))      # FPE_vs_IK_network (same folder)
from Model.Nykamp_Model import FPE_Population
import FPE_vs_IK_network as net
matplotlib.use('Agg')
warnings.filterwarnings('ignore', message='.*non-interactive.*')
warnings.filterwarnings('ignore', category=RuntimeWarning)  # overflow warnings of the closures on coarse grids

OUT_DIR = Path(__file__).resolve().parent / f'FPE_Izhikevich_low_noise{net.method_suffix()}'
OUT_DIR.mkdir(exist_ok=True)


def set_input(kind, out_dir=None, frequency=None):
    """input type 'step' or 'sine' (of FPE_vs_IK_network) and the result folder (default:
    FPE_Izhikevich_low_noise_<kind>, 'step' keeps FPE_Izhikevich_low_noise)"""
    global OUT_DIR
    net.set_input(kind, out_dir=net.OUT_DIR, frequency=frequency)
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / ('FPE_Izhikevich_low_noise' + ('' if kind == 'step' else f'_{kind}')
                                                     + net.method_suffix())
    OUT_DIR = Path(out_dir)
    OUT_DIR.mkdir(exist_ok=True)

########################################################################################################################
# SETTINGS
########################################################################################################################
CLOSURES = ['mean', 'conditional', 'conditional_variance']
SIGMAS = list(np.logspace(-1, 1, 5))          # mV (as in FPE_vs_IK_network.py)
DVS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]       # mV, every dv has to put c = -50 and -45 mV on the grid (v_min = -90)
DT = 0.01                                     # ms, time step of the dv study
DV_DT_STUDY, DTS = 0.02, [0.05, 0.025, 0.01, 0.005]   # dt convergence (conditional closure)
SIGMAS_3POP = [1.0, float(np.sqrt(10))]       # 3 populations: the noise levels at which the FPE failed
DVS_3POP = [0.1, 0.05, 0.02]
N_LIST = [20_000, 100_000]                    # IK neurons per population, the last one is the reference
N_PROCESSES = 4
H_TYP = net.I0 / net.RS['C']                  # typical drift (mV/ms) of the firing RS population for the Peclet number
TAU_NOISE = 10.                               # ms (IZHIKEVICH_DEFAULTS)
CLOSURE_LABELS = {'mean': r'mean $\langle u\rangle$', 'conditional': r'cond., $\sigma_u^2=0$',
                  'conditional_variance': r'cond., $\sigma_u^2(t)$'}
CLOSURE_SHORT = {'mean': 'mean', 'conditional': 'cond.', 'conditional_variance': 'cond.+var.'}


def peclet(dv, sigma):
    """cell Peclet number |h| dv / (2 D) with the typical drift H_TYP"""
    return H_TYP * dv / (2 * sigma ** 2 / TAU_NOISE)


def sigma_num(dv):
    """noise added by the upwinded SG flux (large Pe): tau_noise * |h| dv / 2 = sigma_num^2"""
    return np.sqrt(TAU_NOISE * H_TYP * np.asarray(dv) / 2)


def run(task):
    """one timed simulation: (setup, sigma, 'IK', n, None) or (setup, sigma, closure, dv, dt)"""
    setup, sigma, kind, x, dt = task
    if kind == 'IK':
        t0 = time.perf_counter()
        r, _ = net.ik_network(setup, sigma, x, seed=x)
        cpu = time.perf_counter() - t0
        return dict(task=task, time=cpu, rb=net.binned_rate(r, net.MC_DT))
    pop_types, _, W = net.SETUPS[setup]
    model = FPE_Population(dict(
        soma_model='IK', adaptation_closure=kind, T=net.T, dt=dt, dv=x, population_type=pop_types,
        connectivity_matrix=W, voltage_sigma=sigma, input_function=net.input_current, input_function_idx=[0],
        init_pdf_sigma=1.0, save_p='none', tqdm_disable=True,
        **{name: list(v) for name, v in net.neuron_parameters(setup).items()}),
        name=str(OUT_DIR / f'fpe_{setup}_{kind}_sigma_{sigma:.3g}_dv{x}_dt{dt}'))
    t0 = time.perf_counter()
    model.simulate()
    cpu = time.perf_counter() - t0
    model.clean()
    rb = net.binned_rate(model.r, dt)
    return dict(task=task, time=cpu, rb=rb, n_fallback=model.n_fallback, finite=bool(np.isfinite(rb).all()))


def rel_l2(x, ref):
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref)) if np.isfinite(x).all() else np.nan


########################################################################################################################
# PLOTS
########################################################################################################################
SIGMA_COLORS = dict(zip(SIGMAS, ['#7a1f5c', '#c2185b', '#e8743b', '#2a9d6b', '#1b4f8f']))
DV_RAMP = ['#d9c58f', '#c9a227', '#2a78d6', '#1b4f8f', '#d6612a', '#7a1f5c']


def style(ax):
    ax.grid(which='major', lw=0.4, color='#e0dfdb')
    ax.spines[['top', 'right']].set_visible(False)


def plot_1pop(rows):
    fig, axs = plt.subplots(len(CLOSURES), 4, figsize=(15, 3.0 * len(CLOSURES) + 0.9), constrained_layout=True)
    for i, closure in enumerate(CLOSURES):
        for sigma in SIGMAS:
            sel = sorted([r for r in rows if r['setup'] == '1pop' and r['model'] == closure and r['sigma'] == sigma
                          and r['dt'] == DT], key=lambda r: r['dv'])
            dvs = np.array([r['dv'] for r in sel])
            kw = dict(color=SIGMA_COLORS[sigma], marker='o', ms=4, mfc='white', lw=1.3,
                      label=f'$\\sigma$ = {sigma:.3g} mV')
            axs[i, 0].plot(dvs, [r['err_ref'] for r in sel], **kw)
            g = [(r['dv'], r['err_grid']) for r in sel if r['err_grid'] > 0]
            if g:
                axs[i, 1].plot(*zip(*g), **kw)
            axs[i, 2].plot(peclet(dvs, sigma), [r['err_ref'] for r in sel], **kw)
            axs[i, 3].plot(dvs, [r['time'] for r in sel], **kw)
            ik = next(r for r in rows if r['setup'] == '1pop' and r['model'] == 'IK' and r['sigma'] == sigma
                      and r['n'] == N_LIST[0])
            axs[i, 0].axhline(ik['err_ref'], color=SIGMA_COLORS[sigma], lw=0.8, ls='--')
        ik_time = np.mean([r['time'] for r in rows if r['setup'] == '1pop' and r['model'] == 'IK'
                           and r['n'] == N_LIST[0]])
        axs[i, 3].axhline(ik_time, color='#8a8984', lw=1, ls='--', label=f'IK n={N_LIST[0]:,} (mean)')
        axs[i, 2].axvline(1, color='#8a8984', lw=0.8, ls=':')
        axs[i, 0].set_ylabel(f'{CLOSURE_LABELS[closure]}\nrel. L2 error of the rate', fontsize=9)
        titles = [f'error vs. IK n={N_LIST[-1]:,}\n(dashed: IK n={N_LIST[0]:,})',
                  f'error vs. finest grid ($\\Delta v$={DVS[-1]} mV)', 'error vs. cell Péclet number',
                  'computation time (s)']
        xlabels = [r'$\Delta v$ (mV)', r'$\Delta v$ (mV)',
                   r'Pe $= |h|\,\Delta v\,/\,(2D)$, $h = I_0/C$', r'$\Delta v$ (mV)']
        for j, ax in enumerate(axs[i]):
            ax.set_xscale('log')
            ax.set_yscale('log')
            style(ax)
            if i == 0:
                ax.set_title(titles[j], fontsize=9)
            if i == len(CLOSURES) - 1:
                ax.set_xlabel(xlabels[j])
    handles, labels = axs[0, 0].get_legend_handles_labels()
    h2, l2 = axs[0, 3].get_legend_handles_labels()
    fig.legend(handles + h2[-1:], labels + l2[-1:], loc='outside lower center', ncol=len(SIGMAS) + 1,
               frameon=False, fontsize=8)
    fig.suptitle(f'1 population, {net.input_label()}, $\\Delta t$ = {DT} ms. Numerical noise of the SG flux: '
                 f'$\\sigma_{{num}}$ = {sigma_num(0.1):.2f} mV at $\\Delta v$ = 0.1 mV, '
                 f'{sigma_num(0.01):.2f} mV at $\\Delta v$ = 0.01 mV', fontsize=9)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT_DIR / f'low_noise_1pop.{ext}', dpi=200)
    plt.close(fig)


def plot_dt(rows):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for sigma in SIGMAS:
        sel = sorted([r for r in rows if r['setup'] == '1pop' and r['model'] == 'conditional' and r['sigma'] == sigma
                      and r['dv'] == DV_DT_STUDY and r['study'] == 'dt'], key=lambda r: r['dt'])
        kw = dict(color=SIGMA_COLORS[sigma], marker='o', ms=4, mfc='white', lw=1.3, label=f'$\\sigma$ = {sigma:.3g} mV')
        g = [(r['dt'], r['err_dt']) for r in sel if r['err_dt'] > 0]
        if g:
            axs[0].plot(*zip(*g), **kw)
        axs[1].plot([r['dt'] for r in sel], [r['err_ref'] for r in sel], **kw)
    dts = np.array(DTS[:-1])
    axs[0].plot(dts, 0.5 * (dts / dts[0]) ** 2, color='#8a8984', ls=':', lw=0.9, label=r'$\propto\Delta t^2$')
    axs[0].set_title(f'error vs. $\\Delta t$ = {DTS[-1]} ms (cond., $\\Delta v$ = {DV_DT_STUDY} mV)', fontsize=9)
    axs[1].set_title(f'error vs. IK n={N_LIST[-1]:,}', fontsize=9)
    for ax in axs:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\Delta t$ (ms)')
        style(ax)
    axs[0].set_ylabel('rel. L2 error of the rate')
    axs[0].legend(fontsize=7, frameon=False)
    fig.savefig(OUT_DIR / 'low_noise_dt.png', dpi=200)
    plt.close(fig)


def plot_rates_1pop(results):
    edges = np.arange(0, net.T + 1e-9, net.RATE_BIN_MS)
    tb = 0.5 * (edges[1:] + edges[:-1])
    fig, axs = plt.subplots(len(SIGMAS), 1, figsize=(10, 2.0 * len(SIGMAS) + 0.8), sharex=True,
                            constrained_layout=True)
    for ax, sigma in zip(axs, SIGMAS):
        ref = results[('1pop', sigma, 'IK', N_LIST[-1], None)]
        ax.plot(tb, ref['rb'][0], color='#3d3c39', lw=1.6, label=f'IK n={N_LIST[-1]:,} ({ref["time"]:.0f} s)')
        for c, dv in zip(DV_RAMP, DVS):
            res = results[('1pop', sigma, 'conditional', dv, DT)]
            ax.plot(tb, res['rb'][0], color=c, lw=0.9, label=f'FPE cond. $\\Delta v$={dv} ({res["time"]:.0f} s)')
        ax.set_ylabel('rate (Hz)')
        ax.set_title(f'$\\sigma$ = {sigma:.3g} mV, Pe($\\Delta v$=0.1) = {peclet(0.1, sigma):.2g}', fontsize=9,
                     loc='left')
        ax.axvline(net.T_ON, color='#b4b2a9', lw=0.6, ls=':')
        ax.set_xlim(0, 130)
        style(ax)
        ax.legend(fontsize=6.5, frameon=False, loc='upper right', ncol=2)
    axs[-1].set_xlabel('t (ms)')
    fig.savefig(OUT_DIR / 'rates_1pop.png', dpi=200)
    plt.close(fig)


def plot_rates_3pop(results):
    edges = np.arange(0, net.T + 1e-9, net.RATE_BIN_MS)
    tb = 0.5 * (edges[1:] + edges[:-1])
    for sigma in SIGMAS_3POP:
        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
        ref = results[('3pop', sigma, 'IK', N_LIST[-1], None)]
        small = results[('3pop', sigma, 'IK', N_LIST[0], None)]
        for j, ax in enumerate(axs):
            ax.plot(tb, ref['rb'][j], color='#3d3c39', lw=1.6, label=f'IK n={N_LIST[-1]:,} ({ref["time"]:.0f} s)')
            ax.plot(tb, small['rb'][j], color='#b4b3ad', lw=1.0, label=f'IK n={N_LIST[0]:,} ({small["time"]:.0f} s)')
            for c, dv in zip(DV_RAMP[2:], DVS_3POP):
                res = results[('3pop', sigma, 'conditional', dv, DT)]
                ax.plot(tb, res['rb'][j], color=c, lw=1.0,
                        label=f'FPE cond. $\\Delta v$={dv} ({res["time"]:.0f} s)')
            ax.set_ylabel(f'{net.POP_NAMES["3pop"][j]}\nrate (Hz)')
            ax.axvline(net.T_ON, color='#b4b2a9', lw=0.6, ls=':')
            style(ax)
        axs[0].legend(fontsize=7, frameon=False, loc='upper right', ncol=2)
        axs[-1].set_xlabel('t (ms)')
        fig.suptitle(f'3 populations with feedback, $\\sigma$ = {sigma:.3g} mV, $\\Delta t$ = {DT} ms', fontsize=10)
        fig.savefig(OUT_DIR / f'rates_3pop_sigma_{sigma:.3g}.png', dpi=200)
        plt.close(fig)


########################################################################################################################
# TABLE
########################################################################################################################
def latex_table(rows):
    nd = len(DVS)
    lines = [r'\begin{tabular}{l l r ' + 'r' * nd + ' r}', r'\toprule',
             r' & & & \multicolumn{' + str(nd) + r'}{c}{rel.\ L2 error vs.\ IK ' + f'{N_LIST[-1] // 1000}k' +
             r' for $\Delta v$ (mV) =} & \\', r'\cmidrule(lr){4-' + str(nd + 3) + '}',
             r'setup & $\sigma$ (mV) & closure & ' + ' & '.join(str(d) for d in DVS) +
             f' & IK {N_LIST[0] // 1000}k ' + r'\\']
    for setup, sigmas, dvs, closures in [('1pop', SIGMAS, DVS, CLOSURES),
                                         ('3pop', SIGMAS_3POP, DVS_3POP, ['conditional'])]:
        lines.append(r'\midrule')
        for sigma in sigmas:
            ik = next(r for r in rows if r['setup'] == setup and r['model'] == 'IK' and r['sigma'] == sigma
                      and r['n'] == N_LIST[0])
            for c in closures:
                errs = []
                for dv in DVS:
                    r = [x for x in rows if x['setup'] == setup and x['model'] == c and x['sigma'] == sigma
                         and x['dv'] == dv and x['dt'] == DT and x['study'] == 'dv']
                    errs.append('--' if not r else ('fail' if np.isnan(r[0]['err_ref']) else
                                                    f"{r[0]['err_ref']:.3f}"))
                lines.append(f'{setup} & {sigma:.3g} & {CLOSURE_SHORT[c]} & ' + ' & '.join(errs) +
                             f" & {ik['err_ref']:.3f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    times = {dv: np.mean([r['time'] for r in rows if r['setup'] == '1pop' and r['model'] == 'conditional'
                          and r['dv'] == dv and r['dt'] == DT and r['study'] == 'dv']) for dv in DVS}
    ik_t = {n: np.mean([r['time'] for r in rows if r['setup'] == '1pop' and r['model'] == 'IK' and r['n'] == n])
            for n in N_LIST}
    caption = (f'Error of the FPE against the IK network with {N_LIST[-1]:,} neurons per population for decreasing '
               f'noise and voltage grids, $\\Delta t$ = {DT} ms, input {net.input_label()} into exc 1. Mean computation '
               f'time of the conditional closure (1 pop.): ' +
               ', '.join(f'{times[dv]:.0f} s ($\\Delta v$={dv})' for dv in DVS) +
               f'; IK network: {ik_t[N_LIST[0]]:.0f} s (n={N_LIST[0]:,}), {ik_t[N_LIST[-1]]:.0f} s '
               f'(n={N_LIST[-1]:,}). {N_PROCESSES} parallel processes.')
    return '\n'.join(lines), caption


def save_table_pdf(tabular, caption, fname):
    doc = (r'\documentclass[border=8pt]{standalone}' + '\n' + r'\usepackage{booktabs}' + '\n' +
           r'\begin{document}' + '\n' + r'\begin{minipage}{19cm}\small' + '\n' + tabular + '\n\n' + r'\vspace{6pt}' +
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
    """run all simulations, print and save table and figures"""
    tasks = [('3pop', s, 'IK', n, None) for s in SIGMAS_3POP for n in N_LIST]
    tasks += [('1pop', s, 'IK', n, None) for s in SIGMAS for n in N_LIST]
    fpe_tasks = {('1pop', s, c, dv, DT) for s in SIGMAS for c in CLOSURES for dv in DVS}
    fpe_tasks |= {('1pop', s, 'conditional', DV_DT_STUDY, dt) for s in SIGMAS for dt in DTS}
    fpe_tasks |= {('3pop', s, 'conditional', dv, DT) for s in SIGMAS_3POP for dv in DVS_3POP}
    # most expensive first (load balancing): IK networks, then FPE by number of grid points x time steps
    tasks += sorted(fpe_tasks, key=lambda x: -len(net.SETUPS[x[0]][0]) / (x[3] * x[4]))
    t_start = time.perf_counter()
    results = {}
    with Pool(N_PROCESSES) as pool:
        for q, res in enumerate(pool.imap_unordered(run, tasks), start=1):
            results[res['task']] = res
            extra = f", {res['n_fallback']} BE steps" if 'n_fallback' in res else ''
            print(f'[{q}/{len(tasks)}] {res["task"]}: {res["time"]:.1f} s{extra}', flush=True)
    print(f'total wall time {time.perf_counter() - t_start:.0f} s')

    rows = []
    for (setup, sigma, kind, x, dt), res in results.items():
        ref = results[(setup, sigma, 'IK', N_LIST[-1], None)]
        row = dict(setup=setup, sigma=sigma, model=kind, time=res['time'], err_ref=np.nan, err_grid=np.nan,
                   err_dt=np.nan, n=None, dv=None, dt=net.MC_DT, study='', n_fallback=res.get('n_fallback', ''))
        if kind == 'IK':
            row.update(n=x, err_ref=rel_l2(res['rb'], ref['rb']) if x != N_LIST[-1] else np.nan)
        else:
            row.update(dv=x, dt=dt, err_ref=rel_l2(res['rb'], ref['rb']),
                       study='dv' if dt == DT else 'dt', pe=peclet(x, sigma))
            finest = results.get((setup, sigma, kind, (DVS if setup == '1pop' else DVS_3POP)[-1], DT))
            if dt == DT and finest is not None:
                row['err_grid'] = rel_l2(res['rb'], finest['rb'])
            if setup == '1pop' and kind == 'conditional' and x == DV_DT_STUDY:
                row['err_dt'] = rel_l2(res['rb'], results[(setup, sigma, kind, x, DTS[-1])]['rb'])
                if dt == DT:  # the run of the dv study is also part of the dt study
                    rows.append(dict(row, study='dt'))
        rows.append(row)
    rows.sort(key=lambda r: (r['setup'], r['sigma'], r['model'], r['dv'] or 0, r['dt'], r['n'] or 0))

    fields = ['setup', 'sigma', 'model', 'study', 'n', 'dv', 'dt', 'pe', 'time', 'err_ref', 'err_grid', 'err_dt',
              'n_fallback']
    with open(OUT_DIR / 'low_noise_table.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    np.savez(OUT_DIR / 'low_noise_rates.npz',
             **{'_'.join(str(v) for v in task): res['rb'] for task, res in results.items()})

    print(f"\n{'setup':5s} {'sigma':>6s} {'model':21s} {'n / dv':>8s} {'dt':>6s} {'Pe':>8s} {'time (s)':>8s} "
          f"{'err ref':>8s} {'err grid':>8s} {'BE steps':>8s}")
    for r in rows:
        if r['study'] == 'dt' and r['dt'] == DT:
            continue
        nx = f"{r['n']:,}" if r['model'] == 'IK' else str(r['dv'])
        pe = f"{r['pe']:8.2g}" if 'pe' in r else ' ' * 8
        print(f"{r['setup']:5s} {r['sigma']:6.3g} {r['model']:21s} {nx:>8s} {r['dt']:6} {pe} {r['time']:8.1f} "
              f"{r['err_ref']:8.4f} {r['err_grid']:8.4f} {r['n_fallback']!s:>8s}")

    plot_1pop(rows)
    plot_dt(rows)
    plot_rates_1pop(results)
    plot_rates_3pop(results)
    save_table_pdf(*latex_table(rows), 'low_noise_table')
    print(f'results saved in {OUT_DIR}')


if __name__ == '__main__':
    main()

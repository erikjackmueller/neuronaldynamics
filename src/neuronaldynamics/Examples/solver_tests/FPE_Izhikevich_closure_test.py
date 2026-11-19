"""
Test of the three adaptation closures of FPE_Izhikevich (FPE_Population with soma_model='izhikevich') against a Monte
Carlo simulation of N independent Izhikevich neurons with the same white noise (D = voltage_sigma^2 / tau_noise):

- 'mean':                 E[u | v] = <u>(t)                         (mean adaptation approximation)
- 'conditional':          q = rho E[u | v] as second 1D field, Var(u | v) = 0           (Ly & Tranchina 2007)
- 'conditional_variance': as 'conditional', Var(u | v) = sigma_u2(t) from the exact equation of <u^2>

Two inputs, both T = 200 ms, input on for the first 100 ms, then off:
- step:  I(t) = I0
- sine:  I(t) = I0 + I0 * sin(2 pi f t)

in several regimes (SCENARIOS). The number of Monte Carlo neurons N_NEURONS is the same for all runs.

Computation time: every simulation (each FPE closure and the Monte Carlo reference) is timed with time.perf_counter()
around the simulation only. The times are shown in the title of every regime in the figures and written to a LaTeX
(booktabs) table, compiled to pdf (pdflatex / latexmk / tectonic, if none is available the table is rendered with
matplotlib). Simulations run in N_PROCESSES processes (one simulation per process, numpy limited to one thread each),
N_PROCESSES = 1 gives the cleanest timings.

Measures (rate and <u> of the FPE against the Monte Carlo reference):
- rel. RMS error of the rate (both smoothed with a 1 ms box filter) relative to the mean Monte Carlo rate
- RMS error of <u> (pA)
- spikes per neuron while the input is on (0-100 ms) and off (100-200 ms)
- total variance of u over the population (conditional closures: var_u = sigma_u2 + Var(E[u | v])) against the
  Monte Carlo variance, and corr(u, v) of the Monte Carlo neurons while the input is on

Results are written to the folder FPE_Izhikevich_closures next to this script:
- closure_test_<input>.png/.pdf                rate, <u> and Var(u) of all regimes, computation times in the titles
- closure_test_table.tex / .pdf                LaTeX table of computation times and errors (both inputs)
- closure_test_metrics.csv                     all measures
- closure_test_<input>_results.npz             time courses of all runs
- <scenario>_<input>_<closure>.hdf5            FPE results (save_sim_results)
"""
import os
# one thread per simulation process (fair timings), has to be set before numpy is imported
for _var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_var, '1')
import csv
import time
import shutil
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/neuronaldynamics, for direct calls
from Model.Nykamp_Model import FPE_Izhikevich, IZHIKEVICH_DEFAULTS
SHOW = os.environ.get('SHOW_PLOTS', '1') == '1'  # SHOW_PLOTS=0: no window (e.g. in a batch run)
matplotlib.use('TkAgg' if SHOW else 'Agg')

MC_METHOD = 'heun'  # Monte Carlo integrator: 'heun' (stochastic Heun, interpolated spike times) or 'euler'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'FPE_Izhikevich_closures' + ('' if MC_METHOD == 'euler' else '_heun'))

########################################################################################################################
# SETTINGS
########################################################################################################################
RUN_STEP = True
RUN_SINE = True
CLOSURES = ['mean', 'conditional', 'conditional_variance']
N_NEURONS = 20000           # Monte Carlo neurons, the same for all regimes and inputs
N_PROCESSES = 4             # parallel simulations (1: cleanest timings)

dt = 0.01
dv = 0.02
T = 200.                    # ms
T_ON = 100.                 # input on for t < T_ON, then off
SINE_FREQUENCY = 0.1        # 1/ms (100 Hz)

# file name: (title, parameters differing from IZHIKEVICH_DEFAULTS (RS neuron), I0 in pA)
SCENARIOS = {
    'rs_default':      ('RS default (a=0.03, d=100)', dict(), 100.),
    'rs_strong_noise': ('RS, strong noise (sigma=6)', dict(voltage_sigma=6.), 100.),
    'rs_high_rate':    ('RS, high rate (I0=300)', dict(), 300.),
    'slow':            ('slow adaptation (a=0.005, d=17)', dict(a=0.005, d=17.), 100.),
    'fast':            ('fast adaptation (a=0.1, d=100)', dict(a=0.1), 200.),
}

CLOSURE_LABELS = {'mean': r'mean $\langle u\rangle$',
                  'conditional': r'conditional, $\sigma_u^2=0$',
                  'conditional_variance': r'conditional, $\sigma_u^2(t)$'}
CLOSURE_SHORT = {'mean': 'mean', 'conditional': 'cond.', 'conditional_variance': 'cond.+var.'}


def time_axis():
    return np.arange(0, T, dt)


def input_current(experiment, key):
    """
    input current (pA): I0 (step) or I0 + I0 sin(2 pi f t) (sine) for t < T_ON, 0 afterwards
    """
    t = time_axis()
    i0 = SCENARIOS[key][2]
    signal = np.full_like(t, i0) if experiment == 'step' else i0 + i0 * np.sin(2 * np.pi * SINE_FREQUENCY * t)
    return np.where(t < T_ON, signal, 0.)


def monte_carlo(par, t, I, stats_window, N=N_NEURONS, seed=0):
    """
    simulation of N Izhikevich neurons (integrator MC_METHOD, see mc_step)
    :param stats_window: (t_start, t_end) in ms in which corr(u, v) over the neurons is averaged
    :return: rate (1/ms), <u> (pA), Var(u) (pA^2, every 10th time step, linearly interpolated), corr(u, v)
    """
    rng = np.random.default_rng(seed)
    D = par['voltage_sigma'] ** 2 / par['tau_noise']
    v = rng.normal(par['v_r'], 1.0, N)
    u = par['b'] * (v.mean() - par['v_r']) * np.ones(N)
    r = np.zeros(len(t))
    u_mean = np.zeros(len(t))
    var_idx, var_u = [0], [0.]
    corr = []
    for i in range(1, len(t)):
        v, u, spikes = mc_step(par, v, u, I[i], dt, np.sqrt(2 * D * dt) * rng.standard_normal(N))
        r[i] = spikes.sum() / (N * dt)
        u_mean[i] = u.mean()
        if i % 10 == 0:
            var_idx.append(i)
            var_u.append(u.var())
        if stats_window[0] < t[i] <= stats_window[1] and i % 100 == 0:
            corr.append(np.corrcoef(u, v)[0, 1])
    var_u = np.interp(np.arange(len(t)), var_idx, var_u)
    return r, u_mean, var_u, float(np.mean(corr)) if corr else np.nan


def mc_step(par, v, u, i_in, dt, xi):
    """
    one step of N Izhikevich neurons with the noise increments xi, MC_METHOD 'heun' (stochastic Heun with
    interpolated spike times, as FPE_vs_IK_network.ik_network) or 'euler' (Euler-Maruyama, spikes on the time grid)
    :return: v, u, spikes (bool per neuron)
    """
    f_v = lambda v, u: (par['k'] * (v - par['v_r']) * (v - par['v_t']) - u + i_in) / par['C']
    f_u = lambda v, u: par['a'] * (par['b'] * (v - par['v_r']) - u)
    dvdt, dudt = f_v(v, u), f_u(v, u)
    if MC_METHOD == 'euler':
        v, u = v + dt * dvdt + xi, u + dt * dudt
        spikes = v >= par['v_peak']
        v[spikes] = par['c']
        u[spikes] += par['d']
        return v, u, spikes
    v_p = np.minimum(v + dt * dvdt + xi, 10 * par['v_peak'])
    u_p = u + dt * dudt
    v_new = v + 0.5 * dt * (dvdt + f_v(v_p, u_p)) + xi
    u_new = u + 0.5 * dt * (dudt + f_u(v_p, u_p))
    spikes = v_new >= par['v_peak']
    # crossing time t_n + theta dt (linear interpolation), reset there and drift over the rest of the step
    theta = np.clip((par['v_peak'] - v) / np.maximum(v_new - v, 1e-12), 0., 1.)
    u_s = u + theta * (u_new - u) + par['d']
    rest = (1. - theta) * dt
    v_s = par['c'] + rest * f_v(par['c'], u_s)
    u_s = u_s + rest * f_u(par['c'], u_s)
    return np.where(spikes, v_s, v_new), np.where(spikes, u_s, u_new), spikes


def smooth(x, ms):
    """
    box filter of width ms, normalised by the number of samples in the window (no drop at the edges)
    """
    kernel = np.ones(max(int(ms / dt), 1))
    return np.convolve(x, kernel, 'same') / np.convolve(np.ones_like(x), kernel, 'same')


def run(task):
    """
    one timed simulation: task = (experiment, scenario, closure), closure 'mc' is the Monte Carlo reference
    """
    experiment, key, closure = task
    t = time_axis()
    I = input_current(experiment, key)
    scenario_parameters = SCENARIOS[key][1]
    if closure == 'mc':
        par = dict(IZHIKEVICH_DEFAULTS)
        par.update(scenario_parameters)
        t0 = time.perf_counter()
        r, u, var_u, corr = monte_carlo(par, t, I, stats_window=(T_ON / 2, T_ON))
        cpu = time.perf_counter() - t0
        return dict(task=task, time=cpu, r=r, u=u, var_u=var_u, corr=corr)
    fpe = FPE_Izhikevich(dict(scenario_parameters, T=T, dt=dt, dv=dv, input_function=I, save_p='none',
                              init_pdf_sigma=1.0, tqdm_disable=True, adaptation_closure=closure),
                         name=os.path.join(OUT_DIR, f'{key}_{experiment}_{closure}'))
    t0 = time.perf_counter()
    fpe.simulate()
    cpu = time.perf_counter() - t0
    var_u = fpe.var_u[0] if fpe.var_u is not None else np.full(len(t), np.nan)
    return dict(task=task, time=cpu, r=fpe.r[0], u=fpe.u_mean[0], var_u=var_u)


def metrics(res, ref):
    """
    errors of an FPE run against the Monte Carlo reference and spike counts
    """
    t = time_axis()
    inner = (t > 1) & (t < T - 1)  # without the edges of the box filter
    r_s, r_ref = smooth(res['r'], 1)[inner], smooth(ref['r'], 1)[inner]
    on = t < T_ON
    return dict(rel_rms_rate=np.sqrt(np.mean((r_s - r_ref) ** 2)) / r_ref.mean(),
                rms_u=np.sqrt(np.mean((res['u'] - ref['u']) ** 2)),
                spikes_on=res['r'][on].sum() * dt, spikes_off=res['r'][~on].sum() * dt,
                var_u_end_on=res['var_u'][np.searchsorted(t, T_ON) - 1])


########################################################################################################################
# PLOT (style as in FPE_vs_LIF_timing.py: reference dark grey, light grid, no top / right spines)
########################################################################################################################
MC_STYLE = dict(color='#3d3c39', lw=0.8, alpha=0.85)
CLOSURE_STYLES = {'mean': dict(color='#2a78d6', lw=1.5),
                  'conditional': dict(color='#d6612a', lw=1.5),
                  'conditional_variance': dict(color='#2a9d6b', lw=1.3, ls='--')}
INPUT_STYLE = dict(color='#8a8984', lw=1.0)
GRID_COLOR = '#e0dfdb'
TEXT_COLOR = '#5f5e5a'


def style_axis(ax):
    ax.grid(which='major', lw=0.4, color=GRID_COLOR)
    ax.spines[['top', 'right']].set_visible(False)
    ax.axvline(T_ON, color='#b4b2a9', lw=0.6, ls=':')


def plot_experiment(experiment, results, times, corrs):
    """
    one sub figure per regime (rate, <u>, Var(u)), title with the computation times, input of all regimes on top
    """
    keys = list(SCENARIOS)
    fig = plt.figure(figsize=(11, 2.2 * len(keys) + 1.6), layout='constrained')
    subfigs = fig.subfigures(len(keys) + 1, 1, height_ratios=[0.75] + [1] * len(keys), hspace=0.02)
    t = time_axis()

    ax_in = subfigs[0].subplots(1, 1)
    for key in keys:
        ax_in.plot(t, input_current(experiment, key), label=key, lw=1.0)
    ax_in.set_ylabel('I (pA)')
    ax_in.set_title(f'input ({experiment}, on for t < {T_ON:.0f} ms)', fontsize=9)
    ax_in.legend(fontsize=7, frameon=False, ncol=len(keys), loc='upper right')
    style_axis(ax_in)

    for row, key in enumerate(keys, start=1):
        ref = results[(experiment, key, 'mc')]
        time_str = '   '.join([f'MC {times[(experiment, key, "mc")]:.1f} s'] +
                               [f'{CLOSURE_SHORT[c]} {times[(experiment, key, c)]:.1f} s' for c in CLOSURES])
        subfigs[row].suptitle(f'{SCENARIOS[key][0]}  |  computation time: {time_str}', fontsize=9, x=0.01,
                              ha='left')
        axs = subfigs[row].subplots(1, 3, sharex=True)
        axs[0].plot(t, smooth(ref['r'], 1) * 1e3, **MC_STYLE, label=f'Monte Carlo n={N_NEURONS:,}')
        axs[1].plot(t, ref['u'], **MC_STYLE)
        axs[2].plot(t, ref['var_u'], **MC_STYLE)
        for c in CLOSURES:
            res = results[(experiment, key, c)]
            axs[0].plot(t, res['r'] * 1e3, **CLOSURE_STYLES[c], label=f'FPE {CLOSURE_LABELS[c]}')
            axs[1].plot(t, res['u'], **CLOSURE_STYLES[c])
            if c != 'mean':
                axs[2].plot(t, res['var_u'], **CLOSURE_STYLES[c])
        axs[0].set_ylabel('rate (Hz)')
        axs[1].set_ylabel(r'$\langle u \rangle$ (pA)')
        axs[2].set_ylabel(r'Var$(u)$ (pA$^2$)')
        axs[2].text(0.99, 0.95, f'corr(u,v) MC = {corrs[(experiment, key)]:.2f}', transform=axs[2].transAxes,
                    ha='right', va='top', fontsize=7, color=TEXT_COLOR)
        for ax in axs:
            style_axis(ax)
            if row == len(keys):
                ax.set_xlabel('t (ms)')
        if row == 1:
            axs[0].set_title('population rate', fontsize=9)
            axs[1].set_title(r'mean adaptation $\langle u \rangle$', fontsize=9)
            axs[2].set_title('variance of u over the population', fontsize=9)
            handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=len(handles), frameon=False, fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'closure_test_{experiment}.{ext}'), dpi=300)


########################################################################################################################
# TABLE (LaTeX, booktabs)
########################################################################################################################
def latex_table(rows, experiments):
    """
    LaTeX table of computation times and errors, one block per input
    """
    nc = len(CLOSURES)
    head = ' & '.join(CLOSURE_SHORT[c] for c in CLOSURES)
    lines = [r'\begin{tabular}{l r ' + 'r' * nc + ' ' + 'r' * nc + ' ' + 'r' * nc + '}', r'\toprule',
             r' & \multicolumn{' + str(nc + 1) + r'}{c}{computation time (s)} & \multicolumn{' + str(nc) +
             r'}{c}{rel.\ RMS error rate} & \multicolumn{' + str(nc) + r'}{c}{RMS error $\langle u\rangle$ (pA)} \\',
             r'\cmidrule(lr){2-' + str(nc + 2) + r'} \cmidrule(lr){' + str(nc + 3) + '-' + str(2 * nc + 2) +
             r'} \cmidrule(lr){' + str(2 * nc + 3) + '-' + str(3 * nc + 2) + '}',
             f'regime & MC & {head} & {head} & {head} \\\\']
    for experiment in experiments:
        lines += [r'\midrule', r'\multicolumn{' + str(3 * nc + 2) + r'}{l}{\textit{' + experiment + r' input}} \\']
        for key in SCENARIOS:
            sel = {row['closure']: row for row in rows if row['experiment'] == experiment and row['scenario'] == key}
            err_r = [sel[c]['rel_rms_rate'] for c in CLOSURES]
            err_u = [sel[c]['rms_u'] for c in CLOSURES]

            def fmt(values, f):
                best = int(np.nanargmin(values))
                return ' & '.join((r'\textbf{' + format(x, f) + '}') if k == best else format(x, f)
                                  for k, x in enumerate(values))
            name = SCENARIOS[key][0].replace('sigma', r'$\sigma$').replace('I0', r'$I_0$')
            lines.append(f"{name} & {sel['mc']['time']:.1f} & " +
                         ' & '.join(f"{sel[c]['time']:.1f}" for c in CLOSURES) +
                         f' & {fmt(err_r, ".3f")} & {fmt(err_u, ".2f")} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def save_table_pdf(tabular, fname):
    """
    write tabular as standalone .tex and compile it to pdf, without a LaTeX engine the table is drawn with matplotlib
    """
    caption = (f'Computation time and error of the FPE closures against the Monte Carlo reference (MC, '
               f'{N_NEURONS:,} neurons), T = {T:.0f} ms, input on for {T_ON:.0f} ms, dt = {dt} ms, dv = {dv} mV, '
               f'{N_PROCESSES} parallel process(es). Bold: smallest error of a regime.')
    doc = (r'\documentclass[border=8pt]{standalone}' + '\n' + r'\usepackage{booktabs}' + '\n' +
           r'\begin{document}' + '\n' + r'\begin{minipage}{24cm}\small' + '\n' + tabular + '\n\n' + r'\vspace{6pt}' +
           '\n' + r'\footnotesize ' + caption.replace('dt', r'$\Delta t$').replace('dv', r'$\Delta v$') + '\n' +
           r'\end{minipage}' + '\n' + r'\end{document}' + '\n')
    tex_path = os.path.join(OUT_DIR, f'{fname}.tex')
    with open(tex_path, 'w') as f:
        f.write(doc)
    for engine in [['latexmk', '-pdf', '-quiet'], ['pdflatex', '-interaction=nonstopmode'], ['tectonic']]:
        if shutil.which(engine[0]):
            proc = subprocess.run(engine + [os.path.basename(tex_path)], cwd=OUT_DIR, capture_output=True)
            if proc.returncode == 0:
                print(f'compiled {fname}.pdf with {engine[0]}')
                return
    print('no LaTeX engine found, table rendered with matplotlib')
    render_table_matplotlib(tabular, os.path.join(OUT_DIR, f'{fname}.pdf'), caption)


def render_table_matplotlib(tabular, path, caption):
    """
    booktabs like rendering of the rows of tabular (fallback without LaTeX)
    """
    rows = [line for line in tabular.split('\n') if line.endswith(r'\\')]
    cells = []
    for line in rows:
        clean = line.rstrip('\\').replace(r'\textbf{', '').replace(r'\textit{', '').replace('}', '')
        clean = clean.replace(r'\multicolumn{', '').replace(r'\langle u\rangle', '<u>').replace('$', '')
        clean = clean.replace(r'rel.\ ', 'rel. ').replace(r'\sigma', 'sigma')
        cells.append([c.strip() for c in clean.split('&')])
    fig, ax = plt.subplots(figsize=(13, 0.32 * len(cells) + 1))
    ax.axis('off')
    n_col = max(len(c) for c in cells)
    y = 1.0
    dy = 1 / (len(cells) + 3)
    ax.axhline(y, color='k', lw=1.2)
    for k, row in enumerate(cells):
        y -= dy
        for col, text in enumerate(row):
            ax.text(0.0 if col == 0 else 0.22 + 0.78 * (col - 0.5) / (n_col - 1), y, text, ha='left' if col == 0
                    else 'center', va='center', fontsize=8)
        if k in [1, 2]:
            ax.axhline(y - dy / 2, color='k', lw=0.5)
    ax.axhline(y - dy / 2, color='k', lw=1.2)
    ax.text(0, y - 1.5 * dy, caption, fontsize=7, va='top', wrap=True)
    fig.savefig(path)
    plt.close(fig)


if __name__ == '__main__':
    experiments = [e for e, on in [('step', RUN_STEP), ('sine', RUN_SINE)] if on]
    if not experiments:
        raise ValueError('nothing to do, set RUN_STEP and/or RUN_SINE to True')
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = [(e, key, c) for e in experiments for key in SCENARIOS for c in ['mc'] + CLOSURES]
    with Pool(N_PROCESSES) as pool:
        results = {res['task']: res for res in pool.imap_unordered(run, tasks)}
    times = {task: res['time'] for task, res in results.items()}
    corrs = {(e, key): results[(e, key, 'mc')]['corr'] for e in experiments for key in SCENARIOS}

    rows = []
    for (e, key, c), res in results.items():
        row = dict(experiment=e, scenario=key, closure=c, time=res['time'], corr_uv_mc=corrs[(e, key)])
        if c != 'mc':
            row.update(metrics(res, results[(e, key, 'mc')]))
        else:
            t = time_axis()
            row.update(spikes_on=res['r'][t < T_ON].sum() * dt, spikes_off=res['r'][t >= T_ON].sum() * dt,
                       var_u_end_on=res['var_u'][np.searchsorted(t, T_ON) - 1])
        rows.append(row)
    order = {k: i for i, k in enumerate(['mc'] + CLOSURES)}
    rows.sort(key=lambda r: (experiments.index(r['experiment']), list(SCENARIOS).index(r['scenario']),
                             order[r['closure']]))
    fields = ['experiment', 'scenario', 'closure', 'time', 'rel_rms_rate', 'rms_u', 'spikes_on', 'spikes_off',
              'var_u_end_on', 'corr_uv_mc']
    with open(os.path.join(OUT_DIR, 'closure_test_metrics.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, '') for k in fields} for row in rows])

    print(f"{'input':5s} {'scenario':16s} {'closure':21s} {'time (s)':>8s} {'rel.RMS r':>9s} {'RMS <u>':>8s} "
          f"{'spk on':>7s} {'spk off':>7s} {'Var(u)':>8s}")
    for row in rows:
        print(f"{row['experiment']:5s} {row['scenario']:16s} {row['closure']:21s} {row['time']:8.1f} "
              f"{row.get('rel_rms_rate', np.nan):9.3f} {row.get('rms_u', np.nan):8.2f} {row['spikes_on']:7.3f} "
              f"{row['spikes_off']:7.3f} {row['var_u_end_on']:8.0f}")

    for e in experiments:
        np.savez(os.path.join(OUT_DIR, f'closure_test_{e}_results.npz'), t=time_axis(),
                 **{f'{key}_input': input_current(e, key) for key in SCENARIOS},
                 **{f'{key}_{c}_{name}': results[(e, key, c)][name] for key in SCENARIOS
                    for c in ['mc'] + CLOSURES for name in ['r', 'u', 'var_u']})
        plot_experiment(e, results, times, corrs)
    save_table_pdf(latex_table(rows, experiments), 'closure_test_table')
    print(f'saved results to {OUT_DIR}')
    if SHOW:
        plt.show()

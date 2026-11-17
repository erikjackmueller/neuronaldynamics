"""
Test of the mean adaptation approximation of FPE_Izhikevich against a Monte Carlo simulation of N independent
Izhikevich neurons with the same white noise (D = voltage_sigma^2 / tau_noise), for two inputs:
- step:  current step at t_on, 800 ms: spike frequency adaptation (onset peak, decay, adapted rate)
- sine:  I(t) = offset + amplitude * sin(2 pi f t) for the first 30 ms, then 0, 50 ms: fast time dependent input and
         the decay of the activity after the input stops

The approximation replaces u of every neuron by the population mean <u>. It should hold when the spread of u over the
population is small compared to its mean. For Poisson-like spiking CV(u) ~ sqrt(a / (2 r)), i.e. when
- the adaptation is slow compared to the firing rate (a << r, many spikes per adaptation time constant)
- the noise is strong (no synchronisation of the spikes and with that of u over the population)
The default RS neuron at low rates (a = 0.03/ms, r ~ 0.015/ms, CV(u) ~ 1) is the reference case where it should fail.

The equation for <u> itself is exact (it is linear in v and u), the approximation is only in the drift of p(v), which
uses <u> instead of the conditional mean E[u | v]. corr(u, v) over the Monte Carlo neurons measures how much u depends
on v and is therefore the more direct diagnostic than CV(u) (e.g. strong noise decorrelates u and v despite CV(u) ~ 1).

Measures of the step test (FPE and Monte Carlo): onset peak rate, adapted rate (last 200 ms), adaptation index
1 - adapted / peak, adaptation time constant tau_u (time after the step until <u> has reached (1 - 1/e) of its change,
more robust than a time constant of the noisy Monte Carlo rate), relative RMS error of the rate after the step (both
smoothed with a 5 ms box filter), and CV(u) and corr(u, v) of the Monte Carlo neurons in the adapted state.
Measures of the sine test: peak rate, mean rate during (0-30 ms) and after (30-50 ms) the sine, spikes per neuron,
<u> at the end, relative RMS error of the rate (1 ms box filter) and corr(u, v) of the Monte Carlo neurons during the
sine (10-30 ms).

Results are written to the folder FPE_Izhikevich next to this script:
- adaptation_test_metrics.csv / sine_test_metrics.csv   tables of the measures
- adaptation_test_results.npz / sine_test_results.npz   time courses of r and <u> of FPE and Monte Carlo (and input)
- adaptation_test.png/.pdf / sine_test.png/.pdf         rates and <u> of all scenarios
- <scenario>_step.hdf5 / <scenario>_sine.hdf5           FPE_Izhikevich results (save_sim_results)

RUN_STEP / RUN_SINE switch the two tests on and off. All simulations run in parallel (one process each), runtime
~5-10 min with the step test, ~1 min for the sine test only.
"""
import os
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/neuronaldynamics, for direct calls
from Model.Nykamp_Model import FPE_Izhikevich
matplotlib.use('TkAgg')

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FPE_Izhikevich_2')

# which tests to run
RUN_STEP = False  # 800 ms step test (spike frequency adaptation), ~5-10 min
RUN_SINE = True  # 50 ms sine test, ~1 min

dt = 0.01
dv = 0.02
n_neurons = 20000

# step test
T_STEP, t_on = 800, 50
# sine test: offset + amplitude * sin(2 pi f t) for t < sine_duration, frequency in 1/ms (0.1/ms = 100 Hz)
T_SINE = 200
SINE = dict(duration=100., offset=200., amplitude=200., frequency=0.1)

# file name: (title, parameters differing from FPE_Izhikevich.izhikevich_defaults (RS neuron), step current in pA)
SCENARIOS = {
    'rs_default':      ('RS default (a=0.03, d=100)', dict(), 100.),
    'rs_strong_noise': ('RS, strong noise (sigma=6)', dict(voltage_sigma=6.), 100.),
    'rs_high_rate':    ('RS, high rate (I=300)', dict(), 300.),
    'slow':            ('slow (a=0.005, d=17)', dict(a=0.005, d=17.), 100.),
    'very_slow':       ('very slow (a=0.002, d=6.7)', dict(a=0.002, d=6.7), 100.),
}
# rs_high_rate differs from rs_default only by the step current, the sine input is the same for all scenarios
SINE_SCENARIOS = [key for key in SCENARIOS if key != 'rs_high_rate']


def time_axis(experiment):
    return np.arange(0, T_STEP if experiment == 'step' else T_SINE, dt)


def input_current(experiment, key):
    """
    input current (pA) of an experiment ('step' or 'sine') and scenario
    """
    t = time_axis(experiment)
    if experiment == 'step':
        return np.where(t > t_on, SCENARIOS[key][2], 0.)
    return np.where(t < SINE['duration'], SINE['offset'] + SINE['amplitude'] * np.sin(2 * np.pi * SINE['frequency'] * t),
                    0.)


def monte_carlo(par, t, I, stats_window, N=n_neurons, seed=0):
    """
    Euler-Maruyama simulation of N Izhikevich neurons
    :param stats_window: (t_start, t_end) in ms in which CV(u) and corr(u, v) over the neurons are averaged
    :return: rate (1/ms), <u> (pA), CV(u), corr(u, v)
    """
    rng = np.random.default_rng(seed)
    D = par['voltage_sigma'] ** 2 / par['tau_noise']
    v = rng.normal(par['v_r'], 1.0, N)
    u = par['b'] * (v.mean() - par['v_r']) * np.ones(N)
    r = np.zeros(len(t))
    u_mean = np.zeros(len(t))
    cv = []
    corr = []
    for i in range(1, len(t)):
        dvdt = (par['k'] * (v - par['v_r']) * (v - par['v_t']) - u + I[i]) / par['C']
        dudt = par['a'] * (par['b'] * (v - par['v_r']) - u)
        v = v + dt * dvdt + np.sqrt(2 * D * dt) * rng.standard_normal(N)
        u = u + dt * dudt
        spikes = v >= par['v_peak']
        v[spikes] = par['c']
        u[spikes] += par['d']
        r[i] = spikes.sum() / (N * dt)
        u_mean[i] = u.mean()
        if stats_window[0] < t[i] <= stats_window[1] and i % 100 == 0:
            cv.append(u.std() / abs(u.mean()))
            corr.append(np.corrcoef(u, v)[0, 1])
    return r, u_mean, float(np.mean(cv)), float(np.mean(corr))


def smooth(x, ms):
    """
    box filter of width ms, normalised by the number of samples in the window (no drop at the edges)
    """
    kernel = np.ones(int(ms / dt))
    return np.convolve(x, kernel, 'same') / np.convolve(np.ones_like(x), kernel, 'same')


def rel_rms(r_fpe, r_mc, mask, ms):
    """
    RMS difference of the smoothed rates relative to the mean Monte Carlo rate within mask
    """
    r_fpe_s, r_mc_s = smooth(r_fpe, ms)[mask], smooth(r_mc, ms)[mask]
    return np.sqrt(np.mean((r_fpe_s - r_mc_s) ** 2)) / r_mc_s.mean()


def adaptation_metrics(t, r):
    """
    peak rate, adapted rate (Hz) and adaptation index of a rate after the step
    """
    rs = smooth(r, 5) * 1e3
    after_step = (t > t_on) & (t < T_STEP - 5)  # the edge of the box filter is excluded
    i_peak = np.argmax(np.where(after_step, rs, -1))
    peak = rs[i_peak]
    adapted = rs[t > T_STEP - 200].mean()
    return dict(peak=peak, adapted=adapted, index=1 - adapted / peak)


def adaptation_time_constant(t, u):
    """
    time (ms) after the step until <u> has reached (1 - 1/e) of its change from the step to the adapted state
    """
    u_step = u[np.searchsorted(t, t_on)]
    u_adapted = u[t > T_STEP - 100].mean()
    level = u_step + (1 - 1 / np.e) * (u_adapted - u_step)
    crossed = (u >= level) if u_adapted >= u_step else (u <= level)
    idx = np.where((t > t_on) & crossed)[0]
    return t[idx[0]] - t_on if idx.size else np.nan


def sine_metrics(t, r, u):
    """
    peak rate, mean rate during and after the sine (Hz), spikes per neuron and <u> at the end (pA)
    """
    rs = smooth(r, 1) * 1e3
    during = t < SINE['duration']
    return dict(peak=rs[t < T_SINE - 1].max(), rate_during=r[during].mean() * 1e3, rate_after=r[~during].mean() * 1e3,
                spikes_per_neuron=r.sum() * dt, u_end=u[-1])


def run(task):
    experiment, key = task
    t = time_axis(experiment)
    I = input_current(experiment, key)
    scenario_parameters = SCENARIOS[key][1]
    par = dict(FPE_Izhikevich.izhikevich_defaults)
    par.update(scenario_parameters)

    fpe = FPE_Izhikevich(dict(scenario_parameters, T=T_STEP if experiment == 'step' else T_SINE, dt=dt, dv=dv, input_function=I, save_p='none',
                              init_pdf_sigma=1.0, tqdm_disable=True), name=os.path.join(OUT_DIR, f'{key}_{experiment}'))
    fpe.simulate()
    stats_window = (T_STEP - 200, T_STEP) if experiment == 'step' else (10, SINE['duration'])
    r_mc, u_mc, cv, corr = monte_carlo(par, t, I, stats_window)
    return dict(experiment=experiment, key=key, t=t, I=I, r_fpe=fpe.r[0], u_fpe=fpe.u_mean[0], r_mc=r_mc, u_mc=u_mc,
                cv_u_mc=cv, corr_uv_mc=corr)


def save_results(results, prefix):
    np.savez(os.path.join(OUT_DIR, f'{prefix}_results.npz'), t=results[0]['t'],
             **{f"{res['key']}_{name}": res[name] for res in results
                for name in ['I', 'r_fpe', 'u_fpe', 'r_mc', 'u_mc', 'cv_u_mc', 'corr_uv_mc']})


def save_table(rows, prefix):
    with open(os.path.join(OUT_DIR, f'{prefix}_metrics.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# plot style as in FPE_vs_LIF_timing.py: FPE blue, reference (Monte Carlo) dark grey, light grid, no top / right spines
FPE_STYLE = dict(color='#2a78d6', lw=1.6, label='FPE_Izhikevich')
MC_STYLE = dict(color='#3d3c39', lw=0.8, alpha=0.85, label=f'Monte Carlo n={n_neurons:,} (reference)')
INPUT_STYLE = dict(color='#8a8984', lw=1.0, label='input current')
GRID_COLOR = '#e0dfdb'
TEXT_COLOR = '#5f5e5a'


def style_axis(ax):
    ax.grid(which='major', lw=0.4, color=GRID_COLOR)
    ax.spines[['top', 'right']].set_visible(False)


def plot_results(results, prefix, rate_smoothing_ms, plot_input=False):
    """
    one row per scenario: population rate (left) and <u> (right) of FPE and Monte Carlo, optionally with the (common)
    input current in an extra row on top, saved as <prefix>.png / .pdf in OUT_DIR
    :param rate_smoothing_ms: width of the box filter of the Monte Carlo rate
    :param plot_input: plot the input current of the first scenario on top (for inputs shared by all scenarios)
    """
    n = len(results)
    n_in = int(plot_input)
    fig = plt.figure(figsize=(9, 1.7 * n + 0.9 * n_in + 0.7), constrained_layout=True)
    gs = fig.add_gridspec(n + n_in, 2, height_ratios=[0.5] * n_in + [1] * n)
    ax_ref = None
    axs = np.empty((n + n_in, 2), dtype=object)
    for row in range(n + n_in):
        for col in range(2):
            axs[row, col] = fig.add_subplot(gs[row, col], sharex=ax_ref)
            ax_ref = ax_ref or axs[row, col]
            style_axis(axs[row, col])
            if row < n + n_in - 1:
                axs[row, col].tick_params(labelbottom=False)

    if plot_input:
        for col in range(2):
            axs[0, col].plot(results[0]['t'], results[0]['I'], **INPUT_STYLE)
        axs[0, 0].set_ylabel('input\nI (pA)')

    for row, res in enumerate(results, start=n_in):
        name, _, parameters = SCENARIOS[res['key']][0].partition(' (')
        t = res['t']
        ax_r, ax_u = axs[row]
        ax_r.plot(t, smooth(res['r_mc'], rate_smoothing_ms) * 1e3, **MC_STYLE)
        ax_r.plot(t, res['r_fpe'] * 1e3, **FPE_STYLE)
        ax_r.set_ylabel(f'{name}\nrate (Hz)')
        ax_r.text(0.99, 0.95, f"{parameters.rstrip(')')}\nCV(u)={res['cv_u_mc']:.2f}, corr(u,v)={res['corr_uv_mc']:.2f}",
                  transform=ax_r.transAxes, ha='right', va='top', fontsize=7, color=TEXT_COLOR,
                  bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
        ax_u.plot(t, res['u_mc'], **MC_STYLE)
        ax_u.plot(t, res['u_fpe'], **FPE_STYLE)
        ax_u.set_ylabel(r'$\langle u \rangle$ (pA)')

    for ax in axs[-1]:
        ax.set_xlabel('t (ms)')
    axs[0, 0].set_title('population rate')
    axs[0, 1].set_title(r'mean adaptation current $\langle u \rangle$')
    handles, labels = axs[n_in, 0].get_legend_handles_labels()
    if plot_input:
        h_in, l_in = axs[0, 0].get_legend_handles_labels()
        handles, labels = handles + h_in, labels + l_in
    fig.legend(handles, labels, loc='outside upper center', ncol=len(handles), frameon=False)
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'{prefix}.{ext}'), dpi=300)


if __name__ == '__main__':
    if not (RUN_STEP or RUN_SINE):
        raise ValueError('nothing to do, set RUN_STEP and/or RUN_SINE to True')
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = [('step', key) for key in SCENARIOS if RUN_STEP] + [('sine', key) for key in SINE_SCENARIOS if RUN_SINE]
    with Pool(min(len(tasks), os.cpu_count())) as pool:
        results = pool.map(run, tasks)
    step_results = [res for res in results if res['experiment'] == 'step']
    sine_results = [res for res in results if res['experiment'] == 'sine']

    ####################################################################################################################
    # STEP TEST
    ####################################################################################################################
    if RUN_STEP:
        rows = []
        for res in step_results:
            t = res['t']
            m_fpe, m_mc = adaptation_metrics(t, res['r_fpe']), adaptation_metrics(t, res['r_mc'])
            m_fpe['tau_u'] = adaptation_time_constant(t, res['u_fpe'])
            m_mc['tau_u'] = adaptation_time_constant(t, res['u_mc'])
            rows.append(dict(scenario=res['key'], cv_u_mc=res['cv_u_mc'], corr_uv_mc=res['corr_uv_mc'],
                             rel_rms_rate=rel_rms(res['r_fpe'], res['r_mc'], t > t_on, 5),
                             **{f'{m}_fpe': m_fpe[m] for m in m_fpe}, **{f'{m}_mc': m_mc[m] for m in m_mc}))
        save_results(step_results, 'adaptation_test')
        save_table(rows, 'adaptation_test')

        print('STEP TEST')
        print(f"{'scenario':16s} {'CV(u)':>6s} {'corr':>6s} | {'peak (Hz)':>15s} | {'adapted (Hz)':>15s} | "
              f"{'tau_u (ms)':>13s} | {'rel. RMS':>8s}")
        print(f"{'':16s} {'MC':>6s} {'(u,v)':>6s} | {'FPE':>7s} {'MC':>7s} | {'FPE':>7s} {'MC':>7s} | "
              f"{'FPE':>6s} {'MC':>6s} |")
        for row in rows:
            print(f"{row['scenario']:16s} {row['cv_u_mc']:6.2f} {row['corr_uv_mc']:6.2f} | "
                  f"{row['peak_fpe']:7.2f} {row['peak_mc']:7.2f} | {row['adapted_fpe']:7.2f} {row['adapted_mc']:7.2f} | "
                  f"{row['tau_u_fpe']:6.1f} {row['tau_u_mc']:6.1f} | {row['rel_rms_rate']:8.3f}")
        plot_results(step_results, 'adaptation_test', rate_smoothing_ms=2)

    ####################################################################################################################
    # SINE TEST
    ####################################################################################################################
    if RUN_SINE:
        rows = []
        for res in sine_results:
            t = res['t']
            m_fpe, m_mc = sine_metrics(t, res['r_fpe'], res['u_fpe']), sine_metrics(t, res['r_mc'], res['u_mc'])
            rows.append(dict(scenario=res['key'], corr_uv_mc=res['corr_uv_mc'],
                             rel_rms_rate=rel_rms(res['r_fpe'], res['r_mc'], t < T_SINE - 1, 1),
                             **{f'{m}_fpe': m_fpe[m] for m in m_fpe}, **{f'{m}_mc': m_mc[m] for m in m_mc}))
        save_results(sine_results, 'sine_test')
        save_table(rows, 'sine_test')

        print(f"\nSINE TEST ({SINE['offset']:.0f} + {SINE['amplitude']:.0f} sin(2 pi "
              f"{SINE['frequency'] * 1e3:.0f} Hz t) pA for {SINE['duration']:.0f} ms, T = {T_SINE} ms)")
        print(f"{'scenario':16s} {'corr':>6s} | {'peak (Hz)':>15s} | {'rate 0-30 (Hz)':>15s} | "
              f"{'rate 30-50 (Hz)':>15s} | {'spikes/neuron':>15s} | {'<u> end (pA)':>15s} | {'rel. RMS':>8s}")
        print(f"{'':16s} {'(u,v)':>6s} |" + f" {'FPE':>7s} {'MC':>7s} |" * 5)
        for row in rows:
            print(f"{row['scenario']:16s} {row['corr_uv_mc']:6.2f} | "
                  f"{row['peak_fpe']:7.1f} {row['peak_mc']:7.1f} | "
                  f"{row['rate_during_fpe']:7.1f} {row['rate_during_mc']:7.1f} | "
                  f"{row['rate_after_fpe']:7.1f} {row['rate_after_mc']:7.1f} | "
                  f"{row['spikes_per_neuron_fpe']:7.3f} {row['spikes_per_neuron_mc']:7.3f} | "
                  f"{row['u_end_fpe']:7.2f} {row['u_end_mc']:7.2f} | {row['rel_rms_rate']:8.3f}")
        plot_results(sine_results, 'sine_test', rate_smoothing_ms=0.5, plot_input=True)

    print(f'saved results to {OUT_DIR}')
    plt.show()

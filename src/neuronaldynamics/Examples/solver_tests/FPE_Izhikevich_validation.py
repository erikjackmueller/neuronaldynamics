"""
Validation of FPE_Izhikevich (Fokker-Planck model of Izhikevich neurons with mean adaptation) against a Monte Carlo
simulation of N independent Izhikevich neurons with the same white noise (D = voltage_sigma^2 / tau_noise).

- QIF case (a = b = d = 0): the reduction is exact, FPE and Monte Carlo agree
- RS case (Izhikevich 2007 regular spiking): the mean adaptation approximation neglects the spread of u over the
  population, the onset transient agrees, the damped rate oscillations caused by spike synchronised adaptation are
  missed by the FPE

Monte Carlo integrator MC_METHOD: 'heun' (stochastic Heun with interpolated spike times, default) or 'euler'
(Euler-Maruyama with spikes on the time grid, the old reference).

Results are written to the folder FPE_Izhikevich_heun ('euler': FPE_Izhikevich) next to this script:
- FPE_Izhikevich_validation.png/.pdf   rates and <u> of FPE and Monte Carlo
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/neuronaldynamics, for direct calls
from Model.Nykamp_Model import FPE_Izhikevich
SHOW = os.environ.get('SHOW_PLOTS', '1') == '1'  # SHOW_PLOTS=0: no window (e.g. in a batch run)
matplotlib.use('TkAgg' if SHOW else 'Agg')

MC_METHOD = 'heun'  # Monte Carlo integrator: 'heun' (stochastic Heun, interpolated spike times) or 'euler'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FPE_Izhikevich' + ('' if MC_METHOD == 'euler' else '_heun'))
n_neurons = 40000

# plot style as in FPE_vs_LIF_timing.py: FPE blue, reference (Monte Carlo) dark grey, light grid, no top / right spines
FPE_STYLE = dict(color='#2a78d6', lw=1.6, label='FPE_Izhikevich')
MC_STYLE = dict(color='#3d3c39', lw=0.8, alpha=0.85, label=f'Monte Carlo n={n_neurons:,} (reference)')
GRID_COLOR = '#e0dfdb'


def monte_carlo(par, I, T, dt, N=n_neurons, seed=0):
    """
    simulation of N Izhikevich neurons (integrator MC_METHOD, see mc_step), returns the population rate (1/ms) and
    <u> (pA)
    """
    rng = np.random.default_rng(seed)
    D = par['voltage_sigma'] ** 2 / par['tau_noise']
    v = rng.normal(par['v_r'], 1.0, N)
    u = par['b'] * (v.mean() - par['v_r']) * np.ones(N)
    Nt = int(round(T / dt))
    r = np.zeros(Nt)
    u_mean = np.zeros(Nt)
    for i in range(1, Nt):
        v, u, spikes = mc_step(par, v, u, I[i], dt, np.sqrt(2 * D * dt) * rng.standard_normal(N))
        r[i] = spikes.sum() / (N * dt)
        u_mean[i] = u.mean()
    return r, u_mean


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


def smooth(x, ms, dt):
    """
    box filter of width ms, normalised by the number of samples in the window (no drop at the edges)
    """
    kernel = np.ones(int(ms / dt))
    return np.convolve(x, kernel, 'same') / np.convolve(np.ones_like(x), kernel, 'same')


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    T, dt = 300, 0.01
    t = np.arange(0, T, dt)
    # (row label, parameters different from the defaults (FPE_Izhikevich.izhikevich_defaults, RS neuron), step current)
    cases = [('QIF\n(a=b=d=0)', dict(a=0., b=0., d=0.), 80.),
             ('RS\n(Izhikevich 2007)', dict(), 100.)]

    fig, axs = plt.subplots(len(cases), 2, figsize=(9, 1.9 * len(cases) + 0.8), sharex=True, constrained_layout=True)
    for row, (label, case_parameters, i_step) in enumerate(cases):
        I = np.where(t > 20, i_step, 0.)
        par = dict(FPE_Izhikevich.izhikevich_defaults)
        par.update(case_parameters)

        fpe = FPE_Izhikevich(dict(case_parameters, T=T, dt=dt, dv=0.01, input_function=I, save_p='end',
                                  init_pdf_sigma=1.0), name=os.path.join(OUT_DIR, 'FPE_Izhikevich_validation'))
        fpe.simulate()
        r_mc, u_mc = monte_carlo(par, I, T, dt)

        late = t > 150
        print(f"{label.replace(chr(10), ' ')}: stationary rate FPE {fpe.r[0, late].mean() * 1e3:.2f} Hz, "
              f"MC {r_mc[late].mean() * 1e3:.2f} Hz; <u> FPE {fpe.u_mean[0, late].mean():.2f} pA, "
              f"MC {u_mc[late].mean():.2f} pA; mass {fpe.p[0, :, -1].sum() * fpe.dv:.5f}")

        ax_r, ax_u = axs[row]
        ax_r.plot(t, smooth(r_mc, 1, dt) * 1e3, **MC_STYLE)
        ax_r.plot(t, fpe.r[0] * 1e3, **FPE_STYLE)
        ax_r.set_ylabel(f'{label}\nrate (Hz)')
        ax_r.text(0.99, 0.05, f'step {i_step:.0f} pA at t = 20 ms', transform=ax_r.transAxes, ha='right',
                  va='bottom', fontsize=7, color='#5f5e5a')
        ax_u.plot(t, u_mc, **MC_STYLE)
        ax_u.plot(t, fpe.u_mean[0], **FPE_STYLE)
        ax_u.set_ylabel(r'$\langle u \rangle$ (pA)')

    for ax in axs.ravel():
        ax.grid(which='major', lw=0.4, color=GRID_COLOR)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel('t (ms)')
    axs[0, 0].set_title('population rate')
    axs[0, 1].set_title(r'mean adaptation current $\langle u \rangle$')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=len(handles), frameon=False)
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'FPE_Izhikevich_validation.{ext}'), dpi=300)
    print(f'saved results to {OUT_DIR}')
    if SHOW:
        plt.show()

"""
Exact mean-field model of Montbrio, Pazo and Roxin (2015) for Izhikevich neurons (QIF form with finite parameters,
Lorentzian heterogeneity of the input current, mean adaptation as in Chen & Campbell 2022) compared with the IK
network references and the FPE closures of the other solver tests.

Model (per population, r in 1/ms, v in mV, u in pA): with Lorentzian distributed inputs eta (centre eta_bar = I(t),
half width Delta in pA) the Lorentzian ansatz of the voltage density is invariant and gives exactly
    dr/dt     = Delta k / (pi C^2) + (k / C) r (2 v - v_r - v_t)
    C dv/dt   = k (v - v_r)(v - v_t) - u + I(t) + I_syn(t) - pi^2 C^2 r^2 / k
    du/dt     = a (b (v - v_r) - u) + d r                    (mean adaptation, as the 'mean' closure of the FPE)
for v_peak = +inf and reset = -inf. The IK neurons of the tests have v_peak = 35 mV, reset c and Gaussian white noise
instead, so the model is an approximation for them; Delta has no exact relation to the noise sigma and is fitted to
the reference (best case for the model).

Cases (references from the other tests, same networks and seeds):
    1 population, sigma = 2 mV, step input (FPE_Izhikevich_resolution.py: RS default, high rate, fast adaptation),
      reference: Monte Carlo with 100k neurons (net.MC_METHOD, default stochastic Heun)
    3 populations with feedback, sigma = 3.16 and 1 mV, step and 20 Hz sine (FPE_Izhikevich_3pop_improvements.py),
      reference: IK network with 100k neurons per population (net.MC_METHOD; one Delta for all populations)

Outputs in Examples/solver_tests/MPR_comparison_heun (Euler references: MPR_comparison): mpr_rates_<case>.png, mpr_table.csv

Usage (from src/neuronaldynamics):  python Examples/solver_tests/MPR_comparison.py
"""
import sys
import csv
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Model.Nykamp_Model import IZHIKEVICH_DEFAULTS
import FPE_vs_IK_network as net
import FPE_Izhikevich_3pop_improvements as imp
import FPE_Izhikevich_resolution as res
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / f'MPR_comparison{net.method_suffix()}'
OUT_DIR.mkdir(exist_ok=True)
DT = 0.005                                     # ms (Heun)
DELTAS = np.logspace(-1, 2.5, 36)              # pA, fit range of the Lorentzian half width


def mpr(pars, i_ext, W, delta, dt=DT, T=200.):
    """
    integrate the MPR equations with mean adaptation for n_pop populations (Heun), coupling as in FPE_vs_IK_network
    :param pars: dict of arrays (C, k, v_r, v_t, a, b, d) with one value per population
    :param i_ext: function t -> (n_pop,) input current in pA
    :return: rates (n_pop, Nt) in 1/ms on the grid dt
    """
    t = np.arange(0, T, dt)
    n_pop = len(pars['C'])
    kernel = net.alpha_kernel(dt)
    n_k = len(kernel)
    C, k, v_r, v_t, a, b, d = [np.asarray(pars[x], dtype=float) for x in ['C', 'k', 'v_r', 'v_t', 'a', 'b', 'd']]
    r = np.zeros((n_pop, len(t)))
    v = v_r.copy()
    u = b * (v - v_r)
    rr = np.full(n_pop, 1e-6)

    def rhs(t_, rr, v, u, i_syn):
        dr = delta * k / (np.pi * C**2) + k / C * rr * (2 * v - v_r - v_t)
        dv = (k * (v - v_r) * (v - v_t) - u + i_ext(t_) + i_syn - np.pi**2 * C**2 * rr**2 / k) / C
        du = a * (b * (v - v_r) - u) + d * rr
        return dr, dv, du

    for i in range(1, len(t)):
        nk = min(i, n_k)
        r_conv = r[:, i - nk:i][:, ::-1] @ kernel[:nk] * dt
        i_syn = W.T @ r_conv
        k1 = rhs(t[i - 1], rr, v, u, i_syn)
        pred = [x + dt * dx for x, dx in zip((rr, v, u), k1)]
        k2 = rhs(t[i], *pred, i_syn)
        rr, v, u = [x + 0.5 * dt * (d1 + d2) for x, d1, d2 in zip((rr, v, u), k1, k2)]
        rr = np.maximum(rr, 0.0)
        r[:, i] = rr
    return r


def binned(r, dt=DT):
    return net.binned_rate(r, dt)


def fit(pars, i_ext, W, ref_rb, measure):
    """best Delta (grid search) for the error measure"""
    best = None
    for delta in DELTAS:
        rb = binned(mpr(pars, i_ext, W, delta))
        if not np.isfinite(rb).all():
            continue
        err = measure(rb, ref_rb)
        if best is None or err < best[1]:
            best = (delta, err, rb)
    return best


def main():
    rows = []
    edges = np.arange(0, net.T + 1e-9, net.RATE_BIN_MS)
    tb = 0.5 * (edges[1:] + edges[:-1])
    rel_l2 = lambda x, ref: float(np.linalg.norm(x - ref) / np.linalg.norm(ref))
    smooth_l2 = lambda x, ref: imp.measures(x, ref)['rel_l2_smooth']

    # 1 population (resolution test, sigma = 2 mV, unsmoothed error as in that test)
    d = np.load(HERE / f'FPE_Izhikevich_resolution{net.method_suffix()}' / 'resolution_rates.npz')
    fpe_err = {}
    with open(HERE / f'FPE_Izhikevich_resolution{net.method_suffix()}' / 'resolution_table.csv') as f:
        for row in csv.DictReader(f):
            if row['model'] != 'MC' and row['dv'] == '0.1' and row['dt'] == '0.05':
                fpe_err[(row['regime'], row['model'])] = float(row['err_ref'])
    for regime, (title, par, i0) in res.REGIMES.items():
        ref = d[f'{regime}_mc_100000_None']
        mc20 = d[f'{regime}_mc_20000_None']
        p = dict(IZHIKEVICH_DEFAULTS, **par)
        pars = {x: [p[x]] for x in ['C', 'k', 'v_r', 'v_t', 'a', 'b', 'd']}
        i_ext = lambda t_, i0=i0: np.array([i0 if t_ < res.T_ON else 0.])
        delta, err, rb = fit(pars, i_ext, np.zeros((1, 1)), ref, rel_l2)
        rows.append(dict(case=f'1pop {regime}', measure='rel. L2 (1 ms)', delta=delta, mpr=err,
                         fpe_mean=fpe_err[(regime, 'mean')], fpe_cond=fpe_err[(regime, 'conditional')],
                         fpe_var=fpe_err[(regime, 'conditional_variance')], ik20k=rel_l2(mc20, ref)))
        fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True)
        ax.plot(tb, ref, color='#3d3c39', lw=1.5, label='Monte Carlo 100k')
        ax.plot(tb, rb[0], color='#7a1f5c', lw=1.2, label=f'MPR + mean adaptation, fitted $\\Delta$={delta:.2g} pA')
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('rate (Hz)')
        ax.set_title(f'1 population, {title}, $\\sigma$ = 2 mV', fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        fig.savefig(OUT_DIR / f'mpr_rates_1pop_{regime}.png', dpi=200)
        plt.close(fig)

    # 3 populations (improvements test, smoothed error)
    d = np.load(HERE / f'FPE_Izhikevich_3pop_improvements{net.method_suffix()}' / 'improvements_rates.npz')
    fpe = {}
    with open(HERE / f'FPE_Izhikevich_3pop_improvements{net.method_suffix()}' / 'improvements_table.csv') as f:
        for row in csv.DictReader(f):
            fpe[(row['input'], float(row['sigma']), row['model'], row['n'])] = float(row['rel_l2_smooth'])
    pars = net.neuron_parameters('3pop')
    W = net.SETUPS['3pop'][2]
    for kind, freq in imp.INPUTS.items():
        for sigma in imp.SIGMAS:
            ref = d[f'{kind}_{sigma}_IK_100000']
            imp.set_input(kind)
            i_ext = lambda t_: np.array([float(net.input_current(np.array([t_]))[0]), 0., 0.])
            delta, err, rb = fit(pars, i_ext, W, ref, smooth_l2)
            rows.append(dict(case=f'3pop {kind} sigma={sigma:.3g}', measure='rel. L2 smoothed (2 ms)', delta=delta,
                             mpr=err, fpe_mean=np.nan, fpe_cond=fpe[(kind, sigma, 'base', '')],
                             fpe_var=fpe[(kind, sigma, 'var', '')],
                             ik20k=fpe[(kind, sigma, 'IK', '20000')]))
            fig, axs = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True, constrained_layout=True)
            for j, ax in enumerate(axs):
                ax.plot(tb, ref[j], color='#3d3c39', lw=1.5, label='IK network 100k')
                ax.plot(tb, rb[j], color='#7a1f5c', lw=1.2, label=f'MPR + mean adaptation, $\\Delta$={delta:.2g} pA')
                ax.set_ylabel(f'{net.POP_NAMES["3pop"][j]}\nrate (Hz)')
            axs[0].legend(fontsize=7, frameon=False)
            axs[-1].set_xlabel('t (ms)')
            fig.suptitle(f'3 populations, {kind}, $\\sigma$ = {sigma:.3g} mV', fontsize=10)
            fig.savefig(OUT_DIR / f'mpr_rates_3pop_{kind}_sigma_{sigma:.3g}.png', dpi=200)
            plt.close(fig)

    with open(OUT_DIR / 'mpr_table.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'case':30s} {'measure':24s} {'Delta':>7s} {'MPR':>7s} {'FPE mean':>8s} {'FPE cond':>8s} "
          f"{'FPE var':>8s} {'IK 20k':>7s}")
    for r in rows:
        print(f"{r['case']:30s} {r['measure']:24s} {r['delta']:7.3g} {r['mpr']:7.3f} {r['fpe_mean']:8.3f} "
              f"{r['fpe_cond']:8.3f} {r['fpe_var']:8.3f} {r['ik20k']:7.3f}")


if __name__ == '__main__':
    main()

"""
Where does the remaining error of the two-node closure in the hardest case come from?

Case: 3 populations, sigma = 1 mV, step input (FPE_2D_reference_test.CASES['3pop_step_1']). Against the converged
reference (FPE_reference_rebuild.py, Heun, dt 0.005 ms, 100k) the 1D closures with FCT are 12 % off, the 2D solver with
FCT 5 %. Volley centroids show that the first volleys of all populations are right and that the shift starts with the
second volley of exc 1 (~66 ms, 1D 0.3 ms early); exc 2 and inh inherit it (the inhibitory population has no adaptation,
its 1D equation is exact). So the error is made by the adaptation of exc 1 between its first and second volley.

The 1D closures evolve p, q = p E[u | v], q2 = p E[u^2 | v] (and q3 = p E[u^3 | v] for 'conditional_quadrature'); the
v-flux of q_k contains p E[u^(k+1) | v], so the only closure is that of the top moment: Gaussian third moment for
'conditional_second_moment', two-node fourth central moment c4 = c2^2 + c3^2 / c2 for 'conditional_quadrature'. The
2D solution (no closure) gives all conditional moments, so it can test the closure assumption directly and show where
the 1D fields E[u | v], Var[u | v] depart from it.

Runs (saved in FPE_closure_diagnostics/):
    2d.npz      2D solver, FCT, dv 0.1, du 2, dt 0.01: per snapshot (every SNAP_EVERY ms) and population p(v) and the
                conditional moments m1 = E[u | v], c2, c3, c4 (central), and the u distribution of the firing neurons
                (outflow at v_peak per u line) in FLUX_BIN ms bins
    1d_<closure>.npz   FCT, dv 0.1, dt 0.005: p, m1, c2 at the same snapshot times, rates, <u>, Var u

    dt_<closure>_<dt>.npz   1D closure with FCT, dv 0.1, on the time steps DTS_1D (rates, <u>, Var u)
    network.npz        IK network, Heun, MC_DT 0.01 ms, N_NET neurons per population, seed of the reference: rates and
                       <u> of every population (the truth for the adaptation)

Usage (from src/neuronaldynamics):
    python Examples/solver_tests/FPE_closure_diagnostics.py run          (2D and 1D fields)
    python Examples/solver_tests/FPE_closure_diagnostics.py run_dt       (1D on DTS_1D and the network)
    python Examples/solver_tests/FPE_closure_diagnostics.py analyze
    python Examples/solver_tests/FPE_closure_diagnostics.py analyze_dt
    python Examples/solver_tests/FPE_closure_diagnostics.py run_snap     (network snapshots vs converged 1D fields)
    python Examples/solver_tests/FPE_closure_diagnostics.py analyze_snap
    python Examples/solver_tests/FPE_closure_diagnostics.py run_dv       (two-node closure on DVS_1D at DT_CONV)
    python Examples/solver_tests/FPE_closure_diagnostics.py analyze_dv
"""
import os
for _var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_var, '1')
import sys
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
from Model.Nykamp_Model import FPE_Population
import FPE_vs_IK_network as net
import FPE_2D_reference_test as t2d
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'FPE_closure_diagnostics'
OUT_DIR.mkdir(exist_ok=True)
CASE = '3pop_step_1'
REF = HERE / 'FPE_reference_rebuild' / 'ref_3pop_step_1_heun_dt0.005_100k.npz'
SNAP_EVERY = 1.0            # ms
FLUX_BIN = 0.1              # ms
DV, DU, DT_2D, DT_1D = 0.1, 2., 0.01, 0.005
CLOSURES = ['conditional_quadrature', 'conditional_second_moment']
P_FLOOR = 1e-6              # conditional moments only where p > P_FLOOR * max p
DTS_1D = [0.01, 0.005, 0.0025, 0.00125]
DVS_1D = [0.05, 0.025, 0.0125]          # mV, grid study of the two-node closure at DT_CONV (run_dv)
N_NET, NET_SEED, NET_DT = 100_000, 600_000, 0.01
POP = ['exc 1', 'exc 2', 'inh']


def snapshot_times():
    return np.arange(0., net.T, SNAP_EVERY)


def conditional_moments(P, u, du, floor=P_FLOOR):
    """P (Nu, Nv) -> p(v), E[u|v], central moments c2, c3, c4 (nan where p is tiny)"""
    p = du * P.sum(axis=0)
    ok = p > floor * p.max()
    m1 = np.full_like(p, np.nan)
    c = {k: np.full_like(p, np.nan) for k in (2, 3, 4)}
    m1[ok] = du * (u @ P[:, ok]) / p[ok]
    du_ = u[:, None] - m1[None, ok]
    for k in (2, 3, 4):
        c[k][ok] = du * np.sum(du_**k * P[:, ok], axis=0) / p[ok]
    return p, m1, c[2], c[3], c[4]


def run(task):
    pars = t2d.setup_case(CASE)
    t0 = time.perf_counter()
    if task == '2d':
        m = FPE_Izhikevich2D(dict(pars, dt=DT_2D, dv=DV, du=DU, init_pdf_sigma=1.0, flux_limiter='fct',
                                  snapshot_times=snapshot_times(), save_flux_u=True))
        m.simulate()
        out = dict(t=m.t, v=m.v, t_snap=m.t_snap, r=m.r, rb=net.binned_rate(m.r, DT_2D), u_mean=m.u_mean,
                   var_u=m.var_u, time=m.cpu)
        nb = int(round(FLUX_BIN / DT_2D))
        for j in range(len(m.u)):
            mom = np.array([conditional_moments(P, m.u[j], DU) for P in m.P_snap[j]])   # (n_snap, 5, Nv)
            out[f'mom_{j}'] = mom
            out[f'u_{j}'] = m.u[j]
            fl = m.flux_u[j][1:]
            fl = fl[:len(fl) // nb * nb].reshape(-1, nb, fl.shape[1]).mean(axis=1)
            out[f'flux_u_{j}'] = fl.astype(np.float32)
        np.savez(OUT_DIR / '2d.npz', **out)
    else:
        every = int(round(SNAP_EVERY / DT_1D))
        m = FPE_Population(dict(pars, soma_model='IK', adaptation_closure=task, flux_limiter='fct', dt=DT_1D, dv=DV,
                                init_pdf_sigma=1.0, save_p='all', p_save_every=every, tqdm_disable=True),
                           name=str(OUT_DIR / f'fpe_{task}'))
        m.simulate()
        cpu = time.perf_counter() - t0
        p, q, q2 = m.p, m.q, m.q2
        with np.errstate(all='ignore'):
            m1 = q / p
            c2 = q2 / p - m1**2
        mask = p > P_FLOOR * p.max(axis=1, keepdims=True)
        m1[~mask], c2[~mask] = np.nan, np.nan
        np.savez(OUT_DIR / f'1d_{task}.npz', t=m.t, v=m.v, t_snap=m.t_p, r=m.r, rb=net.binned_rate(m.r, DT_1D),
                 u_mean=m.u_mean, var_u=m.var_u, p=p, m1=m1, c2=c2, time=cpu)
        m.clean()
    print(f'{task}: {time.perf_counter() - t0:.0f} s', flush=True)
    return task


def run_dt(task):
    """task: ('1d', closure, dt) or ('network',)"""
    t0 = time.perf_counter()
    if task[0] == 'network':
        import FPE_Izhikevich_3pop_improvements as imp
        imp.set_input(t2d.CASES[CASE][2])
        net.MC_DT, net.MC_METHOD = NET_DT, 'heun'
        r, u = net.ik_network('3pop', t2d.CASES[CASE][1], N_NET, seed=NET_SEED)
        np.savez(OUT_DIR / 'network.npz', t=np.arange(0, net.T, NET_DT), rb=net.binned_rate(r, NET_DT), u_mean=u,
                 time=time.perf_counter() - t0)
    else:
        _, closure, dt = task[:3]
        dv = task[3] if len(task) > 3 else DV
        m = FPE_Population(dict(t2d.setup_case(CASE), soma_model='IK', adaptation_closure=closure, flux_limiter='fct',
                                dt=dt, dv=dv, init_pdf_sigma=1.0, save_p='none', tqdm_disable=True),
                           name=str(OUT_DIR / f'fpe_{closure}_{dt:g}_{dv:g}'))
        m.simulate()
        np.savez(OUT_DIR / (f'dt_{closure}_{dt:g}.npz' if dv == DV else f'dv_{closure}_{dt:g}_{dv:g}.npz'), t=m.t,
                 rb=net.binned_rate(m.r, dt), u_mean=m.u_mean, var_u=m.var_u, time=time.perf_counter() - t0)
        m.clean()
    print(f'{task}: {time.perf_counter() - t0:.0f} s', flush=True)
    return task


def centroids(rb):
    """centroids (ms) of the volleys of the reference windows: exc 1 second volley, exc 2 second volley, inh bursts"""
    tb = np.arange(rb.shape[1]) + 0.5
    out = {}
    for name, j, (a, b) in [('exc1 v1', 0, (16, 42)), ('exc1 v2', 0, (42, 83)), ('exc2 v2', 1, (69, 114)),
                            ('inh 64-74', 2, (64, 74)), ('inh 77-96', 2, (77, 96))]:
        w = rb[j, a:b]
        out[name] = float(np.sum(w * tb[a:b]) / max(w.sum(), 1e-12))
    return out


def analyze_dv():
    """grid study of the two-node closure at DT_CONV: error, centroids, inhibitory tail burst, early exc 1 spikes"""
    import FPE_Izhikevich_3pop_improvements as imp
    ref = np.load(REF)['rb']
    c = 'conditional_quadrature'
    rows = [('reference (Heun 100k, dt 0.005)', ref, np.nan)]
    nw = OUT_DIR / 'network.npz'
    if nw.exists():
        rows.append((f'network Heun {N_NET // 1000}k dt {NET_DT:g}', np.load(nw)['rb'], np.nan))
    for dv in [DV] + DVS_1D:
        f = OUT_DIR / (f'dt_{c}_{DT_CONV:g}.npz' if dv == DV else f'dv_{c}_{DT_CONV:g}_{dv:g}.npz')
        if f.exists():
            d = np.load(f)
            rows.append((f'1D two-node dt {DT_CONV:g} dv {dv:g}', d['rb'], float(d['time'])))
    names = list(centroids(ref))
    print(f'{"model":36s} {"time":>6s} {"error":>6s} ' + ' '.join(f'{n:>9s}' for n in names) +
          f' {"inh 41-48":>9s} {"exc1 42-56":>10s}')
    for name, rb, cpu in rows:
        e = imp.measures(rb, ref)['rel_l2_smooth']
        print(f'{name:36s} {cpu:6.0f} {e:6.3f} ' + ' '.join(f'{x:9.2f}' for x in centroids(rb).values()) +
              f' {rb[2, 41:48].sum():9.1f} {rb[0, 42:56].sum():10.1f}')


def analyze_dt():
    import FPE_Izhikevich_3pop_improvements as imp
    ref = np.load(REF)['rb']
    rows = [('reference (Heun 100k, dt 0.005)', ref, None, None)]
    netf = OUT_DIR / 'network.npz'
    nw = dict(np.load(netf)) if netf.exists() else None
    if nw is not None:
        rows.append((f'network Heun {N_NET // 1000}k dt {NET_DT:g}', nw['rb'], nw['t'], nw['u_mean']))
    d2 = OUT_DIR / '2d.npz'
    if d2.exists():
        d = np.load(d2)
        rows.append(('2D FCT du 2 dt 0.01', d['rb'], d['t'], d['u_mean']))
    for c in CLOSURES:
        for dt in DTS_1D:
            f = OUT_DIR / f'dt_{c}_{dt:g}.npz'
            if f.exists():
                d = np.load(f)
                rows.append((f'1D {c[12:]} dt {dt:g}', d['rb'], d['t'], d['u_mean']))
    names = list(centroids(ref))
    print(f'{"model":34s} {"error":>6s} ' + ' '.join(f'{n:>9s}' for n in names) + '   <u> exc 1 - network (pA) at '
          '20 / 30 / 40 / 50 / 60 ms')
    for name, rb, t, u in rows:
        e = imp.measures(rb, ref)['rel_l2_smooth']
        line = f'{name:34s} {e:6.3f} ' + ' '.join(f'{x:9.2f}' for x in centroids(rb).values())
        if u is not None and nw is not None:
            du = [np.interp(ts, t, u[0]) - np.interp(ts, nw['t'], nw['u_mean'][0]) for ts in (20, 30, 40, 50, 60)]
            line += '   ' + ' '.join(f'{x:6.2f}' for x in du)
        print(line)



SNAP_NET = np.arange(30., 66.5, 1.)    # ms, snapshots of the network and of the converged 1D fields
DT_CONV = 0.0025                       # ms, 1D closures converged in dt (run_dt)
V_BINS = np.arange(-80., -29.9, 0.5)   # mV, bins for p(v), E[u|v], sd(u|v)


def run_snap(task):
    """task: 'network' (Heun, NET_DT, N_NET, seed of the reference, v and u of all neurons at SNAP_NET) or a closure
    (FCT, dv 0.1, DT_CONV, p, E[u|v], sd(u|v) binned like the network)"""
    t0 = time.perf_counter()
    if task == 'network':
        import FPE_Izhikevich_3pop_improvements as imp
        imp.set_input(t2d.CASES[CASE][2])
        net.MC_DT, net.MC_METHOD = NET_DT, 'heun'
        r, u_mean, t_s, v_s, u_s = net.ik_network('3pop', t2d.CASES[CASE][1], N_NET, seed=NET_SEED,
                                                  snapshot_times=SNAP_NET)
        out = dict(t_snap=t_s, rb=net.binned_rate(r, NET_DT))
        for j in range(2):
            idx = np.digitize(v_s[:, j], V_BINS) - 1                        # (n_snap, n)
            nb = len(V_BINS) - 1
            cnt = np.zeros((len(t_s), nb))
            m1 = np.full((len(t_s), nb), np.nan)
            sd = np.full((len(t_s), nb), np.nan)
            for s_ in range(len(t_s)):
                ok = (idx[s_] >= 0) & (idx[s_] < nb)
                cnt[s_] = np.bincount(idx[s_][ok], minlength=nb)
                su = np.bincount(idx[s_][ok], weights=u_s[s_, j][ok], minlength=nb)
                su2 = np.bincount(idx[s_][ok], weights=u_s[s_, j][ok]**2, minlength=nb)
                good = cnt[s_] >= 100
                m1[s_, good] = su[good] / cnt[s_, good]
                sd[s_, good] = np.sqrt(np.maximum(su2[good] / cnt[s_, good] - m1[s_, good]**2, 0))
            out[f'p_{j}'] = cnt / (N_NET * np.diff(V_BINS))                 # density per mV
            out[f'm1_{j}'] = m1
            out[f'sd_{j}'] = sd
            out[f'u_mean_{j}'] = u_s[:, j].mean(axis=1)
        np.savez(OUT_DIR / 'network_snap.npz', **out)
    else:
        every = int(round(1.0 / DT_CONV))
        m = FPE_Population(dict(t2d.setup_case(CASE), soma_model='IK', adaptation_closure=task, flux_limiter='fct',
                                dt=DT_CONV, dv=DV, init_pdf_sigma=1.0, save_p='all', p_save_every=every,
                                tqdm_disable=True), name=str(OUT_DIR / f'fpe_snap_{task}'))
        m.simulate()
        v = m.v
        w = np.full(len(v), DV)
        out = dict(t_snap=SNAP_NET)
        for j in range(2):
            ps, m1s, sds = [], [], []
            for ts in SNAP_NET:
                k = int(np.argmin(np.abs(m.t_p - ts)))
                p_, q_, q2_ = m.p[j, :, k], m.q[j, :, k], m.q2[j, :, k]
                idx = np.digitize(v, V_BINS) - 1
                nb = len(V_BINS) - 1
                ok = (idx >= 0) & (idx < nb)
                P = np.bincount(idx[ok], weights=(p_ * w)[ok], minlength=nb)
                Q = np.bincount(idx[ok], weights=(q_ * w)[ok], minlength=nb)
                Q2 = np.bincount(idx[ok], weights=(q2_ * w)[ok], minlength=nb)
                with np.errstate(all='ignore'):
                    mm = Q / P
                    ss = np.sqrt(np.maximum(Q2 / P - mm**2, 0))
                bad = P * N_NET < 100                                           # same cut as the network
                mm[bad], ss[bad] = np.nan, np.nan
                ps.append(P / np.diff(V_BINS)); m1s.append(mm); sds.append(ss)
            out[f'p_{j}'], out[f'm1_{j}'], out[f'sd_{j}'] = np.array(ps), np.array(m1s), np.array(sds)
        np.savez(OUT_DIR / f'snap_{task}.npz', **out)
        m.clean()
    print(f'{task}: {time.perf_counter() - t0:.0f} s', flush=True)
    return task


def analyze_snap():
    nw = dict(np.load(OUT_DIR / 'network_snap.npz'))
    one = {c: dict(np.load(OUT_DIR / f'snap_{c}.npz')) for c in CLOSURES if (OUT_DIR / f'snap_{c}.npz').exists()}
    vc = 0.5 * (V_BINS[1:] + V_BINS[:-1])
    for j, pop in [(0, 'exc 1'), (1, 'exc 2')]:
        print(f'\n{pop}: 1D (dt {DT_CONV:g}) minus network (Heun {N_NET // 1000}k): mean v, fraction above -50 / -45 mV, '
              'p-weighted E[u|v] and sd(u|v) differences (pA), all at the same times')
        print(f'{"t":>5s} {"net <v>":>8s} {"net>-50":>8s} {"net>-45":>8s} ' +
              ' '.join(f'{c[12:20]:>8s}:d<v> {"d>-50":>7s} {"d>-45":>7s} {"dE[u|v]":>8s} {"dsd":>6s}' for c in one))
        for s_, ts in enumerate(nw['t_snap']):
            pn = nw[f'p_{j}'][s_] * np.diff(V_BINS)
            vbar = np.sum(pn * vc) / pn.sum()
            f50, f45 = pn[vc > -50].sum(), pn[vc > -45].sum()
            line = f'{ts:5.0f} {vbar:8.2f} {f50:8.4f} {f45:8.4f} '
            for c, d in one.items():
                p1 = d[f'p_{j}'][s_] * np.diff(V_BINS)
                v1 = np.sum(p1 * vc) / p1.sum()
                ok = np.isfinite(d[f'm1_{j}'][s_]) & np.isfinite(nw[f'm1_{j}'][s_])
                dm = np.sum((d[f'm1_{j}'][s_] - nw[f'm1_{j}'][s_])[ok] * pn[ok]) / pn[ok].sum()
                dsd = np.sum((d[f'sd_{j}'][s_] - nw[f'sd_{j}'][s_])[ok] * pn[ok]) / pn[ok].sum()
                line += (f'{v1 - vbar:13.3f} {p1[vc > -50].sum() - f50:7.4f} {p1[vc > -45].sum() - f45:7.4f} '
                         f'{dm:8.2f} {dsd:6.2f} ')
            print(line)
    # figure: exc 1 at four times
    times = [40., 50., 58., 63.]
    fig, axs = plt.subplots(3, len(times), figsize=(14, 8.5), sharex=True, constrained_layout=True)
    colors = {'conditional_quadrature': '#d6612a', 'conditional_second_moment': '#2a9d6b'}
    for col, ts in enumerate(times):
        s_ = int(np.argmin(np.abs(nw['t_snap'] - ts)))
        axs[0, col].plot(vc, nw['p_0'][s_], color='#3d3c39', lw=1.6, label=f'network Heun {N_NET // 1000}k')
        axs[1, col].plot(vc, nw['m1_0'][s_], color='#3d3c39', lw=1.6)
        axs[2, col].plot(vc, nw['sd_0'][s_], color='#3d3c39', lw=1.6)
        for c, d in one.items():
            lab = '1D two-node' if 'quad' in c else '1D second moment'
            axs[0, col].plot(vc, d['p_0'][s_], color=colors[c], lw=1.1, ls='--', label=lab)
            axs[1, col].plot(vc, d['m1_0'][s_], color=colors[c], lw=1.1, ls='--')
            axs[2, col].plot(vc, d['sd_0'][s_], color=colors[c], lw=1.1, ls='--')
        axs[0, col].set_yscale('log')
        axs[0, col].set_ylim(1e-5, None)
        axs[0, col].set_title(f'exc 1, t = {ts:g} ms', fontsize=9)
        axs[2, col].set_xlabel('v (mV)')
    axs[0, 0].set_ylabel('p(v) (1/mV)')
    axs[1, 0].set_ylabel('E[u|v] (pA)')
    axs[2, 0].set_ylabel('sd(u|v) (pA)')
    axs[0, 0].legend(fontsize=7, frameon=False)
    for ax in axs.ravel():
        ax.set_xlim(-75, -35)
        ax.grid(lw=0.4, color='#e0dfdb')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'3 populations, sigma = 1 mV, step: exc 1 before its second volley, 1D closures (FCT, dv 0.1, '
                 f'dt {DT_CONV:g}) against the network (Heun, {N_NET // 1000}k, 0.5 mV bins)', fontsize=10)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT_DIR / f'network_snapshots.{ext}', dpi=200)
    plt.close(fig)
    print(f'figure saved in {OUT_DIR}')


########################################################################################################################
# ANALYSIS
########################################################################################################################
def load():
    d2 = dict(np.load(OUT_DIR / '2d.npz'))
    d1 = {c: dict(np.load(OUT_DIR / f'1d_{c}.npz')) for c in CLOSURES if (OUT_DIR / f'1d_{c}.npz').exists()}
    return d2, d1


def wmean(x, w):
    ok = np.isfinite(x) & (w > 0)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok]))


def closure_check(d2, j=0):
    """p-weighted relative error of the closed top moment against the 2D conditional moments, per snapshot"""
    rows = []
    for s, ts in enumerate(d2['t_snap']):
        p, m1, c2, c3, c4 = d2[f'mom_{j}'][s]
        ok = np.isfinite(c2) & (c2 > 1e-6)
        if ok.sum() < 3:
            continue
        w = p * ok
        c4_two = c2**2 + c3**2 / np.where(ok, c2, 1)
        skew = c3 / np.where(ok, c2, 1)**1.5
        kurt = c4 / np.where(ok, c2, 1)**2
        rows.append(dict(t=ts, sd=np.sqrt(wmean(c2, w)), skew=wmean(skew, w), kurt=wmean(kurt, w),
                         err_c4_two=wmean(np.abs(c4_two - c4) / c4, w),
                         err_c4_gauss=wmean(np.abs(3 * c2**2 - c4) / c4, w),
                         err_c3_gauss=wmean(np.abs(c3) / c2**1.5, w)))
    return rows


def flux_stats(d2, j=0):
    """mean, sd and skewness of u of the firing neurons in FLUX_BIN bins"""
    fl = d2[f'flux_u_{j}'].astype(float)
    u = d2[f'u_{j}']
    tot = fl.sum(axis=1)
    with np.errstate(all='ignore'):
        m = fl @ u / tot
        c2 = (fl * (u[None] - m[:, None])**2).sum(axis=1) / tot
        c3 = (fl * (u[None] - m[:, None])**3).sum(axis=1) / tot
    t = (np.arange(len(tot)) + 0.5) * FLUX_BIN
    return t, tot * DU, m, np.sqrt(c2), c3 / c2**1.5


def analyze():
    d2, d1 = load()
    ref = np.load(REF)['rb']
    tb = np.arange(ref.shape[1]) + 0.5
    v = d2['v']

    # 1. closure assumption in the 2D solution (exc 1)
    print('closure check on the 2D solution, exc 1: p-weighted means over v (moments where p > '
          f'{P_FLOOR:g} max p)')
    print(f'{"t":>5s} {"sd(u|v)":>8s} {"skew":>7s} {"kurt":>7s} {"c4 two-node":>12s} {"c4 Gauss":>9s}')
    rows = closure_check(d2, 0)
    for r in rows:
        if 20 <= r['t'] <= 75:
            print(f'{r["t"]:5.0f} {r["sd"]:8.2f} {r["skew"]:7.2f} {r["kurt"]:7.2f} {r["err_c4_two"]:12.3f} '
                  f'{r["err_c4_gauss"]:9.3f}')

    # 2. 1D fields against 2D (exc 1): p-weighted differences of E[u|v] and sd(u|v)
    t_snap = d2['t_snap']
    print('\n1D fields against the 2D solution, exc 1: p-weighted mean of E[u|v]_1D - E[u|v]_2D and of '
          'sd_1D - sd_2D (pA), <u> and sd(u) of the population')
    header = f'{"t":>5s} ' + ' '.join(f'{c[12:20]:>8s}:dm {"dsd":>6s} {"d<u>":>6s} {"dsd_u":>6s}' for c in d1)
    print(header + f' {"2D <u>":>7s} {"2D sd_u":>7s}')
    i2 = {round(t, 6): s for s, t in enumerate(t_snap)}
    for s, ts in enumerate(t_snap):
        if not (20 <= ts <= 75):
            continue
        p2, m2, c22 = d2['mom_0'][s][:3]
        k2 = int(round(ts / DT_2D))
        line = f'{ts:5.0f} '
        for c, d in d1.items():
            k = int(np.argmin(np.abs(d['t_snap'] - ts)))
            k1 = int(round(ts / DT_1D))
            n = min(len(d['v']), len(v))
            dm = wmean(d['m1'][0, :n, k] - m2[:n], p2[:n])
            dsd = wmean(np.sqrt(np.maximum(d['c2'][0, :n, k], 0)) - np.sqrt(np.maximum(c22[:n], 0)), p2[:n])
            line += (f'{dm:11.2f} {dsd:6.2f} {d["u_mean"][0, k1] - d2["u_mean"][0, k2]:6.2f} '
                     f'{np.sqrt(d["var_u"][0, k1]) - np.sqrt(d2["var_u"][0, k2]):6.2f} ')
        print(line + f'{d2["u_mean"][0, k2]:7.1f} {np.sqrt(d2["var_u"][0, k2]):7.2f}')

    # 3. u of the firing neurons (2D)
    t_f, rate_f, m_f, sd_f, sk_f = flux_stats(d2, 0)
    for a, b in [(16, 42), (42, 83)]:
        sel = (t_f >= a) & (t_f < b) & np.isfinite(m_f)
        w = rate_f[sel]
        print(f'exc 1 volley [{a},{b}) ms (2D): firing-weighted u at v_peak: mean {wmean(m_f[sel], w):.1f} pA, '
              f'sd {wmean(sd_f[sel], w):.2f} pA, skewness {wmean(sk_f[sel], w):.2f}')

    # FIGURE
    fig, axs = plt.subplots(3, 3, figsize=(13, 9.5), constrained_layout=True)
    colors = {'2D': '#1b4f8f', 'conditional_quadrature': '#d6612a', 'conditional_second_moment': '#2a9d6b'}
    labels = {'2D': '2D FCT (no closure)', 'conditional_quadrature': '1D two-node',
              'conditional_second_moment': '1D second moment'}
    # (a) rates exc 1, exc 2, inh 55-85 ms
    ax = axs[0, 0]
    for j, ls in zip(range(3), ['-', '--', ':']):
        ax.plot(tb, ref[j], color='#3d3c39', lw=1.6, ls=ls, label=f'reference {POP[j]}')
        ax.plot(tb, d2['rb'][j], color=colors['2D'], lw=1.1, ls=ls)
        ax.plot(tb, d1['conditional_quadrature']['rb'][j], color=colors['conditional_quadrature'], lw=1.1, ls=ls)
    ax.set_xlim(55, 90)
    ax.set_ylabel('rate (Hz)')
    ax.set_title('(a) rates (solid exc 1, dashed exc 2, dotted inh)', fontsize=9)
    # (b) <u> and sd(u) of exc 1 over time
    ax = axs[0, 1]
    t2 = d2['t']
    ax.plot(t2, d2['u_mean'][0], color=colors['2D'], label=labels['2D'])
    for c, d in d1.items():
        ax.plot(d['t'], d['u_mean'][0], color=colors[c], ls='--', label=labels[c])
    ax.set_xlim(0, 90)
    ax.set_ylabel(r'exc 1 $\langle u\rangle$ (pA)')
    ax.set_title(r'(b) mean adaptation of exc 1', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax = axs[0, 2]
    ax.plot(t2, np.sqrt(d2['var_u'][0]), color=colors['2D'])
    for c, d in d1.items():
        ax.plot(d['t'], np.sqrt(d['var_u'][0]), color=colors[c], ls='--')
    ax.set_xlim(0, 90)
    ax.set_ylabel('exc 1 sd(u) (pA)')
    ax.set_title('(c) spread of the adaptation of exc 1', fontsize=9)
    # (d-f) snapshots of p(v), E[u|v], sd(u|v) for exc 1
    snaps = [40., 55., 62.]
    for col, ts in enumerate(snaps):
        s = i2[round(ts, 6)]
        p2, m2, c22 = d2['mom_0'][s][:3]
        ax = axs[1, col]
        ax2 = axs[2, col]
        ax.plot(v, p2, color=colors['2D'], lw=1.3)
        ax2.plot(v, m2, color=colors['2D'], lw=1.3, label='E[u|v]')
        ax2.plot(v, m2 + np.sqrt(c22), color=colors['2D'], lw=0.7, ls=':')
        ax2.plot(v, m2 - np.sqrt(c22), color=colors['2D'], lw=0.7, ls=':')
        for c, d in d1.items():
            k = int(np.argmin(np.abs(d['t_snap'] - ts)))
            ax.plot(d['v'], d['p'][0, :, k], color=colors[c], lw=1.0, ls='--')
            ax2.plot(d['v'], d['m1'][0, :, k], color=colors[c], lw=1.0, ls='--')
            ax2.plot(d['v'], d['m1'][0, :, k] + np.sqrt(np.maximum(d['c2'][0, :, k], 0)), color=colors[c], lw=0.6,
                     ls=':')
            ax2.plot(d['v'], d['m1'][0, :, k] - np.sqrt(np.maximum(d['c2'][0, :, k], 0)), color=colors[c], lw=0.6,
                     ls=':')
        ax.set_xlim(-75, -30)
        ax2.set_xlim(-75, -30)
        ax.set_yscale('log')
        ax.set_ylim(1e-5, None)
        ax.set_title(f'({"def"[col]}) exc 1, t = {ts:g} ms: p(v)', fontsize=9)
        ax2.set_xlabel('v (mV)')
        ax2.set_ylabel(r'E[u|v] $\pm$ sd (pA)')
    for ax in axs.ravel():
        ax.grid(lw=0.4, color='#e0dfdb')
        ax.spines[['top', 'right']].set_visible(False)
    axs[0, 0].set_xlabel('t (ms)')
    fig.suptitle('3 populations, $\\sigma$ = 1 mV, step: 1D closures (FCT, dv 0.1, dt 0.005) against the 2D solution '
                 '(FCT, dv 0.1, du 2, dt 0.01)', fontsize=10)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT_DIR / f'closure_diagnostics.{ext}', dpi=200)
    plt.close(fig)
    print(f'figure saved in {OUT_DIR}')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'analyze'
    if mode == 'run':
        tasks = sys.argv[2:] or ['2d'] + CLOSURES
        with Pool(len(tasks)) as pool:
            for _ in pool.imap_unordered(run, tasks):
                pass
    if mode == 'run_dt':
        tasks = [('network',)] + [('1d', c, dt) for c in CLOSURES for dt in DTS_1D[::-1]
                                  if not (OUT_DIR / f'dt_{c}_{dt:g}.npz').exists()]
        with Pool(4) as pool:
            for _ in pool.imap_unordered(run_dt, tasks):
                pass
    if mode == 'run_dv':
        tasks = [('1d', 'conditional_quadrature', DT_CONV, dv) for dv in DVS_1D[::-1]
                 if not (OUT_DIR / f'dv_conditional_quadrature_{DT_CONV:g}_{dv:g}.npz').exists()]
        with Pool(max(len(tasks), 1)) as pool:
            for _ in pool.imap_unordered(run_dt, tasks):
                pass
    if mode == 'run_snap':
        with Pool(3) as pool:
            for _ in pool.imap_unordered(run_snap, ['network'] + CLOSURES):
                pass
    if mode in ('run_dv', 'analyze_dv'):
        analyze_dv()
    elif mode in ('run_snap', 'analyze_snap'):
        analyze_snap()
    elif mode in ('run_dt', 'analyze_dt'):
        analyze_dt()
    else:
        analyze()

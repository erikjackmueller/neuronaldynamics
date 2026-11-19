import time
import scipy
import scipy.linalg
import scipy.special
import scipy.sparse
import scipy.sparse.linalg
import warnings
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from scipy.stats import gamma, norm, lognorm
import os
import h5py
import matplotlib
import matplotlib.animation as animation
matplotlib.use('TkAgg')


def plot_p_panel(fig, subplot_args, t_p, v, p_k, heat_map, z_limit, cmap, labelsize, ylims=None):
    """
    Draw p(v, t) of one population into a new subplot of fig: heat map (or surface) if several time steps were
    saved, p(v) if only one time step was saved (save_p='end') or a note if p was not saved (save_p='none').
    ylims: (v_min, v_max) of the voltage axis of the heat map / surface, None: whole v grid
    :return: axis, z_limit
    """
    if p_k is None:
        ax = fig.add_subplot(*subplot_args)
        ax.text(0.5, 0.5, "p not saved\n(save_p='end' or 'all')", ha='center', va='center',
                transform=ax.transAxes, fontsize=labelsize)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax, z_limit
    if p_k.shape[-1] == 1:
        ax = fig.add_subplot(*subplot_args)
        ax.plot(v, p_k[:, 0])
        ax.set_xlabel("Membrane potential (mV)", fontsize=labelsize)
        ax.set_ylabel(f"p at t = {t_p[0]:.2f} ms", fontsize=labelsize)
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        return ax, z_limit
    X, Y = np.meshgrid(t_p, v)
    if heat_map:
        ax = fig.add_subplot(*subplot_args)
        if z_limit is None:
            z_limit = np.abs(p_k).max()
        c = ax.pcolormesh(X, Y, p_k, cmap=cmap, vmin=0, vmax=z_limit)
        fig.colorbar(c, ax=ax)
    else:
        ax = fig.add_subplot(*subplot_args, projection='3d')
        ax.plot_surface(X, Y, p_k, cmap=cmap, linewidth=0, antialiased=False, rcount=100, ccount=100)
        ax.set_zlim3d(0, 1)
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.set_xlabel("t (ms)", fontsize=labelsize)
    ax.set_ylabel("Membrane potential (mV)", fontsize=labelsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    return ax, z_limit


# Izhikevich neurons (soma_model='izhikevich' / 'IK' in FPE_Population, or FPE_Izhikevich): default parameters (regular
# spiking neuron of Izhikevich 2007), parameters that can differ per population and the closures of the adaptation
IZHIKEVICH_DEFAULTS = dict(C=100., k=0.7, v_r=-60., v_t=-40., v_peak=35., v_min=-90., a=0.03, b=-2., c=-50., d=100.,
                           tau_noise=10., voltage_sigma=2., u_init=None, adaptation_closure='mean', u_cond_floor=1e-6,
                           v_grid='uniform', dv_coarse=0.5, v_fine_range=None, grid_growth=1.05, use_numba=True,
                           flux_limiter='none', n_corrector=0)
IZHIKEVICH_PER_POPULATION = ['C', 'k', 'v_r', 'v_t', 'a', 'b', 'c', 'd', 'tau_noise']
ADAPTATION_CLOSURES = ['mean', 'conditional', 'conditional_variance', 'conditional_second_moment',
                       'conditional_quadrature']
QUADRATURE_MAX_SKEWNESS = 10.  # |third central moment| <= this * s^3 in the two-node closure
FLUX_LIMITERS = ['none', 'fct']

# numba (optional): compiled kernels of the Izhikevich time step, without numba the same functions run as plain python
# (correct, but much slower)
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

# fast math without the assumptions 'no inf' / 'no nan' (the kernels compare with inf)
_FASTMATH = {'nsz', 'arcp', 'contract', 'afn', 'reassoc'}


def izhikevich_grid(v_min, v_peak, dv, anchors, dv_coarse=0.5, fine_range=None, growth=1.05):
    """
    non-uniform voltage grid for the Izhikevich FPE: uniform with spacing dv in fine_range = (lo, hi), outside the
    spacing grows geometrically by the factor growth up to dv_coarse towards v_min and v_peak.

    Reason (numerical diffusion of the Scharfetter-Gummel flux): a density packet crossing a cell of width h with speed
    |mu| gets the voltage variance ~h^2 per cell, i.e. a timing variance ~h^2 / mu^2. For a given number of grid points
    the timing error is smallest for h ~ |mu|: fine where the neurons are slow (around v_r, v_t and the reset, where the
    population spends its time), coarse in the fast upswing to v_peak and far below rest.

    :param anchors: voltages that have to be grid points (reset potentials c), all inside fine_range and on the lattice
        of the first anchor with spacing dv
    :return: grid points (increasing, v[0] = v_min, v[-1] = v_peak)
    """
    anchors = np.atleast_1d(np.asarray(anchors, dtype=float))
    c0 = anchors[0]
    if fine_range is None:
        raise ValueError('fine_range has to be given')
    lo, hi = fine_range
    # fine region on the lattice c0 + n dv, containing all anchors
    lo = c0 - np.ceil((c0 - min(lo, anchors.min())) / dv - 1e-9) * dv
    hi = c0 + np.ceil((max(hi, anchors.max()) - c0) / dv - 1e-9) * dv
    lo, hi = max(lo, v_min), min(hi, v_peak)
    n_fine = int(round((hi - lo) / dv))
    fine = lo + dv * np.arange(n_fine + 1)
    for c_ in anchors:
        if not np.isclose(fine, c_).any():
            raise ValueError(f'reset potential {c_} is not on the fine lattice (dv = {dv}, anchor {c0})')

    def graded(start, end):
        # points from start (excluded) to end (included) with growing spacing, rescaled to end exactly at end
        if np.isclose(start, end):
            return np.zeros(0)
        direction = np.sign(end - start)
        pts, x, h = [], start, dv
        while (end - x) * direction > 1e-12:
            h = min(h * growth, dv_coarse)
            x = x + direction * h
            pts.append(x)
        pts = np.array(pts)
        return start + (pts - start) * (end - start) / (pts[-1] - start)

    v = np.concatenate([graded(lo, v_min)[::-1], fine, graded(hi, v_peak)])
    return v


@njit(cache=True, fastmath=_FASTMATH)
def _sg_pair(z, harmonic):
    """Scharfetter-Gummel weights (g_plus, g_minus) for the potential difference z = U_{i+1} - U_i"""
    if harmonic:
        if z >= 0:
            e = np.exp(-z)
            return 2 * e / (1 + e), 2 / (1 + e)
        e = np.exp(z)
        return 2 / (1 + e), 2 * e / (1 + e)
    za = abs(z)
    if za < 1e-8:
        bs = 1 - za / 2
    elif za > 700:
        bs = 0.0
    else:
        bs = za / np.expm1(za)
    bl = bs + za  # B(-z) = B(z) + z
    if z >= 0:
        return bs, bl
    return bl, bs


@njit(cache=True, fastmath=_FASTMATH)
def _thomas2(main, lower, upper, b1, b2, x1, x2, cp):
    """tridiagonal solve (no pivoting, diagonally dominant M-matrix) for two right hand sides,
    lower[k] = A[k + 1, k], upper[k] = A[k, k + 1]"""
    n = main.shape[0]
    inv = 1.0 / main[0]
    cp[0] = upper[0] * inv
    x1[0] = b1[0] * inv
    x2[0] = b2[0] * inv
    for i in range(1, n):
        inv = 1.0 / (main[i] - lower[i - 1] * cp[i - 1])
        if i < n - 1:
            cp[i] = upper[i] * inv
        x1[i] = (b1[i] - lower[i - 1] * x1[i - 1]) * inv
        x2[i] = (b2[i] - lower[i - 1] * x2[i - 1]) * inv
    for i in range(n - 2, -1, -1):
        x1[i] -= cp[i] * x1[i + 1]
        x2[i] -= cp[i] * x2[i + 1]


@njit(cache=True, fastmath=_FASTMATH)
def _iz_density_step(p_old, p_prev, has_prev, v, h, w, quad, k, C, i_total, D, dt, conditional, u_scalar,
                     u_last, u_prev, has_u_prev, reset_idx, harmonic, g_plus, g_minus, main, lower, upper, b_rhs,
                     s_vec, y, z, cp, p_new):
    """
    one implicit step of the voltage density of an Izhikevich population (finite volumes on the grid v with cell widths
    h and control volumes w, Scharfetter-Gummel flux, BDF2 with backward Euler fallback, reinjection at v[reset_idx]
    by Sherman-Morrison, clipping of negative values)
    :return: r, alpha_t (1.5 BDF2, 1 backward Euler), fallback (0/1), clipped (0/1), coef, mass, mean voltage
    main, lower, upper and s_vec are left set for the adaptation step
    """
    N = v.shape[0]
    # SG weights from the cell averaged drift, the adaptation linearly extrapolated to the new time step
    for i in range(N - 1):
        if conditional:
            if has_u_prev:
                us = 0.5 * ((2 * u_last[i] - u_prev[i]) + (2 * u_last[i + 1] - u_prev[i + 1]))
            else:
                us = 0.5 * (u_last[i] + u_last[i + 1])
        else:
            us = u_scalar
        drift = (k * quad[i] - us + i_total) / C
        g_plus[i], g_minus[i] = _sg_pair(-h[i] * drift / D, harmonic)
    p_old[0] = 0.0
    p_old[N - 1] = 0.0
    for i in range(1, N - 1):
        lower[i - 1] = -dt * D / (w[i] * h[i - 1]) * g_plus[i - 1]
        upper[i] = -dt * D / (w[i] * h[i]) * g_minus[i]
    upper[0] = 0.0
    lower[N - 2] = 0.0
    main[0] = 1.0
    main[N - 1] = 1.0
    coef = D / h[N - 2] * g_plus[N - 2]
    for i in range(N):
        s_vec[i] = 0.0
    s_vec[reset_idx] = 1.0 / w[reset_idx]

    use_bdf2 = has_prev
    fallback = 0
    r = 0.0
    alpha = 1.0
    mn = 0.0
    mx = 0.0
    while True:
        alpha = 1.5 if use_bdf2 else 1.0
        for i in range(1, N - 1):
            main[i] = alpha + dt * D / w[i] * (g_plus[i] / h[i] + g_minus[i - 1] / h[i - 1])
            b_rhs[i] = 2 * p_old[i] - 0.5 * p_prev[i] if use_bdf2 else p_old[i]
        b_rhs[0] = 0.0
        b_rhs[N - 1] = 0.0
        _thomas2(main, lower, upper, b_rhs, s_vec, y, z, cp)
        r = coef * y[N - 2] / (1 - dt * coef * z[N - 2])
        mn = np.inf
        mx = -np.inf
        for i in range(N):
            p_new[i] = y[i] + dt * r * z[i]
            mn = min(mn, p_new[i])
            mx = max(mx, p_new[i])
        if use_bdf2 and mn < -1e-6 * mx:
            use_bdf2 = False
            fallback = 1
            continue
        break

    clipped = 0
    if mn < 0:
        if mn < -1e-6 * mx:
            clipped = 1
        mass_before = 0.0
        for i in range(N):
            mass_before += w[i] * p_new[i]
        mass_after = 0.0
        for i in range(N):
            if p_new[i] < 0:
                p_new[i] = 0.0
            mass_after += w[i] * p_new[i]
        for i in range(N):
            p_new[i] *= mass_before / mass_after
    mass = 0.0
    vm = 0.0
    for i in range(N):
        mass += w[i] * p_new[i]
        vm += w[i] * v[i] * p_new[i]
    return r, alpha, fallback, clipped, coef, mass, vm / mass


@njit(cache=True, fastmath=_FASTMATH)
def _iz_q_step(q_old, q_prev, use_bdf2, p_new, v, h, w, main, lower, upper, s_vec, dt, a, b, v_r, d, r, coef, s2,
               C, extra_flux, main_q, b_q, y, z, cp, q_new):
    """
    one implicit step of q = p E[u | v] (conditional closures), same SG operator as p (main, lower, upper of
    _iz_density_step) plus the relaxation a on the diagonal, reinjection of the outflow of q plus d r at the reset
    (Sherman-Morrison), with extra_flux the additional flux -Var(u | v) p / C (explicit, Var(u | v) = s2 per grid
    point: constant for 'conditional_variance', a field for 'conditional_second_moment')
    :return: outflow of q at v_peak (r * mean u of the spiking neurons)
    """
    N = v.shape[0]
    q_old[0] = 0.0
    q_old[N - 1] = 0.0
    for i in range(N):
        b_q[i] = (2 * q_old[i] - 0.5 * q_prev[i] if use_bdf2 else q_old[i]) + dt * a * b * (v[i] - v_r) * p_new[i]
    q_out_extra = 0.0
    if extra_flux:
        flux_left = 0.0  # no flux through the lower boundary
        for i in range(1, N - 1):
            flux_right = -0.5 * (s2[i] * p_new[i] + s2[i + 1] * p_new[i + 1]) / C
            b_q[i] -= dt * (flux_right - flux_left) / w[i]
            flux_left = flux_right
        q_out_extra = -0.5 * (s2[N - 2] * p_new[N - 2] + s2[N - 1] * p_new[N - 1]) / C
    b_q[0] = 0.0
    b_q[N - 1] = 0.0
    for i in range(N):
        main_q[i] = main[i]
    for i in range(1, N - 1):
        main_q[i] += dt * a
    _thomas2(main_q, lower, upper, b_q, s_vec, y, z, cp)
    reinjection = (coef * y[N - 2] + q_out_extra + d * r) / (1 - dt * coef * z[N - 2])
    for i in range(N):
        q_new[i] = y[i] + dt * reinjection * z[i]
    return reinjection - d * r


@njit(cache=True, fastmath=_FASTMATH)
def _iz_q2_step(q2_old, q2_prev, use_bdf2, q_new, p_new, u_cond, s2, v, h, w, main, lower, upper, s_vec, dt, a, b,
                v_r, d, r, q_out, coef, C, main_q, b_q, y, z, cp, q2_new):
    """
    one implicit step of q2 = p E[u^2 | v] ('conditional_second_moment'):
    dq2/dt = -d/dv [mu q2 - D dq2/dv - 2 E[u | v] Var(u | v) p / C] + 2 a (b (v - v_r) q - q2)
             + delta(v - c) (outflow of q2 + 2 d outflow of q + d^2 r)
    (exact up to the Gaussian closure of the third moment, E[u^3 | v] = m^3 + 3 m s^2), same SG operator as p plus 2 a
    on the diagonal, the extra flux explicit (E[u | v], Var(u | v) of the previous step, extrapolated)
    """
    N = v.shape[0]
    q2_old[0] = 0.0
    q2_old[N - 1] = 0.0
    for i in range(N):
        b_q[i] = (2 * q2_old[i] - 0.5 * q2_prev[i] if use_bdf2 else q2_old[i]) + \
                 dt * 2 * a * b * (v[i] - v_r) * q_new[i]
    flux_left = 0.0
    for i in range(1, N - 1):
        flux_right = -(u_cond[i] * s2[i] * p_new[i] + u_cond[i + 1] * s2[i + 1] * p_new[i + 1]) / C
        b_q[i] -= dt * (flux_right - flux_left) / w[i]
        flux_left = flux_right
    out_extra = -(u_cond[N - 2] * s2[N - 2] * p_new[N - 2] + u_cond[N - 1] * s2[N - 1] * p_new[N - 1]) / C
    b_q[0] = 0.0
    b_q[N - 1] = 0.0
    for i in range(N):
        main_q[i] = main[i]
    for i in range(1, N - 1):
        main_q[i] += 2 * dt * a
    _thomas2(main_q, lower, upper, b_q, s_vec, y, z, cp)
    reinjection = (coef * y[N - 2] + out_extra + 2 * d * q_out + d * d * r) / (1 - dt * coef * z[N - 2])
    for i in range(N):
        q2_new[i] = y[i] + dt * reinjection * z[i]


@njit(cache=True, fastmath=_FASTMATH)
def _iz_moment_step(qk_old, qk_prev, use_bdf2, src, extra, v, h, w, main, lower, upper, s_vec, dt, ka, b, v_r, coef,
                    jump, main_q, b_q, y, z, cp, qk_new):
    """
    one implicit step of q_k = p E[u^k | v] for k >= 2 ('conditional_quadrature'):
    dq_k/dt = -d/dv [mu q_k - D dq_k/dv + F_k] + k a (b (v - v_r) q_{k-1} - q_k) + delta(v - c) (J_{q_k} + jump)
    :param src: q_{k-1} at the new time step, ka = k * a
    :param extra: extra flux density per grid point, F_k at an interface = mean of the two values (explicit)
    :param jump: reinjected moment from the jumps u -> u + d: sum_{j<k} binom(k, j) d^(k-j) J_{q_j}
    :return: outflow J_{q_k} at v_peak
    """
    N = v.shape[0]
    qk_old[0] = 0.0
    qk_old[N - 1] = 0.0
    for i in range(N):
        b_q[i] = (2 * qk_old[i] - 0.5 * qk_prev[i] if use_bdf2 else qk_old[i]) + dt * ka * b * (v[i] - v_r) * src[i]
    flux_left = 0.0
    for i in range(1, N - 1):
        flux_right = 0.5 * (extra[i] + extra[i + 1])
        b_q[i] -= dt * (flux_right - flux_left) / w[i]
        flux_left = flux_right
    out_extra = 0.5 * (extra[N - 2] + extra[N - 1])
    b_q[0] = 0.0
    b_q[N - 1] = 0.0
    for i in range(N):
        main_q[i] = main[i]
    for i in range(1, N - 1):
        main_q[i] += dt * ka
    _thomas2(main_q, lower, upper, b_q, s_vec, y, z, cp)
    reinjection = (coef * y[N - 2] + out_extra + jump) / (1 - dt * coef * z[N - 2])
    for i in range(N):
        qk_new[i] = y[i] + dt * reinjection * z[i]
    return reinjection - jump


@njit(cache=True, fastmath=_FASTMATH)
def _iz_moments(p, q, q2, has_q2, v, w, v_r, floor_rel, u_cond, s2):
    """
    mass, <v>, <u>, population mean of E[u | v]^2 - <u>^2, <(v - v_r) u>, <u^2> (from q2) and population mean of
    Var(u | v) (from q2); fills u_cond = E[u | v] and, with q2, s2 = Var(u | v) (both blended towards the population
    values where p is very small: (x + eps X) / (p + eps), eps = floor_rel * max(p))
    """
    N = v.shape[0]
    mass = 0.0
    vm = 0.0
    um = 0.0
    u2m = 0.0
    p_max = 0.0
    for i in range(N):
        mass += w[i] * p[i]
        vm += w[i] * v[i] * p[i]
        um += w[i] * q[i]
        if has_q2:
            u2m += w[i] * q2[i]
        p_max = max(p_max, p[i])
    vm /= mass
    um /= mass
    u2m /= mass
    floor = floor_rel * p_max
    var_cond = 0.0
    uv = 0.0
    sig = 0.0
    for i in range(N):
        u_cond[i] = (q[i] + floor * um) / (p[i] + floor)
        var_cond += w[i] * p[i] * u_cond[i] ** 2
        uv += w[i] * (v[i] - v_r) * q[i]
        if has_q2:
            s2[i] = max((q2[i] + floor * u2m) / (p[i] + floor) - u_cond[i] ** 2, 0.0)
            sig += w[i] * p[i] * s2[i]
    return mass, vm, um, var_cond / mass - um ** 2, uv / mass, u2m, sig / mass


@njit(cache=True, fastmath=_FASTMATH)
def _fct_correct(fields, g_plus, g_minus, h, w, D, dt_eff, alpha, A, Rp, Rm, flux, low, ratio_eps):
    """
    flux-corrected transport (Zalesak 1979, as explicit correction of the implicit low order step, Kuzmin) for the
    Scharfetter-Gummel flux: the antidiffusive flux A_{k+1/2} = F_central - F_SG = (D / h_k) (P coth P - 1)
    (x_{k+1} - x_k), P coth P = (g_plus + g_minus) / 2, is added with the limiter alpha in [0, 1] such that no field
    gets new local extrema (bounds from the low order solution of the neighbours, keeps p >= 0). The limiter is the
    minimum over all fields (rows of fields, synchronized limiting) and is 0 at the first and last interface (boundary
    values, firing rate and reinjection stay those of the low order step). Flux form: the integrals sum(w * x) are
    conserved.
    Failsafe for the moment fields (rows 1, ... = p E[u^k | v]): where the corrected ratio x_f / p leaves the range of
    the low order ratios of the neighbours (only points with p > ratio_eps * max(p)), the correction is switched off
    at both interfaces of the point and the correction is repeated (at most 4 times, then low order).
    :param fields: (n_fields, N) low order solutions (row 0: p), corrected in place
    :param dt_eff: dt / alpha_t (alpha_t = 1.5 for BDF2, 1 for backward Euler)
    :param low: work array of the shape of fields
    """
    nf, N = fields.shape
    for f in range(nf):
        for i in range(N):
            low[f, i] = fields[f, i]
    for k in range(N - 1):
        alpha[k] = 1.0
    alpha[0] = 0.0
    alpha[N - 2] = 0.0
    for f in range(nf):
        x = low[f]
        for k in range(N - 1):
            A[k] = D / h[k] * (0.5 * (g_plus[k] + g_minus[k]) - 1.0) * (x[k + 1] - x[k])
        Rp[0] = 0.0
        Rm[0] = 0.0
        Rp[N - 1] = 0.0
        Rm[N - 1] = 0.0
        for i in range(1, N - 1):
            x_max = max(x[i - 1], max(x[i], x[i + 1]))
            x_min = min(x[i - 1], min(x[i], x[i + 1]))
            p_plus = max(0.0, A[i - 1]) + max(0.0, -A[i])
            p_minus = min(0.0, A[i - 1]) + min(0.0, -A[i])
            q_plus = (x_max - x[i]) * w[i] / dt_eff
            q_minus = (x_min - x[i]) * w[i] / dt_eff
            Rp[i] = 1.0 if p_plus <= 0 else min(1.0, q_plus / p_plus)
            Rm[i] = 1.0 if p_minus >= 0 else min(1.0, q_minus / p_minus)
        for k in range(1, N - 2):
            if A[k] >= 0:
                lim = min(Rm[k], Rp[k + 1])  # antidiffusive flux from k to k + 1
            else:
                lim = min(Rp[k], Rm[k + 1])
            alpha[k] = min(alpha[k], lim)

    p_max = 0.0
    for i in range(N):
        p_max = max(p_max, low[0, i])
    thr = ratio_eps * p_max
    for iteration in range(5):
        if iteration == 4:
            for k in range(N - 1):
                alpha[k] = 0.0
        for f in range(nf):
            x = low[f]
            for k in range(N - 1):
                flux[k] = alpha[k] * D / h[k] * (0.5 * (g_plus[k] + g_minus[k]) - 1.0) * (x[k + 1] - x[k])
            for i in range(N):
                fields[f, i] = x[i]
            for i in range(1, N - 1):
                fields[f, i] += dt_eff / w[i] * (flux[i - 1] - flux[i])
        if nf == 1 or iteration == 4:
            break
        violated = False
        for i in range(1, N - 1):
            if fields[0, i] <= thr:
                continue
            for f in range(1, nf):
                r_min = np.inf
                r_max = -np.inf
                for jj in range(i - 1, i + 2):
                    if low[0, jj] > thr:
                        ratio = low[f, jj] / low[0, jj]
                        r_min = min(r_min, ratio)
                        r_max = max(r_max, ratio)
                if r_min > r_max:
                    continue
                ratio = fields[f, i] / fields[0, i]
                tol = 1e-9 * (abs(r_min) + abs(r_max) + 1.0)
                if ratio > r_max + tol or ratio < r_min - tol:
                    alpha[i - 1] = 0.0
                    alpha[i] = 0.0
                    violated = True
        if not violated:
            break



########################################################################################################################
# numpy / scipy versions of the kernels above (use_numba=False or numba not installed): same arithmetic, vectorized
########################################################################################################################
def _solve_banded2(main, lower, upper, b1, b2):
    """tridiagonal solve for two right hand sides with the banded LU of scipy, returns (x1, x2)"""
    ab = np.zeros((3, main.shape[0]))
    ab[0, 1:] = upper
    ab[1] = main
    ab[2, :-1] = lower
    x = scipy.linalg.solve_banded((1, 1), ab, np.column_stack([b1, b2]), check_finite=False)
    return x[:, 0], x[:, 1]


def _sg_weights_np(dU, harmonic):
    """vectorized _sg_pair: (g_plus, g_minus) for the potential differences dU"""
    if harmonic:
        return 2 * scipy.special.expit(-dU), 2 * scipy.special.expit(dU)
    za = np.abs(dU)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        bs = za / np.expm1(za)
    bs = np.where(za < 1e-8, 1 - za / 2, bs)
    bs = np.where(za > 700, 0.0, bs)
    bl = bs + za
    positive = dU >= 0
    return np.where(positive, bs, bl), np.where(positive, bl, bs)


def _iz_density_step_np(p_old, p_prev, has_prev, v, h, w, quad, k, C, i_total, D, dt, conditional, u_scalar,
                        u_last, u_prev, has_u_prev, reset_idx, harmonic, g_plus, g_minus, main, lower, upper, b_rhs,
                        s_vec, y, z, cp, p_new):
    """numpy version of _iz_density_step (same arguments and return values)"""
    N = v.shape[0]
    if conditional:
        u = 2 * u_last - u_prev if has_u_prev else u_last
        us = 0.5 * (u[1:] + u[:-1])
    else:
        us = u_scalar
    drift = (k * quad - us + i_total) / C
    g_plus[:], g_minus[:] = _sg_weights_np(-h * drift / D, harmonic)
    p_old[0] = 0.0
    p_old[N - 1] = 0.0
    lower[:-1] = -dt * D / (w[1:-1] * h[:-1]) * g_plus[:-1]
    lower[-1] = 0.0
    upper[1:] = -dt * D / (w[1:-1] * h[1:]) * g_minus[1:]
    upper[0] = 0.0
    main[0] = main[-1] = 1.0
    coef = D / h[-1] * g_plus[-1]
    s_vec[:] = 0.0
    s_vec[reset_idx] = 1.0 / w[reset_idx]
    use_bdf2 = has_prev
    fallback = 0
    while True:
        alpha = 1.5 if use_bdf2 else 1.0
        main[1:-1] = alpha + dt * D / w[1:-1] * (g_plus[1:] / h[1:] + g_minus[:-1] / h[:-1])
        b_rhs[:] = 2 * p_old - 0.5 * p_prev if use_bdf2 else p_old
        b_rhs[0] = b_rhs[-1] = 0.0
        y[:], z[:] = _solve_banded2(main, lower, upper, b_rhs, s_vec)
        r = coef * y[-2] / (1 - dt * coef * z[-2])
        p_new[:] = y + dt * r * z
        mn, mx = p_new.min(), p_new.max()
        if use_bdf2 and mn < -1e-6 * mx:
            use_bdf2 = False
            fallback = 1
            continue
        break
    clipped = 0
    if mn < 0:
        clipped = int(mn < -1e-6 * mx)
        mass_before = np.dot(w, p_new)
        p_new[p_new < 0] = 0.0
        p_new *= mass_before / np.dot(w, p_new)
    mass = np.dot(w, p_new)
    return r, alpha, fallback, clipped, coef, mass, np.dot(w * v, p_new) / mass


def _iz_q_step_np(q_old, q_prev, use_bdf2, p_new, v, h, w, main, lower, upper, s_vec, dt, a, b, v_r, d, r, coef, s2,
                  C, extra_flux, main_q, b_q, y, z, cp, q_new):
    """numpy version of _iz_q_step"""
    q_old[0] = q_old[-1] = 0.0
    b_q[:] = (2 * q_old - 0.5 * q_prev if use_bdf2 else q_old) + dt * a * b * (v - v_r) * p_new
    out_extra = 0.0
    if extra_flux:
        flux = -0.5 * (s2[:-1] * p_new[:-1] + s2[1:] * p_new[1:]) / C
        out_extra = flux[-1]
        flux[0] = 0.0
        b_q[1:-1] -= dt * (flux[1:] - flux[:-1]) / w[1:-1]
    b_q[0] = b_q[-1] = 0.0
    main_q[:] = main
    main_q[1:-1] += dt * a
    y[:], z[:] = _solve_banded2(main_q, lower, upper, b_q, s_vec)
    reinjection = (coef * y[-2] + out_extra + d * r) / (1 - dt * coef * z[-2])
    q_new[:] = y + dt * reinjection * z
    return reinjection - d * r


def _iz_q2_step_np(q2_old, q2_prev, use_bdf2, q_new, p_new, u_cond, s2, v, h, w, main, lower, upper, s_vec, dt, a, b,
                   v_r, d, r, q_out, coef, C, main_q, b_q, y, z, cp, q2_new):
    """numpy version of _iz_q2_step"""
    q2_old[0] = q2_old[-1] = 0.0
    b_q[:] = (2 * q2_old - 0.5 * q2_prev if use_bdf2 else q2_old) + dt * 2 * a * b * (v - v_r) * q_new
    f = u_cond * s2 * p_new
    flux = -(f[:-1] + f[1:]) / C
    out_extra = flux[-1]
    flux[0] = 0.0
    b_q[1:-1] -= dt * (flux[1:] - flux[:-1]) / w[1:-1]
    b_q[0] = b_q[-1] = 0.0
    main_q[:] = main
    main_q[1:-1] += 2 * dt * a
    y[:], z[:] = _solve_banded2(main_q, lower, upper, b_q, s_vec)
    reinjection = (coef * y[-2] + out_extra + 2 * d * q_out + d * d * r) / (1 - dt * coef * z[-2])
    q2_new[:] = y + dt * reinjection * z


def _iz_moment_step_np(qk_old, qk_prev, use_bdf2, src, extra, v, h, w, main, lower, upper, s_vec, dt, ka, b, v_r, coef,
                       jump, main_q, b_q, y, z, cp, qk_new):
    """numpy version of _iz_moment_step"""
    qk_old[0] = qk_old[-1] = 0.0
    b_q[:] = (2 * qk_old - 0.5 * qk_prev if use_bdf2 else qk_old) + dt * ka * b * (v - v_r) * src
    flux = 0.5 * (extra[:-1] + extra[1:])
    out_extra = flux[-1]
    flux[0] = 0.0
    b_q[1:-1] -= dt * (flux[1:] - flux[:-1]) / w[1:-1]
    b_q[0] = b_q[-1] = 0.0
    main_q[:] = main
    main_q[1:-1] += dt * ka
    y[:], z[:] = _solve_banded2(main_q, lower, upper, b_q, s_vec)
    reinjection = (coef * y[-2] + out_extra + jump) / (1 - dt * coef * z[-2])
    qk_new[:] = y + dt * reinjection * z
    return reinjection - jump


def _iz_moments_np(p, q, q2, has_q2, v, w, v_r, floor_rel, u_cond, s2):
    """numpy version of _iz_moments"""
    mass = np.dot(w, p)
    vm = np.dot(w * v, p) / mass
    um = np.dot(w, q) / mass
    u2m = np.dot(w, q2) / mass if has_q2 else 0.0
    floor = floor_rel * p.max()
    u_cond[:] = (q + floor * um) / (p + floor)
    var_cond = np.dot(w * p, u_cond**2) / mass - um**2
    uv = np.dot(w * (v - v_r), q) / mass
    sig = 0.0
    if has_q2:
        s2[:] = np.maximum((q2 + floor * u2m) / (p + floor) - u_cond**2, 0.0)
        sig = np.dot(w * p, s2) / mass
    return mass, vm, um, var_cond, uv, u2m, sig


def _fct_correct_np(fields, g_plus, g_minus, h, w, D, dt_eff, alpha, A, Rp, Rm, flux, low, ratio_eps):
    """numpy version of _fct_correct (Zalesak limiter, synchronized, failsafe on the ratios)"""
    nf, N = fields.shape
    low[:] = fields
    fac = D / h * (0.5 * (g_plus + g_minus) - 1.0)
    alpha[:] = 1.0
    alpha[0] = alpha[-1] = 0.0
    inner = slice(1, N - 1)
    for f in range(nf):
        x = low[f]
        A[:] = fac * (x[1:] - x[:-1])
        x_max = np.maximum(np.maximum(x[:-2], x[1:-1]), x[2:])
        x_min = np.minimum(np.minimum(x[:-2], x[1:-1]), x[2:])
        p_plus = np.maximum(0.0, A[:-1]) + np.maximum(0.0, -A[1:])
        p_minus = np.minimum(0.0, A[:-1]) + np.minimum(0.0, -A[1:])
        q_plus = (x_max - x[1:-1]) * w[inner] / dt_eff
        q_minus = (x_min - x[1:-1]) * w[inner] / dt_eff
        Rp[:] = 0.0
        Rm[:] = 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            Rp[inner] = np.where(p_plus > 0, np.minimum(1.0, q_plus / p_plus), 1.0)
            Rm[inner] = np.where(p_minus < 0, np.minimum(1.0, q_minus / p_minus), 1.0)
        k = slice(1, N - 2)
        lim = np.where(A[k] >= 0, np.minimum(Rm[1:N - 2], Rp[2:N - 1]), np.minimum(Rp[1:N - 2], Rm[2:N - 1]))
        alpha[k] = np.minimum(alpha[k], lim)
    thr = ratio_eps * low[0].max()
    for iteration in range(5):
        if iteration == 4:
            alpha[:] = 0.0
        for f in range(nf):
            x = low[f]
            flux[:] = alpha * fac * (x[1:] - x[:-1])
            fields[f] = x
            fields[f, 1:-1] += dt_eff / w[inner] * (flux[:-1] - flux[1:])
        if nf == 1 or iteration == 4:
            break
        ok_low = low[0] > thr
        node = fields[0, 1:-1] > thr
        violated = np.zeros(N, dtype=bool)
        with np.errstate(divide='ignore', invalid='ignore'):
            for f in range(1, nf):
                ratio = np.where(ok_low, low[f] / np.where(ok_low, low[0], 1.0), np.nan)
                r_min = np.fmin(np.fmin(ratio[:-2], ratio[1:-1]), ratio[2:])
                r_max = np.fmax(np.fmax(ratio[:-2], ratio[1:-1]), ratio[2:])
                r_new = fields[f, 1:-1] / np.where(node, fields[0, 1:-1], 1.0)
                tol = 1e-9 * (np.abs(r_min) + np.abs(r_max) + 1.0)
                violated[1:-1] |= node & np.isfinite(r_min) & ((r_new > r_max + tol) | (r_new < r_min - tol))
        if not violated.any():
            break
        idx = np.nonzero(violated)[0]
        alpha[idx - 1] = 0.0
        alpha[idx] = 0.0


def _iz_simulate_set_batch(cfg, i_ext):
    """
    time loop of FPE_Population.simulate_set() for Izhikevich neurons: n_sim independent simulations of the coupled
    populations, i_ext (n_sim, n_pop, Nt) in pA. All n_sim * n_pop populations (blocks) are advanced together: the
    elementwise operations of a time step (input, extrapolation of the adaptation fields, closures, extra fluxes,
    moments of the jumps, updates of <u>) act on (n_blocks, Nx) arrays at once, the implicit solves, the FCT limiter
    and the moments are done block by block with the kernels of simulate() (same arithmetic as simulate())
    :param cfg: settings and per population parameters from FPE_Population._simulate_set_izhikevich
    :return: dict with r, u_mean, v_mean, sigma_u2, var_u, u2_mean (n_blocks, Nt), p, q, q2 (n_blocks, Nx, Nt_p) or
     None, n_clipped, n_fallback
    """
    if cfg['numba']:
        density_step, q_step, q2_step = _iz_density_step, _iz_q_step, _iz_q2_step
        moments, fct_correct, moment_step = _iz_moments, _fct_correct, _iz_moment_step
    else:
        density_step, q_step, q2_step = _iz_density_step_np, _iz_q_step_np, _iz_q2_step_np
        moments, fct_correct, moment_step = _iz_moments_np, _fct_correct_np, _iz_moment_step_np
    n_sim, n_pop, Nt = i_ext.shape
    B = n_sim * n_pop  # block j of simulation s: s * n_pop + j
    v, h, w, dt = cfg['v'], cfg['h'], cfg['w'], cfg['dt']
    Nx = v.shape[0]
    conditional, with_variance, second_moment, quadrature = cfg['flags']
    harmonic, fct, n_corrector, floor_rel = cfg['harmonic'], cfg['fct'], cfg['n_corrector'], cfg['u_cond_floor']
    save_p, p_save_every = cfg['save_p'], cfg['p_save_every']
    W, kernel = cfg['connectivity_matrix'], cfg['delay_kernel']
    n_k_max = len(kernel)

    # parameters per block
    C_m, k_iz, v_r, a, b, d, tau_noise = [np.tile(cfg[name], n_sim) for name in
                                          ['C_m', 'k_iz', 'v_r', 'a', 'b', 'd', 'tau_noise']]
    reset_idx = np.tile(cfg['reset_idx'], n_sim)
    quad = np.tile(cfg['quad_mean'], (n_sim, 1))
    sigma = np.tile(cfg['sigma'], (n_sim, 1))
    i_ext = i_ext.reshape(B, Nt)

    Nt_p = len(range(0, Nt, p_save_every)) if save_p == 'all' else 1
    p = np.zeros((B, Nx, Nt_p))
    q = np.zeros((B, Nx, Nt_p)) if conditional else None
    q2 = np.zeros((B, Nx, Nt_p)) if second_moment else None
    r, u_mean, v_mean, sigma_u2, var_u, u2_mean = [np.zeros((B, Nt)) for _ in range(6)]

    # initial conditions (the same for every simulation)
    p_last = np.tile(cfg['p0'], (n_sim, 1))
    u0 = np.tile(cfg['u_mean0'], n_sim)
    v_mean[:, 0] = np.tile(cfg['v_mean0'], n_sim)
    u_mean[:, 0] = u0
    u2_mean[:, 0] = u0**2
    q_last = u0[:, np.newaxis] * p_last
    q2_last = u0[:, np.newaxis]**2 * p_last
    q3_last = u0[:, np.newaxis]**3 * p_last
    u_cond_last = np.repeat(u0[:, np.newaxis], Nx, axis=1)
    c3_last, s2_last = np.zeros((B, Nx)), np.zeros((B, Nx))
    p_prev = q_prev = q2_prev = q3_prev = u_cond_prev = c3_prev = s2_prev = None
    p[:, :, 0] = p_last
    if conditional:
        q[:, :, 0] = q_last
    if second_moment:
        q2[:, :, 0] = q2_last
    n_clipped = 0
    n_fallback = 0

    # work arrays: per block where the step of the density leaves them set for the adaptation fields, else shared
    n_fields = 1 + conditional + second_moment + quadrature
    g_plus, g_minus, lower, upper = [np.zeros((B, Nx - 1)) for _ in range(4)]
    main, s_vec = np.zeros((B, Nx)), np.zeros((B, Nx))
    main_q, b_rhs, b_q, y, z, cp = [np.zeros(Nx) for _ in range(6)]
    fct_work = [np.zeros(Nx - 1), np.zeros(Nx - 1), np.zeros(Nx), np.zeros(Nx), np.zeros(Nx - 1)]
    fct_low = np.empty((n_fields, Nx))
    s2_zero = np.zeros((B, Nx))
    blocks = range(B)

    for i in tqdm(range(1, Nt), f'simulating {n_sim} simulations of {n_pop} Izhikevich populations '
                                f'({"" if n_sim == 1 else "set, "}{B} blocks) for {Nt - 1} time steps',
                  disable=cfg['tqdm_disable']):
        # delayed recurrent input from the rates of the previous steps, coupling within every simulation
        n_k = min(i + 1, n_k_max)
        r_conv = r[:, i - n_k + 1:i + 1][:, ::-1] @ kernel[:n_k] * dt
        i_total = i_ext[:, i] + (r_conv.reshape(n_sim, n_pop) @ W).reshape(-1)
        D = np.maximum(sigma[:, i]**2 / tau_noise, 1e-10)
        has_prev = p_prev is not None
        p_old, q_old, q2_old, q3_old = p_last, q_last, q2_last, q3_last

        # predictor (pass 0): adaptation in the drift and in the extra fluxes linearly extrapolated to t_i,
        # corrector passes: the adaptation fields of the previous pass at t_i (as in simulate())
        for it in range(1 + n_corrector):
            if it == 0:
                u_scalar = 2 * u_mean[:, i - 1] - u_mean[:, i - 2] if i >= 2 else u_mean[:, i - 1].copy()
                has_u_prev = u_cond_prev is not None
                u_src_last = u_cond_last
                u_src_prev = u_cond_prev if has_u_prev else u_cond_last
                u_ex = 2 * u_cond_last - u_cond_prev if has_u_prev else u_cond_last
                if with_variance:
                    s2_star = 2 * sigma_u2[:, i - 1] - sigma_u2[:, i - 2] if i >= 2 else sigma_u2[:, i - 1]
                    s2_ex = np.repeat(np.maximum(s2_star, 0.)[:, np.newaxis], Nx, axis=1)
                elif second_moment:
                    s2_ex = np.maximum(2 * s2_last - s2_prev, 0.) if s2_prev is not None else s2_last
                else:
                    s2_ex = s2_zero
                if quadrature:
                    c3_ex = 2 * c3_last - c3_prev if c3_prev is not None else c3_last
            else:
                u_scalar = u_mean[:, i].copy()
                has_u_prev = False
                if conditional:
                    u_src_last = u_src_prev = u_ex = u_cond
                if with_variance:
                    s2_ex = np.repeat(sigma_u2[:, i, np.newaxis], Nx, axis=1)
                elif second_moment:
                    s2_ex = s2_new
                if quadrature:
                    c3_ex = c3_new

            # density: BDF2 with backward Euler fallback, reinjection by Sherman-Morrison (per block)
            p_new = np.empty((B, Nx))
            alpha_t, coef, mass = np.empty(B), np.empty(B), np.empty(B)
            for bk in blocks:
                r[bk, i], alpha_t[bk], fallback, clipped, coef[bk], mass[bk], v_mean[bk, i] = density_step(
                    p_old[bk], p_prev[bk] if has_prev else p_old[bk], has_prev, v, h, w, quad[bk], k_iz[bk], C_m[bk],
                    i_total[bk], D[bk], dt, conditional, u_scalar[bk], u_src_last[bk], u_src_prev[bk], has_u_prev,
                    reset_idx[bk], harmonic, g_plus[bk], g_minus[bk], main[bk], lower[bk], upper[bk], b_rhs,
                    s_vec[bk], y, z, cp, p_new[bk])
                n_fallback += fallback
                n_clipped += clipped
            use_bdf2 = alpha_t == 1.5
            u_old1 = u_mean[:, i - 1]
            b_u = np.where(use_bdf2, 2 * u_old1 - 0.5 * u_mean[:, i - 2], u_old1)

            if conditional:
                # q = p E[u | v], with the extra flux -Var(u | v) p / C for the variance closures
                q_new = np.empty((B, Nx))
                q_out = np.empty(B)
                for bk in blocks:
                    q_out[bk] = q_step(q_old[bk], q_prev[bk] if use_bdf2[bk] else q_old[bk], use_bdf2[bk], p_new[bk],
                                       v, h, w, main[bk], lower[bk], upper[bk], s_vec[bk], dt, a[bk], b[bk], v_r[bk],
                                       d[bk], r[bk, i], coef[bk], s2_ex[bk], C_m[bk], with_variance or second_moment,
                                       main_q, b_q, y, z, cp, q_new[bk])
                if second_moment and not quadrature:
                    q2_new = np.empty((B, Nx))
                    for bk in blocks:
                        q2_step(q2_old[bk], q2_prev[bk] if use_bdf2[bk] else q2_old[bk], use_bdf2[bk], q_new[bk],
                                p_new[bk], u_ex[bk], s2_ex[bk], v, h, w, main[bk], lower[bk], upper[bk], s_vec[bk], dt,
                                a[bk], b[bk], v_r[bk], d[bk], r[bk, i], q_out[bk], coef[bk], C_m[bk], main_q, b_q, y,
                                z, cp, q2_new[bk])
                elif quadrature:
                    # moments 2 and 3, two-node closure c4 = s2^2 + c3^2 / s2 (see simulate())
                    N_rate = r[:, i]
                    c3_lim = QUADRATURE_MAX_SKEWNESS * s2_ex**1.5
                    c3_ex = np.clip(c3_ex, -c3_lim, c3_lim)
                    c4_ex = s2_ex**2 + np.where(s2_ex > 0, c3_ex**2 / np.maximum(s2_ex, 1e-300), 0.)
                    C_col = C_m[:, np.newaxis]
                    extra2 = -(2 * u_ex * s2_ex + c3_ex) * p_new / C_col
                    jump2 = 2 * d * q_out + d**2 * N_rate
                    q2_new = np.empty((B, Nx))
                    q2_out = np.empty(B)
                    for bk in blocks:
                        q2_out[bk] = moment_step(
                            q2_old[bk], q2_prev[bk] if use_bdf2[bk] else q2_old[bk], use_bdf2[bk], q_new[bk],
                            extra2[bk], v, h, w, main[bk], lower[bk], upper[bk], s_vec[bk], dt, 2 * a[bk], b[bk],
                            v_r[bk], coef[bk], jump2[bk], main_q, b_q, y, z, cp, q2_new[bk])
                    extra3 = -(3 * u_ex**2 * s2_ex + 3 * u_ex * c3_ex + c4_ex) * p_new / C_col
                    jump3 = 3 * d * q2_out + 3 * d**2 * q_out + d**3 * N_rate
                    q3_new = np.empty((B, Nx))
                    for bk in blocks:
                        moment_step(q3_old[bk], q3_prev[bk] if use_bdf2[bk] else q3_old[bk], use_bdf2[bk], q2_new[bk],
                                    extra3[bk], v, h, w, main[bk], lower[bk], upper[bk], s_vec[bk], dt, 3 * a[bk],
                                    b[bk], v_r[bk], coef[bk], jump3[bk], main_q, b_q, y, z, cp, q3_new[bk])

            if fct:
                # flux-corrected transport, the same limiter for all fields of a block
                fields = np.empty((B, n_fields, Nx))
                fields[:, 0] = p_new
                if conditional:
                    fields[:, 1] = q_new
                if second_moment:
                    fields[:, 2] = q2_new
                if quadrature:
                    fields[:, 3] = q3_new
                for bk in blocks:
                    fct_correct(fields[bk], g_plus[bk], g_minus[bk], h, w, D[bk], dt / alpha_t[bk], *fct_work, fct_low,
                                floor_rel)
                p_new = fields[:, 0].copy()
                if conditional:
                    q_new = fields[:, 1].copy()
                if second_moment:
                    q2_new = fields[:, 2].copy()
                if quadrature:
                    q3_new = fields[:, 3].copy()
                v_mean[:, i] = np.sum(w * v * p_new, axis=1) / mass

            if not conditional:
                # mean adaptation, BDF2 (implicit in <u>, uses the new <v> and r)
                u_mean[:, i] = (b_u + dt * (a * b * (v_mean[:, i] - v_r) + d * r[:, i])) / (alpha_t + dt * a)
            else:
                u_cond, s2_new = np.empty((B, Nx)), np.empty((B, Nx))
                var_cond_mean, uv, u2_field, sig_field = np.empty(B), np.empty(B), np.empty(B), np.empty(B)
                q2_src = q2_new if second_moment else q_new
                for bk in blocks:
                    mass[bk], v_mean[bk, i], u_mean[bk, i], var_cond_mean[bk], uv[bk], u2_field[bk], sig_field[bk] = \
                        moments(p_new[bk], q_new[bk], q2_src[bk], second_moment, v, w, v_r[bk], floor_rel, u_cond[bk],
                                s2_new[bk])
                if with_variance:
                    # exact equation of <u^2> (BDF2, implicit in <u^2>), Var(u | v) = sigma_u2 for all v
                    u2_old1 = u2_mean[:, i - 1]
                    b_u2 = np.where(use_bdf2, 2 * u2_old1 - 0.5 * u2_mean[:, i - 2], u2_old1)
                    u2_mean[:, i] = (b_u2 + dt * (2 * a * b * uv + 2 * d * q_out + d**2 * r[:, i])) / \
                                    (alpha_t + 2 * dt * a)
                    sigma_u2[:, i] = np.maximum(u2_mean[:, i] - u_mean[:, i]**2 - var_cond_mean, 0.)
                    var_u[:, i] = sigma_u2[:, i] + var_cond_mean
                elif second_moment:
                    u2_mean[:, i] = u2_field
                    sigma_u2[:, i] = sig_field
                    var_u[:, i] = u2_field - u_mean[:, i]**2
                    if quadrature:
                        # third central moment E[(u - m)^3 | v] (regularized like E[u | v]), bounded skewness
                        floor = floor_rel * p_new.max(axis=1)[:, np.newaxis]
                        u3_mean = np.sum(w * q3_new, axis=1) / mass
                        e2 = (q2_new + floor * u2_field[:, np.newaxis]) / (p_new + floor)
                        e3 = (q3_new + floor * u3_mean[:, np.newaxis]) / (p_new + floor)
                        c3_new = e3 - 3 * u_cond * e2 + 2 * u_cond**3
                        c3_max = QUADRATURE_MAX_SKEWNESS * s2_new**1.5
                        c3_new = np.clip(c3_new, -c3_max, c3_max)
                else:
                    var_u[:, i] = var_cond_mean
                    u2_mean[:, i] = var_u[:, i] + u_mean[:, i]**2

        # state for the next time step
        save_now = save_p == 'all' and i % p_save_every == 0
        if conditional:
            if second_moment:
                q2_prev, q2_last = q2_old, q2_new
                s2_prev, s2_last = s2_last, s2_new
                if save_now:
                    q2[:, :, i // p_save_every] = q2_new
            if quadrature:
                q3_prev, q3_last = q3_old, q3_new
                c3_prev, c3_last = c3_last, c3_new
            q_prev, q_last = q_old, q_new
            u_cond_prev, u_cond_last = u_cond_last, u_cond
            if save_now:
                q[:, :, i // p_save_every] = q_new
        p_prev, p_last = p_old, p_new
        if save_now:
            p[:, :, i // p_save_every] = p_new

    if save_p == 'end':
        p[:, :, 0] = p_last
        if conditional:
            q[:, :, 0] = q_last
        if second_moment:
            q2[:, :, 0] = q2_last
    elif save_p == 'none':
        p = q = q2 = None
    return dict(r=r, u_mean=u_mean, v_mean=v_mean, sigma_u2=sigma_u2, var_u=var_u, u2_mean=u2_mean, p=p, q=q, q2=q2,
                n_clipped=n_clipped, n_fallback=n_fallback)


def external_input(input_function, input_function_idx, voltage_sigma, t, n_pop, static_noise=True):
    """
    External input of FPE_Population and FPE_mini (same parameter dict for both), one row per population

    input_function: callable of t or array, (Nt,): input into the populations input_function_idx (the others get 0),
        (n_pop, Nt): its own input for every population (input_function_idx is ignored)
    voltage_sigma: noise (mV), scalar, one value per population or (n_pop, Nt); static_noise=False: voltage_sigma times
        the input current (old behaviour of FPE_Population)
    :return: i_ext (n_pop, Nt) in the unit of input_function, sigma (n_pop, Nt) in mV
    """
    Nt = len(t)
    x = np.asarray(input_function(t) if callable(input_function) else input_function, dtype=float)
    i_ext = np.zeros((n_pop, Nt))
    if x.ndim == 2:
        assert x.shape == (n_pop, Nt), f'input_function has to be of shape {(Nt,)} or {(n_pop, Nt)}, got {x.shape}'
        i_ext[:] = x
    else:
        assert x.shape[0] == Nt, f'Dimension missmatch between input with dimension {x.shape[0]} and simulation ' \
                                 f'time with dimension {Nt}!'
        i_ext[input_function_idx] = x
    sigma = np.asarray(voltage_sigma, dtype=float)
    if sigma.ndim == 2:
        assert sigma.shape == (n_pop, Nt), f'voltage_sigma has to be of shape {(n_pop, Nt)}'
        return i_ext, sigma
    if sigma.size not in [1, n_pop]:
        raise ValueError(f'voltage_sigma has to be a scalar, have one value per population or be of shape '
                         f'{(n_pop, Nt)}')
    sigma = sigma.reshape(-1, 1)
    return i_ext, (sigma * i_ext if not static_noise else np.broadcast_to(sigma, (n_pop, Nt)).copy())


def initial_density(v, u_rest, offset, sigma, weight):
    """
    Initial density of FPE_Population and FPE_mini: Gaussian around u_rest + offset (width sigma) plus weight times a
    uniform density above u_rest, p = 0 at both ends of the grid, integral 1 (uniform grid v)
    """
    gauss = norm.pdf(v, v[np.where(v > u_rest + offset)[0][0] - 1], sigma)
    uniform = (v >= u_rest).astype(float)
    p = gauss / gauss.sum() + weight * uniform / uniform.sum()
    p[0] = p[-1] = 0
    return p / (p.sum() * (v[1] - v[0]))


class FPE_Population:
    """
    Fokker-Planck (diffusion approximation) population model of leaky integrate-and-fire neurons with stochastic
    current input (NNLIF model)

    dp/dt = d/dv [ (v - u_rest - v_ext) / tau * p + D * dp/dv ],    D = voltage_sigma^2 / tau

    with p(u_thr) = 0, the outgoing flux at u_thr is the firing rate r, which is reinjected at u_reset.
    p is a probability density (sum(p) * dv = 1).

    Discretization:
    - Scharfetter-Gummel flux in v (structure preserving scheme of Hu et al. 2021), weights computed from neighbouring
      differences of the potential only, so there is no over-/underflow for small noise. 'bernoulli' (default) is the
      classical Scharfetter-Gummel flux, 'harmonic' the harmonic mean of Hu et al. 2021
    - BDF2 in time (first step backward Euler), the reinjection at u_reset is part of the implicit step
    - BDF2 is not strictly positivity preserving: negative values are clipped to 0 and p is rescaled to its mass
      before clipping, otherwise the mass added by clipping accumulates and the rate is too high by that factor
    - tridiagonal systems are solved with a banded solver

    Coupling between populations: population j gets the additional drive sum_k W[k, j] * (kernel * r_k)(t) in mV
    (connectivity_matrix W in mV*ms), negative entries are inhibitory.
    A refractory period (tau_ref) is not implemented.
    simulate_set() runs a set of independent single population simulations at once (computation='vec' in DI_wave),
    split into block matrices (batch_size) and joblib processes (n_jobs)
    """

    def __init__(self, parameters={}, name=None):

        # soma model: 'LIF' (default) or 'izhikevich' / 'IK' (see FPE_Izhikevich for the parameters and the adaptation
        # closures), for Izhikevich neurons the defaults of IZHIKEVICH_DEFAULTS are used for missing parameters and the
        # grid parameters are mapped from the Izhikevich parameters (u_inh = v_min, u_thr = v_peak, u_rest = v_r,
        # u_reset = c)
        soma_model = self.normalize_soma_model(parameters.get('soma_model', 'LIF'))
        parameters_full = dict(parameters)
        if soma_model == 'izhikevich':
            parameters_full = dict(IZHIKEVICH_DEFAULTS)
            parameters_full['name'] = 'FPE_Izhikevich'
            parameters_full.update(parameters)
            parameters_full['u_inh'] = parameters_full['v_min']
            parameters_full['u_thr'] = parameters_full['v_peak']
            parameters_full['u_rest'] = float(np.atleast_1d(parameters_full['v_r'])[0])
            parameters_full['u_reset'] = float(np.atleast_1d(parameters_full['c'])[0])
        parameters_full['soma_model'] = soma_model

        self.connectivity_matrix = np.zeros((1, 1))
        self.delay_kernel_type = 'alpha'
        self.delay_kernel_parameters = {'n_alpha': 9, 'tau_alpha': 1/3}
        self.name = 'FPE'
        self.u_reset = -70
        self.u_rest = -70
        self.u_thr = -55
        self.u_inh = -75  # lower boundary of the voltage grid
        self.tau_mem = [20]
        self.tau_ref = [1]  # not used in simulate()
        self.dt = 0.01
        self.dv = 0.01
        self.T = 100
        self.tqdm_disable = False
        self.input_function = np.zeros((int(self.T/self.dt)))
        self.input_function_idx = [0, 0]
        self.population_type = ['exc']
        self.voltage_sigma = 1  # noise (mV): scalar, one value per population or (n_populations, Nt)
        self.static_noise = True
        self.c_mem = 1.1e-10  # membrane capacitance (F), 0.11 nF cell total c_mem on average
        self.init_pdf_offset = 0
        self.init_pdf_sigma = 1
        self.init_pdf_weight = 0
        # flux discretization: 'bernoulli' (Scharfetter-Gummel) or 'harmonic' (harmonic mean, Hu et al. 2021)
        self.flux_scheme = 'bernoulli'
        # flux limiter: 'none' (Scharfetter-Gummel flux) or 'fct' (flux-corrected transport: the antidiffusive part
        # of the central flux is added with the Zalesak limiter, removes the numerical diffusion |drift| dv / 2 of the
        # SG flux at large cell Peclet numbers while keeping p >= 0, see _fct_correct)
        self.flux_limiter = 'none'
        # compiled kernels (numba) for the Izhikevich time step and the FCT correction; False or numba not installed:
        # vectorized numpy / scipy versions with the same arithmetic
        self.use_numba = True
        # saving of p: 'none' (only r is saved), 'end' (only the last time step) or 'all' (every p_save_every-th
        # time step), r is always saved at every time step
        self.save_p = 'all'
        self.p_save_every = 1
        self.verbose = 0
        self.labelsize = 15
        # simulate_set(): number of simulations solved together in one matrix (None: all simulations of a process)
        # and number of joblib processes (1: no joblib, -1: all cores)
        self.n_simulations = 1
        self.batch_size = None
        self.n_jobs = 1

        self.__dict__.update(parameters_full)

        self.parameters = parameters

        if name is not None:
            self.name = name

        self.n_populations = len(self.population_type)

        if self.soma_model == 'izhikevich':
            if not self.static_noise:
                raise NotImplementedError('static_noise=False is not supported for Izhikevich neurons, pass '
                                          'voltage_sigma as (n_populations, Nt) array instead')
            if self.adaptation_closure not in ADAPTATION_CLOSURES:
                raise NotImplementedError(f'adaptation_closure {self.adaptation_closure} not implemented! Please chose '
                                          f'from {ADAPTATION_CLOSURES}')

        if self.soma_model == 'LIF' and np.size(self.tau_mem) == 1:
            self.tau_mem = [float(np.ravel(self.tau_mem)[0])] * self.n_populations  # one value per population
        if not 'g_leak' in parameters:
            # compute g_leak from tau and capacitance
            self.g_leak = self.c_mem / (np.array(self.tau_mem) * 1e-3)  # (conversion from ms to s for tau_mem)

        if self.flux_limiter not in FLUX_LIMITERS:
            raise NotImplementedError(f"flux_limiter {self.flux_limiter} not implemented! Please chose from "
                                      f"{FLUX_LIMITERS}")
        if self.flux_limiter == 'fct' and self.flux_scheme != 'bernoulli':
            raise NotImplementedError("flux_limiter 'fct' is implemented for flux_scheme 'bernoulli' only")
        if not self.flux_scheme in ['bernoulli', 'harmonic']:
            raise NotImplementedError(f"flux_scheme {self.flux_scheme} not implemented! Please chose from "
                                      f"['bernoulli', 'harmonic']")
        self.p_save_every = int(self.p_save_every)
        if self.p_save_every < 1:
            raise ValueError(f"p_save_every has to be >= 1, got {self.p_save_every}")
        if not self.save_p in ['none', 'end', 'all']:
            raise NotImplementedError(f"save_p {self.save_p} not implemented! Please chose from "
                                      f"['none', 'end', 'all']")

        # set up arrays for simulation
        self.t = np.arange(0, self.T, self.dt)
        self.v = np.arange(self.u_inh, self.u_thr + self.dv / 2, self.dv)
        if type(self.connectivity_matrix) != np.ndarray:
            self.connectivity_matrix = np.array(self.connectivity_matrix)
        if len(self.connectivity_matrix.shape) == 1:
            self.connectivity_matrix = self.connectivity_matrix[:, np.newaxis]

        # calculate alpha kernel
        self.get_delay_kernel()

        self.simulation_done = False

    @staticmethod
    def normalize_soma_model(soma_model):
        """
        'LIF' or 'izhikevich' from the accepted names of the soma models (case insensitive: 'LIF', 'IK', 'izhikevich')
        """
        name = str(soma_model).lower()
        if name == 'lif':
            return 'LIF'
        if name in ['ik', 'izhikevich']:
            return 'izhikevich'
        raise NotImplementedError(f"soma_model {soma_model} not implemented! Please chose from ['LIF', 'IK', "
                                  f"'izhikevich']")

    def simulate(self):
        if self.soma_model == 'izhikevich':
            return self._simulate_izhikevich()

        self.t = np.arange(0, self.T, self.dt)
        self.v = np.arange(self.u_inh, self.u_thr + self.dv / 2, self.dv)
        Nt = self.t.shape[0]
        Nx = self.v.shape[0]
        dv = self.dv
        n_pop = self.n_populations

        # external input current (A) and noise (mV), one row per population, see external_input
        self.i_ext, self.voltage_sigma = external_input(self.input_function, self.input_function_idx,
                                                        self.voltage_sigma, self.t, n_pop, self.static_noise)
        # injected voltage in mV
        v_ext = self.i_ext / np.array(self.g_leak)[:n_pop, np.newaxis] * 1e3

        v_reset_idx = np.where(np.isclose(self.v, self.u_reset))[0][0]  # index of reset potential in array
        self.v_reset_idx = v_reset_idx
        # reinjection at u_reset as density with integral 1
        s = np.zeros(Nx)
        s[v_reset_idx] = 1 / dv

        # the solver keeps the last two states itself, p is only stored as requested by save_p
        # (every p_save_every-th time step, only the initial state for 'none' and 'end')
        self.t_p = self.t[::self.p_save_every] if self.save_p == 'all' else self.t[:1]
        p = np.zeros((n_pop, Nx, len(self.t_p)))
        r = np.zeros((n_pop, Nt))  # output firing rate (1/ms)
        r_conv = np.zeros(n_pop)
        n_k_max = len(self.delay_kernel)

        ################################################################################################################
        # INITIAL CONDITIONS
        ################################################################################################################
        for j in range(n_pop):
            # Gaussian around u_rest + init_pdf_offset plus a uniform part above u_rest, integral 1
            p[j, :, 0] = initial_density(self.v, self.u_rest, self.init_pdf_offset, self.init_pdf_sigma,
                                         self.init_pdf_weight)

        p_last_step = [p[j, :, 0].copy() for j in range(n_pop)]
        p_prev_step = [None] * n_pop
        n_clipped = 0

        ################################################################################################################
        # TIME STEPPING
        ################################################################################################################
        main = np.zeros(Nx)
        lower = np.zeros(Nx - 1)
        upper = np.zeros(Nx - 1)
        fct_h, fct_w = np.full(Nx - 1, dv), np.full(Nx, dv)
        fct_work = [np.zeros(Nx - 1), np.zeros(Nx - 1), np.zeros(Nx), np.zeros(Nx), np.zeros(Nx - 1)]
        for i, t_ in enumerate(tqdm(self.t[1:], f"simulating {self.population_type} neuron populations for"
                                                f" {Nt - 1} time steps", disable=self.tqdm_disable), start=1):

            # delayed spike densities of all populations faster computation with dot product
            n_k = min(i + 1, n_k_max)
            for k in range(n_pop):
                r_conv[k] = np.dot(r[k, i - n_k + 1:i + 1][::-1], self.delay_kernel[:n_k]) * self.dt

            for j in range(n_pop):
                drift_coeff = v_ext[j, i] + np.dot(self.connectivity_matrix[:, j], r_conv)
                # minimal diffusion only to avoid division by zero, the scheme is upwind for D -> 0
                diffusion_coeff = max(self.voltage_sigma[j, i]**2 / self.tau_mem[j], 1e-10)
                c = diffusion_coeff * self.dt / dv**2

                # flux J_{i+1/2} = (D/dv) * (g_plus_{i+1/2} p_i - g_minus_{i+1/2} p_{i+1})
                g_plus, g_minus = self.SG_weights(self.v, drift_coeff, diffusion_coeff, x_rest=self.u_rest,
                                                  tau=self.tau_mem[j], scheme=self.flux_scheme)

                # BDF2: (3 p^{n+1} - 4 p^n + p^{n-1}) / (2 dt) = L p^{n+1}, first step with backward Euler
                p_old = p_last_step[j]
                p_old[0] = p_old[-1] = 0.0
                if p_prev_step[j] is not None:
                    alpha_t = 1.5
                    b = 2 * p_old - 0.5 * p_prev_step[j]
                else:
                    alpha_t = 1
                    b = p_old.copy()

                main[1:-1] = alpha_t + c * (g_plus[1:] + g_minus[:-1])
                lower[:-1] = -c * g_plus[:-1]
                upper[1:] = -c * g_minus[1:]
                # boundary conditions: p = 0 at both ends of the grid
                main[0] = main[-1] = 1
                b[0] = b[-1] = 0.0

                # reinjection at u_reset is part of the implicit step: A p = b + dt * n(p) / dv, with
                # n(p) = coef * p[-2] the flux through the threshold interface, solved with the Sherman-Morrison
                # formula
                coef = (diffusion_coeff / dv) * g_plus[-1]
                yz = self.solve_tridiag(main, lower, upper, np.column_stack([b, s]))
                y, z = yz[:, 0], yz[:, 1]
                r[j, i] = coef * y[-2] / (1 - self.dt * coef * z[-2])
                p_new = y + self.dt * r[j, i] * z

                # BDF2 is not strictly positivity preserving, clip small undershoots, only count the ones above round
                # off level (relative to the maximum of p)
                if p_new.min() < 0:
                    if p_new.min() < -1e-6 * p_new.max():
                        n_clipped += 1
                        if self.verbose > 0 and n_clipped == 1:
                            print(f'WARNING: negative p ({p_new.min():.2e}) at t = {t_:.3f}ms is clipped to 0, '
                                  f'consider a smaller time step')
                    # rescale to the mass before clipping, clipping adds mass
                    mass_before_clip = p_new.sum()
                    p_new = np.maximum(p_new, 0)
                    p_new *= mass_before_clip / p_new.sum()

                if self.flux_limiter == 'fct':
                    # flux-corrected transport on the (positive) low order solution, see _fct_correct
                    fields = p_new[np.newaxis, :].copy()
                    (_fct_correct if self._numba_active() else _fct_correct_np)(
                        fields, g_plus, g_minus, fct_h, fct_w, diffusion_coeff, self.dt / alpha_t, *fct_work,
                        np.empty((1, Nx)), 0.)
                    p_new = fields[0]

                p_prev_step[j] = p_old
                p_last_step[j] = p_new
                if self.save_p == 'all' and i % self.p_save_every == 0:
                    p[j, :, i // self.p_save_every] = p_new

        if self.verbose > 0 and n_clipped > 0:
            print(f'negative p values clipped in {n_clipped} time steps')

        if self.save_p == 'end':
            for j in range(n_pop):
                p[j, :, 0] = p_last_step[j]
            self.t_p = self.t[-1:]
        elif self.save_p == 'none':
            p = None
            self.t_p = None

        self.r = r
        self.p = p
        self.save_sim_results(r=r, p=p, p_plot=p)
        self.simulation_done = True

    def simulate_set(self):
        """
        Simulate a set of independent simulations of one population each (e.g. different input currents or
        parameters), same scheme as simulate()

        input_function: (n_simulations, Nt) input currents in A
        tau_mem, voltage_sigma, init_pdf_offset, init_pdf_sigma, init_pdf_weight: scalar or one value per simulation
        (voltage_sigma also as (n_simulations, Nt))
        connectivity_matrix: self coupling of each simulation (scalar or one value per simulation, in mV*ms)

        The simulations are split into batches of batch_size simulations, each batch is solved as one block
        tridiagonal system (matrix splitting), the batches are distributed over n_jobs processes with joblib:
        - batch_size=None, n_jobs=1: all simulations in one matrix (default)
        - batch_size=1, n_jobs>1: joblib only, every simulation solved on its own
        - batch_size=None, n_jobs>1: simulations split evenly over the processes, one matrix per process
        n_jobs=-1 uses all cores

        Izhikevich neurons (soma_model='izhikevich' / 'IK'): see _simulate_set_izhikevich, a set of independent
        simulations of the coupled populations (all closures, flux_limiter 'fct' and n_corrector as in simulate())
        """
        if self.soma_model == 'izhikevich':
            return self._simulate_set_izhikevich()
        if self.flux_limiter != 'none':
            raise NotImplementedError("flux_limiter 'fct' is implemented in simulate() only, not in simulate_set()")
        self.t = np.arange(0, self.T, self.dt)
        self.v = np.arange(self.u_inh, self.u_thr + self.dv / 2, self.dv)
        Nt = self.t.shape[0]

        input_current = np.atleast_2d(np.asarray(self.input_function, dtype=float))
        assert input_current.shape[1] == Nt, f'Dimension missmatch between input with dimension' \
                                             f' {input_current.shape[1]} and simulation time with dimension {Nt}!'
        n_sim = input_current.shape[0]
        self.n_simulations = n_sim
        self.i_ext = input_current

        def repeat_per_simulation(x):
            # scalar or one value per simulation -> (n_sim,)
            x = np.asarray(x, dtype=float).reshape(-1)
            return np.broadcast_to(x, (n_sim,)).copy() if x.size in [1, n_sim] else None

        tau = repeat_per_simulation(self.tau_mem)
        g_leak = repeat_per_simulation(self.g_leak)
        init_offset = repeat_per_simulation(self.init_pdf_offset)
        init_sigma = repeat_per_simulation(self.init_pdf_sigma)
        init_weight = repeat_per_simulation(self.init_pdf_weight)
        w = repeat_per_simulation(self.connectivity_matrix)
        for name, val in [('tau_mem', tau), ('g_leak', g_leak), ('init_pdf_offset', init_offset),
                          ('init_pdf_sigma', init_sigma), ('init_pdf_weight', init_weight),
                          ('connectivity_matrix', w)]:
            if val is None:
                raise ValueError(f'{name} has to be a scalar or have one value per simulation ({n_sim})')

        sigma = np.asarray(self.voltage_sigma, dtype=float)
        if sigma.ndim == 2:
            assert sigma.shape == (n_sim, Nt), f'voltage_sigma has to be of shape {(n_sim, Nt)}'
        else:
            sigma = repeat_per_simulation(sigma)[:, np.newaxis] * np.ones((1, Nt))
            if not self.static_noise:
                sigma = sigma * input_current

        # injected voltage in mV
        v_ext = input_current / g_leak[:, np.newaxis] * 1e3

        # initial distributions (densities)
        init_dist = np.zeros((n_sim, self.v.shape[0]))
        for k in range(n_sim):
            v0 = self.v[np.where(self.v > self.u_rest + init_offset[k])[0][0] - 1]
            init_dist_1 = norm.pdf(self.v, v0, init_sigma[k])
            init_dist_1 /= init_dist_1.sum()
            init_dist_2 = np.ones_like(self.v)
            init_dist_2[self.v < self.u_rest] = 0
            init_dist_2 /= init_dist_2.sum()
            init_dist[k] = init_dist_1 + init_weight[k] * init_dist_2
            init_dist[k, 0] = init_dist[k, -1] = 0
            init_dist[k] /= init_dist[k].sum() * self.dv

        # split into batches
        n_jobs = os.cpu_count() if self.n_jobs == -1 else max(int(self.n_jobs), 1)
        batch_size = self.batch_size if self.batch_size is not None else int(np.ceil(n_sim / n_jobs))
        batches = [np.arange(k, min(k + batch_size, n_sim)) for k in range(0, n_sim, batch_size)]
        args = dict(v=self.v, dt=self.dt, u_rest=self.u_rest, u_reset=self.u_reset, delay_kernel=self.delay_kernel,
                    flux_scheme=self.flux_scheme, save_p=self.save_p, p_save_every=self.p_save_every)
        batch_inputs = [dict(tau=tau[b], v_ext=v_ext[b], sigma=sigma[b], init_dist=init_dist[b], w=w[b]) for b in batches]

        # results are written into preallocated arrays to keep the memory peak low
        self.t_p = {'all': self.t[::self.p_save_every], 'end': self.t[-1:], 'none': None}[self.save_p]
        r = np.zeros((n_sim, Nt))
        p = np.zeros((n_sim, self.v.shape[0], len(self.t_p))) if self.save_p != 'none' else None
        n_clipped = 0
        if n_jobs == 1:
            for b, b_in in zip(batches, tqdm(batch_inputs, f'simulating {n_sim} simulations in {len(batches)} batches',
                                             disable=self.tqdm_disable or len(batches) == 1)):
                r[b], p_b, n_clip_b = FPE_Population._simulate_batch(**args, **b_in,
                                                                       tqdm_disable=self.tqdm_disable)
                if p is not None:
                    p[b] = p_b
                n_clipped += n_clip_b
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(delayed(FPE_Population._simulate_batch)(**args, **b_in,
                                                                                      tqdm_disable=True)
                                              for b_in in batch_inputs)
            for b, res in zip(batches, results):
                r[b] = res[0]
                if p is not None:
                    p[b] = res[1]
                n_clipped += res[2]
            del results

        if self.verbose > 0 and n_clipped > 0:
            print(f'negative p values clipped in {n_clipped} time steps (summed over batches), '
                  f'consider a smaller time step')

        self.r = r
        self.p = p
        self.save_sim_results(r=r, p=p, p_plot=p)
        self.simulation_done = True

    @staticmethod
    def _simulate_batch(v, dt, u_rest, u_reset, delay_kernel, flux_scheme, save_p, p_save_every, tau, v_ext,
                        sigma, init_dist, w, tqdm_disable=True):
        """
        Solve a batch of n independent simulations as one block tridiagonal system per time step
        :return: r (n, Nt), p (n, Nx, Nt_p) (None for save_p='none'), number of time steps with clipped
         negative p
        """
        n, Nt = v_ext.shape
        Nx = v.shape[0]
        dv = v[1] - v[0]
        v_reset_idx = np.where(np.isclose(v, u_reset))[0][0]
        tau_ = tau[:, np.newaxis]

        r = np.zeros((n, Nt))
        p = np.zeros((n, Nx, len(range(0, Nt, p_save_every)) if save_p == 'all' else 1))
        p[:, :, 0] = init_dist
        p_old = init_dist.copy()
        p_prev = None
        n_clipped = 0
        n_k_max = len(delay_kernel)

        # reinjection at u_reset for every block as second right hand side
        s = np.zeros((n, Nx))
        s[:, v_reset_idx] = 1 / dv
        rhs = np.zeros((n * Nx, 2))
        rhs[:, 1] = s.reshape(-1)
        main = np.ones((n, Nx))
        lower = np.zeros((n, Nx))  # lower[:, k] = A[k + 1, k] within a block, last entry couples blocks -> 0
        upper = np.zeros((n, Nx))  # upper[:, k] = A[k, k + 1] within a block, last entry couples blocks -> 0
        ab = np.zeros((3, n * Nx))

        for i in tqdm(range(1, Nt), f'simulating {n} simulations for {Nt - 1} time steps', disable=tqdm_disable):
            n_k = min(i + 1, n_k_max)
            r_conv = r[:, i - n_k + 1:i + 1][:, ::-1] @ delay_kernel[:n_k] * dt
            drift_coeff = (v_ext[:, i] + w * r_conv)[:, np.newaxis]
            diffusion_coeff = np.maximum(sigma[:, i] ** 2 / tau, 1e-10)[:, np.newaxis]
            c = diffusion_coeff * dt / dv ** 2

            g_plus, g_minus = FPE_Population.SG_weights(v, drift_coeff, diffusion_coeff, x_rest=u_rest, tau=tau_,
                                                        scheme=flux_scheme)

            p_old[:, 0] = p_old[:, -1] = 0.0
            if p_prev is not None:
                alpha_t = 1.5
                b = 2 * p_old - 0.5 * p_prev
            else:
                alpha_t = 1
                b = p_old.copy()
            b[:, 0] = b[:, -1] = 0.0

            main[:, 1:-1] = alpha_t + c * (g_plus[:, 1:] + g_minus[:, :-1])
            lower[:, :-2] = -c * g_plus[:, :-1]
            upper[:, 1:-1] = -c * g_minus[:, 1:]
            ab[0, 1:] = upper.reshape(-1)[:-1]
            ab[1] = main.reshape(-1)
            ab[2, :-1] = lower.reshape(-1)[:-1]
            rhs[:, 0] = b.reshape(-1)
            yz = scipy.linalg.solve_banded((1, 1), ab, rhs, check_finite=False)
            y, z = yz[:, 0].reshape(n, Nx), yz[:, 1].reshape(n, Nx)

            # Sherman-Morrison for the reinjection, separately for every block
            coef = (diffusion_coeff[:, 0] / dv) * g_plus[:, -1]
            r[:, i] = coef * y[:, -2] / (1 - dt * coef * z[:, -2])
            p_new = y + dt * r[:, i, np.newaxis] * z

            if p_new.min() < 0:
                if np.any(p_new.min(axis=1) < -1e-6 * p_new.max(axis=1)):
                    n_clipped += 1
                # clipping adds mass, which BDF2 carries forward: rescale every block to its mass before clipping
                mass_before_clip = p_new.sum(axis=1, keepdims=True)
                p_new = np.maximum(p_new, 0)
                p_new *= mass_before_clip / p_new.sum(axis=1, keepdims=True)

            p_prev = p_old
            p_old = p_new
            if save_p == 'all' and i % p_save_every == 0:
                p[:, :, i // p_save_every] = p_new

        if save_p == 'end':
            p[:, :, 0] = p_old
        elif save_p == 'none':
            p = None
        return r, p, n_clipped

    def get_delay_kernel(self):
        if self. delay_kernel_type == 'alpha':
            t_alpha = self.t[self.t < 10]
            n_alpha = self.delay_kernel_parameters['n_alpha']
            tau_alpha = self.delay_kernel_parameters['tau_alpha']
            self.delay_kernel = np.exp(-t_alpha/tau_alpha) / (tau_alpha * scipy.special.factorial(n_alpha-1)) *\
                    (t_alpha/tau_alpha)**(n_alpha-1)
            self.delay_kernel = self.delay_kernel/np.trapz(self.delay_kernel, dx=self.dt)
            self.delay_kernel_time = t_alpha
        elif self. delay_kernel_type == ('bi-exp' or 'bi-exponential' or 'bi_exp'):
            """
            The bi-exponential kernel is taken from Rusu et al. 2014 [1] where the values for tau_1, tau_2 and g_peak
            are given in a table for AMPA, NMDA, GABA_a and GABA_b synapses in M1. tau_cond is specified as 1ms in the 
            text. In the context of this function the values for an AMPA synapse are:
            (tau_1, tau_2, g_peak, tau_cond) = (0.2, 1.7, 1e-4, 1)
            [1] Rusu, C. V., Murakami, M., Ziemann, U., & Triesch, J. (2014). A model of TMS-induced I-waves in motor
             cortex. Brain Stimulation, 7(3), 401-414. 
            """
            t_kernel = self.t[self.t < 10]
            tau_1 = self.delay_kernel_parameters['tau_1']
            tau_2 = self.delay_kernel_parameters['tau_2']
            tau_cond = self.delay_kernel_parameters['tau_cond']
            g_peak = self.delay_kernel_parameters['g_peak']
            self.delay_kernel = g_peak * (np.exp(-(t_kernel - tau_cond)/tau_1) - np.exp(-(t_kernel - tau_cond)/tau_2))
            self.delay_kernel = self.delay_kernel / np.trapz(self.delay_kernel, dx=self.dt)
            self.delay_kernel_time = t_kernel
        else:
            raise NotImplementedError('Specified delay kernel is not implemented!, please choose from "alpha", "bi-exp"')

    def plot_delay_kernel(self):
        # find time window
        plt.plot(self.delay_kernel_time, self.delay_kernel)
        plt.grid()
        plt.xlabel('t (ms)')
        plt.ylabel(f'{self.delay_kernel_type} delay kernel')
        plt.show()

    def save_sim_results(self, r, p, p_plot):
        with h5py.File(self.name + '.hdf5', 'w') as h5file:
            h5file.create_dataset('t', data=self.t)
            h5file.create_dataset('v', data=self.v)
            h5file.create_dataset('r', data=r)
            h5file.create_dataset('in', data=self.i_ext)
            if p is not None:
                h5file.create_dataset('rho', data=p)
                h5file.create_dataset('rho_plot', data=p_plot)
                if getattr(self, 't_p', None) is not None:
                    h5file.create_dataset('t_rho', data=self.t_p)
            h5file.create_dataset('p_types', data=self.population_type)
            if self.soma_model == 'izhikevich':
                h5file.create_dataset('u_mean', data=self.u_mean)
                h5file.create_dataset('v_mean', data=self.v_mean)
                for name in ['q', 'sigma_u2', 'var_u', 'u2_mean']:
                    if getattr(self, name, None) is not None:
                        h5file.create_dataset(name, data=getattr(self, name))

    def plot(self, fname=None, heat_map=True, plot_idxs=None, savefig=False, plot_input=False, z_limit=None,
             plot_combined=True, animate=False, crop_rate=False, frame_off=False, ylims=None):

        if fname == None:
            fname = self.name
        p = None
        i_in = None

        with h5py.File(fname + '.hdf5', 'r') as h5file:

            t_plot = np.array(h5file['t'])
            # time axis of p, differs from t if p was only stored every n-th time step
            t_p = np.array(h5file['t_rho']) if 't_rho' in h5file else t_plot
            v = np.array(h5file['v'])
            r_plot = np.array(h5file['r'])
            for k in range(r_plot.shape[0]):
                r_plot[k, -1] = r_plot[k, -2]
            p_plot = np.array(h5file['rho_plot']) if 'rho_plot' in h5file else None
            p_types_raw = h5file['p_types']
            p_types = p_types_raw.asstr()[:]
            if 'rho' in h5file:
                p = np.array(h5file['rho'])

        if plot_input:
            with h5py.File(fname + '.hdf5', 'r') as h5file:
                i_in = np.array(h5file['in'])

        p_plot_full = p_plot.copy() if p_plot is not None else None
        p_full = p.copy() if p is not None else None
        r_plot_full = r_plot.copy()
        i_in_full = i_in.copy() if i_in is not None else None
        for j in range(self.n_simulations):
            # TODO: adjust for n_populations at some point

            if self.n_simulations > 1:
                p_plot = p_plot_full[j:j + 1] if p_plot_full is not None else None
                p = p_full[j:j + 1] if p_full is not None else None
                r_plot = r_plot_full[j:j + 1]
                if i_in_full is not None:
                    i_in = i_in_full[j:j + 1]

            if plot_idxs is None:
                n_plots = len(p_types)
                plot_idxs = np.arange(n_plots)
            else:
                n_plots = len(plot_idxs)

            if plot_combined:
                fig = plt.figure(figsize=(10, 3.75*n_plots))
                for i_plot, plot_idx in enumerate(plot_idxs):
                    plot_loc_1 = int(2*i_plot + 1)
                    plot_loc_2 = int(2 * i_plot + 2)
                    ax, z_limit = plot_p_panel(fig, (n_plots, 2, plot_loc_1), t_p, v,
                                                 p_plot[plot_idx] if p_plot is not None else None, heat_map,
                                                 z_limit, 'gnuplot2', self.labelsize, ylims)
                    ax.set_title(f"Membrane potential distribution ({str(p_types[plot_idx])})", fontsize=self.labelsize)

                    ax = fig.add_subplot(n_plots, 2, plot_loc_2)
                    ax.plot(t_plot, r_plot[plot_idx] * 1000)
                    ax.set_title(f"Population activity ({str(p_types[plot_idx])})", fontsize=self.labelsize)
                    ax.set_ylabel("Spike density (1/s)", fontsize=self.labelsize)
                    ax.set_xlabel("t (ms)", fontsize=self.labelsize)
                    ax.tick_params(axis='both', which='major', labelsize=self.labelsize)
                    if crop_rate:
                        ax.set_ylim(0, np.max(1.2*r_plot[plot_idx, 50:] * 1000))
                    if plot_input:
                        rate_height = np.max(r_plot[plot_idx, 50:] * 1000)
                        i_in_plot = i_in[plot_idx].flatten() / np.max(i_in[plot_idx].flatten()) * rate_height  # renormalize for plot with rate
                        ax.plot(t_plot, i_in_plot)
                        ax.legend(['d', 'I'])

                    ax.grid()
                plt.tight_layout()
                if savefig:
                    if self.n_simulations > 1:
                        plot_name = self.name + f'_sim_{j}'
                    else:
                        plot_name = self.name
                    plt.savefig(plot_name + '_plot.png')
                    plt.close()
                else:
                    plt.show()

            else:
                for i_plot, plot_idx in enumerate(plot_idxs):
                    fig_1 = plt.figure(figsize=(6, 4.25))

                    ax, z_limit = plot_p_panel(fig_1, (n_plots, 1, 1), t_p, v,
                                                 p_plot[plot_idx] if p_plot is not None else None, heat_map,
                                                 z_limit, 'jet', self.labelsize, ylims)
                    ax.set_title(f"Membrane potential distribution ({str(p_types[plot_idx])})")
                    plt.tight_layout()
                    if savefig:
                        plt.savefig(self.name + 'voltage_dist_plot.png')
                        plt.close()
                    else:
                        plt.show()
                    fig_2 = plt.figure(figsize=(6, 4.25))
                    ax = fig_2.add_subplot(n_plots, 1, 1)
                    ax.plot(t_plot, r_plot[plot_idx] * 1000, color='darkorange', linewidth=2)
                    ax.set_title(f"Population activity ({str(p_types[plot_idx])})")
                    ax.set_ylabel("Spike density (1/s)", fontsize=self.labelsize)
                    ax.set_xlabel("t (ms)", fontsize=self.labelsize)
                    if plot_input:
                        ax_input = ax.twinx()

                        # rate_height = np.max(r_plot[plot_idx] * 1000)
                        # i_in_plot = i_in[plot_idx].flatten() / np.max(i_in[plot_idx].flatten()) * rate_height  # renormalize for plot with rate
                        ax_input.plot(t_plot, i_in[plot_idx].flatten()/1e3*1e9, color='teal', linewidth=2)
                        ax_input.set_ylabel("Current (nA)", fontsize=self.labelsize)
                        ax_input.set_ylim(0, i_in[plot_idx].flatten().max()/1e3*1e9*1.2)
                        ax_input.tick_params(axis='both', which='major', labelsize=self.labelsize)
                        ax.plot(t_plot, -5*np.ones_like(r_plot[plot_idx]), color='teal')
                        ax.legend(['Spike density', 'Current'])
                    ax.set_ylim(0, r_plot[plot_idx].max()*1e3*1.2)
                    ax.set_xlim(0, t_plot.max())
                    ax.spines['top'].set_visible(False)
                    ax.tick_params(axis='both', which='major', labelsize=self.labelsize)


                    # ax.grid()
                    plt.tight_layout()
                    if savefig:
                        plt.savefig(self.name + 'rate_plot.png')
                        plt.close()
                    else:
                        plt.show()
            if animate and (p_plot is None or p_plot.shape[-1] < 2):
                print("no animation of p possible, p was not saved for several time steps (save_p='all')")
                animate = False
            if animate:
                print('plotting animation of p(v, t)')
                if isinstance(p, np.ndarray):
                    p_animation = p
                else:
                    p_animation = p_plot
                fig = plt.figure(figsize=(10, 4.25*n_plots))
                for i_plot, plot_idx in enumerate(plot_idxs):
                    ax = fig.add_subplot(n_plots, 1, i_plot+1)
                    line = ax.plot(v, p_animation[plot_idx, :, 0])[0]
                    # plt.fill_between(x=v, y1=p_animation[plot_idx, :, 0], color="b", alpha=0.2)
                    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
                    text = ax.text(0.95, 0.9, f't = {t_p[0]:.2f}ms', fontsize=9, bbox=bbox,
                            transform=ax.transAxes, horizontalalignment='right')
                    ax.set(xlim=[v.min(), v.max()], ylim=[0, p_animation.mean()*3], xlabel='v (mV)', ylabel='p')  # p_animation.max()*0.5
                    def update(frame):
                        # for each frame, update the data stored on each artist.
                        p_frame = p_animation[plot_idx, :, frame]

                        # update the line plot:
                        # line.set_xdata(v[:frame])
                        line.set_ydata(p_frame)
                        text.set_text(f't = {t_p[frame]:.2f}ms')
                        # plt.fill_between(x=v, y1=p_frame, color="b", alpha=0.2)
                        return (line, text)

                    ani = animation.FuncAnimation(fig=fig, func=update, frames=p_animation.shape[2], interval=22)
                    # plt.show()
                    ani.save(filename=self.name + '_p_animation.gif', writer="pillow")
                    print(f'saved animation of p to {self.name}_p_animation.gif')
                    plt.close()

    def save_log(self, full_out=False):
        if full_out:
            log_dict = self.__dict__.copy()
        else:
            log_dict = self.parameters
        print(f'saved log to: {self.name}_log.yaml')
        with open(f'{self.name}_log.yaml', 'w') as file:
            yaml.dump(log_dict, file)

    @staticmethod
    def SG_weights(x, a, alpha, x_rest=0.2, tau=1, scheme='bernoulli'):
        """
        compute the flux weights of the Scharfetter-Gummel scheme from neighbouring differences of U only
        (M = exp(-U), U = ((1/2) x - x_rest - a) x / (alpha tau)), avoids over-/underflow of M for small noise values
        flux at interface i+1/2: F = (alpha/dx) * (g_plus * p_i - g_minus * p_{i+1})
        :param x: x
        :param a: drift
        :param alpha: diffusion
        :param x_rest: resting point
        :param tau: membrane time constant
        :param scheme: 'harmonic': harmonic mean of M (Hu et al. 2021),
         'bernoulli': classical Scharfetter-Gummel flux with Bernoulli function B(z) = z / (exp(z) - 1)
        :return: g_plus, g_minus (both of length len(x) - 1)
        """
        # U_{i+1} - U_i, exact for the quadratic U
        x_mid = 0.5 * (x[1:] + x[:-1])
        dU = np.diff(x) * (x_mid - x_rest - a) / (alpha * tau)
        return FPE_Population.SG_weights_from_dU(dU, scheme=scheme)

    @staticmethod
    def SG_weights_from_dU(dU, scheme='bernoulli'):
        """
        Scharfetter-Gummel flux weights from the potential differences dU = U_{i+1} - U_i at the cell interfaces, with
        U = -int(drift / D) dv, for any drift: flux at interface i+1/2: F = (D/dx) * (g_plus * p_i - g_minus * p_{i+1})
        :param dU: potential differences (len(x) - 1)
        :param scheme: 'harmonic' or 'bernoulli' (see SG_weights)
        :return: g_plus, g_minus (both of length len(x) - 1)
        """
        if scheme == 'harmonic':
            # M_{i+1/2}/M_i = 2 / (1 + exp(dU)), M_{i+1/2}/M_{i+1} = 2 / (1 + exp(-dU))
            return 2 * scipy.special.expit(-dU), 2 * scipy.special.expit(dU)
        elif scheme == 'bernoulli':
            # B(-z) = B(z) + z: only one evaluation of B, on |dU| to avoid cancellation
            dU_abs = np.abs(dU)
            b_small = FPE_Population.bernoulli(dU_abs)
            b_large = b_small + dU_abs
            positive = dU >= 0
            return np.where(positive, b_small, b_large), np.where(positive, b_large, b_small)
        else:
            raise NotImplementedError(f"flux_scheme {scheme} not implemented! Please chose from "
                                      f"['bernoulli', 'harmonic']")

    @staticmethod
    def bernoulli(z):
        """
        Bernoulli function B(z) = z / (exp(z) - 1), with B(0) = 1
        """
        z = np.asarray(z, dtype=float)
        small = np.abs(z) < 1e-8
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            res = z / np.expm1(z)
        res[small] = 1 - z[small] / 2
        # exp(z) overflows for large z, B -> 0
        res[np.isnan(res)] = 0
        return res

    @staticmethod
    def solve_tridiag(main, lower, upper, b):
        """
        solve a tridiagonal system with the banded solver of scipy
        :param main: main diagonal (N)
        :param lower: lower diagonal (N - 1), lower[k] = A[k + 1, k]
        :param upper: upper diagonal (N - 1), upper[k] = A[k, k + 1]
        :param b: right hand side (N) or (N, n_rhs)
        :return: solution x of A x = b
        """
        ab = np.zeros((3, main.shape[0]))
        ab[0, 1:] = upper
        ab[1] = main
        ab[2, :-1] = lower
        return scipy.linalg.solve_banded((1, 1), ab, b, check_finite=False)

    def _closure_flags(self):
        """
        (conditional, with_variance, second_moment, quadrature) of the adaptation closure: conditional closures evolve
        q = p E[u | v]; q2 = p E[u^2 | v] is a field for 'conditional_second_moment' (Gaussian third moment) and
        'conditional_quadrature' (also q3 = p E[u^3 | v], two-node closure of the fourth moment)
        """
        conditional = self.adaptation_closure in ['conditional', 'conditional_variance', 'conditional_second_moment',
                                                  'conditional_quadrature']
        quadrature = self.adaptation_closure == 'conditional_quadrature'
        second_moment = self.adaptation_closure == 'conditional_second_moment' or quadrature
        return conditional, self.adaptation_closure == 'conditional_variance', second_moment, quadrature

    def _izhikevich_setup(self):
        """
        common part of simulate() and simulate_set() for Izhikevich neurons: time and voltage grid (sets self.t,
        self.v, self.v_cell_width, self.v_control_volume, self.v_reset_idx), noise, reset points, cell averaged drift
        and the initial conditions of every population
        :return: dict with the parameters per population (C_m, k_iz, v_r, v_t, a, b, c, d, tau_noise), v, h, w,
         sigma (n_pop, Nt), reset_idx, quad_mean (n_pop, Nx - 1), p0 (n_pop, Nx), v_mean0, u_mean0 (n_pop)
        """
        self.t = np.arange(0, self.T, self.dt)
        Nt = self.t.shape[0]
        n_pop = self.n_populations
        C_m, k_iz, v_r, v_t, a, b, c, d, tau_noise = [self._per_population(name) for name in IZHIKEVICH_PER_POPULATION]

        # voltage grid: uniform (spacing dv) or non-uniform ('refined': spacing dv in v_fine_range, growing to
        # dv_coarse outside, see izhikevich_grid; or the grid points as array). Finite volumes: cells [v_i, v_{i+1}]
        # of width h_i, control volume w_i = (h_{i-1} + h_i) / 2 around every grid point
        if isinstance(self.v_grid, str) and self.v_grid == 'uniform':
            self.v = np.arange(self.u_inh, self.u_thr + self.dv / 2, self.dv)
        elif isinstance(self.v_grid, str) and self.v_grid == 'refined':
            fine_range = self.v_fine_range if self.v_fine_range is not None else (v_r.min() - 20, v_t.max() + 10)
            self.v = izhikevich_grid(self.u_inh, self.u_thr, self.dv, c, dv_coarse=self.dv_coarse,
                                     fine_range=fine_range, growth=self.grid_growth)
        elif not isinstance(self.v_grid, str):
            self.v = np.asarray(self.v_grid, dtype=float)
            if np.any(np.diff(self.v) <= 0):
                raise ValueError('v_grid has to be strictly increasing')
            self.u_inh, self.u_thr = self.v[0], self.v[-1]
        else:
            raise NotImplementedError(f"v_grid {self.v_grid} not implemented! Please chose 'uniform', 'refined' or "
                                      f"pass the grid points as array")
        v = self.v
        Nx = v.shape[0]
        h = np.diff(v)
        w = np.zeros(Nx)
        w[1:-1] = 0.5 * (h[1:] + h[:-1])
        w[0], w[-1] = 0.5 * h[0], 0.5 * h[-1]
        self.v_cell_width = h
        self.v_control_volume = w

        sigma = np.asarray(self.voltage_sigma, dtype=float)
        if sigma.ndim == 2:
            assert sigma.shape == (n_pop, Nt), f'voltage_sigma has to be of shape {(n_pop, Nt)}'
        else:
            sigma = np.broadcast_to(sigma.reshape(-1, 1), (n_pop, Nt)) if sigma.size in [1, n_pop] else None
            if sigma is None:
                raise ValueError(f'voltage_sigma has to be a scalar, have one value per population or be of shape '
                                 f'{(n_pop, Nt)}')

        # reinjection at c (per population), the source of the reinjected mass is 1 / w at the reset point
        reset_idx = []
        for j in range(n_pop):
            idx = np.where(np.isclose(v, c[j]))[0]
            if idx.size == 0:
                raise ValueError(f'reset potential c = {c[j]} is not on the voltage grid (v_min={self.u_inh}, '
                                 f'dv={self.dv}, v_grid={self.v_grid if isinstance(self.v_grid, str) else "array"})')
            reset_idx.append(int(idx[0]))
        self.v_reset_idx = reset_idx[0]

        # cell average of (v - v_r)(v - v_t) over [v_i, v_{i+1}]: exact SG potential differences for the quadratic drift
        v_mid = 0.5 * (v[1:] + v[:-1])
        quad_mean = np.array([(v_mid - v_r[j]) * (v_mid - v_t[j]) + h**2 / 12 for j in range(n_pop)])

        # initial conditions: gaussian around v_r (plus init_pdf_weight times a uniform part above v_r), all neurons
        # with the same u (adaptation at rest unless u_init is given)
        p0 = np.zeros((n_pop, Nx))
        v_mean0 = np.zeros(n_pop)
        for j in range(n_pop):
            v0 = v[np.where(v > v_r[j] + self.init_pdf_offset)[0][0] - 1]
            init_dist_1 = norm.pdf(v, v0, self.init_pdf_sigma)
            init_dist_1 /= np.sum(w * init_dist_1)
            init_dist_2 = np.ones_like(v)
            init_dist_2[v < v_r[j]] = 0
            init_dist_2 /= np.sum(w * init_dist_2)
            init_dist = init_dist_1 + (self.init_pdf_weight * init_dist_2)
            init_dist[0] = init_dist[-1] = 0
            init_dist /= np.sum(w * init_dist)
            p0[j] = init_dist
            v_mean0[j] = np.sum(w * v * init_dist)
        u_mean0 = b * (v_mean0 - v_r) if self.u_init is None else self._per_population('u_init')
        return dict(C_m=C_m, k_iz=k_iz, v_r=v_r, v_t=v_t, a=a, b=b, c=c, d=d, tau_noise=tau_noise, v=v, h=h, w=w,
                    sigma=sigma, reset_idx=np.array(reset_idx), quad_mean=quad_mean, p0=p0, v_mean0=v_mean0,
                    u_mean0=u_mean0)

    def _simulate_izhikevich(self):
        """
        simulate() for soma_model='izhikevich' (FPE_Izhikevich), see the docstring of FPE_Izhikevich for the model, the
        adaptation closures and the discretization
        """
        st = self._izhikevich_setup()
        C_m, k_iz, v_r, v_t, a, b, c, d, tau_noise = [st[name] for name in ['C_m', 'k_iz', 'v_r', 'v_t', 'a', 'b', 'c',
                                                                            'd', 'tau_noise']]
        v, h, w, sigma, reset_idx, quad_mean = st['v'], st['h'], st['w'], st['sigma'], st['reset_idx'], st['quad_mean']
        Nt = self.t.shape[0]
        Nx = v.shape[0]
        dt = self.dt
        n_pop = self.n_populations
        conditional, with_variance, second_moment, quadrature = self._closure_flags()
        harmonic = self.flux_scheme == 'harmonic'
        fct = self.flux_limiter == 'fct'

        # external input current in pA, (Nt,) into the populations input_function_idx or (n_pop, Nt)
        self.i_ext = external_input(self.input_function, self.input_function_idx, 0., self.t, n_pop)[0]

        self.t_p = self.t[::self.p_save_every] if self.save_p == 'all' else self.t[:1]
        p = np.zeros((n_pop, Nx, len(self.t_p)))
        q = np.zeros((n_pop, Nx, len(self.t_p))) if conditional else None
        q2 = np.zeros((n_pop, Nx, len(self.t_p))) if second_moment else None
        r = np.zeros((n_pop, Nt))
        u_mean = np.zeros((n_pop, Nt))
        v_mean = np.zeros((n_pop, Nt))
        # conditional closures: sigma_u2 = population average of Var(u | v) (closure variable, 0 for 'conditional'),
        # var_u = total variance of u over the population = sigma_u2 + variance of E[u | v], u2_mean = <u^2>
        sigma_u2 = np.zeros((n_pop, Nt))
        var_u = np.zeros((n_pop, Nt))
        u2_mean = np.zeros((n_pop, Nt))
        r_conv = np.zeros(n_pop)
        n_k_max = len(self.delay_kernel)

        ################################################################################################################
        # INITIAL CONDITIONS
        ################################################################################################################
        p[:, :, 0] = st['p0']
        v_mean[:, 0] = st['v_mean0']
        u_mean[:, 0] = st['u_mean0']
        # all neurons start with the same u: E[u | v] = <u>, no variance
        u2_mean[:, 0] = u_mean[:, 0]**2

        p_last_step = [p[j, :, 0].copy() for j in range(n_pop)]
        p_prev_step = [None] * n_pop
        q_last_step = [u_mean[j, 0] * p[j, :, 0] for j in range(n_pop)]
        q_prev_step = [None] * n_pop
        u_cond_last = [np.full(Nx, u_mean[j, 0]) for j in range(n_pop)]
        u_cond_prev = [None] * n_pop
        # 'conditional_second_moment': q2 = p E[u^2 | v] and Var(u | v) = q2 / p - E[u | v]^2 (0 at the start)
        q2_last_step = [u_mean[j, 0]**2 * p[j, :, 0] for j in range(n_pop)]
        q2_prev_step = [None] * n_pop
        # 'conditional_quadrature': q3 = p E[u^3 | v] and the third central moment c3 (0 at the start)
        q3_last_step = [u_mean[j, 0]**3 * p[j, :, 0] for j in range(n_pop)]
        q3_prev_step = [None] * n_pop
        c3_last = [np.zeros(Nx) for _ in range(n_pop)]
        c3_prev = [None] * n_pop
        s2_last = [np.zeros(Nx) for _ in range(n_pop)]
        s2_prev = [None] * n_pop
        if conditional:
            q[:, :, 0] = np.array(q_last_step)
        if second_moment:
            q2[:, :, 0] = np.array(q2_last_step)
        n_clipped = 0
        n_fallback = 0  # time steps (per population) repeated with backward Euler

        ################################################################################################################
        # TIME STEPPING (python loop, the steps of every population in the kernels _iz_density_step, _iz_q_step,
        # _iz_q2_step, _fct_correct and _iz_moments, compiled with numba if available)
        ################################################################################################################
        # kernels: compiled with numba (use_numba=True and numba installed) or vectorized numpy / scipy versions
        # (same arithmetic, slower for small grids)
        if self._numba_active():
            density_step, q_step, q2_step = _iz_density_step, _iz_q_step, _iz_q2_step
            moments, fct_correct, moment_step = _iz_moments, _fct_correct, _iz_moment_step
        else:
            density_step, q_step, q2_step = _iz_density_step_np, _iz_q_step_np, _iz_q2_step_np
            moments, fct_correct, moment_step = _iz_moments_np, _fct_correct_np, _iz_moment_step_np
        n_corrector = int(self.n_corrector)
        n_fields = 1 + conditional + second_moment + quadrature
        fct_work = [np.zeros(Nx - 1), np.zeros(Nx - 1), np.zeros(Nx), np.zeros(Nx), np.zeros(Nx - 1)]
        s2_zero = np.zeros(Nx)
        g_plus, g_minus = np.zeros(Nx - 1), np.zeros(Nx - 1)
        main, main_q = np.zeros(Nx), np.zeros(Nx)
        lower, upper = np.zeros(Nx - 1), np.zeros(Nx - 1)
        b_rhs, b_q, s_vec, y, z, cp = [np.zeros(Nx) for _ in range(6)]
        for i, t_ in enumerate(tqdm(self.t[1:], f"simulating {self.population_type} Izhikevich populations "
                                                f"({self.adaptation_closure} closure) for {Nt - 1} time steps",
                                    disable=self.tqdm_disable), start=1):

            n_k = min(i + 1, n_k_max)
            for k in range(n_pop):
                r_conv[k] = np.dot(r[k, i - n_k + 1:i + 1][::-1], self.delay_kernel[:n_k]) * dt

            for j in range(n_pop):
                i_total = self.i_ext[j, i] + np.dot(self.connectivity_matrix[:, j], r_conv)
                diffusion_coeff = max(sigma[j, i]**2 / tau_noise[j], 1e-10)
                p_old = p_last_step[j]
                has_prev = p_prev_step[j] is not None
                q_old = q_last_step[j]
                q2_old = q2_last_step[j]
                q3_old = q3_last_step[j]

                # predictor (pass 0): the adaptation in the drift and in the extra fluxes is linearly extrapolated to
                # t_i (second order consistent with BDF2); corrector passes (n_corrector > 0): the step is repeated
                # with the adaptation fields of the previous pass at t_i
                for it in range(1 + n_corrector):
                    if it == 0:
                        u_scalar = 2 * u_mean[j, i - 1] - u_mean[j, i - 2] if i >= 2 else u_mean[j, i - 1]
                        has_u_prev = u_cond_prev[j] is not None
                        u_src_last = u_cond_last[j]
                        u_src_prev = u_cond_prev[j] if has_u_prev else u_cond_last[j]
                        u_ex = 2 * u_cond_last[j] - u_cond_prev[j] if has_u_prev else u_cond_last[j]
                        if with_variance:
                            s2_star = 2 * sigma_u2[j, i - 1] - sigma_u2[j, i - 2] if i >= 2 else sigma_u2[j, i - 1]
                            s2_ex = np.full(Nx, max(s2_star, 0.))
                        elif second_moment:
                            s2_ex = np.maximum(2 * s2_last[j] - s2_prev[j], 0.) if s2_prev[j] is not None \
                                else s2_last[j]
                        else:
                            s2_ex = s2_zero
                        if quadrature:
                            c3_ex = 2 * c3_last[j] - c3_prev[j] if c3_prev[j] is not None else c3_last[j]
                    else:
                        u_scalar = u_mean[j, i]
                        has_u_prev = False
                        if conditional:
                            u_src_last = u_src_prev = u_ex = u_cond
                        if with_variance:
                            s2_ex = np.full(Nx, sigma_u2[j, i])
                        elif second_moment:
                            s2_ex = s2_new
                        if quadrature:
                            c3_ex = c3_new

                    p_new = np.empty(Nx)
                    # BDF2 step; if it gives significantly negative densities (under-resolved, e.g. a narrow density
                    # packet crossing v_peak within one step at small noise), the step is repeated with backward
                    # Euler: with the SG M-matrix it keeps p >= 0 and r >= 0, and for the conditional closures
                    # E[u | v] stays within the range of its old values and sources (discrete maximum principle)
                    r[j, i], alpha_t, fallback, clipped, coef, mass, v_mean[j, i] = density_step(
                        p_old, p_prev_step[j] if has_prev else p_old, has_prev, v, h, w, quad_mean[j], k_iz[j],
                        C_m[j], i_total, diffusion_coeff, dt, conditional, u_scalar, u_src_last, u_src_prev,
                        has_u_prev, reset_idx[j], harmonic, g_plus, g_minus, main, lower, upper, b_rhs, s_vec, y, z,
                        cp, p_new)
                    n_fallback += fallback
                    if clipped:
                        n_clipped += 1
                        if self.verbose > 0 and n_clipped == 1:
                            print(f'WARNING: negative p at t = {t_:.3f}ms is clipped to 0, consider a smaller time '
                                  f'step')
                    b_u = 2 * u_mean[j, i - 1] - 0.5 * u_mean[j, i - 2] if alpha_t == 1.5 else u_mean[j, i - 1]
                    use_bdf2 = alpha_t == 1.5

                    if conditional:
                        # q = p E[u | v]: same SG operator as p (same drift and noise) plus the relaxation of u,
                        # dq/dt = -d/dv [mu q - D dq/dv - Var(u | v) p / C] + a (b (v - v_r) p - q)
                        #         + delta(v - c) (q outflow at v_peak + d r)
                        # Var(u | v): 0 ('conditional'), sigma_u2(t) ('conditional_variance'), field from q2
                        # ('conditional_second_moment'), explicit
                        q_new = np.empty(Nx)
                        q_out = q_step(q_old, q_prev_step[j] if use_bdf2 else q_old, use_bdf2, p_new, v, h, w, main,
                                       lower, upper, s_vec, dt, a[j], b[j], v_r[j], d[j], r[j, i], coef, s2_ex,
                                       C_m[j], with_variance or second_moment, main_q, b_q, y, z, cp, q_new)
                        if second_moment and not quadrature:
                            q2_new = np.empty(Nx)
                            q2_step(q2_old, q2_prev_step[j] if use_bdf2 else q2_old, use_bdf2, q_new, p_new, u_ex,
                                    s2_ex, v, h, w, main, lower, upper, s_vec, dt, a[j], b[j], v_r[j], d[j], r[j, i],
                                    q_out, coef, C_m[j], main_q, b_q, y, z, cp, q2_new)
                        elif quadrature:
                            # moments 2 and 3 with the exact extra fluxes in terms of the central moments
                            # (m, s2, c3) and the two-node closure c4 = s2^2 + c3^2 / s2 of the fourth moment:
                            # F_2 = -(2 m s2 + c3) p / C,  F_3 = -(3 m^2 s2 + 3 m c3 + c4) p / C
                            N_rate = r[j, i]
                            # the extrapolated c3 is bounded by the skewness limit of the extrapolated s2 (otherwise
                            # c3^2 / s2 can explode where s2 is small)
                            c3_lim = QUADRATURE_MAX_SKEWNESS * s2_ex**1.5
                            c3_ex = np.clip(c3_ex, -c3_lim, c3_lim)
                            c4_ex = s2_ex**2 + np.where(s2_ex > 0, c3_ex**2 / np.maximum(s2_ex, 1e-300), 0.)
                            q2_new = np.empty(Nx)
                            q2_out = moment_step(
                                q2_old, q2_prev_step[j] if use_bdf2 else q2_old, use_bdf2, q_new,
                                -(2 * u_ex * s2_ex + c3_ex) * p_new / C_m[j], v, h, w, main, lower, upper, s_vec, dt,
                                2 * a[j], b[j], v_r[j], coef, 2 * d[j] * q_out + d[j]**2 * N_rate, main_q, b_q, y, z,
                                cp, q2_new)
                            q3_new = np.empty(Nx)
                            moment_step(
                                q3_old, q3_prev_step[j] if use_bdf2 else q3_old, use_bdf2, q2_new,
                                -(3 * u_ex**2 * s2_ex + 3 * u_ex * c3_ex + c4_ex) * p_new / C_m[j], v, h, w, main,
                                lower, upper, s_vec, dt, 3 * a[j], b[j], v_r[j], coef,
                                3 * d[j] * q2_out + 3 * d[j]**2 * q_out + d[j]**3 * N_rate, main_q, b_q, y, z, cp,
                                q3_new)

                    if fct:
                        # flux-corrected transport: antidiffusive flux (central minus SG flux) with the Zalesak
                        # limiter, the same limiter for p, q and q2 (synchronized)
                        fields = np.empty((n_fields, Nx))
                        fields[0] = p_new
                        if conditional:
                            fields[1] = q_new
                        if second_moment:
                            fields[2] = q2_new
                        if quadrature:
                            fields[3] = q3_new
                        fct_correct(fields, g_plus, g_minus, h, w, diffusion_coeff, dt / alpha_t, *fct_work,
                                    np.empty((n_fields, Nx)), self.u_cond_floor)
                        p_new = fields[0].copy()
                        if conditional:
                            q_new = fields[1].copy()
                        if second_moment:
                            q2_new = fields[2].copy()
                        if quadrature:
                            q3_new = fields[3].copy()
                        v_mean[j, i] = np.sum(w * v * p_new) / mass

                    if not conditional:
                        # mean adaptation, BDF2 (implicit in <u>, uses the new <v> and r)
                        u_mean[j, i] = (b_u + dt * (a[j] * b[j] * (v_mean[j, i] - v_r[j]) + d[j] * r[j, i])) / \
                                       (alpha_t + dt * a[j])
                    else:
                        u_cond = np.empty(Nx)
                        s2_new = np.empty(Nx)
                        mass, v_mean[j, i], u_mean[j, i], var_cond_mean, uv, u2_field, sig_field = moments(
                            p_new, q_new, q2_new if second_moment else q_new, second_moment, v, w, v_r[j],
                            self.u_cond_floor, u_cond, s2_new)
                        if with_variance:
                            # exact equation of <u^2> (BDF2, implicit in <u^2>):
                            # d<u^2>/dt = 2 a (b <(v - v_r) u> - <u^2>) + 2 d r <u>_spike + d^2 r
                            b_u2 = 2 * u2_mean[j, i - 1] - 0.5 * u2_mean[j, i - 2] if use_bdf2 \
                                else u2_mean[j, i - 1]
                            u2_mean[j, i] = (b_u2 + dt * (2 * a[j] * b[j] * uv + 2 * d[j] * q_out +
                                                          d[j]**2 * r[j, i])) / (alpha_t + 2 * dt * a[j])
                            # closure: Var(u | v) = sigma_u2, the same for all v
                            sigma_u2[j, i] = max(u2_mean[j, i] - u_mean[j, i]**2 - var_cond_mean, 0.)
                            var_u[j, i] = sigma_u2[j, i] + var_cond_mean
                        elif second_moment:
                            # <u^2> from q2 (its integral follows the exact equation of <u^2>), Var(u | v) as field
                            u2_mean[j, i] = u2_field
                            sigma_u2[j, i] = sig_field
                            var_u[j, i] = u2_field - u_mean[j, i]**2
                            if quadrature:
                                # third central moment E[(u - m)^3 | v] (regularized like E[u | v]), bounded skewness
                                floor = self.u_cond_floor * p_new.max()
                                u3_mean = np.dot(w, q3_new) / mass
                                e2 = (q2_new + floor * u2_field) / (p_new + floor)
                                e3 = (q3_new + floor * u3_mean) / (p_new + floor)
                                c3_new = e3 - 3 * u_cond * e2 + 2 * u_cond**3
                                c3_max = QUADRATURE_MAX_SKEWNESS * s2_new**1.5
                                c3_new = np.clip(c3_new, -c3_max, c3_max)
                        else:
                            var_u[j, i] = var_cond_mean
                            u2_mean[j, i] = var_u[j, i] + u_mean[j, i]**2

                # state for the next time step
                if conditional:
                    if second_moment:
                        q2_prev_step[j] = q2_old
                        q2_last_step[j] = q2_new
                        s2_prev[j] = s2_last[j]
                        s2_last[j] = s2_new
                        if self.save_p == 'all' and i % self.p_save_every == 0:
                            q2[j, :, i // self.p_save_every] = q2_new
                    if quadrature:
                        q3_prev_step[j] = q3_old
                        q3_last_step[j] = q3_new
                        c3_prev[j] = c3_last[j]
                        c3_last[j] = c3_new
                    q_prev_step[j] = q_old
                    q_last_step[j] = q_new
                    u_cond_prev[j] = u_cond_last[j]
                    u_cond_last[j] = u_cond
                    if self.save_p == 'all' and i % self.p_save_every == 0:
                        q[j, :, i // self.p_save_every] = q_new
                p_prev_step[j] = p_old
                p_last_step[j] = p_new
                if self.save_p == 'all' and i % self.p_save_every == 0:
                    p[j, :, i // self.p_save_every] = p_new

        if self.verbose > 0 and n_clipped > 0:
            print(f'negative p values clipped in {n_clipped} time steps')
        if self.verbose > 0 and n_fallback > 0:
            print(f'{n_fallback} time steps repeated with backward Euler (negative p with BDF2)')
        self.n_fallback = n_fallback

        if self.save_p == 'end':
            for j in range(n_pop):
                p[j, :, 0] = p_last_step[j]
                if conditional:
                    q[j, :, 0] = q_last_step[j]
                if second_moment:
                    q2[j, :, 0] = q2_last_step[j]
            self.t_p = self.t[-1:]
        elif self.save_p == 'none':
            p = None
            q = None
            q2 = None
            self.t_p = None

        self.r = r
        self.p = p
        self.q = q
        self.q2 = q2
        self.u_mean = u_mean
        self.v_mean = v_mean
        self.sigma_u2 = sigma_u2 if conditional else None
        self.var_u = var_u if conditional else None
        self.u2_mean = u2_mean if conditional else None
        self.save_sim_results(r=r, p=p, p_plot=p)
        self.simulation_done = True

    def _simulate_set_izhikevich(self):
        """
        simulate_set() for Izhikevich neurons: n_sim independent simulations of the n_pop coupled populations (same
        parameters, connectivity_matrix, noise and grid, different input currents), all n_sim * n_pop populations are
        advanced together in one time loop (_iz_simulate_set_batch): the elementwise operations of a step act on all
        populations at once, the implicit solves, FCT and moments use the kernels of simulate() per population, so a
        single simulation (n_sim = 1) gives the same result as simulate().

        input_function (pA): callable or (Nt,) array (one simulation, as in simulate()), (n_sim, Nt) (input into the
        populations input_function_idx) or (n_sim, n_pop, Nt) (input into every population)
        batch_size, n_jobs: as for LIF neurons, batches of simulations distributed over joblib processes

        Results: r, u_mean, v_mean (and sigma_u2, var_u, u2_mean for the conditional closures) of shape
        (n_sim, n_pop, Nt), p, q, q2 of shape (n_sim, n_pop, Nx, Nt_p); the hdf5 file holds them as n_sim * n_pop
        populations (simulation s, population j at index s * n_pop + j)
        """
        st = self._izhikevich_setup()
        Nt, n_pop = self.t.shape[0], self.n_populations
        Nx = self.v.shape[0]
        idx = self.input_function_idx

        # input currents (n_sim, n_pop, Nt)
        if callable(self.input_function):
            i_set = np.zeros((1, n_pop, Nt))
            i_set[0, idx] = self.input_function(self.t)
        else:
            x = np.asarray(self.input_function, dtype=float)
            x = x[np.newaxis] if x.ndim == 1 else x
            if x.ndim == 2:
                i_set = np.zeros((x.shape[0], n_pop, Nt))
                i_set[:, idx] = x[:, np.newaxis, :]
            elif x.ndim == 3 and x.shape[1] == n_pop:
                i_set = x.copy()
            else:
                raise ValueError(f'input_function has to be callable or of shape (Nt,), (n_sim, Nt) or '
                                 f'(n_sim, {n_pop}, Nt), not {x.shape}')
        assert i_set.shape[-1] == Nt, f'Dimension missmatch between input with dimension {i_set.shape[-1]} and ' \
                                      f'simulation time with dimension {Nt}!'
        n_sim = i_set.shape[0]
        self.n_simulations = n_sim
        self.i_ext = i_set

        cfg = dict(st, dt=self.dt, connectivity_matrix=np.asarray(self.connectivity_matrix, dtype=float),
                   delay_kernel=self.delay_kernel, flags=self._closure_flags(), harmonic=self.flux_scheme == 'harmonic',
                   fct=self.flux_limiter == 'fct', n_corrector=int(self.n_corrector), u_cond_floor=self.u_cond_floor,
                   save_p=self.save_p, p_save_every=self.p_save_every, numba=self._numba_active(),
                   tqdm_disable=self.tqdm_disable)

        # batches of simulations, distributed over n_jobs processes
        n_jobs = os.cpu_count() if self.n_jobs == -1 else max(int(self.n_jobs), 1)
        batch_size = self.batch_size if self.batch_size is not None else int(np.ceil(n_sim / n_jobs))
        batches = [np.arange(k, min(k + batch_size, n_sim)) for k in range(0, n_sim, batch_size)]
        if n_jobs == 1:
            results = [_iz_simulate_set_batch(cfg, i_set[bt]) for bt in batches]
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(delayed(_iz_simulate_set_batch)(dict(cfg, tqdm_disable=True), i_set[bt])
                                              for bt in batches)

        def collect(name, shape):
            if results[0][name] is None:
                return None
            return np.concatenate([res[name] for res in results]).reshape(n_sim, n_pop, *shape)

        self.r, self.u_mean, self.v_mean = [collect(name, (Nt,)) for name in ['r', 'u_mean', 'v_mean']]
        conditional = cfg['flags'][0]
        self.sigma_u2, self.var_u, self.u2_mean = [collect(name, (Nt,)) if conditional else None
                                                   for name in ['sigma_u2', 'var_u', 'u2_mean']]
        Nt_p = results[0]['p'].shape[-1] if results[0]['p'] is not None else 0
        self.p, self.q, self.q2 = [collect(name, (Nx, Nt_p)) for name in ['p', 'q', 'q2']]
        self.t_p = {'all': self.t[::self.p_save_every], 'end': self.t[-1:], 'none': None}[self.save_p]
        self.n_clipped = sum(res['n_clipped'] for res in results)
        self.n_fallback = sum(res['n_fallback'] for res in results)
        if self.verbose > 0 and self.n_clipped > 0:
            print(f'negative p values clipped in {self.n_clipped} time steps (summed over populations)')
        if self.verbose > 0 and self.n_fallback > 0:
            print(f'{self.n_fallback} time steps repeated with backward Euler (negative p with BDF2)')

        # hdf5 file: the n_sim * n_pop populations in the format of save_sim_results
        flat = lambda x: None if x is None else x.reshape(n_sim * n_pop, *x.shape[2:])
        with h5py.File(self.name + '.hdf5', 'w') as h5file:
            h5file.create_dataset('t', data=self.t)
            h5file.create_dataset('v', data=self.v)
            h5file.create_dataset('r', data=flat(self.r))
            h5file.create_dataset('in', data=flat(self.i_ext))
            if self.p is not None:
                h5file.create_dataset('rho', data=flat(self.p))
                h5file.create_dataset('rho_plot', data=flat(self.p))
                h5file.create_dataset('t_rho', data=self.t_p)
            h5file.create_dataset('p_types', data=list(self.population_type) * n_sim)
            for name in ['u_mean', 'v_mean', 'q', 'sigma_u2', 'var_u', 'u2_mean']:
                if getattr(self, name) is not None:
                    h5file.create_dataset(name, data=flat(getattr(self, name)))
        self.simulation_done = True

    def _numba_active(self):
        """True if the numba kernels are used (use_numba=True and numba installed), warns once otherwise"""
        if self.use_numba and not NUMBA_AVAILABLE:
            if not getattr(FPE_Population, '_numba_warning_done', False):
                warnings.warn('numba is not installed, the numpy / scipy versions of the kernels are used '
                              '(use_numba=False to silence this warning)')
                FPE_Population._numba_warning_done = True
            return False
        return bool(self.use_numba)

    def _per_population(self, name):
        """
        parameter name as array with one value per population (from a scalar or one value per population)
        """
        x = np.asarray(getattr(self, name), dtype=float).reshape(-1)
        if x.size not in [1, self.n_populations]:
            raise ValueError(f'{name} has to be a scalar or have one value per population ({self.n_populations})')
        return np.broadcast_to(x, (self.n_populations,)).copy()

    def clean(self):
        if self.simulation_done:
            os.remove(self.name + '.hdf5')


class FPE_Izhikevich(FPE_Population):
    """
    Fokker-Planck population model of Izhikevich neurons (simple model, Izhikevich 2007) with stochastic current input,
    the same as FPE_Population with soma_model='izhikevich' (or 'IK')

    C dv/dt = k (v - v_r)(v - v_t) - u + I + noise,    du/dt = a (b (v - v_r) - u),
    at v >= v_peak: v <- c, u <- u + d

    Integrating the two dimensional density P(v, u) over u gives the exact equation of the voltage density
    p(v) = int P du,

    dp/dt = -d/dv [ mu(v, t) p - D dp/dv ],    mu = (k (v - v_r)(v - v_t) - E[u | v] + I) / C,
    D = voltage_sigma^2 / tau_noise

    with p(v_peak) = 0, the outgoing flux at v_peak is the firing rate r, which is reinjected at c. Only the conditional
    mean E[u | v] of the adaptation enters, it is approximated by one of three closures (adaptation_closure):

    - 'mean' (default): E[u | v] = <u>(t), mean adaptation approximation (e.g. Nicola & Campbell 2013, Augustin et al.
      2017), d<u>/dt = a (b (<v> - v_r) - <u>) + d r (exact), good for slow adaptation, misses the correlation of u and
      v (neurons that just spiked sit at c with a larger u)
    - 'conditional': the first conditional moment q(v, t) = p E[u | v] is evolved as a second one dimensional field
      (moment closure of Ly & Tranchina 2007),
      dq/dt = -d/dv [ mu q - D dq/dv - Var(u | v) p / C ] + a (b (v - v_r) p - q) + delta(v - c) (J_q(v_peak) + d r)
      with the closure Var(u | v) = 0 (all neurons at the same v have the same u)
    - 'conditional_variance': as 'conditional', with Var(u | v) = sigma_u2(t), the same for all v. sigma_u2 is the
      population average of Var(u | v), computed from the exact equation of the second moment
      d<u^2>/dt = 2 a (b <(v - v_r) u> - <u^2>) + 2 d r <u>_spike + d^2 r,
      sigma_u2 = <u^2> - int p E[u | v]^2 dv

    - 'conditional_second_moment': as 'conditional', with Var(u | v) as function of v: the second conditional moment
      q2(v, t) = p E[u^2 | v] is evolved as a third field,
      dq2/dt = -d/dv [ mu q2 - D dq2/dv - 2 E[u | v] Var(u | v) p / C ] + 2 a (b (v - v_r) q - q2)
               + delta(v - c) (J_q2(v_peak) + 2 d J_q(v_peak) + d^2 r),
      Var(u | v) = q2 / p - E[u | v]^2, with the Gaussian closure of the third moment E[u^3 | v] = m^3 + 3 m s^2
      (moment closure of order 3 in the counting of Ly & Tranchina 2007: E[u | v] and E[u^2 | v] are evolved, the
      third conditional moment is closed)

    The equation of <u> is exact in all four cases (the conditional closures reinject exactly the u that leaves at
    v_peak), as the equation of <u^2> for 'conditional_variance' and 'conditional_second_moment'. a = d = 0 (and b = 0) gives the quadratic integrate-and-fire model, for which all closures are exact.

    Discretization as in FPE_Population (Scharfetter-Gummel flux, BDF2, Sherman-Morrison for the reinjection), as
    finite volumes on a uniform or non-uniform grid. If BDF2 gives significantly negative densities the step is
    repeated with backward Euler (n_fallback counts these steps). The
    potential differences of the SG flux use the exact cell average of the quadratic drift. <u> (and <u^2>) are
    integrated with BDF2 after every density step. q is solved after p with the same tridiagonal operator (plus a dt on
    the diagonal), the drift uses <u> or E[u | v] = q / p linearly extrapolated to the new time step. Where p is
    very small E[u | v] is blended towards <u>: (q + eps <u>) / (p + eps), eps = u_cond_floor * max(p).

    Units: v in mV, t in ms, C in pF, k in nS/mV, a in 1/ms, b in nS, u, d and I in pA, r in 1/ms.
    input_function: input current in pA (not in A as for LIF neurons)
    connectivity_matrix W in pA*ms: population j gets the additional input current sum_k W[k, j] * (kernel * r_k)(t)
    C, k, v_r, v_t, a, b, c, d, tau_noise: scalar or one value per population, v_peak and v_min (lower boundary of the
    grid) are shared, every c has to lie on the voltage grid.
    voltage_sigma: scalar, one value per population or (n_populations, Nt), static_noise=False is not supported.
    u_init: initial <u> (scalar or one value per population), None: b (<v>(0) - v_r) (adaptation at rest), all neurons
    start with the same u.
    Results: r, p, u_mean, v_mean, for the conditional closures also q (saved like p), sigma_u2 (closure variance,
    0 for 'conditional'), var_u (total variance of u over the population) and u2_mean.
    Only simulate() is implemented (no simulate_set()).

    Grid and speed:
    v_grid: 'uniform' (spacing dv, default), 'refined' (spacing dv in v_fine_range, default (min(v_r) - 20,
    max(v_t) + 10) mV, growing by grid_growth per cell up to dv_coarse towards v_min and v_peak, see izhikevich_grid)
    or the grid points as array (v[0] = v_min, v[-1] = v_peak, every c on the grid).
    use_numba: run the time step kernels compiled with numba (default True, falls back to plain python if numba is
    not installed).
    flux_limiter: 'none' (Scharfetter-Gummel flux) or 'fct' (flux-corrected transport, see _fct_correct: removes the
    numerical diffusion ~|mu| dv / 2 of the SG flux at large cell Peclet numbers, the same limiter for p, q and q2,
    keeps p >= 0 and conserves mass).
    """

    izhikevich_defaults = IZHIKEVICH_DEFAULTS
    per_population = IZHIKEVICH_PER_POPULATION

    def __init__(self, parameters={}, name=None):
        super().__init__(dict(parameters, soma_model='izhikevich'), name=name)
        self.parameters = parameters


class FPE_mini:
    """
    Minimal Fokker-Planck (population density) model of coupled populations of leaky integrate-and-fire neurons, solved
    with the original structure preserving scheme of

        J. Hu, J.-G. Liu, Y. Xie and Z. Zhou. A structure preserving numerical scheme for Fokker-Planck equations of
        neuron networks: numerical analysis and exploration. Journal of Computational Physics 433:110195, 2021

    (building on J.-G. Liu, L. Wang and Z. Zhou, Mathematics of Computation 87(311):1165-1189, 2018), in the
    formulation of the first Hu-2021 solver of Nykamp_Model_1: Maxwellian M and its harmonic mean M_harm as values, the
    matrix written out from them and solved with a sparse solver. Educational version: backward Euler, nothing
    optimised. FPE_Population is the revised solver (Bernoulli flux, BDF2, implicit reinjection, FCT, Izhikevich).

    Model of population j = 0 ... n_pop - 1:
        tau_j dV = (u_rest + mu_j(t) - V) dt + sigma_j(t) sqrt(2 tau_j) dW,     V >= u_thr: spike, V -> u_reset

        dp_j/dt = -dJ_j/dv,    J_j = h_j(v, t) p_j - D_j dp_j/dv,    h_j = (u_rest + mu_j(t) - v) / tau_j,
        D_j = sigma_j^2 / tau_j

    with p_j(u_inh) = p_j(u_thr) = 0, the firing rate r_j(t) = J_j(u_thr, t) (1/ms) and the reinjection of r_j at
    u_reset. The input mu_j(t) (mV) is the external current of population j divided by its leak conductance plus the
    delayed rates of all populations (as in FPE_Population),

        mu_j(t) = I_j(t) / g_leak_j + sum_k W[k, j] (alpha * r_k)(t),    g_leak_j = c_mem / tau_j,

    W = connectivity_matrix in mV ms (from k to j, < 0 inhibitory), alpha: kernel with n_alpha = 9, tau_alpha = 1/3 ms
    (mean delay 3 ms, integral 1).

    Discretisation on the uniform grid v_i = u_inh + i dv, i = 0 ... N - 1 (v_{N-1} = u_thr), for every population:
    1. Potential and Maxwellian: h = -D dU/dv with U(v) = (v / 2 - u_rest - mu) v / (D tau), M = exp(-U). Then
       J = -D M d/dv (p / M): the flux vanishes exactly where p / M is constant (the equilibrium). U is only defined up to
       a constant, which cancels in all ratios of M; it is shifted to min U = 0 so that 0 < M <= 1 (without the shift,
       M = exp(-U) overflows for sigma below about 2 mV; the first code rescaled v to (0, 1) to avoid this). U grows
       like (v - u_rest - mu)^2 / (2 sigma^2), so for small noise M still underflows far from the minimum (sigma = 1 mV:
       U ~ 870 at u_min); U is therefore capped at U_MAX = 700. Where the cap acts, p ~ exp(-700): the scheme treats
       these cells as drift-free, which does not change the solution. Storing M as values is the limitation of this
       formulation; FPE_Population computes the flux weights from differences of U only.
    2. Flux through the interface i+1/2 with the harmonic mean M_harm_{i+1/2} = 2 / (1 / M_i + 1 / M_{i+1}):
           J_{i+1/2} = (D / dv) M_harm_{i+1/2} (p_i / M_i - p_{i+1} / M_{i+1})
    3. Backward Euler, (p^{n+1}_i - p^n_i) / dt = -(J_{i+1/2} - J_{i-1/2}) / dv, i.e. with c = D dt / dv^2 (dv squared:
       one 1/dv from the flux, one from the divergence; the first code used D dt / dv, wrong by a factor 1/dv)
           -c M_harm_{i-1/2} / M_{i-1} p_{i-1} + (1 + c (M_harm_{i-1/2} + M_harm_{i+1/2}) / M_i) p_i
               - c M_harm_{i+1/2} / M_{i+1} p_{i+1} = p^n_i,
       p_0 = p_{N-1} = 0. All off-diagonal entries are negative and the matrix is diagonally dominant (an M-matrix):
       p >= 0 for every dt.
    4. Firing rate: the flux of the scheme through the last interface (p_{N-1} = 0) at the new time step,
           r^{n+1} = (D / dv) M_harm_{N-3/2} / M_{N-2} p^{n+1}_{N-2},
       i.e. exactly the mass that left through the threshold in the step, divided by dt.
    5. Reinjection after the step: r^{n+1} dt is added at the grid point u_reset (density r dt / dv), so the mass stays
       1 (up to what leaves at u_inh, negligible on a grid that is wide enough).
    First order in dt and dv (backward Euler, reinjection split from the step).

    Parameters: the parameter dict of FPE_Population (it can be passed to both without changes, keys that FPE_mini does
    not use are ignored); units ms, mV, A, F:
        T, dt, dv: simulated time, time step, grid spacing
        population_type: list of the populations (its length is the number of populations)
        tau_mem: membrane time constant, scalar or one value per population; c_mem: membrane capacitance
        u_rest, u_reset, u_thr, u_inh: resting / reset potential, threshold, lower end of the grid (u_reset has to be a
            grid point)
        input_function, input_function_idx, voltage_sigma, static_noise: external input current and noise, scalar,
            per population or per population and time step, see external_input
        connectivity_matrix: W (n_pop, n_pop) in mV ms; delay_kernel_parameters: n_alpha, tau_alpha of the kernel
        init_pdf_offset, init_pdf_sigma, init_pdf_weight: initial density, see initial_density
        save_p, p_save_every: 'all': p every p_save_every-th time step, otherwise only the final density

    Results after simulate(), in the format of FPE_Population (written to name.hdf5 by save_sim_results, so that
    FPE_Population(parameters, name=name).plot() plots them):
        t (ms), v (mV), r (n_pop, Nt) firing rates (1/ms), mu (n_pop, Nt) total input (mV), i_ext (n_pop, Nt) input
        current (A), sigma (n_pop, Nt) noise (mV), p (n_pop, N, n_saved) densities (1/mV), t_p their times
    """

    U_MAX = 700.        # cap of the shifted potential, exp(-700) ~ 1e-304 is still a normal float
    DEFAULTS = dict(T=100., dt=0.01, dv=0.01, population_type=['exc'], tau_mem=20., c_mem=1.1e-10, u_rest=-70.,
                    u_reset=-70., u_thr=-55., u_inh=-75., input_function=None, input_function_idx=[0], voltage_sigma=1.,
                    static_noise=True, connectivity_matrix=0., delay_kernel_parameters={'n_alpha': 9, 'tau_alpha': 1/3},
                    init_pdf_offset=0., init_pdf_sigma=1., init_pdf_weight=0., save_p='all', p_save_every=1, verbose=False)

    def __init__(self, parameters={}, name='FPE_mini'):
        self.__dict__.update(self.DEFAULTS)
        self.__dict__.update(parameters)
        self.name, self.n_populations = name, len(self.population_type)

    def simulate(self):
        """
        :return: firing rates r (1/ms), (n_pop, Nt)
        """
        dt, dv, n_pop = self.dt, self.dv, self.n_populations
        t = np.arange(0, self.T, dt)
        v = np.arange(self.u_inh, self.u_thr + dv / 2, dv)
        N = len(v)
        reset_idx = np.where(np.isclose(v, self.u_reset))[0]
        if reset_idx.size == 0:
            raise ValueError(f'u_reset = {self.u_reset} is not a grid point')
        tau = np.broadcast_to(np.asarray(self.tau_mem, dtype=float).reshape(-1), (n_pop,))
        W = np.broadcast_to(np.asarray(self.connectivity_matrix, dtype=float), (n_pop, n_pop))
        # external input current (A) and noise (mV) per population, input in mV: I / g_leak
        i_in = np.zeros(len(t)) if self.input_function is None else self.input_function
        i_ext, sigma = external_input(i_in, self.input_function_idx, self.voltage_sigma, t, n_pop, self.static_noise)
        mu_ext = i_ext / (self.c_mem / (tau[:, None] * 1e-3)) * 1e3

        # alpha kernel of the coupling, normalised to integral 1 on the time grid
        n_a, tau_a = self.delay_kernel_parameters['n_alpha'], self.delay_kernel_parameters['tau_alpha']
        t_a = t[t < 10]
        kernel = np.exp(-t_a / tau_a) / (tau_a * scipy.special.factorial(n_a - 1)) * (t_a / tau_a)**(n_a - 1)
        kernel /= np.trapz(kernel, dx=dt)

        # initial condition of every population: Gaussian around u_rest, p = 0 at both ends, integral of p = 1
        p = np.array([initial_density(v, self.u_rest, self.init_pdf_offset, self.init_pdf_sigma,
                                      self.init_pdf_weight)] * n_pop)
        r, mu = np.zeros((n_pop, len(t))), np.zeros((n_pop, len(t)))
        save_all = self.save_p == 'all'
        saved, t_saved = [p.copy()], [t[0]]
        main, lower, upper = np.zeros(N), np.zeros(N - 1), np.zeros(N - 1)
        for n in tqdm(range(1, len(t)), 'solving fpe', disable=not self.verbose):
            # input of the step, with the delayed rates of all populations up to the previous step (dot product with
            # the kernel)
            n_k = min(n + 1, len(kernel))
            r_conv = r[:, n - n_k + 1:n + 1][:, ::-1] @ kernel[:n_k] * dt
            mu[:, n] = mu_ext[:, n] + W.T @ r_conv

            for j in range(n_pop):
                D = max(sigma[j, n]**2 / tau[j], 1e-10)                    # minimal D against division by zero
                c = D * dt / dv**2                                          # grid constant
                # potential and Maxwellian (U shifted to min U = 0, capped at U_MAX against underflow)
                U = (0.5 * v - self.u_rest - mu[j, n]) * v / (D * tau[j])
                M = np.exp(-np.minimum(U - U.min(), self.U_MAX))
                M_harm = 2 / (1 / M[:-1] + 1 / M[1:])                       # harmonic mean at the interfaces

                # backward Euler matrix: interior rows i = 1 ... N - 2, first and last row: p = 0
                main[1:-1] = 1 + c * (M_harm[:-1] + M_harm[1:]) / M[1:-1]
                lower[:-1] = -c * M_harm[:-1] / M[:-2]                     # A[i, i-1], i = 1 ... N - 2
                upper[1:] = -c * M_harm[1:] / M[2:]                        # A[i, i+1], i = 1 ... N - 2
                main[0] = main[-1] = 1
                # sparse matrix (A.todense() gives the matrix as readable array)
                A = scipy.sparse.diags([main, lower, upper], offsets=[0, -1, 1], shape=(N, N), format='csr')
                b = p[j].copy()
                b[0] = b[-1] = 0
                p[j] = scipy.sparse.linalg.spsolve(A, b)

                r[j, n] = (D / dv) * M_harm[-1] / M[-2] * p[j, -2]       # flux through the last interface
                p[j, reset_idx[0]] += r[j, n] * dt / dv                    # reinjection at u_reset

            if save_all and n % self.p_save_every == 0:
                saved.append(p.copy())
                t_saved.append(t[n])
        if not save_all:
            saved, t_saved = [p.copy()], [t[-1]]

        self.t, self.v, self.r, self.mu, self.i_ext, self.sigma = t, v, r, mu, i_ext, sigma
        self.p, self.t_p = np.moveaxis(np.array(saved), 0, -1), np.array(t_saved)
        self.save_sim_results()
        return r

    def save_sim_results(self):
        # format of FPE_Population.save_sim_results
        with h5py.File(self.name + '.hdf5', 'w') as h5file:
            for key, x in dict(t=self.t, v=self.v, r=self.r, rho=self.p, rho_plot=self.p, t_rho=self.t_p,
                               p_types=self.population_type).items():
                h5file.create_dataset(key, data=x)
            h5file.create_dataset('in', data=self.i_ext)


class Nykamp_Model:
    """
    Class for the population model of Nykamp and Tranchina
    Probability flux over time is used to model synaptic membrane and firing rates of neuron
    populations
    """

    def __init__(self, parameters, name='Nykamp'):

        self.connectivity_matrix = np.ones([2, 2])
        self.tau_alpha = 1/3
        self.n_alpha = 9

        self.dt = 0.1
        self.dv = 0.01
        self.T = 100

        self.name = name

        if 'connectivity_matrix' in parameters:
            self.connectivity_matrix = parameters['connectivity_matrix']
        if 'u_rest' in parameters:
            self.u_rest = parameters['u_rest']
        if 'u_reset' in parameters:
            self.u_reset = parameters['u_reset']
        else:
            self.u_reset = parameters['u_rest']
        if 'u_thr' in parameters:
            self.u_thr = parameters['u_thr']
        if 'u_exc' in parameters:
            self.u_exc = parameters['u_exc']
        if 'u_inh' in parameters:
            self.u_inh = parameters['u_inh']
        if 'tau_mem' in parameters:
            self.tau_mem = parameters['tau_mem']
        if 'tau_ref' in parameters:
            self.tau_ref = parameters['tau_ref']
        if 'mu_gamma' in parameters:
            self.mu_gamma = parameters['mu_gamma']
        if 'var_coeff_gamma' in parameters:
            self.var_coeff_gamma = parameters['var_coeff_gamma']
        if 'tau_alpha' in parameters:
            self.tau_alpha = parameters['tau_alpha']
        if 'n_alpha' in parameters:
            self.n_alpha = parameters['n_alpha']
        if 'input_function' in parameters:
            self.input_function = parameters['input_function']
        if 'input_function_type' in parameters:
            self.input_function_type = parameters['input_function_type']
        if 'input_function_idx' in parameters:
            self.input_function_idx = parameters['input_function_idx']
        if 'population_type' in parameters:
            self.population_type = parameters['population_type']

        self.n_populations = len(self.population_type)


    def simulate(self, dv=None, dt=None, T=None, dv_fine=None):

        if dv is not None:
            self.dv = dv
        if dt is not None:
            self.dt = dt
        if T is not None:
            self.T = T
        if dv_fine is None:
            self.dv_fine = dv
        else:
            self.dv_fine = dv_fine

        self.t = np.arange(0, T, dt)
        self.v = np.arange(self.u_inh, self.u_thr + dv / 2, dv)

        self.input = np.zeros([self.n_populations, self.n_populations, self.t.shape[0]])
        # set up input_function over time
        self.input[self.input_function_idx[0], self.input_function_idx[1]] = self.input_function(t=self.t)

        # calculate alpha kernel
        self.get_alpha_kernel()

        # self.c1ee, self.c2ee, self.c1ei, self.c2ei,self.c1ie, self.c2ie, self.c1ii,\
        #     self.c2ii = self.get_coeffs_classic()

        # calculate diffusion coefficients
        self.get_coeffs()
        self.diffusion_coefficients_dv = np.gradient(self.diffusion_coefficients, self.dv, axis=-1)

        self.c1ee = self.diffusion_coefficients[0]
        self.c2ee = self.diffusion_coefficients[1]
        self.c1ei = self.diffusion_coefficients[2]
        self.c2ei = self.diffusion_coefficients[3]

        self.c1ie = self.diffusion_coefficients[4]
        self.c2ie = self.diffusion_coefficients[5]
        self.c1ii = self.diffusion_coefficients[6]
        self.c2ii = self.diffusion_coefficients[7]

        self.c1ee_v = self.diffusion_coefficients_dv[0]
        self.c2ee_v = self.diffusion_coefficients_dv[1]
        self.c1ei_v = self.diffusion_coefficients_dv[2]
        self.c2ei_v = self.diffusion_coefficients_dv[3]

        self.c1ie_v = self.diffusion_coefficients_dv[4]
        self.c2ie_v = self.diffusion_coefficients_dv[5]
        self.c1ii_v = self.diffusion_coefficients_dv[6]
        self.c2ii_v = self.diffusion_coefficients_dv[7]

        for i in range(len(self.population_type)):
            # initialize arrays
            v_reset_idx = np.where(np.isclose(self.v, self.u_reset))[0][0]  # index of reset potential in array
            self.v_reset_idx = v_reset_idx

            #TODO: check how many of those are replications and eventually use a single array for all of these
            if self.population_type[i] == 'exc':

                p_exc = np.zeros((len(self.v), len(self.t))) # probability density of discontinuous membrane potential
                p_exc_delta = np.zeros(len(self.t)) # probability density of membrane potential
                ref_exc_delta_idx = int(self.tau_ref[0] / self.dt)  # number of time steps of refractory delay
                r_exc = np.zeros(len(self.t))  # output firing rate
                r_exc_delayed = np.zeros(len(self.t) + ref_exc_delta_idx)  # delayed output firing rate
                v_in_ee = np.zeros(len(self.t))
                v_in_ei = np.zeros(len(self.t))

                p_exc[:, 0] = scipy.stats.norm.pdf(self.v, self.u_rest, 1)
                p_exc[0, 0] = 0
                p_exc[-1, 0] = 0


            elif self.population_type[i] == 'inh':
                p_inh = np.zeros((len(self.v), len(self.t)))  # probability density of membrane potential
                p_inh_delta = np.zeros(len(self.t))  # probability density of discontinuous membrane potential
                ref_inh_delta_idx = int(self.tau_ref[1] / self.dt)  # number of time steps of refractory delay
                r_inh = np.zeros(len(self.t))  # output firing rate
                r_inh_delayed = np.zeros(len(self.t) + ref_inh_delta_idx)  # delayed output firing rate
                v_in_ie = np.zeros(len(self.t))
                v_in_ii = np.zeros(len(self.t))

                # initialize p with a gaussian distribution around the resting potential
                p_inh[:, 0] = scipy.stats.norm.pdf(self.v, self.u_rest, 1)
                p_inh[0, 0] = 0
                p_inh[-1, 0] = 0

            Fee_v = self.dFdv[0]
            Fei_v = self.dFdv[1]
            Fie_v = self.dFdv[2]
            Fii_v = self.dFdv[3]

        # Determine population dynamics (diffusion approximation)
        for i, t_ in enumerate(tqdm(self.t[:-1])):
            # for i, t_ in enumerate(t[:-1]):

            # excitatory population
            # ================================================================================================================
            # input to excitatory population
            if i > 0:
                r_exc_conv = np.convolve(r_exc[:(i + 1)], self.alpha)[-len(self.alpha)] * self.dt
                r_inh_conv = np.convolve(r_inh[:(i + 1)], self.alpha)[-len(self.alpha)] * self.dt
            else:
                r_exc_conv = 0
                r_inh_conv = 0

            v_in_ee[i] = self.connectivity_matrix[0, 0] * r_exc_conv + self.input[0, 0][i]
            v_in_ei[i] = self.connectivity_matrix[0, 1] * r_inh_conv + self.input[0, 1][i]
            v_in_ie[i] = self.connectivity_matrix[1, 0] * r_exc_conv + self.input[1, 0][i]
            v_in_ii[i] = self.connectivity_matrix[1, 1] * r_inh_conv + self.input[1, 1][i]

            # v_in_ee[i] = self.input[0, 0][i] + self.connectivity_matrix[0, 0] * r_exc_conv
            # v_in_ei[i] = self.connectivity_matrix[0, 1] * r_inh_conv

            # coefficients for finite difference matrices
            # c1, c2 are over all v steps and i is a time step

            f0_exc = self.dt / 2 * (1 / self.tau_mem[0] - v_in_ee[i] * self.c1ee_v + v_in_ei[i] * self.c1ei_v)

            f1_exc = self.dt / (4 * self.dv) * (
                    (self.v - self.u_rest) / self.tau_mem[0] + v_in_ee[i] * (-self.c1ee + self.c2ee_v) + v_in_ei[i] * (
                    self.c1ei + self.c2ei_v))
            f2_exc = self.dt / (2 * self.dv ** 2) * (v_in_ee[i] * self.c2ee + v_in_ei[i] * self.c2ei)


            # LHS matrix (t+dt)
            A_exc = np.diag(1 + 2 * f2_exc - f0_exc) + np.diagflat((-f2_exc - f1_exc)[:-1], 1) + np.diagflat(
                (f1_exc - f2_exc)[1:], -1)
            A_exc[0, 1] = -2 * f1_exc[1]
            A_exc[-1, -2] = 2 * f1_exc[-2]

            # RHS matrix (t)
            B_exc = np.diag(1 - 2 * f2_exc + f0_exc) + np.diagflat((f2_exc + f1_exc)[:-1], 1) + np.diagflat(
                (f2_exc - f1_exc)[1:], -1)
            B_exc[0, 1] = 2 * f1_exc[1]
            B_exc[-1, -2] = -2 * f1_exc[-2]

            # contribution to dp/dt from p_delta at u_res
            g_exc = p_exc_delta[i] * (-v_in_ee[i] * Fee_v + v_in_ei[i] * Fei_v)

            # calculate firing rate
            r_exc[i] = v_in_ee[i] * (self.c2ee[-1] * p_exc[-2, i] / self.dv + self.gamma_ee.sf(
                (self.u_thr - self.u_rest) / (self.u_exc - self.u_rest)) * p_exc_delta[i])
            if r_exc[i] < 0:
                r_exc[i] = 0
            r_exc_delayed[i + ref_exc_delta_idx] = r_exc[i]

            p_exc[:, i + 1] = np.linalg.solve(A_exc, np.matmul(B_exc, p_exc[:, i]))
            p_exc[:, i + 1] += self.dt * g_exc
            p_exc_delta[i + 1] = p_exc_delta[i] + self.dt * (
                        -(v_in_ee[i] + v_in_ei[i]) * p_exc_delta[i] + r_exc_delayed[i])

            # inhibitory population
            # ================================================================================================================
            # input to inhibitory population


            # coefficients for finite difference matrices
            f0_inh = self.dt / 2 * (1 / self.tau_mem[1] - v_in_ie[i] * self.c1ie_v + v_in_ii[i] * self.c1ii_v)
            f1_inh = self.dt / (4 * self.dv) * (
                        (self.v - self.u_rest) / self.tau_mem[1] + v_in_ie[i] * (-self.c1ie + self.c2ie_v) + v_in_ii[
                    i] * (self.c1ii + self.c2ii_v))
            f2_inh = self.dt / (2 * self.dv ** 2) * (v_in_ie[i] * self.c2ie + v_in_ii[i] * self.c2ii)

            # LHS matrix (t+dt)
            A_inh = np.diag(1 + 2 * f2_inh - f0_inh) + np.diagflat((-f2_inh - f1_inh)[:-1], 1) + np.diagflat(
                (f1_inh - f2_inh)[1:], -1)
            A_inh[0, 1] = -2 * f1_inh[1]
            A_inh[-1, -2] = 2 * f1_inh[-2]

            # RHS matrix (t)
            B_inh = np.diag(1 - 2 * f2_inh + f0_inh) + np.diagflat((f2_inh + f1_inh)[:-1], 1) + np.diagflat(
                (f2_inh - f1_inh)[1:], -1)
            B_inh[0, 1] = 2 * f1_inh[1]
            B_inh[-1, -2] = -2 * f1_inh[-2]

            # contribution to dp/dt from p_delta at u_res
            g_inh = p_inh_delta[i] * (-v_in_ie[i] * Fie_v + v_in_ii[i] * Fii_v)

            # calculate firing rate
            r_inh[i] = v_in_ie[i] * (self.c2ie[-1] * p_inh[-2, i] / self.dv + self.gamma_ie.sf(
                (self.u_thr - self.u_rest) / (self.u_exc - self.u_rest)) * p_inh_delta[i])
            if r_inh[i] < 0:
                # print(f"WARNING: r_inh < 0 ! (r_inh = {r_inh[i]}) ... Setting r_inh to 0")
                r_inh[i] = 0
            r_inh_delayed[i + ref_inh_delta_idx] = r_inh[i]

            # update p and p_delta
            p_inh[:, i + 1] = np.linalg.solve(A_inh, np.matmul(B_inh, p_inh[:, i]))
            p_inh[:, i + 1] += dt * g_inh
            p_inh_delta[i + 1] = p_inh_delta[i] + dt * (
                        -(v_in_ie[i] + v_in_ii[i]) * p_inh_delta[i] + r_inh_delayed[i])

        self.v_in_ee = v_in_ee
        self.v_in_ei = v_in_ei
        self.v_in_ie = v_in_ie
        self.v_in_ii = v_in_ii
        self.r_exc_delayed = r_exc_delayed
        self.ref_exc_delta_idx = ref_exc_delta_idx


        p_plot_exc = p_exc
        p_plot_exc[v_reset_idx, :] = p_exc[v_reset_idx, :] + p_exc_delta[:]
        r_plot_exc = r_exc[:]

        p_plot_inh = p_inh
        p_plot_inh[v_reset_idx, :] = p_inh[v_reset_idx, :] + p_inh_delta[:]
        r_plot_inh = r_inh[:]

        self.p_exc_delta = p_exc_delta
        self.p_exc = p_exc
        self.p_inh = p_inh
        self.p_inh_delta = p_inh_delta
        self.r_exc = r_exc
        self.r_inh = r_inh

        with h5py.File(self.name + '.hdf5', 'w') as h5file:
            h5file.create_dataset('t', data=self.t)
            h5file.create_dataset('v', data=self.v)
            h5file.create_dataset('r_exc', data=r_plot_exc)
            h5file.create_dataset('r_inh', data=r_plot_inh)
            h5file.create_dataset('rho_plot_exc', data=p_plot_exc)
            h5file.create_dataset('rho_plot_inh', data=p_plot_inh)

        # loop version of code
        self.get_coeffs_1()

        # first init all arrays
        v_reset_idx = np.where(np.isclose(self.v, self.u_reset))[0][0]  # index of reset potential in array
        self.v_reset_idx = v_reset_idx
        ref_delta_idxs = np.array([int(self.tau_ref[k] / self.dt) for k in range(self.n_populations)])
        max_ref_delta_indx = np.max(ref_delta_idxs)
        p = np.zeros((self.n_populations, len(self.v), len(self.t)))
        p_delta = np.zeros((self.n_populations, len(self.t)))
        r = np.zeros((self.n_populations, len(self.t)))  # output firing rate
        r_delayed = np.zeros((self.n_populations, (len(self.t) + max_ref_delta_indx)))
        v_in = np.zeros((self.n_populations, self.n_populations, len(self.t))) # ref_exc_delta_idx used to be here

        r_conv = np.zeros(self.n_populations)
        exc_idxs = [i for i, type in enumerate(self.population_type) if type == 'exc']
        inh_idxs = [i for i, type in enumerate(self.population_type) if type == 'inh']

        for i in range(len(self.population_type)):
            # initialize arrays
            p[i, :, 0] = scipy.stats.norm.pdf(self.v, self.u_rest, 1)
            p[i, 0, 0] = 0
            p[i, -1, 0] = 0

        # Determine population dynamics (diffusion approximation)
        for i, t_ in enumerate(tqdm(self.t[:-1], f'simulating Nykamp model for {self.t[:-1].shape[0]} time steps')):

            # if i > 0:
            #     r_conv = self.mat_convolve(r[:(i + 1), :], self.alpha)[:, :, -len(self.alpha)] * self.dt
            # v_in[:, :, i] = self.connectivity_matrix * r_conv + self.in2D

            for j, type_j in enumerate(self.population_type):

                # as of now r_conv has only one dimension, same as r
                # each entry is convoluted by its representing kernel
                if i > 0:
                    r_conv[j] = np.convolve(r[j, :(i + 1)], self.alpha)[-len(self.alpha)] * self.dt
                v_in[:, j, i] = self.connectivity_matrix[:, j] * r_conv + self.input[:, j, i]

                # v_in is the incoming FIRING RATE!
                # it is not a voltage so consider renaming it nu_in


                # coefficients for finite difference matrices
                # c1, c2 are over all v steps and i is a time step
                # here the values need to be probably split between excitatory and inhibitory types
                if type_j == 'exc':
                    c1ee = self.c[j, 0, 0]
                    c1ei = self.c[j, 0, 1]
                    c2ee = self.c[j, 1, 0]
                    c2ei = self.c[j, 1, 1]
                    c1ee_v = self.c_v[j, 0, 0]
                    c1ei_v = self.c_v[j, 0, 1]
                    c2ee_v = self.c_v[j, 1, 0]
                    c2ei_v = self.c_v[j, 1, 1]
                    # TODO: this can be collapsed into drawing out the coeffs, since they can be take out of the sum
                    f0_exc = self.dt / 2 * (1 / self.tau_mem[0] + np.sum(- v_in[exc_idxs, j, i]) * c1ee_v
                                            + np.sum(v_in[inh_idxs, j, i]) * c1ei_v)
                    f1_exc = self.dt / (4 * self.dv) * ((self.v - self.u_rest) / self.tau_mem[0] +
                                                        np.sum(v_in[exc_idxs, j, i]) * (-c1ee + c2ee_v) +
                                                        np.sum(v_in[inh_idxs, j, i]) * (c1ei + c2ei_v))
                    f2_exc = self.dt / (2 * self.dv ** 2) * (np.sum(v_in[exc_idxs, j, i]) * c2ee +
                                                             np.sum(v_in[inh_idxs, j, i]) * c2ei)

                    # LHS matrix (t+dt)
                    A_exc = np.diag(1 + 2 * f2_exc - f0_exc) + np.diagflat((-f2_exc - f1_exc)[:-1], 1) + np.diagflat(
                        (f1_exc - f2_exc)[1:], -1)
                    A_exc[0, 1] = -2 * f1_exc[1]
                    A_exc[-1, -2] = 2 * f1_exc[-2]

                    # RHS matrix (t)
                    B_exc = np.diag(1 - 2 * f2_exc + f0_exc) + np.diagflat((f2_exc + f1_exc)[:-1], 1) + np.diagflat(
                        (f2_exc - f1_exc)[1:], -1)
                    B_exc[0, 1] = 2 * f1_exc[1]
                    B_exc[-1, -2] = -2 * f1_exc[-2]

                    # contribution to dp/dt from p_delta at u_res
                    g_exc = p_delta[j, i] * (-np.sum(v_in[exc_idxs, j, i]) * self.dFdv[j, 0] +
                                               np.sum(v_in[inh_idxs, j, i]) * self.dFdv[j, 1])

                    # calculate firing rate
                    r[j, i] = np.sum(v_in[exc_idxs, j, i]) * (c2ee[-1] * p[j, -2, i] / self.dv +
                                                               self.gamma_funcs[j].sf((self.u_thr - self.u_rest) / (
                                                                           self.u_exc - self.u_rest)) *
                                                               p_delta[j, i])
                    if r[j, i] < 0:
                        r[j, i] = 0
                    r_delayed[j, i + ref_delta_idxs[j]] = r[j, i]

                    p[j, :, i + 1] = np.linalg.solve(A_exc, np.matmul(B_exc, p[j, :, i]))
                    p[j, :, i + 1] += self.dt * g_exc
                    p_delta[j, i + 1] = p_delta[j, i] + self.dt * (
                            -(np.sum(v_in[exc_idxs, j, i]) + np.sum(v_in[inh_idxs, j, i])) *
                            p_delta[j, i] + r_delayed[j, i])
                else:
                    # inhibitory population
                    # ================================================================================================================

                    c1ie = self.c[j, 0, 0]
                    c1ii = self.c[j, 0, 1]
                    c2ie = self.c[j, 1, 0]
                    c2ii = self.c[j, 1, 1]
                    c1ie_v = self.c_v[j, 0, 0]
                    c1ii_v = self.c_v[j, 0, 1]
                    c2ie_v = self.c_v[j, 1, 0]
                    c2ii_v = self.c_v[j, 1, 1]

                    # coefficients for finite difference matrices
                    f0_inh = self.dt / 2 * (1 / self.tau_mem[1] - np.sum(v_in[exc_idxs, j, i]) * c1ie_v +
                                            np.sum(v_in[inh_idxs, j, i]) * c1ii_v)
                    f1_inh = self.dt / (4 * self.dv) * (
                            (self.v - self.u_rest) / self.tau_mem[1] + np.sum(v_in[exc_idxs, j, i]) * (-c1ie + c2ie_v) +
                            np.sum(v_in[inh_idxs, j, i]) * (c1ii + c2ii_v))
                    f2_inh = self.dt / (2 * self.dv ** 2) * (np.sum(v_in[exc_idxs, j, i]) * c2ie +
                                                             np.sum(v_in[inh_idxs, j, i]) * c2ii)

                    # LHS matrix (t+dt)
                    A_inh = np.diag(1 + 2 * f2_inh - f0_inh) + np.diagflat((-f2_inh - f1_inh)[:-1], 1) + np.diagflat(
                        (f1_inh - f2_inh)[1:], -1)
                    A_inh[0, 1] = -2 * f1_inh[1]
                    A_inh[-1, -2] = 2 * f1_inh[-2]

                    # RHS matrix (t)
                    B_inh = np.diag(1 - 2 * f2_inh + f0_inh) + np.diagflat((f2_inh + f1_inh)[:-1], 1) + np.diagflat(
                        (f2_inh - f1_inh)[1:], -1)
                    B_inh[0, 1] = 2 * f1_inh[1]
                    B_inh[-1, -2] = -2 * f1_inh[-2]

                    # contribution to dp/dt from p_delta at u_res
                    g_inh = p_delta[j, i] * (-np.sum(v_in[exc_idxs, j, i]) * self.dFdv[j, 0] +
                                               np.sum(v_in[inh_idxs, j, i]) * self.dFdv[j, 1])

                    # calculate firing rate
                    if i == 250:
                        a=1
                    r[j, i] = np.sum(v_in[exc_idxs, j, i]) * (c2ie[-1] * p[j, -2, i] / self.dv +
                                                              self.gamma_funcs[j].sf((self.u_thr - self.u_rest) / (
                                                                          self.u_exc - self.u_rest)) *
                                                              p_delta[j, i])
                    if r[j, i] < 0:
                        # print(f"WARNING: r_inh < 0 ! (r_inh = {r_inh[i]}) ... Setting r_inh to 0")
                        r[j, i] = 0
                    r_delayed[j, i + ref_delta_idxs[j]] = r[j, i]

                    # update p and p_delta
                    p[j, :, i + 1] = np.linalg.solve(A_inh, np.matmul(B_inh, p[j, :, i]))
                    p[j, :, i + 1] += dt * g_inh
                    p_delta[j, i + 1] = p_delta[j, i] + dt * (
                            -(np.sum(v_in[exc_idxs, j, i]) + np.sum(v_in[inh_idxs, j, i])) *
                            p_delta[j, i] + r_delayed[j, i])

        p_plot_exc = p[0]
        p_plot_exc[v_reset_idx, :] = p[0, v_reset_idx, :] + p_delta[0, :]
        r_plot_exc = r[0, :]

        p_plot_inh = p[1]
        p_plot_inh[v_reset_idx, :] = p[1, v_reset_idx, :] + p_delta[1, :]
        r_plot_inh = r[1, :]

        if np.allclose(self.p_exc, p[0]) and np.allclose(self.p_inh, p[1]):

            print('works!')
        with h5py.File(self.name + '_1.hdf5', 'w') as h5file:
            h5file.create_dataset('t', data=self.t)
            h5file.create_dataset('v', data=self.v)
            h5file.create_dataset('r_exc', data=r_plot_exc)
            h5file.create_dataset('r_inh', data=r_plot_inh)
            h5file.create_dataset('rho_plot_exc', data=p_plot_exc)
            h5file.create_dataset('rho_plot_inh', data=p_plot_inh)

    def mat_convolve(self, x, kernel):
        """
        Function that convolves an array of time series with the same kernel unsing np.convolve

        Parameter
        ---------
        x : np.array of float (2D)
            input array
        kernel : np.array of float (1D)
            convolution kernel

        Returns
        -------
        B : np.array of float (2D)
            convolved input array
        """

        n_x = x.shape[0]
        n_y = x.shape[1]

        # use extra convolution to determine shape of output array
        n_b = np.convolve(x[0, 0], kernel).shape[0]

        b = np.zeros((n_x, n_y, n_b))
        for i in range(n_x):
            for j in range(n_y):
                b[i, j] = np.convolve(x[i, j], kernel)
        return b

    def get_coeffs(self):

        # TODO: make these matrix operations, get all diffusion coeffs at once?
        v = np.arange(self.u_inh, self.u_thr + self.dv / 2, self.dv)
        n_v = v.shape[0]

        n_coeff = 4 * self.n_populations
        self.diffusion_coefficients = np.zeros([n_coeff, n_v])
        self.diffusion_coefficients_dv = np.zeros([n_coeff, n_v])
        self.dFdv = np.zeros([int(n_coeff / 2), n_v])

        self.scales = []
        for k in range(self.n_populations):
            if self.population_type[k] == 'exc':

                # init arrays for diffuson coeffs
                c1ee = np.zeros(len(v))
                c2ee = np.zeros(len(v))
                c1ei = np.zeros(len(v))
                c2ei = np.zeros(len(v))

                # synapse parameters
                mu_ee = self.mu_gamma[k, 0]
                mu_ei = self.mu_gamma[k, 1]

                coeff_var_ee = self.var_coeff_gamma[k, 0]
                coeff_var_ei = self.var_coeff_gamma[k, 1]
                var_ee = (coeff_var_ee * mu_ee) ** 2
                var_ei = (coeff_var_ei * mu_ei) ** 2

                scale_ee = var_ee / mu_ee
                scale_ei = var_ei / mu_ei

                self.scales.append(scale_ee)
                self.scales.append(scale_ei)

                # conductance jump distributions
                gamma_ee = gamma(a=coeff_var_ee ** (-2), loc=0, scale=scale_ee)
                gamma_ei = gamma(a=coeff_var_ei ** (-2), loc=0, scale=scale_ei)

                for i, v_ in enumerate(v):

                    vpe = np.arange(self.u_inh, v_ + self.dv / 2, self.dv)
                    int_c1ee = gamma_ee.sf((v_ - vpe) / (self.u_exc - vpe))
                    int_c2ee = int_c1ee * (v_ - vpe)
                    c1ee[i] = np.trapz(x=vpe, y=int_c1ee)
                    c2ee[i] = np.trapz(x=vpe, y=int_c2ee)

                    if i > 0:
                        vpi = np.arange(v_, self.u_thr + self.dv / 2, self.dv)
                        int_c1ei = gamma_ei.sf((v_ - vpi) / (self.u_inh - vpi))
                        int_c2ei = int_c1ei * (vpi - v_)
                        c1ei[i] = np.trapz(x=vpi, y=int_c1ei)
                        c2ei[i] = np.trapz(x=vpi, y=int_c2ei)

                c1ei[0] = c1ei[1]
                c2ei[0] = 0

                self.diffusion_coefficients[4*k:4*(k+1)] = np.array([c1ee, c2ee, c1ei, c2ei])
                self.gamma_ee = gamma_ee
                Fee_v = np.gradient(gamma_ee.sf(x=(self.v - self.u_rest) / (self.u_exc - self.u_rest)), self.dv) \
                        * np.heaviside(self.v - self.u_rest, 0.5)
                Fei_v = np.gradient(gamma_ei.sf(x=(self.v - self.u_rest) / (self.u_inh - self.u_rest)), self.dv) \
                        * np.heaviside(self.u_rest - self.v, 0.5)

                self.dFdv[2*k:2*(k+1)] = np.array([Fee_v, Fei_v])


            elif self.population_type[k] == 'inh':


                # init arrays for diffusion coeffs
                c1ie = np.zeros(len(v))
                c2ie = np.zeros(len(v))
                c1ii = np.zeros(len(v))
                c2ii = np.zeros(len(v))

                # synapse parameters
                mu_ie = self.mu_gamma[k, 0]
                mu_ii = self.mu_gamma[k, 1]

                coeff_var = self.var_coeff_gamma[k, 0]
                var_ie = (coeff_var * mu_ie) ** 2
                var_ii = (coeff_var * mu_ii) ** 2

                scale_ie = var_ie / mu_ie
                scale_ii = var_ii / mu_ii

                # conductance jump distributions
                gamma_ie = gamma(a=coeff_var ** (-2), loc=0, scale=scale_ie)
                gamma_ii = gamma(a=coeff_var ** (-2), loc=0, scale=scale_ii)

                for i, v_ in enumerate(v):

                    vpe = np.arange(self.u_inh, v_ + self.dv / 2, self.dv)
                    int_c1ie = gamma_ie.sf((v_ - vpe) / (self.u_exc - vpe))
                    int_c2ie = int_c1ie * (v_ - vpe)
                    c1ie[i] = np.trapz(x=vpe, y=int_c1ie)
                    c2ie[i] = np.trapz(x=vpe, y=int_c2ie)

                    if i > 0:
                        vpi = np.arange(v_, self.u_thr + self.dv / 2, self.dv)
                        int_c1ii = gamma_ii.sf((v_ - vpi) / (self.u_inh - vpi))
                        int_c2ii = int_c1ii * (vpi - v_)
                        c1ii[i] = np.trapz(x=vpi, y=int_c1ii)
                        c2ii[i] = np.trapz(x=vpi, y=int_c2ii)


                c1ii[0] = c1ii[1]
                c2ii[0] = 0

                self.diffusion_coefficients[4*k:4*(k+1)] = c1ie, c2ie, c1ii, c2ii
                self.gamma_ie = gamma_ie
                Fie_v = np.gradient(gamma_ie.sf(x=(self.v - self.u_rest) / (self.u_exc - self.u_rest)), self.dv) \
                                   * np.heaviside(self.v - self.u_rest, 0.5)
                Fii_v = np.gradient(gamma_ii.sf(x=(self.v - self.u_rest) / (self.u_inh - self.u_rest)), self.dv) \
                                   * np.heaviside(self.u_rest - self.v, 0.5)
                self.dFdv[2*k:2*(k+1)] = np.array([Fie_v, Fii_v])

            else:
                raise NotImplementedError('population types must be "exc" or "inh"!')

    def get_coeffs_1(self):

        # TODO: make these matrix operations, get all diffusion coeffs at once?
        v = np.arange(self.u_inh, self.u_thr + self.dv / 2, self.dv)
        n_v = v.shape[0]

        self.c = np.zeros((self.n_populations, 2, 2, n_v))
        self.c_v = np.zeros_like(self.c)
        self.dFdv = np.zeros([self.n_populations, 2, n_v])

        self.gamma_funcs = []
        for k in range(self.n_populations):
            if self.population_type[k] == 'exc':

                # init arrays for diffuson coeffs
                c1ee = np.zeros(len(v))
                c2ee = np.zeros(len(v))
                c1ei = np.zeros(len(v))
                c2ei = np.zeros(len(v))

                # conductance jump distributions
                # input synapse parameters into the distribution of conductance jumps
                gamma_ee = gamma(a=self.var_coeff_gamma[k, 0] ** (-2), loc=0,
                                 scale=self.var_coeff_gamma[k, 0]**2*self.mu_gamma[k, 0])
                gamma_ei = gamma(a=self.var_coeff_gamma[k, 1] ** (-2),
                                 loc=0, scale=self.var_coeff_gamma[k, 1]**2*self.mu_gamma[k, 1])

                if self.synapse_pdf_type == 'gamma':
                    synapse_pdf_ee = gamma(a=self.synapse_pdf_params[0, k, 0] ** (-2), loc=0,
                                     scale=self.synapse_pdf_params[0, k, 0] ** 2 * self.synapse_pdf_params[1, k, 0])
                    synapse_pdf_ei = gamma(a=self.synapse_pdf_params[0, k, 1] ** (-2),
                                     loc=0, scale=self.synapse_pdf_params[0, k, 1] ** 2 * self.synapse_pdf_params[1, k, 1])



                for i, v_ in enumerate(v):

                    vpe = np.arange(self.u_inh, v_ + self.dv / 2, self.dv)
                    int_c1ee = synapse_pdf_ee.sf((v_ - vpe) / (self.u_exc - vpe))
                    int_c2ee = int_c1ee * (v_ - vpe)
                    c1ee[i] = np.trapz(x=vpe, y=int_c1ee)
                    c2ee[i] = np.trapz(x=vpe, y=int_c2ee)

                    if i > 0:
                        vpi = np.arange(v_, self.u_thr + self.dv / 2, self.dv)
                        int_c1ei = synapse_pdf_ei.sf((v_ - vpi) / (self.u_inh - vpi))
                        int_c2ei = int_c1ei * (vpi - v_)
                        c1ei[i] = np.trapz(x=vpi, y=int_c1ei)
                        c2ei[i] = np.trapz(x=vpi, y=int_c2ei)

                c1ei[0] = c1ei[1]
                c2ei[0] = 0

                self.c[k] = np.array([[[c1ee, c1ei], [c2ee, c2ei]]])
                c1ee_v = np.gradient(c1ee, self.v)
                c1ei_v = np.gradient(c1ei, self.v)
                c2ee_v = np.gradient(c2ee, self.v)
                c2ei_v = np.gradient(c2ei, self.v)
                self.c_v[k] = np.array([[[c1ee_v, c1ei_v], [c2ee_v, c2ei_v]]])
                self.synapse_pdf_funcs.append(synapse_pdf_ee)
                Fee_v = np.gradient(synapse_pdf_ee.sf(x=(self.v - self.u_rest) / (self.u_exc - self.u_rest)), self.dv) \
                        * np.heaviside(self.v - self.u_rest, 0.5)
                Fei_v = np.gradient(synapse_pdf_ei.sf(x=(self.v - self.u_rest) / (self.u_inh - self.u_rest)), self.dv) \
                        * np.heaviside(self.u_rest - self.v, 0.5)

                self.dFdv[k] = np.array([Fee_v, Fei_v])


            elif self.population_type[k] == 'inh':


                # init arrays for diffusion coeffs
                c1ie = np.zeros(len(v))
                c2ie = np.zeros(len(v))
                c1ii = np.zeros(len(v))
                c2ii = np.zeros(len(v))

                # conductance jump distributions
                synapse_pdf_ie = gamma(a=self.var_coeff_gamma[k, 0] ** (-2), loc=0,
                                 scale=self.var_coeff_gamma[k, 0] ** 2 * self.mu_gamma[k, 0])
                synapse_pdf_ii = gamma(a=self.var_coeff_gamma[k, 1] ** (-2),
                                 loc=0, scale=self.var_coeff_gamma[k, 1] ** 2 * self.mu_gamma[k, 1])

                for i, v_ in enumerate(v):

                    vpe = np.arange(self.u_inh, v_ + self.dv / 2, self.dv)
                    int_c1ie = synapse_pdf_ie.sf((v_ - vpe) / (self.u_exc - vpe))
                    int_c2ie = int_c1ie * (v_ - vpe)
                    c1ie[i] = np.trapz(x=vpe, y=int_c1ie)
                    c2ie[i] = np.trapz(x=vpe, y=int_c2ie)

                    if i > 0:
                        vpi = np.arange(v_, self.u_thr + self.dv / 2, self.dv)
                        int_c1ii = synapse_pdf_ii.sf((v_ - vpi) / (self.u_inh - vpi))
                        int_c2ii = int_c1ii * (vpi - v_)
                        c1ii[i] = np.trapz(x=vpi, y=int_c1ii)
                        c2ii[i] = np.trapz(x=vpi, y=int_c2ii)

                c1ii[0] = c1ii[1]
                c2ii[0] = 0

                self.c[k] = np.array([[c1ie, c1ii], [c2ie, c2ii]])
                c1ie_v = np.gradient(c1ie, self.v)
                c1ii_v = np.gradient(c1ii, self.v)
                c2ie_v = np.gradient(c2ie, self.v)
                c2ii_v = np.gradient(c2ii, self.v)
                self.c_v[k] = np.array([[[c1ie_v, c1ii_v], [c2ie_v, c2ii_v]]])
                self.synapse_pdf_funcs.append(synapse_pdf_ie)
                Fie_v = np.gradient(synapse_pdf_ie.sf(x=(self.v - self.u_rest) / (self.u_exc - self.u_rest)), self.dv) \
                                   * np.heaviside(self.v - self.u_rest, 0.5)
                Fii_v = np.gradient(synapse_pdf_ii.sf(x=(self.v - self.u_rest) / (self.u_inh - self.u_rest)), self.dv) \
                                   * np.heaviside(self.u_rest - self.v, 0.5)
                self.dFdv[k] = np.array([Fie_v, Fii_v])

            else:
                raise NotImplementedError('population types must be "exc" or "inh"!')

    def get_alpha_kernel(self):
        self.t_alpha = self.t[self.t < 10]
        self.alpha = np.exp(-self.t_alpha/self.tau_alpha) / (self.tau_alpha * scipy.special.factorial(self.n_alpha-1)) *\
                (self.t_alpha/self.tau_alpha)**(self.n_alpha-1)
        self.alpha = self.alpha/np.trapz(self.alpha, dx=self.dt)

class Nykamp_Model_1:
    """
    Class for the population model of Nykamp and Tranchina
    Probability flux over time is used to model synaptic membrane and firing rates of neuron
    populations
    """

    def __init__(self, parameters={}, name=None):

        self.connectivity_matrix = np.ones((1, 1))
        self.delay_kernel_type = 'alpha'
        self.delay_kernel_parameters = {'n_alpha': 9, 'tau_alpha': 1/3}
        self.name = 'Nykamp_example'
        self.u_reset = -70
        self.u_rest = -70
        self.u_thr = -55
        self.u_exc = 0
        self.u_inh = -75
        self.tau_mem = [20]
        self.tau_ref = [1]
        self.mu_gamma = np.array([[0.008, 0.027]])
        self.var_coeff_gamma = 0.5*np.ones((1, 2))
        self.dt = 0.1
        self.dv = 0.01
        self.T = 100
        self.tqdm_disable = False
        self.sparse_mat = True
        self.input_function = np.zeros((int(self.T/self.dt)))
        self.input_function_idx = [0, 0]
        self.multiple_inputs = False
        self.population_type = ['exc']
        self.synapse_pdf_type = 'gamma'
        self.verbose = 0
        self.solver = 'nykamp'
        self.implemented_pdf_types = ['gamma', 'normal', 'log-normal']
        self.voltage_sigma = 5
        self.init_pdf_offset = 0
        self.init_pdf_sigma = 0.1
        self.static_noise = False
        self.flux_scheme = 'bernoulli'
        self.time_scheme = 'bdf2'
        self.linear_solver = 'banded'
        # saving of p (Hu-2021 solver only, the Nykamp solver always saves all time steps): 'none' (only r is
        # saved), 'end' (only the last time step) or 'all' (every p_save_every-th time step)
        self.save_p = 'all'
        self.p_save_every = 1

        self.g_eext_factor = 1
        self.c_eext1_factor = 1
        self.c_eext2_factor = 1
        self.current_factor = 1

        self.init_pdf_weight = 1
        self.thr_idx = -1
        self.verbose = 0
        self.labelsize=15

        # input current
        self.input_type = 'rate'

        self.__dict__.update(parameters)

        self.parameters = parameters

        if name is not None:
            self.name = name

        self.n_populations = len(self.population_type)

        # leakage conductance
        if not 'neuron_size' in parameters:
            self.neuron_size = 3.0 * np.ones(self.n_populations)  # unit is mm average is here for l5pt cells

        if not 'g_leak' in parameters:
            # avg_c_mem_per_mm = 0.1  # generally accepted avg
            # self.c_mem = self.neuron_size * avg_c_mem_per_mm
            self.c_mem = 1.1e-10  # (0.5 nF cell total c_mem on average) # this estimate is now for the L5 TTP cells
            # from papers

            # compute g_leak from tau and capacitance
            # self.g_leak = [1e-5]*self.n_populations
            # self.g_leak = self.c_mem/self.tau_mem * 1e-3  # (conversion to mS from µS)
            self.g_leak = self.c_mem / (np.array(self.tau_mem)*1e-3)  # (conversion from ms t s for tau_mem)
            # self.g_leak = np.array(1/30e6).reshape((1, 1))
        if self.synapse_pdf_type == 'gamma':
            self.synapse_pdf_params = np.array([self.var_coeff_gamma, self.mu_gamma])
        # set up arrays for simulation
        self.t = np.arange(0, self.T, self.dt)
        self.v = np.arange(self.u_inh, self.u_thr + self.dv / 2, self.dv)
        if len(self.connectivity_matrix.shape) == 1:
            self.connectivity_matrix = self.connectivity_matrix[:, np.newaxis]

        # calculate alpha kernel
        self.get_delay_kernel()

        # initialize breaking variable for numerically unstable solutions
        self.break_outer = False

        if type(self.connectivity_matrix) != np.ndarray:
            self.connectivity_matrix = np.array(self.connectivity_matrix)

        self.simulation_done = False

        if not self.input_type in ['rate', 'current', 'current-2', 'stochastic-current']:
            raise NotImplementedError(f"Input type {self.input_type} not implemented! Please chose from "
                                      f"['rate', 'current', 'current-2', 'stochastic-current']")

        if not self.flux_scheme in ['harmonic', 'harmonic-stable', 'bernoulli']:
            raise NotImplementedError(f"flux_scheme {self.flux_scheme} not implemented! Please chose from "
                                      f"['harmonic', 'harmonic-stable', 'bernoulli']")
        if not self.time_scheme in ['backward-euler', 'bdf2']:
            raise NotImplementedError(f"time_scheme {self.time_scheme} not implemented! Please chose from "
                                      f"['backward-euler', 'bdf2']")
        if not self.linear_solver in ['sparse', 'banded']:
            raise NotImplementedError(f"linear_solver {self.linear_solver} not implemented! Please chose from "
                                      f"['sparse', 'banded']")
        self.p_save_every = int(self.p_save_every)
        if self.p_save_every < 1:
            raise ValueError(f"p_save_every has to be >= 1, got {self.p_save_every}")
        if self.p_save_every > 1 and self.solver.lower() != 'hu-2021':
            raise NotImplementedError("p_save_every > 1 is only implemented for the Hu-2021 solver")
        if not self.save_p in ['none', 'end', 'all']:
            raise NotImplementedError(f"save_p {self.save_p} not implemented! Please chose from "
                                      f"['none', 'end', 'all']")

    def simulate(self):

        self.t = np.arange(0, self.T, self.dt)
        self.v = np.arange(self.u_inh, self.u_thr + self.dv / 2, self.dv)

        self.input = np.zeros([self.n_populations, self.n_populations, self.t.shape[0]])
        self.i_ext = np.zeros([self.n_populations, self.t.shape[0]])
        # set up input_function for rate over time or current over time
        if not callable(self.input_function) and type(self.input_function) == np.ndarray:
            assert self.input_function.shape[0] == self.t.shape[0], f'Dimension missmatch between input with dimension'\
                                                                    f' {self.input_function.shape[0]} and simulation' \
                                                                    f' time with dimension {self.t.shape[0]}!'
            input_function_copy = self.input_function
            self.input_function = lambda x: input_function_copy  # hotfix that just returns the array for any input
        if self.input_type == 'rate':
            if self.multiple_inputs == True:
                for k in range(len(self.input_function_idx)):
                    self.input[self.input_function_idx[k][0], self.input_function_idx[k][1]] = self.input_function[k](t=self.t)
            else:
                try:
                    self.input[self.input_function_idx[0], self.input_function_idx[1]] = self.input_function(x=self.t)
                except:
                    self.input[self.input_function_idx[0], self.input_function_idx[1]] = self.input_function(t=self.t)
        elif self.input_type in ['current', 'current-2', 'stochastic-current']:
            self.i_ext[self.input_function_idx] = self.input_function(self.t) * self.current_factor
            if not isinstance(self.voltage_sigma, np.ndarray):
                if self.static_noise:
                    self.voltage_sigma = np.tile(self.voltage_sigma, (self.n_populations, self.t.shape[0]))
                else:
                    self.voltage_sigma = self.voltage_sigma*self.i_ext


        if self.verbose > 1:
            t0_coeff = time.time()

        self.get_coeffs()
        self.r = None

        if self.verbose > 1:
            t1_coeff = time.time()
            print(f'time for coeffs: {t1_coeff - t0_coeff:.3f}s')

        # first init all arrays
        v_reset_idx = np.where(np.isclose(self.v, self.u_reset))[0][0]  # index of reset potential in array
        self.v_reset_idx = v_reset_idx
        ref_delta_idxs = np.array([int(np.round(self.tau_ref[k] / self.dt)) for k in range(self.n_populations)])
        max_ref_delta_indx = np.max(ref_delta_idxs)
        # save_p defines if p is saved during the computation, if it is not needed 'none' or only the last step
        # is needed 'end' computation is faster (less memory overhead), in general it is a very useful tool to see
        # the voltage distribution. Using the old Nykamp solver requires it in any case.
        save_p = self.save_p if self.solver.lower() == 'hu-2021' else 'all'
        self.t_p = self.t[::self.p_save_every] if save_p == 'all' else self.t[:1]
        p = np.zeros((self.n_populations, len(self.v), len(self.t_p)))
        p_delta = np.zeros((self.n_populations, len(self.t)))
        r = np.zeros((self.n_populations, len(self.t)))  # output firing rate
        r_delayed = np.zeros((self.n_populations, (len(self.t) + max_ref_delta_indx)))
        v_in = np.zeros((self.n_populations, self.n_populations, len(self.t))) # ref_exc_delta_idx used to be here

        r_conv = np.zeros(self.n_populations)
        exc_idxs = [i for i, type in enumerate(self.population_type) if type == 'exc']
        inh_idxs = [i for i, type in enumerate(self.population_type) if type == 'inh']

        self.c1eext = np.zeros_like(self.v)
        self.c2eext = np.zeros_like(self.v)
        self.c1eext_v = np.zeros_like(self.v)
        self.c2eext_v = np.zeros_like(self.v)

        ################################################################################################################
        # INITIAL CONDITIONS
        ################################################################################################################
        for i in range(len(self.population_type)):
            # initial distribution
            if self.solver.lower() == 'nykamp':
                init_dist = scipy.stats.norm.pdf(self.v, self.u_rest + self.init_pdf_offset, self.init_pdf_sigma)
                init_dist /= init_dist.sum()
                # init_dist /= 10
                p[i, :, 0] = init_dist
                p[i, 0, 0] = 0
                p[i, -1, 0] = 0

        # transform to v = (0, 1) range if Hu- Solver is used which cannot handle negative u_rest values
        if self.solver.lower() == 'hu-2021':
            v_ = np.linspace(0, 1,self.v.shape[0])
            dv_ = np.diff(v_)[0]

            v_rest_idx_orig = np.where(self.v > self.u_rest)[0][0] - 1
            u_rest_ = v_[v_rest_idx_orig]

            v_reset_idx_orig = np.where(self.v >self.u_reset)[0][0] - 1
            u_reset_ = v_[v_reset_idx_orig]

            v_range_orig = self.v.max() - self.v.min()
            v_range_new = 1.0
            hu_scaling_factor = dv_/self.dv

            v_ = self.v
            dv_ = self.dv
            u_rest_ = self.u_rest
            u_reset_ = self.u_reset
            hu_scaling_factor = 1

            for i in range(len(self.population_type)):
            # initial distribution
                sigma0_new = self.init_pdf_sigma * hu_scaling_factor
                v0_idx_original = np.where(self.v > self.u_rest + self.init_pdf_offset)[0][0] - 1
                v0_new = v_[v0_idx_original]
                # Gaussian component
                init_dist_1 = norm.pdf(v_, v0_new, sigma0_new)
                init_dist_1 /= init_dist_1.sum()

                # Uniform distributed component
                init_dist_2 = np.ones_like(v_)
                init_dist_2[np.where(v_ < u_rest_)] = 0
                init_dist_2 /= init_dist_2.sum()
                init_dist = init_dist_1 + (self.init_pdf_weight * init_dist_2)
                init_dist[0] = init_dist[-1] = 0
                # scale it such that the integral dv over the dist sums to 1: it is dependent on dv!
                init_dist /= init_dist.sum() * self.dv

                p[i, :, 0] = init_dist
                p[i, 0, 0] = 0
                p[i, -1, 0] = 0

            # states at the last two time steps, needed for the time stepping
            p_last_step = [p[k, :, 0].copy() for k in range(self.n_populations)]
            p_prev_step = [None] * self.n_populations
            n_clipped = 0

        c_count = 0
        c_v_warn_count = 0
        added_noise = np.zeros(self.t.shape[0] - 1)
        injected_voltage = np.zeros(self.t.shape[0] - 1)

        # Determine population dynamics (diffusion approximation)
        for i, t_ in enumerate(tqdm(self.t[:-1], f"simulating {self.population_type} neuron populations for"
                                                 f" {self.t[:-1].shape[0]} time steps", disable=self.tqdm_disable)):

            # if i > 0:
            #     r_conv = self.mat_convolve(r[:(i + 1), :], self.alpha)[:, :, -len(self.alpha)] * self.dt
            # v_in[:, :, i] = self.connectivity_matrix * r_conv + self.in2D

            # if self.voltage_idx is not None:
            #     # map shift to dv
            #     v_shift = int(self.input_voltage[i] / self.dv)
            #     p[self.voltage_idx, :, i] = np.roll(p[self.voltage_idx, :, i], v_shift)


            if self.break_outer:
                break

            for j, type_j in enumerate(self.population_type):

                # as of now r_conv has only one dimension, same as r
                # each entry is convoluted by its representing kernel
                if i > 0:
                    # kernel length, if simulated tim so far is faster, compute only kernel part that fits in simulation history
                    n_k = min(i + 1, len(self.delay_kernel))
                    r_conv[j] = np.dot(r[j, i - n_k + 1:i + 1][::-1], self.delay_kernel[:n_k]) * self.dt
                v_in[:, j, i] = self.connectivity_matrix[:, j] * r_conv + self.input[:, j, i]

                # coefficients for finite difference matrices
                # c1, c2 are over all v steps and i is a time step
                # here the values are split between excitatory and inhibitory types
                if type_j == 'exc':
                    # excitatory population
                    # ==================================================================================================
                    c1ee = self.c[j, 0, 0]
                    c1ei = self.c[j, 0, 1]
                    c2ee = self.c[j, 1, 0]
                    c2ei = self.c[j, 1, 1]
                    c1ee_v = self.c_v[j, 0, 0]
                    c1ei_v = self.c_v[j, 0, 1]
                    c2ee_v = self.c_v[j, 1, 0]
                    c2ei_v = self.c_v[j, 1, 1]

                    g_eext = 0
                    F_ext_delta = 0
                    v_in_i_ext = 0
                    v_ext = 0

                    if self.input_type == 'current':
                        ################################################################################################
                        # EXTERNAL COEFFS
                        # Additional external coefficients and pdfs to handle constant current input as dirac
                        # distributed in voltage space
                        ################################################################################################
                        v_ext = self.i_ext[j, i] / self.g_leak[j]
                        mask1 = np.where(self.v < v_ext + self.u_inh)[0]
                        mask2 = np.where(self.v > v_ext + self.u_inh)[0]

                        self.c1eext[mask1] = self.v[mask1] - self.u_inh
                        self.c1eext[mask2] = v_ext
                        self.c1eext_v[mask1] = 1
                        self.c1eext_v[mask2] = 0

                        self.c2eext[mask1] = 0.5 * (self.v[mask1] - self.u_inh) ** 2
                        self.c2eext[mask2] = 0.5 * v_ext ** 2
                        self.c2eext_v[mask1] = (self.v[mask1] - self.u_inh)
                        self.c2eext_v[mask2] = 0

                        self.c1eext = self.c_eext1_factor*self.c1eext
                        self.c1eext_v = self.c_eext1_factor*self.c1eext_v
                        self.c2eext = self.c_eext2_factor*self.c2eext  # was 0.2
                        self.c2eext_v = self.c_eext2_factor*self.c2eext_v  # was 0.2

                        # self.c1eext[0] = 0
                        # self.c2eext[0] = 0
                        # self.c1eext_v[0] = 0
                        # self.c2eext_v[0] = 0
                        # self.c1eext[-1] = 0
                        # self.c2eext[-1] = 0
                        # self.c1eext_v[-1] = 0
                        # self.c2eext_v[-1] = 0

                        # all new delay component terms
                        # apply dirac delta at v = self.u_rest + v_ext
                        g_eext = np.zeros_like(self.v)
                        # dirac_index = np.where(self.v > self.u_reset + v_ext)[0]
                        # if dirac_index.size == 0:
                        #     # dirac_index = -2  # set dirac to latest point in v-space to preserve flux?
                        #     dirac_index = -50 # safety distance if gauss_func is used in g_eext
                        #     pass
                        # else:
                        #     dirac_index = dirac_index[0]
                        dirac_index = np.where(self.v > self.u_reset)[0][0] # insert a v_reset
                        if p_delta[j, i-100] > 0:
                            a=1
                        # g_eext[dirac_index] = - p_delta[j, i] #* 100 # 100 was the area under the curve of the pdf
                        # g_eext = self.gauss_func(x = self.v, mu=(v_ext + self.u_reset), sigma=0.1)
                        g_eext = self.gauss_func(x=self.v, mu=self.v[dirac_index], sigma=0.1)
                        g_eext[:10] = 0
                        g_eext[-10:] = 0
                        # g_eext = self.gauss_func(x=self.v, mu=self.u_reset+5, sigma=2)
                        g_eext /= g_eext.sum()
                        g_eext = g_eext * -p_delta[j, i] * self.g_eext_factor
                        # F_ext_delta = np.heaviside(-self.u_thr + v_ext + self.u_reset, 0.5)
                        F_ext_delta = 1*self.sigmoid(-self.u_thr + v_ext + self.u_reset, r=0.01)
                        v_in_i_ext = 1

                    current_2_val = 0
                    if self.input_type == 'current-2':
                        v_ext = self.i_ext[j, i] / self.g_leak[j] * 1e3  # conversion from V to mV
                        current_2_val = 1
                    elif self.input_type == 'stochastic-current':
                        ################################################################################################
                        # EXTERNAL COEFFS
                        # Additional external coefficients and pdfs to handle random current input with mean and variance
                        ################################################################################################
                        v_ext = self.i_ext[j, i] / self.g_leak[j] * 1e3  # conversion from V to mV
                        sigma = self.voltage_sigma[j, i]**2/self.tau_mem[j]
                        v_shape = self.v.shape[0]
                        self.c1eext = np.repeat(v_ext, v_shape)
                        self.c1eext_v = 0
                        self.c2eext = np.repeat(sigma, v_shape)
                        self.c2eext_v = 0

                        # This part is not used in the Hu-solver (skipped there)
                        if self.solver.lower() != 'hu-2021':
                            dirac_index = np.where(self.v > self.u_reset)[0][0]  # insert a v_reset
                            g_eext = self.gauss_func(x=self.v, mu=self.v[dirac_index], sigma=0.1)
                            g_eext[:10] = 0
                            g_eext[-10:] = 0
                            g_eext /= g_eext.sum()
                            g_eext = g_eext * -p_delta[j, i] * self.g_eext_factor
                            F_ext_delta = 1 * self.sigmoid(-self.u_thr + v_ext + self.u_reset, r=0.01)
                        v_in_i_ext = 1


                    # TODO: this can be collapsed into drawing out the coeffs, since they can be taken out of the sum
                    #  check if this is correct
                    if self.solver.lower() == 'nykamp':
                        f0_exc = self.dt / 2 * (1 / self.tau_mem[0] + np.sum(- v_in[exc_idxs, j, i]) * c1ee_v
                                                + np.sum(v_in[inh_idxs, j, i]) * c1ei_v - self.c1eext_v)
                        f1_exc = self.dt / (4 * self.dv) * ((self.v - self.u_rest - current_2_val*v_ext) / self.tau_mem[0] +
                                                            - self.c1eext + self.c2eext_v +  # new external inputs
                                                            np.sum(v_in[exc_idxs, j, i]) * (-c1ee + c2ee_v) +
                                                            np.sum(v_in[inh_idxs, j, i]) * (c1ei + c2ei_v))
                        f2_exc = self.dt / (2 * self.dv ** 2) * (np.sum(v_in[exc_idxs, j, i]) * c2ee +
                                                                 np.sum(v_in[inh_idxs, j, i]) * c2ei + self.c2eext)
                        # else:


                        A_exc = self.get_A(f0_exc, f1_exc, f2_exc)
                        B_exc = self.get_B(f0_exc, f1_exc, f2_exc)

                        # contribution to dp/dt from p_delta at u_res

                        g_exc = p_delta[j, i] * (-np.sum(v_in[exc_idxs, j, i]) * self.dFdv[j, 0] +
                                                   np.sum(v_in[inh_idxs, j, i]) * self.dFdv[j, 1]) - v_in_i_ext * g_eext


                        # calculate firing rate
                        r_j = np.sum(v_in[exc_idxs, j, i]) * (c2ee[-1] * p[j, -2, i] / self.dv +
                                                                   self.synapse_pdf_funcs[j].sf(
                                                                       (self.u_thr - self.u_rest) / (
                                                                               self.u_exc - self.u_rest)) *
                                                                   p_delta[j, i])
                        r_ext = + self.c2eext[-1] * p[j, -2, i] / self.dv + F_ext_delta * p_delta[j, i]
                        r[j, i] = r_j + r_ext

                        if i == 600:
                            a=1

                        # if r[j, i] < 0:
                        #     r[j, i] = 0
                        if np.isnan(r[j, i]):
                            a = 0
                        if np.abs(r[j, i] - r[j, i-1]) > 30e-3 and i > 1:
                            l = 12

                        if i == 0:
                            neg_p_counter = 0
                            p_shrink_counter = 0
                            p_inflate_counter = 0
                            p_area_start = np.sum(p[j, :, i])
                        if neg_p_counter == 0 and np.mean(p[j, :, i]) < -1:
                            print('\nWarning!, negative p detected!')
                            neg_p_counter = 1
                        mean_p_area = np.sum(p[j, :, :], axis=0)
                        mean_p_idx = int(2*ref_delta_idxs[j])
                        if i > mean_p_idx and self.verbose > 1:
                            if np.mean(mean_p_area[i-mean_p_idx:i]) < p_area_start*0.2 and p_shrink_counter == 0:
                                # print(f'\n Warning!, p is draining, i = {i}')
                                p_shrink_counter = 1
                            elif mean_p_area[i-1] > p_area_start*1.1 and p_inflate_counter == 0:
                                # print(f'\n Warning!, p is inlfating, i = {i}')
                                p_inflate_counter = 1

                        if r[j, i] > 1e6:
                            if self.sparse_mat:
                                A = np.array(A_exc.todense())
                                B = np.array(B_exc.todense())
                                kappa_A = np.linalg.cond(A)
                                kappa_B = np.linalg.cond(B)
                            else:
                                kappa_A = np.linalg.cond(A_exc)
                                kappa_B = np.linalg.cond(B_exc)

                            dt_proposal = self.dt/kappa_A*50  # may use 75 for boarder to instability

                            print(f'Exiting simulation: rate of {r[j, i]:5f} detected \n Condition number for A and B are: '
                                  f'{kappa_A}, {kappa_B} \n Try dt<= {dt_proposal:.6f}')
                            self.break_outer = True
                            break

                        r_delayed[j, i + ref_delta_idxs[j]] = r[j, i]

                        # test some stability criteria for the matrices
                        # A = np.array(A_exc.todense())
                        # B = np.array(B_exc.todense())
                        # b = np.array(B_exc.dot(p[j, :, i]))
                        #

                        # kappa = np.linalg.cond(B)
                        #
                        # #C_exc = np.linalg.inv(A_exc) @ B_exc
                        # n_s = int(self.v.shape[0])
                        # u, s, v = scipy.sparse.linalg.svds(B_exc)
                        # u, s, v = np..linalg.svd(B_exc)
                        # min_svd = np.min(s)
                        # B_exc = u.dot(s).dot(v.T)

                        if not self.sparse_mat:
                            p[j, :, i + 1] = np.linalg.solve(A_exc, np.matmul(B_exc, p[j, :, i]))
                            # p[j, :, i + 1] = scipy.linalg.solve_banded((1, 1), A_exc, np.matmul(B_exc, p[j, :, i]))
                        else:
                            p[j, :, i + 1] = scipy.sparse.linalg.spsolve(A_exc, B_exc.dot(p[j, :, i]))
                            # p[j, :, i + 1] = scipy.sparse.linalg.lsmr(A_exc, B_exc.dot(p[j, :, i]),
                            #                                             damp=0.2, maxiter=100)[0]

                        # update p and p_delta by their time derivative components from discontinuous terms
                        p[j, :, i + 1] += self.dt * g_exc
                        p_delta[j, i + 1] = p_delta[j, i] + self.dt * (
                                -(np.sum(v_in[exc_idxs, j, i]) + np.sum(v_in[inh_idxs, j, i]) + v_in_i_ext) *
                                p_delta[j, i] + r_delayed[j, i])
                        # p_delta[j, i + 1] = p_delta[j, i] + self.dt * (-100*p_delta[j, i] + r_delayed[j, i])


                        if i == 50:
                            a = 1
                        b = p[j, :, i + 1]
                        if (self.v[p[j, :, i + 1] < 0]).shape[0] > 1:
                            a = 1
                        if i == 180:
                            l=20

                    elif self.solver.lower() == 'hu-2021':
                    ####################################################################################################
                    # Hu-solver 2021
                    ####################################################################################################
                        if isinstance(self.c1eext_v, np.ndarray) and isinstance(self.c2eext_v, np.ndarray):
                            if ((not np.allclose(self.c1eext_v, np.zeros_like(self.c1eext_v)) and
                                    not np.allclose(self.c2eext_v, np.zeros_like(self.c2eext_v))) and
                                    c_v_warn_count < 1):
                                print(f'WARNING!: v-derivatives of coefficients are non-zero, but will be ignored!')
                                c_v_warn_count +=1

                        if i > 0:
                            Nx = v_.shape[0]
                            main = np.zeros(Nx)
                            lower = np.zeros(Nx - 1)
                            upper = np.zeros(Nx - 1)

                            # conversion of coefficients from Nykamp to Hu-formulation
                            mu_hu_solver = np.repeat(v_ext, Nx)
                            drift_coeff_vec = (-np.sum(- v_in[exc_idxs, j, i]) * c1ee_v + np.sum(v_in[inh_idxs, j, i]) * c1ei_v +\
                                          self.c1eext)
                            drift_coeff = drift_coeff_vec[0]
                            # scaling factor to drift coeff is applied in SG Flux itself

                            diffusion_coeff = + self.c2eext[0] * hu_scaling_factor**2  # square due to 2nd derivative

                            # ensure minimal diffusion coeff for numerical stability, in case of no drift input
                            diffusion_coeff = np.array(max(diffusion_coeff, 1e-5))

                            # discretization for Hu-2021
                            c = diffusion_coeff * self.dt / dv_**2
                            # critval = 0.3 * (drift_coeff / 10)
                            if self.flux_scheme in ['harmonic', 'harmonic-stable']:
                                critval = 1e-5 * max(drift_coeff, 1)
                                if c < critval:
                                    c = critval
                                    diffusion_coeff_original = diffusion_coeff.copy()
                                    diffusion_coeff = c * dv_ / self.dt
                                    sigma_orig = np.sqrt(diffusion_coeff_original* hu_scaling_factor**2 * self.tau_mem[j])
                                    sigma_new = np.sqrt(diffusion_coeff * hu_scaling_factor**2 * self.tau_mem[j])
                                    added_noise[i] = sigma_new
                                    if c_count < 1 & self.verbose > 0:
                                        c_count += 1
                                        print(f'increasing diffusion_coeff from to achieve numerical stability at noise val {sigma_orig:.5f}mV')
                                # {diffusion_coeff_original} to {diffusion_coeff:.5f} (sigma_v from {sigma_orig:.5f} to {sigma_new:.5f})
                                # if drift_coeff == 0:
                                #     diffusion_coeff = 1# self.voltage_sigma[j, 0]
                            injected_voltage[i] = drift_coeff
                            if injected_voltage[i] == 0:
                                # diffusion_coeff = 0.01
                                diffusion_coeff_original = diffusion_coeff.copy()
                                diffusion_coeff = diffusion_coeff_original
                            if i == self.t.shape[0] - 2:
                                print(f'largest injected voltage value encountered: {injected_voltage.max():.5f}mV')
                                if c_count > 0:
                                    print(f'largest noise value encountered: {added_noise.max():.5f}mV')
                            # Scharfetter-Gummel Flux
                            # Ms = self.SG_Flux(v_, drift_coeff, diffusion_coeff, x_rest=u_rest_, s_x=hu_scaling_factor)
                            # do rescaling of entire flux rather than only v here_ expression is partial_v / s_x (h*p)
                            if self.flux_scheme == 'harmonic':
                                Ms = self.SG_Flux(self.v, drift_coeff, diffusion_coeff, x_rest=self.u_rest,
                                                  s_x=hu_scaling_factor, tau=self.tau_mem[j])

                                # Calculate harmonic means
                                M_harm = 2 / ((1 / Ms[:-1]) + (1 / Ms[1:]))

                                if Ms.max() == 0:
                                    raise ValueError('Zero Flux detected!')
                                if len(np.where(c*M_harm == np.inf)[0]) > 0:
                                    raise ValueError('Infinite flux detected!')
                                if np.isnan(c*M_harm).any():
                                    raise ValueError('NaN values in flux detected!')

                                # flux weights at interfaces i+1/2: M_{i+1/2}/M_i and M_{i+1/2}/M_{i+1}
                                g_plus = M_harm / Ms[:-1]
                                g_minus = M_harm / Ms[1:]
                            else:
                                # overflow free version, uses only differences of U between neighboring nodes
                                g_plus, g_minus = self.SG_weights(self.v, drift_coeff, diffusion_coeff,
                                                                  x_rest=self.u_rest, s_x=hu_scaling_factor,
                                                                  tau=self.tau_mem[j], scheme=self.flux_scheme)

                            # BDF2: (3 p^{n+1} - 4 p^n + p^{n-1}) / (2 dt) = L p^{n+1}, first step with backward Euler
                            use_bdf2 = self.time_scheme == 'bdf2' and p_prev_step[j] is not None
                            alpha_t = 1.5 if use_bdf2 else 1

                            # compute matrix terms
                            # flux F_{i+1/2} = (D/dv) * (g_plus_{i+1/2} p_i - g_minus_{i+1/2} p_{i+1})
                            r1 = -c * g_plus[:-1]
                            r2 = alpha_t + c * (g_plus[1:] + g_minus[:-1])
                            r3 = -c * g_minus[1:]

                            main[1:-1] = r2
                            lower[:-1] = r1
                            upper[1:] = r3

                            # Boundary Conditions
                            main[0] = 1
                            main[-1] = 1
                            p_old = p_last_step[j]  # state at time step i - 1
                            p_old[0] = p_old[-1] = 0.0

                            if use_bdf2:
                                b = 2 * p_old - 0.5 * p_prev_step[j]
                            else:
                                b = p_old.copy()
                            b[0] = b[-1] = 0.0

                            if self.time_scheme == 'bdf2':
                                # reinjection at v_reset is part of the implicit step: A p = b + dt * s * r(p), with
                                # r(p) = coef * p[-2] the flux through the threshold interface and s a delta at
                                # v_reset, solved with the Sherman-Morrison formula (two tridiagonal solves)
                                spike_coef = (diffusion_coeff / dv_) * g_plus[-1]
                                s = np.zeros(Nx)
                                s[v_reset_idx] = 1 / dv_
                                yz = self.solve_tridiag(main, lower, upper, np.column_stack([b, s]))
                                y, z = yz[:, 0], yz[:, 1]
                                r[j, i] = spike_coef * y[-2] / (1 - self.dt * spike_coef * z[-2])
                                p_new = y + self.dt * r[j, i] * z

                                # BDF2 is not strictly positivity preserving, clip small undershoots, only count the
                                # ones above round off level (relative to the maximum of p)

                                if p_new.min() < 0:
                                    if p_new.min() < -1e-6 * p_new.max():
                                        n_clipped += 1
                                        if self.verbose > 0 and n_clipped == 1:
                                            print(f'WARNING: negative p ({p_new.min():.2e}) at t = {t_:.3f}ms is '
                                                  f'clipped to 0, consider a smaller time step')
                                    # clipping adds mass, which BDF2 carries forward: rescale to the mass before
                                    # clipping
                                    mass_before_clip = p_new.sum()
                                    p_new = np.maximum(p_new, 0)
                                    p_new *= mass_before_clip / p_new.sum()
                            else:
                                # solve for p
                                if self.linear_solver == 'banded':
                                    p_new = self.solve_tridiag(main, lower, upper, b)
                                else:
                                    A = scipy.sparse.diags(diagonals=[main, lower, upper], offsets=[0, -1, 1],
                                                           shape=(Nx, Nx), format='csr')
                                    p_new = scipy.sparse.linalg.spsolve(A, b)

                                # firing rate / outgoing flux / spike density
                                # if i > 1:
                                #     r[i] = (u[i-2].sum() - u[i-1].sum())
                                if self.flux_scheme == 'harmonic':
                                    r[j, i] = ((diffusion_coeff / dv_) * (p_old[-2]) +
                                            ((v_[self.thr_idx] - u_rest_) + drift_coeff) * p_old[self.thr_idx])
                                else:
                                    # flux of the scheme through the threshold interface (p = 0 at threshold),
                                    # stays correct for small diffusion where the drift carries the neurons over
                                    # threshold
                                    r[j, i] = (diffusion_coeff / dv_) * g_plus[-1] * p_new[-2]

                                J_out = r[j, i] * (np.heaviside((v_ + dv_) - u_reset_, 1) -
                                                   np.heaviside((v_ - dv_) - u_reset_, 1))
                                if J_out.max() != 0:
                                    J_out /= J_out.sum()
                                    J_out *= (1 - p_new.sum()*dv_) /dv_
                                p_new += J_out

                            p_prev_step[j] = p_old
                            p_last_step[j] = p_new
                            if save_p == 'all' and i % self.p_save_every == 0:
                                p[j, :, i // self.p_save_every] = p_new
                    if save_p == 'all' and self.p_save_every == 1:
                        p[j, :, -1] = p[j,:, -2]
                else:
                    # inhibitory population
                    # ==================================================================================================

                    c1ie = self.c[j, 0, 0]
                    c1ii = self.c[j, 0, 1]
                    c2ie = self.c[j, 1, 0]
                    c2ii = self.c[j, 1, 1]
                    c1ie_v = self.c_v[j, 0, 0]
                    c1ii_v = self.c_v[j, 0, 1]
                    c2ie_v = self.c_v[j, 1, 0]
                    c2ii_v = self.c_v[j, 1, 1]

                    # coefficients for finite difference matrices
                    f0_inh = self.dt / 2 * (1 / self.tau_mem[1] - np.sum(v_in[exc_idxs, j, i]) * c1ie_v +
                                            np.sum(v_in[inh_idxs, j, i]) * c1ii_v)
                    f1_inh = self.dt / (4 * self.dv) * (
                            (self.v - self.u_rest) / self.tau_mem[1] +
                            # - (self.i_ext[j, i] / self.g_leak[j]) +
                            np.sum(v_in[exc_idxs, j, i]) * (-c1ie + c2ie_v) +
                            np.sum(v_in[inh_idxs, j, i]) * (c1ii + c2ii_v))
                    f2_inh = self.dt / (2 * self.dv ** 2) * (np.sum(v_in[exc_idxs, j, i]) * c2ie +
                                                             np.sum(v_in[inh_idxs, j, i]) * c2ii)

                    A_inh = self.get_A(f0_inh, f1_inh, f2_inh)
                    B_inh = self.get_B(f0_inh, f1_inh, f2_inh)

                    # contribution to dp/dt from p_delta at u_res
                    g_inh = p_delta[j, i] * (-np.sum(v_in[exc_idxs, j, i]) * self.dFdv[j, 0] +
                                               np.sum(v_in[inh_idxs, j, i]) * self.dFdv[j, 1])

                    # calculate firing rate
                    r_j = np.sum(v_in[exc_idxs, j, i]) * (c2ie[-1] * p[j, -2, i] / self.dv +
                                                              self.synapse_pdf_funcs[j].sf((self.u_thr - self.u_rest) / (
                                                                          self.u_exc - self.u_rest)) *
                                                              p_delta[j, i])
                    r_ext = 0 #c1ie[-1]*(1/self.g_leak[j])*self.i_ext[j, i] * p[j, -2, i] / self.dv

                    r[j, i] = r_j + r_ext
                    if r[j, i] < 0.1:
                        print(f"WARNING: r_inh < 0 ! (r_inh = {r[j, i]}) ... Setting r_inh to 0")
                        r[j, i] = 0
                    r_delayed[j, i + ref_delta_idxs[j]] = r[j, i]

                    # update p and p_delta
                    if not self.sparse_mat:
                        p[j, :, i + 1] = np.linalg.solve(A_inh, np.matmul(B_inh, p[j, :, i]))
                    else:
                        p[j, :, i + 1] = scipy.sparse.linalg.spsolve(A_inh, B_inh.dot(p[j, :, i]))
                    p[j, :, i + 1] += self.dt * g_inh
                    p_delta[j, i + 1] = p_delta[j, i] + self.dt * (
                            -(np.sum(v_in[exc_idxs, j, i]) + np.sum(v_in[inh_idxs, j, i])) *
                            p_delta[j, i] + r_delayed[j, i])

        # plt.plot(mean_p_area)
        # plt.plot(p_delta[0]*300)
        # plt.legend(['p area', 'p_delta'])
        # plt.show()

        if self.solver.lower() == 'hu-2021':
            # last stored time point is never reached by the time loop, use the last computed state
            if (save_p == 'all' and self.p_save_every > 1 and
                    (len(self.t_p) - 1) * self.p_save_every == len(self.t) - 1):
                for k in range(self.n_populations):
                    p[k, :, -1] = p_last_step[k]
            if save_p == 'end':
                for k in range(self.n_populations):
                    p[k, :, 0] = p_last_step[k]
                self.t_p = self.t[-1:]
            if self.verbose > 0 and n_clipped > 0:
                print(f'negative p values clipped in {n_clipped} time steps')

        self.r = r
        if save_p == 'none':
            p = p_plot = None
            self.t_p = None
        else:
            p_delta_saved = p_delta[:, ::self.p_save_every] if save_p == 'all' else p_delta[:, -1:]
            p_plot = p.copy()
            p_plot[:, v_reset_idx, :] += p_delta_saved
        self.p = p

        self.save_sim_results(r=r, p=p, p_plot=p_plot)
        self.simulation_done = True

    def get_A(self, f0, f1, f2):
        if not self.sparse_mat:
            A = np.diag(1 + 2 * f2 - f0) + np.diagflat((-f2 - f1)[:-1], 1) + np.diagflat(
                (f1 - f2)[1:], -1)
            A[0, 1] = -2 * f1[1]
            A[-1, -2] = 2 * f1[-2]
        else:
            n_v = f0.shape[0]
            main = 1 + 2 * f2 - f0
            lower = (f1 - f2)[1:]
            upper = (-f2 - f1)[:-1]
            # Insert boundary conditions
            lower[-1] = 2 * f1[-2]
            upper[0] = -2 * f1[1]
            # lower[0] = 0
            # upper[-1] = 0

            A = scipy.sparse.diags(
                diagonals=[main, lower, upper],
                offsets=[0, -1, 1], shape=(n_v, n_v),
                format='csr')
        return A

    def get_B(self, f0, f1, f2):
        if not self.sparse_mat:

            B = np.diag(1 - 2 * f2 + f0) + np.diagflat((f2 + f1)[:-1], 1) + np.diagflat(
                (f2 - f1)[1:], -1)
            B[0, 1] = 2 * f1[1]
            B[-1, -2] = -2 * f1[-2]
        else:
            n_v = f0.shape[0]
            main = 1 - 2 * f2 + f0
            upper = (f2 + f1)[:-1]
            lower = (f2 - f1)[1:]
            # Insert boundary conditions
            upper[0] = 2 * f1[1]
            lower[-1] = -2 * f1[-2]
            # lower[0] = 0
            # upper[-1] = 0

            B = scipy.sparse.diags(
                diagonals=[main, lower, upper],
                offsets=[0, -1, 1], shape=(n_v, n_v),
                format='csr')
            # B_ = B.todense()
        return B

    def mat_convolve(self, x, kernel):
        """
        Function that convolves an array of time series with the same kernel unsing np.convolve

        Parameter
        ---------
        x : np.array of float (2D)
            input array
        kernel : np.array of float (1D)
            convolution kernel

        Returns
        -------
        B : np.array of float (2D)
            convolved input array
        """

        n_x = x.shape[0]
        n_y = x.shape[1]

        # use extra convolution to determine shape of output array
        n_b = np.convolve(x[0, 0], kernel).shape[0]

        b = np.zeros((n_x, n_y, n_b))
        for i in range(n_x):
            for j in range(n_y):
                b[i, j] = np.convolve(x[i, j], kernel)
        return b

    def get_coeffs(self):

        v = np.arange(self.u_inh, self.u_thr + self.dv / 2, self.dv)
        n_v = v.shape[0]

        if self.verbose > 0:
            tqdm_disable_coeffs = False
        else:
            tqdm_disable_coeffs = True

        self.c = np.zeros((self.n_populations, 2, 2, n_v))
        self.c_v = np.zeros_like(self.c)
        self.dFdv = np.zeros([self.n_populations, 2, n_v])

        if (not np.allclose(self.connectivity_matrix, np.zeros_like(self.connectivity_matrix))
                or self.input_type == 'rate'):


            self.synapse_pdf_funcs = []
            for k in range(self.n_populations):
                if self.population_type[k] == 'exc':

                    # init arrays for diffuson coeffs
                    c1ee = np.zeros(len(v))
                    c2ee = np.zeros(len(v))
                    c1ei = np.zeros(len(v))
                    c2ei = np.zeros(len(v))

                    # conductance jump distributions
                    # input synapse parameters into synapse dependent voltage jump distribution

                    if self.synapse_pdf_type == 'gamma':
                        synapse_pdf_ee = gamma(a=self.synapse_pdf_params[0, k, 0] ** (-2), loc=0,
                                         scale=self.synapse_pdf_params[0, k, 0]**2*self.synapse_pdf_params[1, k, 0])
                        synapse_pdf_ei = gamma(a=self.synapse_pdf_params[0, k, 1] ** (-2),
                                         loc=0, scale=self.synapse_pdf_params[0, k, 1]**2*self.synapse_pdf_params[1, k, 1])
                    elif self.synapse_pdf_type == 'normal':
                        synapse_pdf_ee = norm(loc=self.synapse_pdf_params[0, k, 0], scale=self.synapse_pdf_params[1, k, 0])
                        synapse_pdf_ei = norm(loc=self.synapse_pdf_params[0, k, 1], scale=self.synapse_pdf_params[1, k, 1])
                    elif self.synapse_pdf_type == 'log-normal':
                        synapse_pdf_ee = lognorm(scale=self.synapse_pdf_params[0, k, 0], s=self.synapse_pdf_params[1, k, 0])
                        synapse_pdf_ei = lognorm(scale=self.synapse_pdf_params[0, k, 1], s=self.synapse_pdf_params[1, k, 1])
                    else:
                        raise NotImplementedError(f'Please choose from implemented pdf types: {self.implemented_pdf_types}!')

                    for i, v_ in tqdm(enumerate(v), f'computing drift and diffusion coefficients for rates of'
                                                    f' exc population', disable=tqdm_disable_coeffs):

                        vpe = np.arange(self.u_inh, v_ + self.dv / 2, self.dv)
                        int_c1ee = synapse_pdf_ee.sf((v_ - vpe) / (self.u_exc - vpe))
                        int_c2ee = int_c1ee * (v_ - vpe)
                        c1ee[i] = np.trapz(x=vpe, y=int_c1ee)
                        c2ee[i] = np.trapz(x=vpe, y=int_c2ee)

                        if i > 0:
                            vpi = np.arange(v_, self.u_thr + self.dv / 2, self.dv)
                            int_c1ei = synapse_pdf_ei.sf((v_ - vpi) / (self.u_inh - vpi))
                            int_c2ei = int_c1ei * (vpi - v_)
                            c1ei[i] = np.trapz(x=vpi, y=int_c1ei)
                            c2ei[i] = np.trapz(x=vpi, y=int_c2ei)

                    c1ei[0] = c1ei[1]
                    c2ei[0] = 0

                    self.c[k] = np.array([[[c1ee, c1ei], [c2ee, c2ei]]])
                    c1ee_v = np.gradient(c1ee, self.v)
                    c1ei_v = np.gradient(c1ei, self.v)
                    c2ee_v = np.gradient(c2ee, self.v)
                    c2ei_v = np.gradient(c2ei, self.v)
                    self.c_v[k] = np.array([[[c1ee_v, c1ei_v], [c2ee_v, c2ei_v]]])
                    self.synapse_pdf_funcs.append(synapse_pdf_ee)
                    Fee_v = np.gradient(synapse_pdf_ee.sf(x=(self.v - self.u_rest) / (self.u_exc - self.u_rest)), self.dv) \
                            * np.heaviside(self.v - self.u_rest, 0.5)
                    Fei_v = np.gradient(synapse_pdf_ei.sf(x=(self.v - self.u_rest) / (self.u_inh - self.u_rest)), self.dv) \
                            * np.heaviside(self.u_rest - self.v, 0.5)

                    self.dFdv[k] = np.array([Fee_v, Fei_v])


                elif self.population_type[k] == 'inh':


                    # init arrays for diffusion coeffs
                    c1ie = np.zeros(len(v))
                    c2ie = np.zeros(len(v))
                    c1ii = np.zeros(len(v))
                    c2ii = np.zeros(len(v))

                    # conductance jump distributions
                    if self.synapse_pdf_type == 'gamma':
                        synapse_pdf_ie = gamma(a=self.synapse_pdf_params[0, k, 0] ** (-2), loc=0,
                                               scale=self.synapse_pdf_params[0, k, 0] ** 2 * self.synapse_pdf_params[
                                                   1, k, 0])
                        synapse_pdf_ii = gamma(a=self.synapse_pdf_params[0, k, 1] ** (-2),
                                               loc=0, scale=self.synapse_pdf_params[0, k, 1] ** 2 * self.synapse_pdf_params[
                                1, k, 1])
                    elif self.synapse_pdf_type == 'normal':
                        synapse_pdf_ie = norm(loc=self.synapse_pdf_params[0, k, 0], scale=self.synapse_pdf_params[1, k, 0])
                        synapse_pdf_ii = norm(loc=self.synapse_pdf_params[0, k, 1], scale=self.synapse_pdf_params[1, k, 1])
                    elif self.synapse_pdf_type == 'log-normal':
                        synapse_pdf_ie = lognorm(scale=self.synapse_pdf_params[0, k, 0], s=self.synapse_pdf_params[1, k, 0])
                        synapse_pdf_ii = lognorm(scale=self.synapse_pdf_params[0, k, 1], s=self.synapse_pdf_params[1, k, 1])
                    else:
                        raise NotImplementedError(
                            f'Please choose from implemented pdf types: {self.implemented_pdf_types}!')

                    for i, v_ in tqdm(enumerate(v), f'computing drift and diffusion coefficients for rates of'
                                                    f' inh population', disable=tqdm_disable_coeffs):
                        vpe = np.arange(self.u_inh, v_ + self.dv / 2, self.dv)
                        int_c1ie = synapse_pdf_ie.sf((v_ - vpe) / (self.u_exc - vpe))
                        int_c2ie = int_c1ie * (v_ - vpe)
                        c1ie[i] = np.trapz(x=vpe, y=int_c1ie)
                        c2ie[i] = np.trapz(x=vpe, y=int_c2ie)

                        if i > 0:
                            vpi = np.arange(v_, self.u_thr + self.dv / 2, self.dv)
                            int_c1ii = synapse_pdf_ii.sf((v_ - vpi) / (self.u_inh - vpi))
                            int_c2ii = int_c1ii * (vpi - v_)
                            c1ii[i] = np.trapz(x=vpi, y=int_c1ii)
                            c2ii[i] = np.trapz(x=vpi, y=int_c2ii)

                    c1ii[0] = c1ii[1]
                    c2ii[0] = 0

                    self.c[k] = np.array([[c1ie, c1ii], [c2ie, c2ii]])
                    c1ie_v = np.gradient(c1ie, self.v)
                    c1ii_v = np.gradient(c1ii, self.v)
                    c2ie_v = np.gradient(c2ie, self.v)
                    c2ii_v = np.gradient(c2ii, self.v)
                    self.c_v[k] = np.array([[[c1ie_v, c1ii_v], [c2ie_v, c2ii_v]]])
                    self.synapse_pdf_funcs.append(synapse_pdf_ie)
                    Fie_v = np.gradient(synapse_pdf_ie.sf(x=(self.v - self.u_rest) / (self.u_exc - self.u_rest)), self.dv) \
                                       * np.heaviside(self.v - self.u_rest, 0.5)
                    Fii_v = np.gradient(synapse_pdf_ii.sf(x=(self.v - self.u_rest) / (self.u_inh - self.u_rest)), self.dv) \
                                       * np.heaviside(self.u_rest - self.v, 0.5)
                    self.dFdv[k] = np.array([Fie_v, Fii_v])

                else:
                    raise NotImplementedError('population types must be "exc" or "inh"!')

    def get_delay_kernel(self):
        if self. delay_kernel_type == 'alpha':
            t_alpha = self.t[self.t < 10]
            n_alpha = self.delay_kernel_parameters['n_alpha']
            tau_alpha = self.delay_kernel_parameters['tau_alpha']
            self.delay_kernel = np.exp(-t_alpha/tau_alpha) / (tau_alpha * scipy.special.factorial(n_alpha-1)) *\
                    (t_alpha/tau_alpha)**(n_alpha-1)
            self.delay_kernel = self.delay_kernel/np.trapz(self.delay_kernel, dx=self.dt)
            self.delay_kernel_time = t_alpha
        elif self. delay_kernel_type == ('bi-exp' or 'bi-exponential' or 'bi_exp'):
            """
            The bi-exponential kernel is taken from Rusu et al. 2014 [1] where the values for tau_1, tau_2 and g_peak
            are given in a table for AMPA, NMDA, GABA_a and GABA_b synapses in M1. tau_cond is specified as 1ms in the 
            text. In the context of this function the values for an AMPA synapse are:
            (tau_1, tau_2, g_peak, tau_cond) = (0.2, 1.7, 1e-4, 1)
            [1] Rusu, C. V., Murakami, M., Ziemann, U., & Triesch, J. (2014). A model of TMS-induced I-waves in motor
             cortex. Brain Stimulation, 7(3), 401-414. 
            """
            t_kernel = self.t[self.t < 10]
            tau_1 = self.delay_kernel_parameters['tau_1']
            tau_2 = self.delay_kernel_parameters['tau_2']
            tau_cond = self.delay_kernel_parameters['tau_cond']
            g_peak = self.delay_kernel_parameters['g_peak']
            self.delay_kernel = g_peak * (np.exp(-(t_kernel - tau_cond)/tau_1) - np.exp(-(t_kernel - tau_cond)/tau_2))
            self.delay_kernel = self.delay_kernel / np.trapz(self.delay_kernel, dx=self.dt)
            self.delay_kernel_time = t_kernel
        else:
            raise NotImplementedError('Specified delay kernel is not implemented!, please choose from "alpha", "bi-exp"')

    def plot_delay_kernel(self):
        # find time window
        plt.plot(self.delay_kernel_time, self.delay_kernel)
        plt.grid()
        plt.xlabel('t (ms)')
        plt.ylabel(f'{self.delay_kernel_type} delay kernel')
        plt.show()

    def save_sim_results(self, r, p, p_plot):
        with h5py.File(self.name + '.hdf5', 'w') as h5file:
            h5file.create_dataset('t', data=self.t)
            h5file.create_dataset('v', data=self.v)
            h5file.create_dataset('r', data=r)
            if self.input_type in ['current', 'current-2', 'stochastic-current']:
                h5file.create_dataset('in', data=self.i_ext)
            else:
                h5file.create_dataset('in', data=self.input)
            if p is not None:
                h5file.create_dataset('rho', data=p)
                h5file.create_dataset('rho_plot', data=p_plot)
                if getattr(self, 't_p', None) is not None:
                    h5file.create_dataset('t_rho', data=self.t_p)
            h5file.create_dataset('p_types', data=self.population_type)

    def plot(self, fname=None, heat_map=True, plot_idxs=None, savefig=False, plot_input=False, z_limit=None,
             plot_combined=True, animate=False, crop_rate=False, frame_off=False):

        if fname == None:
            fname = self.name
        p = None

        with h5py.File(fname + '.hdf5', 'r') as h5file:

            t_plot = np.array(h5file['t'])
            # time axis of p, differs from t if p was only stored every n-th time step
            t_p = np.array(h5file['t_rho']) if 't_rho' in h5file else t_plot
            v = np.array(h5file['v'])
            r_plot = np.array(h5file['r'])
            for k in range(r_plot.shape[0]):
                r_plot[k, -1] = r_plot[k, -2]
            p_plot = np.array(h5file['rho_plot']) if 'rho_plot' in h5file else None
            p_types_raw = h5file['p_types']
            p_types = p_types_raw.asstr()[:]
            if 'rho' in h5file:
                p = np.array(h5file['rho'])

        if plot_input:
            with h5py.File(fname + '.hdf5', 'r') as h5file:
                i_in = np.array(h5file['in'])

        if plot_idxs is None:
            n_plots = len(p_types)
            plot_idxs = np.arange(n_plots)
        else:
            n_plots = len(plot_idxs)

        if plot_combined:
            fig = plt.figure(figsize=(10, 4.25*n_plots))
            for i_plot, plot_idx in enumerate(plot_idxs):
                plot_loc_1 = int(2*i_plot + 1)
                plot_loc_2 = int(2 * i_plot + 2)
                ax, z_limit = plot_p_panel(fig, (n_plots, 2, plot_loc_1), t_p, v,
                                             p_plot[plot_idx] if p_plot is not None else None, heat_map,
                                             z_limit, 'gnuplot2', self.labelsize)

                ax = fig.add_subplot(n_plots, 2, plot_loc_2)
                ax.plot(t_plot, r_plot[plot_idx] * 1000)
                # ax.set_title(f"Population activity ({str(p_types[plot_idx])})", fontsize=self.labelsize)
                ax.set_ylabel("Spike density (1/s)", fontsize=self.labelsize)
                ax.set_xlabel("t (ms)", fontsize=self.labelsize)
                ax.tick_params(axis='both', which='major', labelsize=self.labelsize)
                ax.spines['top'].set_visible(False)
                if crop_rate:
                    ax.set_ylim(0, np.max(1.2*r_plot[plot_idx, 50:] * 1000))
                if plot_input:
                    rate_height = np.max(r_plot[plot_idx, 50:] * 1000)
                    i_in_plot = i_in[plot_idx].flatten() / np.max(i_in[plot_idx].flatten()) * rate_height  # renormalize for plot with rate
                    # ax.plot(t_plot, i_in_plot)
                    ax_in = ax.twinx()

                    # Plot second time series
                    ax_in.plot(t_plot, i_in[plot_idx]*1e9, color='tab:orange', label='i_ext')
                    ax_in.set_ylabel('i_ext (nA)', color='tab:orange', fontsize=self.labelsize)
                    ax_in.tick_params(axis='both', which='major', labelsize=self.labelsize)
                    ax.spines['top'].set_visible(False)
                    # ax.legend(['n', 'I_ext'])

                ax.grid()
            plt.tight_layout()
            if savefig:
                plt.savefig(self.name + '_plot.png')
                plt.close()
            else:
                plt.show()

        else:
            for i_plot, plot_idx in enumerate(plot_idxs):
                fig_1 = plt.figure(figsize=(6, 4.25))

                ax, z_limit = plot_p_panel(fig_1, (n_plots, 1, 1), t_p, v,
                                             p_plot[plot_idx] if p_plot is not None else None, heat_map,
                                             z_limit, 'jet', self.labelsize)
                ax.set_title(f"Membrane potential distribution ({str(p_types[plot_idx])})")
                plt.tight_layout()
                if savefig:
                    plt.savefig(self.name + 'voltage_dist_plot.png')
                    plt.close()
                else:
                    plt.show()
                fig_2 = plt.figure(figsize=(6, 4.25))
                ax = fig_2.add_subplot(n_plots, 1, 1)
                ax.plot(t_plot, r_plot[plot_idx] * 1000, color='darkorange', linewidth=2)
                ax.set_title(f"Population activity ({str(p_types[plot_idx])})")
                ax.set_ylabel("Spike density (1/s)", fontsize=self.labelsize)
                ax.set_xlabel("t (ms)", fontsize=self.labelsize)
                if plot_input:
                    ax_input = ax.twinx()

                    # rate_height = np.max(r_plot[plot_idx] * 1000)
                    # i_in_plot = i_in[plot_idx].flatten() / np.max(i_in[plot_idx].flatten()) * rate_height  # renormalize for plot with rate
                    ax_input.plot(t_plot, i_in[plot_idx].flatten()/1e3*1e9, color='teal', linewidth=2)
                    ax_input.set_ylabel("Current (nA)", fontsize=self.labelsize)
                    ax_input.set_ylim(0, i_in[plot_idx].flatten().max()/1e3*1e9*1.2)
                    ax_input.tick_params(axis='both', which='major', labelsize=self.labelsize)
                    ax.plot(t_plot, -5*np.ones_like(r_plot[plot_idx]), color='teal')
                    ax.legend(['n', 'I_ext'])
                ax.set_ylim(0, r_plot[plot_idx].max()*1e3*1.2)
                ax.set_xlim(0, t_plot.max())
                ax.spines['top'].set_visible(False)
                ax.tick_params(axis='both', which='major', labelsize=self.labelsize)


                # ax.grid()
                plt.tight_layout()
                if savefig:
                    plt.savefig(self.name + 'rate_plot.png')
                    plt.close()
                else:
                    plt.show()
        if animate and (p_plot is None or p_plot.shape[-1] < 2):
            print("no animation of p possible, p was not saved for several time steps (save_p='all')")
            animate = False
        if animate:
            print('plotting animation of p(v, t)')
            if isinstance(p, np.ndarray):
                p_animation = p
            else:
                p_animation = p_plot
            fig = plt.figure(figsize=(10, 4.25*n_plots))
            for i_plot, plot_idx in enumerate(plot_idxs):
                ax = fig.add_subplot(n_plots, 1, i_plot+1)
                line = ax.plot(v, p_animation[plot_idx, :, 0])[0]
                # plt.fill_between(x=v, y1=p_animation[plot_idx, :, 0], color="b", alpha=0.2)
                bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
                text = ax.text(0.95, 0.9, f't = {t_p[0]:.2f}ms', fontsize=9, bbox=bbox,
                        transform=ax.transAxes, horizontalalignment='right')
                ax.set(xlim=[v.min(), v.max()], ylim=[0, p_animation.mean()*3], xlabel='v (mV)', ylabel='p')  # p_animation.max()*0.5
                def update(frame):
                    # for each frame, update the data stored on each artist.
                    p_frame = p_animation[plot_idx, :, frame*5]

                    # update the line plot:
                    # line.set_xdata(v[:frame])
                    line.set_ydata(p_frame)
                    text.set_text(f't = {t_p[frame*5]:.2f}ms')
                    # plt.fill_between(x=v, y1=p_frame, color="b", alpha=0.2)
                    return (line, text)

                ani = animation.FuncAnimation(fig=fig, func=update, frames=int(p_animation.shape[2]/5), interval=50)
                # plt.show()
                ani.save(filename=self.name + '_p_animation.gif', writer="pillow")
                print(f'saved animation of p to {self.name}_p_animation.gif')
                plt.close()

    def save_log(self, full_out=False):
        if full_out:
            log_dict = self.__dict__.copy()
        else:
            log_dict = self.parameters
        print(f'saved log to: {self.name}_log.yaml')
        with open(f'{self.name}_log.yaml', 'w') as file:
            yaml.dump(log_dict, file)

    def sigmoid(self, x, r=20):
        return 1 / (1 + np.exp(-r*x))

    def gauss_func(self,x,  mu=0, sigma=1):
        return (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))

    def SG_Flux(self, x, a, alpha, x_rest=0.2, s_x=1, tau=1):
        """
        compute Scharfetter-Gummel flux functional
        :param x: x
        :param a: drift
        :param alpha: diffusion
        :param x_L: resting point
        :param s_x: scaling factor for x in case of interpolation on new grid / affine transformation to different
         coordinates was used
        :return: M: Scharfetter-Gummel flux functional
        """
        U = s_x * (((1 / 2) * x - x_rest - a) * (x / (alpha * tau)))
        return np.exp(-U)

    def SG_weights(self, x, a, alpha, x_rest=0.2, s_x=1, tau=1, scheme='bernoulli'):
        """
        compute the flux weights of the Scharfetter-Gummel scheme from neighbouring differences of U only (M = exp(-U)),
        avoids over-/underflow of M for small noise values
        flux at interface i+1/2: F = (alpha/dx) * (g_plus * p_i - g_minus * p_{i+1})
        :param x: x
        :param a: drift
        :param alpha: diffusion
        :param x_rest: resting point
        :param s_x: scaling factor for x, see SG_Flux
        :param tau: membrane time constant
        :param scheme: 'harmonic-stable': harmonic mean of M (same matrix as SG_Flux + harmonic mean),
         'bernoulli': classical Scharfetter-Gummel flux with Bernoulli function B(z) = z / (exp(z) - 1)
        :return: g_plus, g_minus (both of length len(x) - 1)
        """
        # U_{i+1} - U_i, exact for the quadratic U in SG_Flux
        x_mid = 0.5 * (x[1:] + x[:-1])
        dU = s_x * np.diff(x) * (x_mid - x_rest - a) / (alpha * tau)
        if scheme == 'harmonic-stable':
            # M_{i+1/2}/M_i = 2 / (1 + exp(dU)), M_{i+1/2}/M_{i+1} = 2 / (1 + exp(-dU))
            return 2 * scipy.special.expit(-dU), 2 * scipy.special.expit(dU)
        elif scheme == 'bernoulli':
            # B(-z) = B(z) + z: only one evaluation of B, on |dU| to avoid cancellation
            dU_abs = np.abs(dU)
            b_small = self.bernoulli(dU_abs)
            b_large = b_small + dU_abs
            positive = dU >= 0
            return np.where(positive, b_small, b_large), np.where(positive, b_large, b_small)
        else:
            raise NotImplementedError(f"flux_scheme {scheme} not implemented! Please chose from "
                                      f"['harmonic', 'harmonic-stable', 'bernoulli']")

    def solve_tridiag(self, main, lower, upper, b):
        """
        solve a tridiagonal system with the banded solver of scipy
        :param main: main diagonal (N)
        :param lower: lower diagonal (N - 1), lower[k] = A[k + 1, k]
        :param upper: upper diagonal (N - 1), upper[k] = A[k, k + 1]
        :param b: right hand side (N) or (N, n_rhs)
        :return: solution x of A x = b
        """
        ab = np.zeros((3, main.shape[0]))
        ab[0, 1:] = upper
        ab[1] = main
        ab[2, :-1] = lower
        return scipy.linalg.solve_banded((1, 1), ab, b, check_finite=False)

    def bernoulli(self, z):
        """
        Bernoulli function B(z) = z / (exp(z) - 1), with B(0) = 1
        param z: np.ndarray / float /int, input value
        return res: np.ndarray bernoulli flux
        """
        z = np.asarray(z, dtype=float)
        small_value_mask = np.abs(z) < 1e-8
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            res = z / np.expm1(z)
        res[small_value_mask] = 1 - z[small_value_mask] / 2
        # exp(z) overflows for large z, B -> 0, take care by setting NaN entries to 0
        res[np.isnan(res)] = 0
        return res

    def clean(self):
        if self.simulation_done:
            os.remove(self.name + '.hdf5')



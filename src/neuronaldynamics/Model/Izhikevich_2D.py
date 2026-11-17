"""
Two-dimensional Fokker-Planck solver for populations of Izhikevich neurons: the density P(v, u, t) is computed on a grid
in the membrane potential v and the adaptation u, without any closure of the adaptation. Reference solution for the
one-dimensional closures of Nykamp_Model.FPE_Population (soma_model='IK').

Model (per population, see FPE_Izhikevich):
    C dv = (k (v - v_r)(v - v_t) - u + I) dt + noise (D = voltage_sigma^2 / tau_noise),    du = a (b (v - v_r) - u) dt,
    at v = v_peak: v -> c, u -> u + d
    dP/dt = -d/dv [ (k (v - v_r)(v - v_t) - u + I) / C P - D dP/dv ] - d/du [ a (b (v - v_r) - u) P ]
            + delta(v - c) J_v(v_peak, u - d)

Discretization:
- grid: v uniform (dv, v_min ... v_peak, c on the grid), u uniform cells of width du in [u_lo, u_hi] (cell centres u_j);
  d has to be a multiple of du. Populations with a = b = d = 0 (no adaptation) get a single u line.
- u transport along the exact characteristics: for fixed v the flow u -> u* + (u - u*) exp(-a dt), u* = b (v - v_r), is
  known exactly, the density is remapped conservatively (piecewise linear reconstruction with minmod slopes, mass of
  every new cell = integral of the reconstruction over the preimage of the cell): mass conserving, positive, no time
  step restriction.
- v transport and noise: for every u line the implicit Scharfetter-Gummel / BDF2 step of FPE_Population, with the
  history transported along the u characteristics (semi-Lagrangian BDF2):
      (3/2 I - dt L_j) P_j^{n+1} = 2 (U_dt P^n)_j - 1/2 (U_2dt P^{n-1})_j + dt S_j,
  backward Euler for the whole population if BDF2 gives significantly negative values. The splitting error is
  proportional to the coupling of the two directions (a b / C) and small.
- flux_limiter='fct': flux-corrected transport in v on every u line after the implicit step (Zalesak limiter of
  Nykamp_Model._fct_correct, bounds from the low order solution, no correction at the first and last interface so that
  outflow and reinjection are those of the low order step): removes the numerical diffusion of the Scharfetter-Gummel
  flux at large cell Peclet numbers (|drift| dv / D >> 2) while keeping P >= 0 and the mass of every line.
- reset: the outflow of line j at v_peak re-enters line j + d / du at c in the same step (lines are solved in
  increasing j; outflow beyond u_hi enters the last line, with Sherman-Morrison if a line feeds itself).

Coupling between populations as in FPE_Population: I_j = I_ext,j + sum_k W[k, j] (alpha * r_k)(t), W in pA ms.

use_numba: compiled kernels (default), False or numba not installed: numpy / scipy versions (same arithmetic).

Diagnostics (off by default): snapshot_times (ms): P of every population at the first time step >= each time, in
self.P_snap (list over populations of arrays (n_snap, Nu, Nv)) at self.t_snap; save_flux_u=True: the outflow at
v_peak per u line and time step, self.flux_u (list over populations of arrays (Nt, Nu), rate density in 1/(ms pA)),
i.e. the distribution of u of the neurons that fire.
"""
import time
import warnings

import numpy as np
import scipy.linalg
import scipy.special
from scipy.stats import norm
from tqdm import tqdm

from Model.Nykamp_Model import (IZHIKEVICH_DEFAULTS, IZHIKEVICH_PER_POPULATION, NUMBA_AVAILABLE, njit, _FASTMATH,
                                _sg_pair, _thomas2, _sg_weights_np, _fct_correct, _fct_correct_np, FPE_Population)


########################################################################################################################
# KERNELS (numba)
########################################################################################################################
@njit(cache=True, fastmath=_FASTMATH)
def _remap_u(P, u_lo, du, ustar, factor, out, skip_thr):
    """
    conservative remap of every v column of P (Nu, Nx) along the exact u flow over one step:
    new cell [e_j, e_{j+1}] receives the mass of the old reconstruction on [u* + (e_j - u*) factor,
    u* + (e_{j+1} - u*) factor], factor = exp(a dt); the first and last new cell also take everything below / above
    """
    Nu, Nx = P.shape
    if Nu == 1:
        for i in range(Nx):
            out[0, i] = P[0, i]
        return
    cum = np.zeros(Nu + 1)
    slope = np.zeros(Nu)
    for i in range(Nx):
        col_max = 0.0
        for j in range(Nu):
            col_max = max(col_max, P[j, i])
        if col_max <= skip_thr:
            # negligible column: carried along unchanged
            for j in range(Nu):
                out[j, i] = P[j, i]
            continue
        for j in range(Nu):
            cum[j + 1] = cum[j] + P[j, i] * du
        for j in range(Nu):
            if j == 0 or j == Nu - 1:
                slope[j] = 0.0
            else:
                left = P[j, i] - P[j - 1, i]
                right = P[j + 1, i] - P[j, i]
                if left * right <= 0:
                    slope[j] = 0.0
                elif abs(left) < abs(right):
                    slope[j] = left / du
                else:
                    slope[j] = right / du
        total = cum[Nu]
        prev_F = 0.0
        for j in range(Nu):
            if j == Nu - 1:
                F = total
            else:
                x = ustar[i] + (u_lo + (j + 1) * du - ustar[i]) * factor[i]
                if x <= u_lo:
                    F = 0.0
                elif x >= u_lo + Nu * du:
                    F = total
                else:
                    k = int((x - u_lo) / du)
                    if k > Nu - 1:
                        k = Nu - 1
                    s = (x - (u_lo + k * du)) / du
                    F = cum[k] + du * (P[k, i] * s + slope[k] * du * (s * s - s) / 2)
            out[j, i] = (F - prev_F) / du
            prev_F = F


@njit(cache=True, fastmath=_FASTMATH)
def _v_lines_step(H, Hprev, has_prev, use_bdf2, u, v, h, w, quad, k, C, i_total, D, dt, reset_idx, shift, harmonic,
                  Pnew, Jout, g_plus, g_minus, main, lower, upper, b_rhs, s_vec, y, z, cp, skip_thr, fct):
    """
    implicit v step of all u lines (Nu, Nx) of one population: (alpha I - dt L_j) P_j = rhs_j + dt S_j e_c / w_c,
    rhs = 2 H - Hprev / 2 (BDF2) or H (backward Euler), S_j = outflow of line j - shift (solved before)
    fct: flux-corrected transport on every solved line (outflow and reinjection unchanged)
    :return: minimum and maximum of the new density (for the fallback test); Jout: outflow of every line
    """
    Nu, Nx = H.shape
    alpha = 1.5 if use_bdf2 else 1.0
    src = np.zeros(Nu)
    f_alpha, f_A, f_flux = np.zeros(Nx - 1), np.zeros(Nx - 1), np.zeros(Nx - 1)
    f_Rp, f_Rm, f_low = np.zeros(Nx), np.zeros(Nx), np.zeros((1, Nx))
    for j in range(Nx):
        s_vec[j] = 0.0
    s_vec[reset_idx] = 1.0 / w[reset_idx]
    mn = np.inf
    mx = -np.inf
    for jl in range(Nu):
        if skip_thr > 0 and src[jl] == 0:
            line_max = 0.0
            for i in range(Nx):
                line_max = max(line_max, H[jl, i], Hprev[jl, i])
            if line_max <= skip_thr:
                # negligible line: carried along unchanged, no outflow
                for i in range(Nx):
                    Pnew[jl, i] = H[jl, i]
                Jout[jl] = 0.0
                continue
        for i in range(Nx - 1):
            drift = (k * quad[i] - u[jl] + i_total) / C
            g_plus[i], g_minus[i] = _sg_pair(-h[i] * drift / D, harmonic)
        for i in range(1, Nx - 1):
            lower[i - 1] = -dt * D / (w[i] * h[i - 1]) * g_plus[i - 1]
            upper[i] = -dt * D / (w[i] * h[i]) * g_minus[i]
            main[i] = alpha + dt * D / w[i] * (g_plus[i] / h[i] + g_minus[i - 1] / h[i - 1])
            b_rhs[i] = (2 * H[jl, i] - 0.5 * Hprev[jl, i]) if use_bdf2 else H[jl, i]
        upper[0] = 0.0
        lower[Nx - 2] = 0.0
        main[0] = 1.0
        main[Nx - 1] = 1.0
        b_rhs[0] = 0.0
        b_rhs[Nx - 1] = 0.0
        b_rhs[reset_idx] += dt * src[jl] / w[reset_idx]
        _thomas2(main, lower, upper, b_rhs, s_vec, y, z, cp)
        coef = D / h[Nx - 2] * g_plus[Nx - 2]
        target = jl + shift
        if target > Nu - 1:
            target = Nu - 1
        if target == jl:
            # the line feeds itself: implicit reinjection with Sherman-Morrison
            jo = coef * y[Nx - 2] / (1 - dt * coef * z[Nx - 2])
            for i in range(Nx):
                Pnew[jl, i] = y[i] + dt * jo * z[i]
        else:
            for i in range(Nx):
                Pnew[jl, i] = y[i]
            jo = coef * y[Nx - 2]
            src[target] += jo
        Jout[jl] = jo
        if fct:
            _fct_correct(Pnew[jl:jl + 1], g_plus, g_minus, h, w, D, dt / alpha, f_alpha, f_A, f_Rp, f_Rm, f_flux,
                         f_low, 0.0)
        for i in range(Nx):
            mn = min(mn, Pnew[jl, i])
            mx = max(mx, Pnew[jl, i])
    return mn, mx


########################################################################################################################
# KERNELS (numpy / scipy)
########################################################################################################################
def _remap_u_np(P, u_lo, du, ustar, factor, out, skip_thr):
    """numpy version of _remap_u (vectorized over the v columns)"""
    Nu, Nx = P.shape
    if Nu == 1:
        out[:] = P
        return
    cum = np.vstack([np.zeros((1, Nx)), np.cumsum(P * du, axis=0)])
    left = np.zeros_like(P)
    right = np.zeros_like(P)
    left[1:] = P[1:] - P[:-1]
    right[:-1] = P[1:] - P[:-1]
    slope = np.where(left * right <= 0, 0.0, np.where(np.abs(left) < np.abs(right), left, right)) / du
    slope[0] = slope[-1] = 0.0
    edges = u_lo + du * np.arange(1, Nu)[:, None]                           # inner edges (Nu - 1, 1)
    x = ustar[None, :] + (edges - ustar[None, :]) * factor[None, :]         # preimages (Nu - 1, Nx)
    kk = np.clip(np.floor((x - u_lo) / du).astype(int), 0, Nu - 1)
    s = (x - (u_lo + kk * du)) / du
    cols = np.broadcast_to(np.arange(Nx), kk.shape)
    F = np.take_along_axis(cum, kk, axis=0) + du * (P[kk, cols] * s + slope[kk, cols] * du * (s * s - s) / 2)
    total = cum[-1]
    F = np.where(x <= u_lo, 0.0, np.where(x >= u_lo + Nu * du, total[None, :], F))
    F = np.vstack([np.zeros((1, Nx)), F, total[None, :]])
    out[:] = np.diff(F, axis=0) / du


def _v_lines_step_np(H, Hprev, has_prev, use_bdf2, u, v, h, w, quad, k, C, i_total, D, dt, reset_idx, shift, harmonic,
                     Pnew, Jout, g_plus, g_minus, main, lower, upper, b_rhs, s_vec, y, z, cp, skip_thr, fct):
    """numpy version of _v_lines_step (one banded solve per u line)"""
    Nu, Nx = H.shape
    alpha = 1.5 if use_bdf2 else 1.0
    src = np.zeros(Nu)
    s_vec[:] = 0.0
    s_vec[reset_idx] = 1.0 / w[reset_idx]
    f_work = [np.zeros(Nx - 1), np.zeros(Nx - 1), np.zeros(Nx), np.zeros(Nx), np.zeros(Nx - 1), np.zeros((1, Nx))]
    for jl in range(Nu):
        if skip_thr > 0 and src[jl] == 0 and max(H[jl].max(), Hprev[jl].max()) <= skip_thr:
            Pnew[jl] = H[jl]
            Jout[jl] = 0.0
            continue
        drift = (k * quad - u[jl] + i_total) / C
        gp, gm = _sg_weights_np(-h * drift / D, harmonic)
        lower[:-1] = -dt * D / (w[1:-1] * h[:-1]) * gp[:-1]
        lower[-1] = 0.0
        upper[1:] = -dt * D / (w[1:-1] * h[1:]) * gm[1:]
        upper[0] = 0.0
        main[1:-1] = alpha + dt * D / w[1:-1] * (gp[1:] / h[1:] + gm[:-1] / h[:-1])
        main[0] = main[-1] = 1.0
        b_rhs[:] = (2 * H[jl] - 0.5 * Hprev[jl]) if use_bdf2 else H[jl]
        b_rhs[0] = b_rhs[-1] = 0.0
        b_rhs[reset_idx] += dt * src[jl] / w[reset_idx]
        ab = np.zeros((3, Nx))
        ab[0, 1:] = upper
        ab[1] = main
        ab[2, :-1] = lower
        yz = scipy.linalg.solve_banded((1, 1), ab, np.column_stack([b_rhs, s_vec]), check_finite=False)
        coef = D / h[-1] * gp[-1]
        target = min(jl + shift, Nu - 1)
        if target == jl:
            jo = coef * yz[-2, 0] / (1 - dt * coef * yz[-2, 1])
            Pnew[jl] = yz[:, 0] + dt * jo * yz[:, 1]
        else:
            Pnew[jl] = yz[:, 0]
            jo = coef * yz[-2, 0]
            src[target] += jo
        Jout[jl] = jo
        if fct:
            _fct_correct_np(Pnew[jl:jl + 1], gp, gm, h, w, D, dt / alpha, *f_work, 0.0)
    return Pnew.min(), Pnew.max()


########################################################################################################################
# SOLVER
########################################################################################################################
class FPE_Izhikevich2D:
    """
    two-dimensional (v, u) Fokker-Planck solver for coupled populations of Izhikevich neurons, see module docstring

    parameters (dict): as FPE_Population with soma_model='IK' (T, dt, dv, population_type, connectivity_matrix,
    voltage_sigma, input_function (callable of t or array), input_function_idx, init_pdf_sigma, Izhikevich parameters
    per population, delay kernel), plus
        du: cell width in u (pA), d has to be a multiple of du (default 2)
        u_range: (u_lo, u_hi) in pA, None: (min(0, b (v_t - v_r), b (v_min - v_r)) - 20, max(...) + 3 d + 20)
        flux_limiter: 'none' (Scharfetter-Gummel flux, default) or 'fct' (flux-corrected transport in v)
        use_numba: compiled kernels (default True)
        skip_rel: u lines and v columns whose density stays below skip_rel * max(P) are carried along unchanged
            (no solve), 0: everything is solved (default 1e-10)
    results: r (n_pop, Nt) in 1/ms, u_mean, v_mean, var_u (n_pop, Nt), P (list of the final densities (Nu, Nx)), u
    (list of the u grids), n_fallback
    """

    def __init__(self, parameters={}, name='FPE_Izhikevich2D'):
        pars = dict(IZHIKEVICH_DEFAULTS)
        pars.update(dict(T=100., dt=0.05, dv=0.1, du=2., u_range=None, population_type=['exc'],
                         connectivity_matrix=np.zeros((1, 1)), input_function=None, input_function_idx=[0],
                         init_pdf_sigma=1., init_pdf_offset=0., tqdm_disable=True, use_numba=True,
                         skip_rel=1e-10, flux_limiter='none', snapshot_times=(), save_flux_u=False))
        pars.update(parameters)
        self.__dict__.update(pars)
        self.name = name
        self.n_populations = len(self.population_type)
        self.connectivity_matrix = np.atleast_2d(np.asarray(self.connectivity_matrix, dtype=float))
        # delay kernel of FPE_Population (alpha function)
        self.delay_kernel = FPE_Population(dict(T=self.T, dt=self.dt)).delay_kernel

    def _per_population(self, name):
        x = np.asarray(getattr(self, name), dtype=float).reshape(-1)
        return np.broadcast_to(x, (self.n_populations,)).copy()

    def simulate(self):
        use_numba = self.use_numba and NUMBA_AVAILABLE
        if self.use_numba and not NUMBA_AVAILABLE:
            warnings.warn('numba is not installed, the numpy / scipy kernels are used')
        remap, lines_step = (_remap_u, _v_lines_step) if use_numba else (_remap_u_np, _v_lines_step_np)
        if self.flux_limiter not in ('none', 'fct'):
            raise ValueError(f"flux_limiter has to be 'none' or 'fct', not {self.flux_limiter!r}")
        fct = self.flux_limiter == 'fct'
        n_pop = self.n_populations
        C, k, v_r, v_t, a, b, c, d, tau_noise = [self._per_population(p) for p in IZHIKEVICH_PER_POPULATION]
        dt = self.dt
        self.t = np.arange(0, self.T, dt)
        Nt = len(self.t)
        v = np.arange(self.v_min, self.v_peak + self.dv / 2, self.dv)
        Nx = len(v)
        h = np.diff(v)
        w = np.zeros(Nx)
        w[1:-1] = 0.5 * (h[1:] + h[:-1])
        w[0], w[-1] = 0.5 * h[0], 0.5 * h[-1]
        self.v = v
        v_mid = 0.5 * (v[1:] + v[:-1])
        quad = [(v_mid - v_r[j]) * (v_mid - v_t[j]) + h**2 / 12 for j in range(n_pop)]
        reset_idx = []
        for j in range(n_pop):
            idx = np.where(np.isclose(v, c[j]))[0]
            if idx.size == 0:
                raise ValueError(f'reset potential c = {c[j]} is not on the voltage grid')
            reset_idx.append(int(idx[0]))

        # u grids
        u_grid, u_lo, shift = [], [], []
        for j in range(n_pop):
            if a[j] == 0 and b[j] == 0 and d[j] == 0:
                u_grid.append(np.array([0.0]))
                u_lo.append(-0.5 * self.du)
                shift.append(0)
                continue
            s_j = d[j] / self.du
            if abs(s_j - round(s_j)) > 1e-9:
                raise ValueError(f'd = {d[j]} has to be a multiple of du = {self.du}')
            if self.u_range is None:
                # u relaxes towards b (v - v_r); neurons spend their time below threshold, so the range of b (v - v_r)
                # for v_min < v < v_t (plus the jumps) is enough, mass beyond the range is kept in the last cells
                ends = [0.0, b[j] * (v_t[j] - v_r[j]), b[j] * (self.v_min - v_r[j])]
                lo, hi = min(ends) - 20, max(ends) + 3 * max(d[j], 0) + 20
            else:
                lo, hi = self.u_range
            n_u = int(np.ceil((hi - lo) / self.du))
            u_grid.append(lo + self.du * (np.arange(n_u) + 0.5))
            u_lo.append(lo)
            shift.append(int(round(s_j)))
        self.u = u_grid

        # input and noise
        i_ext = np.zeros((n_pop, Nt))
        f_in = self.input_function
        i_ext[self.input_function_idx] = f_in(self.t) if callable(f_in) else np.asarray(f_in)
        sigma = np.broadcast_to(np.asarray(self.voltage_sigma, dtype=float).reshape(-1, 1), (n_pop, Nt))

        # initial condition: Gaussian in v around v_r, all neurons with u = b (<v> - v_r) (in one u cell)
        P, P_prev_tr = [], [None] * n_pop
        u_mean = np.zeros((n_pop, Nt))
        v_mean = np.zeros((n_pop, Nt))
        var_u = np.zeros((n_pop, Nt))
        r = np.zeros((n_pop, Nt))
        for j in range(n_pop):
            v0 = v[np.where(v > v_r[j] + self.init_pdf_offset)[0][0] - 1]
            pv = norm.pdf(v, v0, self.init_pdf_sigma)
            pv[0] = pv[-1] = 0
            pv /= np.dot(w, pv)
            u0 = b[j] * (np.dot(w * v, pv) - v_r[j])
            Pj = np.zeros((len(u_grid[j]), Nx))
            ju = int(np.clip(np.floor((u0 - u_lo[j]) / self.du), 0, len(u_grid[j]) - 1))
            Pj[ju] = pv / self.du
            P.append(Pj)
            u_mean[j, 0] = u_grid[j][ju]
            v_mean[j, 0] = np.dot(w * v, pv)

        # work arrays
        g_plus, g_minus, lower, upper = [np.zeros(Nx - 1) for _ in range(4)]
        main, b_rhs, s_vec, y, z, cp = [np.zeros(Nx) for _ in range(6)]
        ustar = [b[j] * (v - v_r[j]) for j in range(n_pop)]
        factor = [np.full(Nx, np.exp(a[j] * dt)) for j in range(n_pop)]
        r_conv = np.zeros(n_pop)
        n_k_max = len(self.delay_kernel)
        self.n_fallback = 0
        snap_idx = sorted({int(np.searchsorted(self.t, ts - 1e-9)) for ts in self.snapshot_times
                           if ts - 1e-9 <= self.t[-1]})
        self.t_snap = self.t[snap_idx]
        self.P_snap = [np.zeros((len(snap_idx), len(u_grid[j]), Nx)) for j in range(n_pop)]
        self.flux_u = [np.zeros((Nt, len(u_grid[j]))) for j in range(n_pop)] if self.save_flux_u else None
        t0 = time.perf_counter()
        for i in tqdm(range(1, Nt), disable=self.tqdm_disable):
            n_k = min(i + 1, n_k_max)
            for kk in range(n_pop):
                r_conv[kk] = np.dot(r[kk, i - n_k + 1:i + 1][::-1], self.delay_kernel[:n_k]) * dt
            for j in range(n_pop):
                i_total = i_ext[j, i] + np.dot(self.connectivity_matrix[:, j], r_conv)
                D = max(sigma[j, i]**2 / tau_noise[j], 1e-10)
                H = np.empty_like(P[j])
                skip_thr = self.skip_rel * P[j].max()
                remap(P[j], u_lo[j], self.du, ustar[j], factor[j], H, skip_thr)          # U_dt P^n
                has_prev = P_prev_tr[j] is not None
                if has_prev:
                    Hprev = np.empty_like(P[j])
                    remap(P_prev_tr[j], u_lo[j], self.du, ustar[j], factor[j], Hprev, skip_thr)  # U_2dt P^{n-1}
                else:
                    Hprev = H
                Pnew = np.empty_like(P[j])
                Jout = np.zeros(len(u_grid[j]))
                mn, mx = lines_step(H, Hprev, has_prev, has_prev, u_grid[j], v, h, w, quad[j], k[j], C[j], i_total,
                                    D, dt, reset_idx[j], shift[j], False, Pnew, Jout, g_plus, g_minus, main, lower,
                                    upper, b_rhs, s_vec, y, z, cp, skip_thr, fct)
                if has_prev and mn < -1e-6 * mx:
                    # backward Euler for the whole population (keeps the mass exact)
                    self.n_fallback += 1
                    mn, mx = lines_step(H, Hprev, has_prev, False, u_grid[j], v, h, w, quad[j], k[j], C[j], i_total,
                                        D, dt, reset_idx[j], shift[j], False, Pnew, Jout, g_plus, g_minus, main,
                                        lower, upper, b_rhs, s_vec, y, z, cp, skip_thr, fct)
                if mn < 0:
                    mass_before = self.du * np.sum(Pnew @ w)
                    np.maximum(Pnew, 0, out=Pnew)
                    Pnew *= mass_before / (self.du * np.sum(Pnew @ w))
                P_prev_tr[j] = H        # U_dt P^n, transported once more in the next step
                P[j] = Pnew
                r[j, i] = self.du * Jout.sum()
                if self.save_flux_u:
                    self.flux_u[j][i] = Jout
                if i in snap_idx:
                    self.P_snap[j][snap_idx.index(i)] = Pnew
                pu = self.du * (Pnew @ w)                  # mass per u cell
                mass = pu.sum()
                u_mean[j, i] = np.dot(pu, u_grid[j]) / mass
                var_u[j, i] = np.dot(pu, u_grid[j]**2) / mass - u_mean[j, i]**2
                v_mean[j, i] = self.du * np.sum(Pnew.sum(axis=0) * w * v) / mass
        self.cpu = time.perf_counter() - t0
        self.r, self.u_mean, self.v_mean, self.var_u, self.P = r, u_mean, v_mean, var_u, P
        self.mass = [self.du * np.sum(Pj @ w) for Pj in P]

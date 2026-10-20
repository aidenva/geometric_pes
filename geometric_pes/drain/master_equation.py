"""Partial-secular Bloch-Redfield master equation for the exchange of relaxed products with N
quantum drain modes treated as a microcanonical bath (exact state counts, so detailed balance is
exact). The exchange time tau_x is the slowest P+/P- antisymmetric relaxation time.
"""
import os
import time

import numpy as np
from scipy.linalg import eig

from .statistics import eigen2d, count_states
from .modes import mode_freqs
from ..config import cache_dir

WC = 5.0


def _cache():
    d = os.path.join(cache_dir(), "qme")
    os.makedirs(d, exist_ok=True)
    return d


def J(w, eta):
    return 0.5 * np.pi * eta * w * np.exp(-w / WC)


def basis(S, hbar, emax, dx_fac=0.8, vpad=6.0, cache=True):
    """2D eigenstates below emax with x, y matrix elements and basin probabilities."""
    tag = f"eig_hb{hbar:g}_e{emax:.2f}_dx{dx_fac:g}_v{vpad:g}.npz"
    path = os.path.join(_cache(), tag)
    if cache and os.path.exists(path):
        d = np.load(path)
        return d["ev"], d["P"], d["X"], d["Y"]
    t0 = time.time()
    ev, P, info = eigen2d(S, hbar, emax, return_vecs=True, dx_fac=dx_fac, vpad=vpad)
    U = info["vecs"]
    X = U.T @ (info["x"][:, None] * U)
    Y = U.T @ (info["y"][:, None] * U)
    if cache:
        os.makedirs(_cache(), exist_ok=True)
        np.savez(path, ev=ev, P=P, X=X, Y=Y)
    print(f"  basis hbar={hbar} emax={emax:.2f}: {len(ev)} states, {info['npts']} DVR points, "
          f"dx={info['dx']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return ev, P, X, Y


class BathDOS:
    """smoothed log density of states of the quantum bath (energies above the ZPE)."""

    def __init__(self, hw, emax, de=None, win=None):
        self.de = de or min(0.002, hw.min() / 4)
        L = count_states(hw, emax + 1.0, self.de)          # log cumulative count on e = k de
        self.e = self.de * np.arange(len(L))
        self.logN = L
        h = 0.5 * (win or max(0.05, 2 * hw.min()))
        self.h = h
        lo = np.interp(np.maximum(self.e - h, 0.0), self.e, L)
        hi = np.interp(self.e + h, self.e, L)
        width = (self.e + h) - np.maximum(self.e - h, 0.0)
        self.logrho = hi + np.log1p(-np.exp(np.minimum(lo - hi, -1e-12))) - np.log(width)

    def lrho(self, E):
        return np.interp(E, self.e, self.logrho, left=-np.inf)

    def lN(self, E):
        return np.interp(E, self.e, self.logN, left=-np.inf)


def occupation(dos, hw, Eb, m0, mmax=400):
    """F = sum_{m >= m0} rho(Eb - m hw) / rho(Eb) (elementwise arrays hw, Eb)."""
    hw = np.asarray(hw, float); Eb = np.asarray(Eb, float)
    F = np.zeros_like(hw)
    l0 = dos.lrho(Eb)
    nmax = np.floor(Eb / np.maximum(hw, 1e-300))
    exact = nmax <= mmax
    # exact sum where few terms
    m = np.arange(mmax + 1)
    if exact.any():
        he, Ee, l0e = hw[exact], Eb[exact], l0[exact]
        F[exact] = 0.0
        for k in m[m0:]:
            ok = Ee - k * he >= 0
            if not ok.any():
                break
            F[exact] += np.where(ok, np.exp(dos.lrho(Ee - k * he) - l0e), 0.0)
    # many terms: Euler-Maclaurin, sum_{m>=0} f(m hw) = N(Eb)/(rho hw) + 1/2 f(0) + ...
    big = ~exact
    if big.any():
        s0 = np.exp(dos.lN(Eb[big]) - l0[big]) / hw[big] + 0.5
        F[big] = s0 - (1.0 if m0 == 1 else 0.0)
    return F


def cmatrix(ev, hbar, eta, E, dos, wmin=1e-6):
    """c[a, b] = bath spectrum for the transition b -> a (bath energy before: E - e_b):
    (2/hbar) J(w) <n+1> if e_b > e_a (emission), (2/hbar) J(|w|) <n> if e_b < e_a,
    and the w -> 0 limit pi eta N(E_b) / (rho(E_b) hbar^2) on the diagonal (pure dephasing)."""
    n = len(ev)
    A, B = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    dE = ev[B] - ev[A]                                   # > 0: emission
    hw = np.maximum(np.abs(dE), wmin * hbar)
    Eb = E - ev[B]
    C = np.zeros((n, n))
    em = dE > 0
    ab = dE < 0
    C[em] = (2 / hbar) * J(hw[em] / hbar, eta) * occupation(dos, hw[em], Eb[em], 0)
    C[ab] = (2 / hbar) * J(hw[ab] / hbar, eta) * occupation(dos, hw[ab], Eb[ab], 1)
    z = ~(em | ab)                                       # exact degeneracy and diagonal
    C[z] = np.pi * eta * np.exp(dos.lN(Eb[z]) - dos.lrho(Eb[z])) / hbar**2
    return C


def redfield(ev, Ss, C, hbar, delta):
    """Bloch-Redfield generator (no Lamb shift) on populations + coherences between
    levels closer than delta (energy). Returns L (complex, dim x dim) and the index pairs.

        d rho / dt = -i [H, rho] / hbar - sum_s [S, Lam rho - rho Lam^T],  Lam_ab = S_ab C_ab / 2
    """
    n = len(ev)
    a, b = np.nonzero(np.abs(ev[:, None] - ev[None, :]) < delta)
    if delta <= 0:
        a = b = np.arange(n)
    A, Bi = a[:, None], b[:, None]
    Cc, D = a[None, :], b[None, :]
    R = np.zeros((len(a), len(a)))
    for S_ in Ss:
        Lam = 0.5 * S_ * C
        M = S_ @ Lam                 # (S Lam)_ac
        M2 = Lam.T @ S_              # (Lam^T S)_db
        R += -1.0 * (Bi == D) * M[A, Cc] + Lam[A, Cc] * S_[D, Bi] + S_[A, Cc] * Lam[Bi, D] - 1.0 * (A == Cc) * M2[D, Bi]
    L = R.astype(complex)
    L[np.arange(len(a)), np.arange(len(a))] += -1j * (ev[a] - ev[b]) / hbar
    return L, a, b


def relaxed_start(pi, P, S, Pminus=0.539):
    """'relaxed products': local equilibrium within each product well, P- share Pminus
    (the frozen QCT-Bose split at theta = 0)."""
    inm = P[:, S.iPm] > 0.5
    inp = P[:, S.iPp] > 0.5
    p0 = np.zeros(len(pi))
    p0[inm] = Pminus * pi[inm] / pi[inm].sum()
    p0[inp] = (1 - Pminus) * pi[inp] / pi[inp].sum()
    return p0


def redfield_case(S, hbar, N, eta, E, delta, ecut=None, dx_fac=0.8, vpad=6.0, win=None, de=None, wmax=None,
                  tau_obs=400.0, Pminus=0.539, nev=8, propagate=False, verbose=True):
    ev, P, X, Y = basis(S, hbar, E, dx_fac=dx_fac, vpad=vpad)
    keep = ev <= (E if ecut is None else min(E, ecut))
    ev, P, X, Y = ev[keep], P[keep], X[np.ix_(keep, keep)], Y[np.ix_(keep, keep)]
    w = mode_freqs(N, eta)
    if wmax is not None:
        w = w[w < wmax]
    hw = hbar * w
    dos = BathDOS(hw, E - ev.min() + 1.0, de=de, win=win)
    C = cmatrix(ev, hbar, eta, E, dos)
    L, a, b = redfield(ev, (X, Y), C, hbar, delta)
    t0 = time.time()
    pop = a == b
    ia = a[pop]
    if propagate:
        lam, Vr = eig(L)
    else:
        from scipy.sparse.linalg import eigs
        lam, Vr = eigs(L, k=min(nev, L.shape[0] - 2), sigma=1e-9, which="LM")
    max_re = float(np.sort(lam.real)[-2]) if propagate else None    # largest besides the stationary one
    o = np.argsort(np.abs(lam))                  # stationary first, then the slowest modes
    lam, Vr = lam[o], Vr[:, o]
    modes = []
    for j in range(min(nev, len(lam))):
        v = np.zeros(len(ev), complex); v[ia] = Vr[pop, j]
        pr = np.array([(v * P[:, q]).sum() for q in range(P.shape[1])]).real
        modes.append(dict(rate=float(-lam[j].real), imag=float(lam[j].imag),
                          proj={S.names[q]: float(pr[q] / max(np.abs(pr).sum(), 1e-300)) for q in range(len(pr))}))
    pi = np.zeros(len(ev)); pi[ia] = Vr[pop, 0].real; pi /= pi.sum()
    xm = [m for m in modes[1:] if m["proj"]["P+"] * m["proj"]["P-"] < 0 and abs(m["proj"]["R"]) < 0.25
          and abs(m["imag"]) < 1e-6 * max(m["rate"], 1e-30)]
    xm = xm[0] if xm else modes[1]
    stat = {S.names[q]: float(pi @ P[:, q]) for q in range(P.shape[1])}
    st_m = stat["P-"] / (stat["P-"] + stat["P+"])
    lx = xm["rate"]
    times = [0.0, tau_obs, 2 * tau_obs]
    Pm_t = [float(st_m + (Pminus - st_m) * np.exp(-lx * t)) for t in times]
    r = dict(hbar=hbar, N=N, eta=eta, E=E, delta=delta, ecut=ecut, nstates=int(len(ev)), wmax=wmax,
             nmodes=int(len(hw)), dim=int(L.shape[0]), dx_fac=dx_fac, vpad=vpad, win=dos.h * 2, de=dos.de,
             tau_x=1.0 / lx, exchange_mode=xm, modes=modes, stat=stat, stat_minus_share=st_m,
             Pminus_t=Pm_t, phi_t=[1 - q for q in Pm_t], iota_minus_tobs=Pm_t[1] / st_m, times=times,
             tau_obs=tau_obs, Pminus0=Pminus, pi_min=float(pi.min()),
             max_growth=max_re)
    if propagate:
        # direct propagation from relaxed products (full spectrum): P+-(t)
        p0 = relaxed_start(pi, P, S, Pminus)
        r0 = np.zeros(L.shape[0], complex); r0[np.nonzero(pop)[0]] = p0[ia]
        cc = np.linalg.solve(Vr, r0)
        prop = []
        for t in times + [50.0, 100.0, 200.0]:
            rt = Vr @ (cc * np.exp(lam * t))
            p = np.zeros(len(ev)); p[ia] = rt[pop].real
            Pp, Pm = float(p @ P[:, S.iPp]), float(p @ P[:, S.iPm])
            prop.append(dict(t=t, P_plus=Pp, P_minus=Pm, phi=Pp / (Pp + Pm), pmin=float(p.min())))
        r["propagated"] = prop
    if verbose:
        extra = ""
        if propagate:
            q = r["propagated"][1]
            extra = f" | propagated P-({tau_obs:g})={q['P_minus'] / (q['P_minus'] + q['P_plus']):.3f} pmin={min(z['pmin'] for z in r['propagated']):.1e} max Re lam={r['max_growth']:.1e}"
        print(f"RF hbar={hbar:<5} N={N:3d} eta={eta:<4} delta={delta:<5} ecut={ecut} wmax={wmax} n={len(ev):4d} dim={L.shape[0]:6d} "
              f"tau_x={1/lx:.4g} stat-={st_m:.4f} P-({tau_obs:g})={Pm_t[1]:.3f} phi={1 - Pm_t[1]:.3f} iota-={r['iota_minus_tobs']:.2f} "
              f"pi_min={pi.min():.1e} ({time.time() - t0:.0f}s){extra}", flush=True)
    return r

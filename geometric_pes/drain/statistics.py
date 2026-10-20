"""Statistical references and rate estimates.

stat_fractions        classical many-mode basin shares, weight int_basin (E - V)^N dA
stat_quantum          the same with exact quantum state counts of the modes (above zero point)
eigen2d               2D eigenstates by sinc DVR (quantum statistics of the reacting coordinates)
p_return              equilibrium probability that the x, y energy is back above a level
tst_tables, rates     microcanonical TST rates (product exchange; ergodic estimate)
"""
import numpy as np

from ..surfaces.branching import V
from ..surfaces.basins import Setup
from ..config import surface as _surface


def stat_fractions(S, E, N):
    """microcanonical basin fractions with N modes at energies E (array) -> (len(E), nbasin)."""
    nx, ny = S.lab.shape
    X = S.gx0 + S.gdx * np.arange(nx)
    Y = S.gy0 + S.gdy * np.arange(ny)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    Vg = V(XX, YY, S.p)
    out = []
    for e in np.atleast_1d(E):
        K = np.clip(e - Vg, 0, None)
        lw = np.where(K > 0, N * np.log(np.maximum(K, 1e-300)), -np.inf)
        w = np.exp(lw - lw.max())
        b = np.array([w[S.lab == i].sum() for i in range(len(S.names))])
        out.append(b / b.sum())
    return np.array(out)


def p_return(key, N, E, E_ts):
    """equilibrium probability, with N classical modes at total energy E, that the x, y energy is
    at or above E_ts: weight of x, y energy e is A(e) (E - e)^(N-1), A(e) the area with V < e"""
    S = Setup(_surface(key) if isinstance(key, str) else key)
    nx, ny = S.lab.shape
    X = S.gx0 + S.gdx * np.arange(nx); Y = S.gy0 + S.gdy * np.arange(ny)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    Vs = np.sort(V(XX, YY, S.p).ravel())
    e = np.linspace(Vs[0], E, 4000)
    w = np.searchsorted(Vs, e) * np.clip(E - e, 0, None) ** (N - 1)
    return float(w[e >= E_ts].sum() / w.sum())


def tst_tables(S, ts12):
    """microcanonical TST exchange rates of the bare 2D surface, P+ -> P- and P- -> P+, on an
    energy grid: k_j(e) = int 2 sqrt(2 (e - V)) ds / (2 pi A_j(e)), the line integral along the
    dividing line through TS12 (normal to its unstable mode), A_j(e) the area of basin j below e"""
    p = S.p
    n = ts12["vecs"][:, 0]; t = np.array([-n[1], n[0]])
    sv = np.linspace(-4, 4, 4001); ds = sv[1] - sv[0]
    pts = ts12["xy"][:, None] + np.outer(t, sv)
    Vl = V(pts[0], pts[1], p)
    nx, ny = S.lab.shape
    X = S.gx0 + S.gdx * np.arange(nx); Y = S.gy0 + S.gdy * np.arange(ny)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    Vg = V(XX, YY, p); dA = S.gdx * S.gdy
    Vp = np.sort(Vg[S.lab == S.iPp]); Vm = np.sort(Vg[S.lab == S.iPm])
    e = np.linspace(ts12["E"], ts12["E"] + 40.0, 2000)
    flux = np.array([np.sum(2 * np.sqrt(2 * np.clip(ei - Vl, 0, None))) * ds for ei in e])
    Ap = np.searchsorted(Vp, e) * dA; Am = np.searchsorted(Vm, e) * dA
    return e, flux / (2 * np.pi * np.maximum(Ap, dA)), flux / (2 * np.pi * np.maximum(Am, dA))


def count_states(hw, emax, de):
    """log of the cumulative number of states N_q(e) on e = 0, de, ..., emax.
    hw: quanta. Beyer-Swinehart with rescaling (returns log N_q)."""
    n = int(np.ceil(emax / de)) + 1
    rho = np.zeros(n); rho[0] = 1.0
    logscale = 0.0
    for h in hw:
        k = int(round(h / de))
        if k == 0:
            raise ValueError("energy bin larger than a quantum; reduce de")
        for a in range(k, n, k):
            b = min(a + k, n)
            rho[a:b] += rho[a - k:b - k]
        m = rho.max(); rho /= m; logscale += np.log(m)
    cum = np.cumsum(rho)
    return np.log(cum) + logscale


def classical_logcount(hw, e):
    e = np.maximum(e, 1e-300)
    N = len(hw)
    from scipy.special import gammaln
    return N * np.log(e) - gammaln(N + 1) - np.log(hw).sum()


class Grid:
    def __init__(self, S):
        nx, ny = S.lab.shape
        X = S.gx0 + S.gdx * np.arange(nx)
        Y = S.gy0 + S.gdy * np.arange(ny)
        XX, YY = np.meshgrid(X, Y, indexing="ij")
        self.Vg = V(XX, YY, S.p)
        self.lab = S.lab
        self.nb = len(S.names)


def basin_fractions_qmodes(G, E, logN, de):
    """classical x, y + quantum modes: W_j = int_basin N_q(E - V) dA."""
    K = E - G.Vg
    ok = K >= 0
    idx = np.clip(np.floor(np.where(ok, K, 0) / de).astype(int), 0, len(logN) - 1)
    lw = np.where(ok, logN[idx], -np.inf)
    w = np.exp(lw - lw[ok].max())
    b = np.array([w[G.lab == i].sum() for i in range(G.nb)])
    return b / b.sum()


def basin_fractions_cl(G, E, N):
    K = np.clip(E - G.Vg, 0, None)
    lw = np.where(K > 0, N * np.log(np.maximum(K, 1e-300)), -np.inf)
    w = np.exp(lw - lw.max())
    b = np.array([w[G.lab == i].sum() for i in range(G.nb)])
    return b / b.sum()


def eigen2d(S, hbar, emax, nx=None, ny=None, return_vecs=False, dx_fac=0.8, vpad=6.0):
    """2D eigenstates below emax by sinc DVR (mass 1); returns energies (from the
    classical zero of V) and basin probabilities (n_states, n_basins). With
    return_vecs the info dict also holds the eigenvectors and the DVR points
    (used by qme.py for coupling matrix elements)."""
    p = S.p
    kmax = np.sqrt(2 * (emax - S.mins[S.iPp]["E"]) + 1.0) / hbar
    dx = dx_fac * np.pi / kmax
    xs = np.arange(-2.4, 2.7, dx)
    ys = np.arange(-3.0, 3.0, dx)
    if p.A_c:
        ys = np.arange(-3.6, 2.6, dx)
    def T1(n, d):
        i = np.arange(n)
        D = i[:, None] - i[None, :]
        with np.errstate(divide="ignore"):
            T = np.where(D == 0, np.pi**2 / 3, 2.0 / np.where(D == 0, 1, D)**2 * (-1.0)**D)
        return hbar**2 / (2 * d * d) * T
    Tx, Ty = T1(len(xs), dx), T1(len(ys), dx)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    Vg = V(XX, YY, p).ravel()
    keep = Vg < emax + vpad                     # prune grid points far above emax
    nxp, nyp = len(xs), len(ys)
    ii, kk = np.unravel_index(np.nonzero(keep)[0], (nxp, nyp))
    n = len(ii)
    H = np.where(kk[:, None] == kk[None, :], Tx[ii[:, None], ii[None, :]], 0.0)
    H += np.where(ii[:, None] == ii[None, :], Ty[kk[:, None], kk[None, :]], 0.0)
    H[np.arange(n), np.arange(n)] += Vg[keep]
    from scipy.linalg import eigh
    ev, vec = eigh(H, subset_by_value=(-np.inf, emax), driver="evr")
    lab = S.basin(XX.ravel()[keep], YY.ravel()[keep])
    P = np.array([(vec[lab == j] ** 2).sum(0) for j in range(len(S.names))]).T
    info = dict(npts=n, dx=dx)
    if return_vecs:
        info.update(vecs=vec, x=XX.ravel()[keep], y=YY.ravel()[keep])
    return ev, P, info


def basin_fractions_qfull(ev, P, E, logN, de):
    K = E - ev
    ok = K >= 0
    idx = np.clip(np.floor(np.where(ok, K, 0) / de).astype(int), 0, len(logN) - 1)
    lw = np.where(ok, logN[idx], -np.inf)
    w = np.exp(lw - lw[ok].max())
    b = (w[:, None] * P).sum(0)
    return b / b.sum()


def thermal_energies(hw, kT, n, rng):
    """mode excitation energies above ZPE at t = 0 (Bose-Einstein), per sample."""
    if kT <= 0:
        return np.zeros(n)
    pz = np.exp(-hw / kT)
    # geometric: n_k ~ P(n) = (1 - pz) pz^n
    u = rng.uniform(size=(n, len(hw)))
    nk = np.floor(np.log1p(-u) / np.log(np.maximum(pz, 1e-300)))
    nk = np.where(pz < 1e-300, 0, nk)
    return (nk * hw).sum(1)


def stat_quantum(S, E, hw, de=None):
    """quantum-mode microcanonical basin fractions at excitation energies E (array)."""
    G = Grid(S)
    d = de or min(0.002, hw.min() / 4)
    emax = float(np.max(E)) - S.mins[S.iPp]["E"] + 1.0
    logN = count_states(hw, emax, d)
    return np.array([basin_fractions_qmodes(G, e, logN, d) for e in np.atleast_1d(E)])


def ts12_line(S, n=4001, half=3.0):
    sad = [s for s in S.sads if set(s["connects"]) == {"P+", "P-"}][0]
    nv = sad["vecs"][:, 0]
    t = np.array([-nv[1], nv[0]])
    u = np.linspace(-half, half, n)
    pts = sad["xy"][:, None] + np.outer(t, u)
    ia, ib = S.iPp, S.iPm
    la = S.basin(pts[0] + 0.02 * nv[0], pts[1] + 0.02 * nv[1])
    lb = S.basin(pts[0] - 0.02 * nv[0], pts[1] - 0.02 * nv[1])
    on = ((la == ia) & (lb == ib)) | ((la == ib) & (lb == ia))
    return u, V(pts[0], pts[1], S.p), on, sad["E"]


def rates(S, E, logN_fn, nK=400):
    """k(P- -> P+), k(P+ -> P-) at total energy E; logN_fn(eps) -> log N_m(eps)."""
    G = Grid(S)
    K = E - G.Vg
    ok = K > 0
    lw = np.full(K.shape, -np.inf)
    lw[ok] = logN_fn(K[ok])
    ref = lw[ok].max()
    w = np.exp(lw - ref)
    dA = S.gdx * S.gdy
    Om = {i: 2 * np.pi * w[G.lab == i].sum() * dA for i in (S.iPp, S.iPm)}
    u, Vl, on, Ets = ts12_line(S)
    du = u[1] - u[0]
    J = 0.0
    for Vi in Vl[on]:
        Km = E - Vi
        if Km <= 0:
            continue
        # int dp_t N(E - V - p^2/2) = 2 int_0^Km dK / sqrt(2K) N(Km - K); substitute K = Km s^2
        s = (np.arange(nK) + 0.5) / nK
        Kk = Km * s**2
        val = np.exp(logN_fn(np.maximum(Km - Kk, 1e-12)) - ref)
        J += 2 * np.sum(val * np.sqrt(2 * Km) / nK) * du       # dK/sqrt(2K) = sqrt(2 Km) ds
    return J / Om[S.iPm], J / Om[S.iPp], Om[S.iPm] / (Om[S.iPm] + Om[S.iPp])


def make_logN(hw, emax, quantum=True):
    if not quantum:
        return lambda e: classical_logcount(hw, e)
    de = min(0.002, hw.min() / 4)
    L = count_states(hw, emax, de)
    return lambda e: L[np.clip(np.floor(np.asarray(e) / de).astype(int), 0, len(L) - 1)]


def reference_shares(S, E, Ns=(0, 4, 16, 64)):
    """statistical shares of R, P+ and P- at energy E with N classical modes (N = 0: the 2D
    surface's own statistics)"""
    out = {}
    for N in Ns:
        f = stat_fractions(S, [E], N)[0]
        out[f"N{N}"] = dict(R=float(f[S.iR]), P_plus=float(f[S.iPp]), P_minus=float(f[S.iPm]))
    return out

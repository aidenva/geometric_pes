#!/usr/bin/env python3
"""M2: the junction - an ET crossing seam feeding a branching ridge.

Two diabatic states (hbar = 1, equal masses M on x and y):

    V1 = kx/2 (x + a)^2 + ky/2 y^2                              donor
    V2 = kx/2 (x - a)^2 - eps + kap(x)/2 y^2 + b(y)/4 y^4 + tau y   acceptor
         - A_w exp(-(x - x_w)^2/(2 s_wx^2) - (y - y_w)^2/(2 s_wy^2))   optional extra well (M6)
    kap(x) = k_in + (k_out - k_in) s((x - x_r)/w)               y-curvature on V2
    D(x,y) = D0 + c (y - y_c)                                   coupling

At y = 0 the seam V1 = V2 sits at x_s = -eps/(2 kx a). V2 is stable in y
before x_r and a double well after it (b_plus for y > 0, b_minus for y < 0,
so the products P+ and P- differ in depth and width). The distance
d = x_r - x_s moves the branching ridge downstream of the seam. D0 = 0, c > 0
puts a conical intersection on the seam at y = y_c; c = 0 gives an avoided
crossing of width D0.

hbar_eff is set by the mass: classical dynamics in units of the vibrational
period are M-independent, quantum dynamics see hbar_eff ~ 1/sqrt(M) (as Q1).

Engines (all closed, no bath):
  quantum2   two-state split operator (exact 2x2 potential step, funnel.model.propagator_2x2)
  quantumBO  single-surface split operator on E_- (no upper state, no geometric phase)
  classical  Newtonian trajectories on E_- from the Wigner distribution of the packet
  fssh       Tully fewest switches on E_-/E_+ (exact 2x2 electronic step, NAC rescaling)

First-arrival branching: product *cores* (basin P+/P- of E_- with
E_- < E_TS12 - core_margin) are absorbing. Quantum: complex absorbing
potential -iW inside the cores, absorbed probability per core. Classical:
trajectory removed on first entry into a core.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..two_state.model import propagator_2x2
from ..two_state.stationary import merge_points



@dataclass(frozen=True)
class J:
    name: str = "J"
    kx: float = 1.0
    a: float = 3.0
    ky: float = 2.0
    eps: float = 6.0
    k_in: float = 2.0
    k_out: float = -4.0
    x_r: float = -1.0       # ridge onset on V2 (seam at x_s = -eps/(2 kx a) = -1)
    w: float = 0.3
    b_plus: float = 1.5
    b_minus: float = 3.0
    tau: float = 0.0
    D0: float = 0.5
    c: float = 0.0
    y_c: float = 0.0
    M: float = 16.0
    # grid / window
    x_min: float = -8.0
    x_max: float = 9.0
    y_min: float = -4.0
    y_max: float = 4.0
    core_margin: float = 0.5
    E_kin0: float = 3.0     # initial kinetic energy along +x (packet starts at the E_- minimum of R)
    # transverse (branching-mode) initial state: "gauss" = the Gaussian used in M2
    # (width (M k_yy)^-1/4, ~94% ground state); "ho" = harmonic eigenstate n_y of the
    # R-well y mode, with E_kin0 lowered by n_y * w_y so the total energy is unchanged
    transverse: str = "gauss"
    n_y: int = 0
    # compensate = False keeps E_kin0 for every n_y (the y mode adds its energy:
    # a thermally excited reactant fed by the same ET kick; used for thermal mixtures)
    compensate: bool = True
    # optional extra Gaussian well on V2 (M6, downstream basin); A_w = 0 leaves V2 unchanged
    A_w: float = 0.0
    x_w: float = 3.0
    y_w: float = 0.0
    s_wx: float = 0.5
    s_wy: float = 0.5

    @property
    def x_s(self):
        return -self.eps / (2 * self.kx * self.a)


def _sw(x, x0, w):
    t = np.tanh((x - x0) / w)
    return 0.5 * (1 + t), 0.5 * (1 - t * t) / w


def diabats(x, y, p):
    """v1, v2, D and their gradients."""
    s, ds = _sw(x, p.x_r, p.w)
    kap = p.k_in + (p.k_out - p.k_in) * s
    b = np.where(y > 0, p.b_plus, p.b_minus)
    v1 = 0.5 * p.kx * (x + p.a) ** 2 + 0.5 * p.ky * y**2
    v2 = 0.5 * p.kx * (x - p.a) ** 2 - p.eps + 0.5 * kap * y**2 + 0.25 * b * y**4 + p.tau * y
    D = p.D0 + p.c * (y - p.y_c) + 0 * x
    g1 = (p.kx * (x + p.a), p.ky * y)
    g2 = (p.kx * (x - p.a) + 0.5 * (p.k_out - p.k_in) * ds * y**2,
          kap * y + b * y**3 + p.tau)
    if p.A_w:
        ux, uy = (x - p.x_w) / p.s_wx, (y - p.y_w) / p.s_wy
        vw = -p.A_w * np.exp(-0.5 * (ux * ux + uy * uy))
        v2 = v2 + vw
        g2 = (g2[0] - vw * ux / p.s_wx, g2[1] - vw * uy / p.s_wy)
    gD = (0 * x, p.c + 0 * y)
    return v1, v2, D, g1, g2, gD


def adiabats(x, y, p):
    v1, v2, D, g1, g2, gD = diabats(x, y, p)
    cc = 0.5 * (v1 + v2)
    aa = 0.5 * (v1 - v2)
    R = np.sqrt(aa * aa + D * D) + 1e-300
    Em, Ep = cc - R, cc + R
    gc = [0.5 * (g1[i] + g2[i]) for i in range(2)]
    ga = [0.5 * (g1[i] - g2[i]) for i in range(2)]
    gR = [(aa * ga[i] + D * gD[i]) / R for i in range(2)]
    gm = (gc[0] - gR[0], gc[1] - gR[1])
    gp = (gc[0] + gR[0], gc[1] + gR[1])
    # mixing angle theta = 1/2 atan2(2D, v1 - v2); NAC = grad theta
    u, wv = v1 - v2, 2 * D
    den = u * u + wv * wv + 1e-300
    nac = tuple(0.5 * (u * 2 * gD[i] - wv * 2 * ga[i]) / den for i in range(2))
    theta = 0.5 * np.arctan2(wv, u)
    return dict(Em=Em, Ep=Ep, gm=gm, gp=gp, nac=nac, theta=theta, v1=v1, v2=v2, D=D)


def Em(x, y, p):
    return adiabats(x, y, p)["Em"]


# ------------------------------------------------------------ stationary points on E_-
def stationary(p, n_starts=300, seed=3):
    rng = np.random.default_rng(seed)
    pts = []

    def f(z):
        g = adiabats(z[0], z[1], p)["gm"]
        return float(g[0] ** 2 + g[1] ** 2)

    for s0 in np.column_stack([rng.uniform(-4.5, 4.5, n_starts), rng.uniform(-3, 3, n_starts)]):
        r = minimize(f, s0, method="L-BFGS-B", bounds=[(-5, 5), (-3.5, 3.5)],
                     options=dict(ftol=1e-22, gtol=1e-14, maxiter=4000))
        if r.fun < 1e-12 and np.all(np.abs(r.x) < [4.9, 3.4]):
            pts.append(r.x)
    pts = merge_points(pts, 1e-3)
    out = []
    for z in pts:
        h = 1e-5
        H = np.zeros((2, 2))
        for i, dd in enumerate(np.eye(2) * h):
            gp_ = np.array(adiabats(*(z + dd), p)["gm"]); gm_ = np.array(adiabats(*(z - dd), p)["gm"])
            H[:, i] = (gp_ - gm_) / (2 * h)
        wv, vec = np.linalg.eigh(0.5 * (H + H.T))
        kind = {0: "min", 1: "saddle", 2: "max"}[int(np.sum(wv < -1e-6))]
        out.append(dict(xy=z, E=float(Em(z[0], z[1], p)), kind=kind, eig=wv, vecs=vec))
    mins = [d for d in out if d["kind"] == "min"]
    for d in mins:
        x, y = d["xy"]
        d["name"] = "R" if x < 0 else ("P+" if y > 0 else "P-")
    for nm in ("P+", "P-"):                       # an extra well (A_w > 0) can add a minimum
        same = [d for d in mins if d["name"] == nm]
        if len(same) > 1:
            min(same, key=lambda d: (d["xy"][0] - p.x_w) ** 2 + (d["xy"][1] - p.y_w) ** 2)["name"] = "W"
    sads = [d for d in out if d["kind"] == "saddle"]
    for d in sads:
        x, y = d["xy"]
        d["name"] = "TS1" if x < p.x_r + 1.0 and abs(y) < 1.0 and x < 1.5 else "TS12"
    return mins, sads


class Grid:
    """Grid, surfaces, basins and cores for one parameter set."""

    def __init__(self, p, nx=None, ny=None):
        self.p = p
        pmax = np.sqrt(2 * p.M * (p.E_kin0 + p.eps + 8.0 + p.A_w))  # generous momentum bound
        dxmax = np.pi / (1.6 * pmax)
        nx = nx or int(2 ** np.ceil(np.log2((p.x_max - p.x_min) / dxmax)))
        ny = ny or int(2 ** np.ceil(np.log2((p.y_max - p.y_min) / dxmax)))
        self.x = np.linspace(p.x_min, p.x_max, nx, endpoint=False)
        self.y = np.linspace(p.y_min, p.y_max, ny, endpoint=False)
        self.dx, self.dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")
        a = adiabats(self.X, self.Y, p)
        self.v1, self.v2, self.D, self.Em, self.Ep, self.theta = (a[k] for k in
                                                                  ("v1", "v2", "D", "Em", "Ep", "theta"))
        self.mins, self.sads = stationary(p)
        self.names = [d["name"] for d in self.mins]
        self.ts12 = [s for s in self.sads if s["name"] == "TS12"]
        self.ts1 = [s for s in self.sads if s["name"] == "TS1"]
        self.lab = self._basins()
        Ecut = (self.ts12[0]["E"] if self.ts12 else max(d["E"] for d in self.mins)) - p.core_margin
        self.Ecut = Ecut
        self.core = {nm: (self.lab == i) & (self.Em < Ecut) for i, nm in enumerate(self.names)
                     if nm in ("P+", "P-")}

    def _basins(self):
        p = self.p
        zx, zy = self.X.copy(), self.Y.copy()
        for _ in range(4000):
            g = adiabats(zx, zy, p)["gm"]
            gn = np.sqrt(g[0] ** 2 + g[1] ** 2) + 1e-12
            st = np.minimum(2e-3, 0.01 / gn)
            zx -= st * g[0]; zy -= st * g[1]
        m = np.array([d["xy"] for d in self.mins])
        dist = (zx[..., None] - m[:, 0]) ** 2 + (zy[..., None] - m[:, 1]) ** 2
        return np.argmin(dist, axis=-1)

    def lookup(self, field, x, y):
        i = np.clip(np.rint((x - self.x[0]) / self.dx).astype(int), 0, len(self.x) - 1)
        k = np.clip(np.rint((y - self.y[0]) / self.dy).astype(int), 0, len(self.y) - 1)
        return field[i, k]

    def packet(self):
        """Gaussian on the R minimum of E_-, ground-state widths, momentum along +x."""
        p = self.p
        r = [d for d in self.mins if d["name"] == "R"][0]
        H = np.diag(r["eig"]) if abs(r["vecs"][0, 0]) > 0.99 else None
        kxx, kyy = (r["eig"][0], r["eig"][1]) if abs(r["vecs"][0, 0]) > 0.99 else (r["eig"][1], r["eig"][0])
        sx = (1.0 / (p.M * kxx)) ** 0.25
        wy = np.sqrt(kyy / p.M)
        if p.transverse == "ho":
            sy = (4.0 * p.M * kyy) ** -0.25                  # ground-state density std
            ekin = p.E_kin0 - (p.n_y * wy if p.compensate else 0.0)
        else:
            sy = (1.0 / (p.M * kyy)) ** 0.25
            ekin = p.E_kin0
        px = np.sqrt(2 * p.M * ekin)
        return dict(x0=r["xy"][0], y0=r["xy"][1], sx=sx, sy=sy, px0=px, py0=0.0, wy=wy)


# ------------------------------------------------------------ quantum
def cap(G, W0):
    """-iW inside each product core, ramped smoothly with the depth below Ecut."""
    W = {}
    for nm, mask in G.core.items():
        depth = np.clip((G.Ecut - G.Em) / 1.0, 0, 1)
        W[nm] = W0 * np.where(mask, depth**2, 0.0)
    return W


def transverse_states(G, x0, K=16):
    """Eigenstates of -1/(2M) d^2/dy^2 + E_-(x0, y) on the y grid (finite differences)."""
    p = G.p
    y = G.y
    n = len(y)
    v = Em(np.full(n, x0), y, p)
    off = -1.0 / (2 * p.M * G.dy**2)
    H = np.diag(v - 2 * off) + np.diag(np.full(n - 1, off), 1) + np.diag(np.full(n - 1, off), -1)
    e, U = np.linalg.eigh(H)
    U = U[:, :K] / np.sqrt(G.dy)                     # normalized: sum |phi|^2 dy = 1
    return e[:K], U


def ho_state(G, n, y0, sy):
    """Normalized harmonic eigenstate n on the y grid (stable recursion), shape (ny,)."""
    xi = (G.y - y0) / (np.sqrt(2) * sy)
    h0 = np.pi ** -0.25 * np.exp(-xi**2 / 2)
    if n == 0:
        out = h0
    else:
        h1 = np.sqrt(2) * xi * h0
        for k in range(2, n + 1):
            h0, h1 = h1, np.sqrt(2 / k) * xi * h1 - np.sqrt((k - 1) / k) * h0
        out = h1
    return out / np.sqrt((out**2).sum() * G.dy)


def quantum(p, G=None, two_state=True, t_final=None, dt=None, W0=3.0, n_out=200, keep=False,
            probe_x=None, K=16, psi0=None, probe_series=0):
    """psi0: optional initial wavefunction on the grid (complex (nx, ny)), used instead
    of the packet built from p (it is normalized here). probe_series = k > 0 also
    records the transverse amplitudes c_0..c_{k-1}(t) at the probe every step."""
    G = G or Grid(p)
    pk = G.packet()
    X, Y = G.X, G.Y
    if psi0 is not None:
        gy = None
    elif p.transverse == "ho":
        gy = ho_state(G, p.n_y, pk["y0"], pk["sy"])[None, :]
    else:
        gy = np.exp(-((Y - pk["y0"]) ** 2) / (4 * pk["sy"] ** 2))
    if psi0 is not None:
        g = np.asarray(psi0, complex)
    else:
        g = gy * np.exp(-((X - pk["x0"]) ** 2) / (4 * pk["sx"] ** 2)
                        + 1j * (pk["px0"] * (X - pk["x0"]) + pk["py0"] * (Y - pk["y0"])))
    g /= np.sqrt((np.abs(g) ** 2).sum() * G.dx * G.dy)
    th = G.theta
    lower = (-np.sin(th), np.cos(th))                        # lower adiabatic eigenvector
    T_vib = 2 * np.pi / np.sqrt(p.kx / p.M)
    t_final = t_final or 2.0 * T_vib
    dt = dt or T_vib / 2000
    kx = 2 * np.pi * np.fft.fftfreq(len(G.x), d=G.dx)
    ky = 2 * np.pi * np.fft.fftfreq(len(G.y), d=G.dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    Tk = np.exp(-1j * (KX**2 + KY**2) / (2 * p.M) * dt / 2)
    W = cap(G, W0)
    Wtot = sum(W.values())
    damp = np.exp(-Wtot * dt)
    if two_state:
        U11, U22, U12 = propagator_2x2(G.v1, G.v2, G.D, dt)
        psi = [g * lower[0], g * lower[1]]
    else:
        UB = np.exp(-1j * G.Em * dt)
        psi = [g]
    absorbed = {nm: 0.0 for nm in W}
    if probe_x is not None:
        i0 = int(np.argmin(np.abs(G.x - probe_x)))
        e_k, Uk = transverse_states(G, G.x[i0], K)
        flux_k = np.zeros(K)                    # time-integrated flux through x = x0, per state
        flux_tot = 0.0
        series = np.zeros((int(round(t_final / dt)), probe_series), complex) if probe_series else None
    nsteps = int(round(t_final / dt))
    every = max(1, nsteps // n_out)
    out = dict(t=[], abs_p=[], abs_m=[], norm=[], P_upper=[], P_R=[], P_edge=[])
    edge = (np.abs(X - X.min()) < 0.5) | (np.abs(X - X.max()) < 0.5) | \
           (np.abs(Y - Y.min()) < 0.3) | (np.abs(Y - Y.max()) < 0.3)
    iR = G.names.index("R")
    for i in range(1, nsteps + 1):
        psi = [np.fft.ifft2(Tk * np.fft.fft2(c)) for c in psi]
        # absorbed probability over this step, split by core: rate 2 W |psi|^2
        dens = sum(np.abs(c) ** 2 for c in psi)
        for nm in W:
            absorbed[nm] += float((dens * (1 - np.exp(-2 * W[nm] * dt))).sum() * G.dx * G.dy)
        if two_state:
            a, b = psi
            psi = [U11 * a + U12 * b, U12 * a + U22 * b]
        else:
            psi = [UB * psi[0]]
        psi = [damp * c for c in psi]
        psi = [np.fft.ifft2(Tk * np.fft.fft2(c)) for c in psi]
        if probe_x is not None:
            if two_state:
                col = (lower[0][i0 - 1:i0 + 2] * psi[0][i0 - 1:i0 + 2] +
                       lower[1][i0 - 1:i0 + 2] * psi[1][i0 - 1:i0 + 2])
            else:
                col = psi[0][i0 - 1:i0 + 2]
            c0 = Uk.T @ col[1] * G.dy
            dc = Uk.T @ (col[2] - col[0]) * G.dy / (2 * G.dx)
            flux_k += np.imag(np.conj(c0) * dc) / p.M * dt
            if probe_series:
                series[i - 1] = c0[:probe_series]
            flux_tot += float(np.sum(np.imag(np.conj(col[1]) * (col[2] - col[0]) / (2 * G.dx))) * G.dy / p.M * dt)
        if i % every == 0:
            dens = sum(np.abs(c) ** 2 for c in psi)
            out["t"].append(i * dt)
            out["abs_p"].append(absorbed.get("P+", 0.0))
            out["abs_m"].append(absorbed.get("P-", 0.0))
            out["norm"].append(float(dens.sum() * G.dx * G.dy))
            out["P_R"].append(float(dens[G.lab == iR].sum() * G.dx * G.dy))
            out["P_edge"].append(float(dens[edge].sum() * G.dx * G.dy))
            if two_state:
                lo = np.abs(lower[0] * psi[0] + lower[1] * psi[1]) ** 2
                out["P_upper"].append(float((dens - lo).sum() * G.dx * G.dy))
            else:
                out["P_upper"].append(0.0)
    res = {k: np.array(v) for k, v in out.items()}
    ap, am = res["abs_p"][-1], res["abs_m"][-1]
    res.update(phi1=ap / (ap + am), yield_=ap + am, T_vib=T_vib, dt=dt,
               edge_max=float(res["P_edge"].max()))
    if probe_x is not None:
        res.update(probe_x=float(G.x[i0]), e_k=e_k, flux_k=flux_k, flux_tot=flux_tot)
        if probe_series:
            res["probe_c"] = series
    if keep:
        res["psi"] = psi
    return res


# ------------------------------------------------------------ classical
def wigner(G, n, rng, how="wigner"):
    """Initial conditions for the packet. Returns (x, y, px, py) or, for an
    excited transverse state sampled from its (signed) Wigner function,
    (x, y, px, py, w) with quasi-probability weights w.

    how = "wigner": exact Wigner distribution (Gaussian in x; for transverse
          = "ho", W_n = (-1)^n/pi exp(-r^2) L_n(2 r^2) in units xi = y sqrt(M w),
          eta = p_y / sqrt(M w), sampled from a Gaussian proposal with signed weights)
    how = "shell": quasiclassical (QCT) - transverse action fixed at (n + 1/2),
          uniform angle; x part Wigner."""
    p = G.p
    pk = G.packet()
    x = pk["x0"] + rng.normal(0, pk["sx"], n)
    px = pk["px0"] + rng.normal(0, 1 / (2 * pk["sx"]), n)
    if p.transverse != "ho":
        y = pk["y0"] + rng.normal(0, pk["sy"], n)
        py = pk["py0"] + rng.normal(0, 1 / (2 * pk["sy"]), n)
        return x, y, px, py
    Mw = p.M * pk["wy"]
    if how == "shell":
        Ey = (p.n_y + 0.5) * pk["wy"]
        ph = rng.uniform(0, 2 * np.pi, n)
        y = pk["y0"] + np.sqrt(2 * Ey / (p.M * pk["wy"] ** 2)) * np.cos(ph)
        py = -np.sqrt(2 * p.M * Ey) * np.sin(ph)
        return x, y, px, py
    from scipy.special import eval_laguerre
    s2 = 0.5 * (2 * p.n_y + 1)                      # proposal variance per coordinate
    xi = rng.normal(0, np.sqrt(s2), n)
    eta = rng.normal(0, np.sqrt(s2), n)
    r2 = xi**2 + eta**2
    Wn = (-1) ** p.n_y / np.pi * np.exp(-r2) * eval_laguerre(p.n_y, 2 * r2)
    q = np.exp(-r2 / (2 * s2)) / (2 * np.pi * s2)
    w = Wn / q
    y = pk["y0"] + xi / np.sqrt(Mw)
    py = eta * np.sqrt(Mw)
    return x, y, px, py, w


def classical(p, G=None, n=20000, mode="BO", t_final=None, dt=None, seed=5, init=None, W0=3.0,
              how="wigner", probe_x=None):
    """mode = 'BO' (Newton on E_-) or 'fssh'. Returns first-arrival branching.

    W0 > 0: each trajectory carries a weight that decays as exp(-2 int W dt) with
    the same absorbing potential W as the quantum run; the absorbed weight per
    core is the branching observable (identical measurement in both).
    W0 = None: hard absorption on first entry into a core."""
    G = G or Grid(p)
    rng = np.random.default_rng(seed)
    ic = init if init is not None else wigner(G, n, rng, how)
    x, y, px, py = ic[:4]
    qw = ic[4] if len(ic) > 4 else np.ones(len(x))   # quasi-probability weights
    n = len(x)
    if probe_x is not None:
        Ey_cross = np.full(n, np.nan)              # transverse energy at the first crossing of x0
        ycut = np.linspace(G.y[0], G.y[-1], 2001)
        vmin = Em(np.full_like(ycut, probe_x), ycut, p).min()
    M = p.M
    T_vib = 2 * np.pi / np.sqrt(p.kx / M)
    t_final = t_final or 2.0 * T_vib
    dt = dt or T_vib / 2000
    state = np.zeros(n, int)
    a0 = adiabats(x, y, p)
    c1 = (-np.sin(a0["theta"])).astype(complex)
    c2 = np.cos(a0["theta"]).astype(complex)
    first = -np.ones(n, int)
    alive = np.ones(n, bool)
    core_p, core_m = G.core.get("P+"), G.core.get("P-")
    if W0 is not None:
        Wf = cap(G, W0)
        Wp, Wm = Wf["P+"], Wf["P-"]
        wt = np.ones(n)
        absp = np.zeros(n); absm = np.zeros(n)
    nsteps = int(round(t_final / dt))
    nhop = nfrus = 0

    def force(x, y, st):
        a = adiabats(x, y, p)
        gx = np.where(st == 0, a["gm"][0], a["gp"][0])
        gy = np.where(st == 0, a["gm"][1], a["gp"][1])
        return -gx, -gy, a

    fx, fy, a = force(x, y, state)
    E0 = (px**2 + py**2) / (2 * M) + np.where(state == 0, a["Em"], a["Ep"])
    for i in range(1, nsteps + 1):
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break
        xs, ys, pxs, pys, sts = x[idx], y[idx], px[idx], py[idx], state[idx]
        pxs = pxs + 0.5 * dt * fx[idx]; pys = pys + 0.5 * dt * fy[idx]
        xo, yo = xs, ys
        xs = xs + dt * pxs / M; ys = ys + dt * pys / M
        if mode == "fssh":
            th_o = adiabats(xo, yo, p)["theta"]
            lo_o = np.abs(-np.sin(th_o) * c1[idx] + np.cos(th_o) * c2[idx]) ** 2
            Pa_old = np.where(sts == 0, lo_o, 1 - lo_o)
            v1, v2, D, *_ = diabats(0.5 * (xs + xo), 0.5 * (ys + yo), p)
            U11, U22, U12 = propagator_2x2(v1, v2, D, dt)
            n1 = U11 * c1[idx] + U12 * c2[idx]
            n2 = U12 * c1[idx] + U22 * c2[idx]
            c1[idx], c2[idx] = n1, n2
        gfx, gfy, a = force(xs, ys, sts)
        if mode == "fssh":
            lo = np.abs(-np.sin(a["theta"]) * n1 + np.cos(a["theta"]) * n2) ** 2
            Pa_new = np.where(sts == 0, lo, 1 - lo)
            gprob = np.clip((Pa_old - Pa_new) / np.maximum(Pa_old, 1e-12), 0, 1)
            h = rng.random(len(idx)) < gprob
            if h.any():
                k = np.flatnonzero(h)
                dE = np.where(sts[k] == 0, a["Ep"][k] - a["Em"][k], a["Em"][k] - a["Ep"][k])
                nx_, ny_ = a["nac"][0][k], a["nac"][1][k]
                nn = np.sqrt(nx_**2 + ny_**2) + 1e-300
                ux, uy = nx_ / nn, ny_ / nn
                pu = pxs[k] * ux + pys[k] * uy
                disc = pu * pu - 2 * M * dE
                ok = disc >= 0
                al = pu - np.sign(pu) * np.sqrt(np.where(ok, disc, 0))
                kk = k[ok]
                pxs[kk] -= al[ok] * ux[ok]; pys[kk] -= al[ok] * uy[ok]
                sts[kk] = 1 - sts[kk]
                nhop += ok.sum(); nfrus += (~ok).sum()
                gfx, gfy, a = force(xs, ys, sts)
        pxs = pxs + 0.5 * dt * gfx; pys = pys + 0.5 * dt * gfy
        if probe_x is not None:
            cr = (xo < probe_x) & (xs >= probe_x) & np.isnan(Ey_cross[idx])
            if cr.any():
                k = np.flatnonzero(cr)
                Ey_cross[idx[k]] = pys[k] ** 2 / (2 * M) + Em(np.full(len(k), probe_x), ys[k], p) - vmin
        x[idx], y[idx], px[idx], py[idx], state[idx] = xs, ys, pxs, pys, sts
        fx[idx], fy[idx] = gfx, gfy
        if W0 is not None:
            ap = 1 - np.exp(-2 * G.lookup(Wp, xs, ys) * dt)
            am = 1 - np.exp(-2 * G.lookup(Wm, xs, ys) * dt)
            w = wt[idx]
            absp[idx] += w * ap
            absm[idx] += w * (1 - ap) * am
            wt[idx] = w * (1 - ap) * (1 - am)
            alive[idx[wt[idx] < 1e-4]] = False
        elif i % 4 == 0:
            inp = G.lookup(core_p, xs, ys)
            inm = G.lookup(core_m, xs, ys)
            first[idx[inp]] = 1
            first[idx[inm]] = 0
            alive[idx[inp | inm]] = False
    a = adiabats(x, y, p)
    E1 = (px**2 + py**2) / (2 * M) + np.where(state == 0, a["Em"], a["Ep"])
    if W0 is not None:
        tot = (absp + absm) * qw
        phi1 = float((absp * qw).sum() / tot.sum())
        # error from the spread of per-trajectory contributions (ratio estimator)
        r = absp * qw - phi1 * tot
        err = float(np.sqrt((r**2).sum()) / tot.sum())
        yld = float(tot.sum() / qw.sum())
    else:
        got = first >= 0
        phi1 = float(np.mean(first[got] == 1)) if got.any() else np.nan
        err = float(np.sqrt(phi1 * (1 - phi1) / max(got.sum(), 1)))
        yld = float(got.mean())
    out_box = (x < G.x[0]) | (x > G.x[-1]) | (y < G.y[0]) | (y > G.y[-1])
    extra = dict(Ey_cross=Ey_cross, qw=qw) if probe_x is not None else {}
    return dict(**extra, phi1=phi1, phi1_err=err, x_out=float(np.mean(out_box)),
                yield_=yld, nhop=int(nhop), nfrus=int(nfrus),
                dE_max=float(np.max(np.abs(E1 - E0)[~alive | True])), n=n)


def describe(p):
    G = Grid(p)
    lines = [f"{p.name}: grid {len(G.x)}x{len(G.y)}, x_s = {p.x_s:.2f}, x_r = {p.x_r:.2f}, "
             f"d = {p.x_r - p.x_s:+.2f}, Ecut = {G.Ecut:.2f}"]
    for d in G.mins + G.sads:
        lines.append(f"  {d['kind']:6s} {d['name']:5s} ({d['xy'][0]:+.3f},{d['xy'][1]:+.3f}) "
                     f"E = {d['E']:+.3f}  eig = {d['eig'][0]:+.3f},{d['eig'][1]:+.3f}")
    pk = G.packet()
    lines.append(f"  packet sx = {pk['sx']:.3f} sy = {pk['sy']:.3f} px0 = {pk['px0']:.2f}")
    gap = (G.Ep - G.Em)
    lines.append(f"  min gap E+ - E- on grid = {gap.min():.3f}")
    return G, "\n".join(lines)

"""Start ensembles and basin cores.

flux_sample    microcanonical flux ensemble on the TS1 dividing line (toward products)
rstart_sample  microcanonical ensemble inside the reactant basin (uphill-transfer test)
core_levels    energy levels defining each basin's core (F_CORE of the way from the minimum to
               its lowest exit saddle; P+/P- use the product-product saddle)
"""
import numpy as np

from ..surfaces.branching import V

F_CORE = 0.5


def flux_sample(S, E, N, rng):
    """Microcanonical flux ensemble on the TS1 line at energy E (toward products)."""
    s = np.linspace(-3, 3, 6001)
    pts = S.ts1["xy"][:, None] + np.outer(S.t, s)
    K = np.clip(E - V(pts[0], pts[1], S.p), 0, None)
    w = 2 * np.sqrt(2 * K)
    cdf = np.cumsum(w); cdf /= cdf[-1]
    sv = np.interp(rng.uniform(size=N), cdf, s)
    x = S.ts1["xy"][0] + sv * S.t[0]
    y = S.ts1["xy"][1] + sv * S.t[1]
    th = np.arcsin(2 * rng.uniform(size=N) - 1)          # density cos(theta)/2
    v = np.sqrt(2 * np.clip(E - V(x, y, S.p), 0, None))
    vn, vt = v * np.cos(th), v * np.sin(th)
    return x, y, vn * S.n[0] + vt * S.t[0], vn * S.n[1] + vt * S.t[1]


def core_levels(S):
    ts12 = [s for s in S.sads if set(s["connects"]) == {"P+", "P-"}]
    lev = {}
    for i, nm in enumerate(S.names):
        out = [s["E"] for s in S.sads if nm in s["connects"]]
        top = ts12[0]["E"] if (ts12 and nm in ("P+", "P-")) else min(out)
        lev[i] = S.mins[i]["E"] + F_CORE * (top - S.mins[i]["E"])
    return np.array([lev[i] for i in range(len(S.names))])


def areas(S, E):
    nx, ny = S.lab.shape
    X = S.gx0 + S.gdx * np.arange(nx)
    Y = S.gy0 + S.gdy * np.arange(ny)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    below = V(XX, YY, S.p) < E
    return np.array([np.sum(below & (S.lab == i)) * S.gdx * S.gdy for i in range(len(S.names))])


def rstart_sample(S, E_xy, n, rng):
    """Microcanonical start inside the R basin at 2D energy E_xy: positions uniform on
    the area where V < E_xy (the 2D microcanonical measure), speed sqrt(2(E_xy - V)),
    random direction."""
    Vf = V
    nx, ny = S.lab.shape
    X = S.gx0 + S.gdx * np.arange(nx)
    Y = S.gy0 + S.gdy * np.arange(ny)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    ok = np.flatnonzero(((S.lab == S.iR) & (Vf(XX, YY, S.p) < E_xy)).ravel())
    xs, ys = [], []
    while sum(len(a) for a in xs) < n:
        k = rng.choice(ok, n)
        x = XX.ravel()[k] + (rng.random(n) - 0.5) * S.gdx
        y = YY.ravel()[k] + (rng.random(n) - 0.5) * S.gdy
        keep = (Vf(x, y, S.p) < E_xy) & (S.basin(x, y) == S.iR)
        xs.append(x[keep]); ys.append(y[keep])
    x = np.concatenate(xs)[:n]; y = np.concatenate(ys)[:n]
    sp = np.sqrt(2 * (E_xy - Vf(x, y, S.p)))
    ang = rng.uniform(0, 2 * np.pi, n)
    return x, y, sp * np.cos(ang), sp * np.sin(ang)

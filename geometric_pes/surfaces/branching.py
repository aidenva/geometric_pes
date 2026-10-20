"""Multi-basin model surface with a valley-ridge inflection (VRI).

A single adiabatic surface V(x, y) in reduced units (mass 1, energies in
units of the reference k_B T = 1). x is the reaction coordinate (reactant
R at x < 0, first transition state TS1 near x = 0, products at x > 0); y is
the branching coordinate.

    u      = y - c(x)                         valley centre line c(x)
    V(x,y) = U(x) + 1/2 kappa(x) u^2 + 1/4 b(x,u) u^4 + 1/6 e(x) u^6 + eps s(x) u

    U(x)     = Vb ((x/L)^2 - 1)^2 - Delta s_D(x)      R well, TS1, exothermic drop
    kappa(x) = kR + (kP - kR) s(x)                    y-curvature: kR > 0 before,
                                                      kP after the branching region
    s(x)     = (1 + tanh((x - xv)/wv)) / 2            switch centred at xv
    c(x)     = A_c (1 + tanh((x - xc)/wc)) / 2        bend of the valley (0 = straight)
    b(x,u)   = bR + (b_+/- - bR) s(x)                 b_plus for u > 0, b_minus for u < 0:
                                                      different product shapes
    e(x)     = eR + (e - eR) s(x)

kP < 0 with b > 0 gives a VRI: the valley floor at u = 0 turns into a ridge
and splits into two product wells P+ (u > 0) and P- (u < 0) separated by a
saddle TS12. kP > 0 with b < 0 and e > 0 keeps a shallow intermediate I at
u = 0 with two exit saddles (a "Curtin-Hammett" intermediate) instead.

This is a direct adiabatic surface rather than the lowest eigenvalue of an
N-state diabatic Hamiltonian: analytic gradients make the Langevin engine
~100x cheaper and every topological feature (VRI location, bend, product
shapes) is set by one parameter. Stationary points are found with the same
|grad V|^2 + Hessian approach as funnel.stationary
and reuse its merge_points helper.

Variants (surfaces A, B, C, Aup1, Aup2) are parameter sets in config/experiments.toml; build them
with geometric_pes.config.surface(name).
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..two_state.stationary import merge_points


@dataclass(frozen=True)
class Surface:
    name: str = "A_vri_straight"
    # reaction coordinate
    Vb: float = 6.0       # double-well height of U (R barrier before the drop)
    L: float = 1.5        # x length scale of U
    Delta: float = 14.0   # exothermic drop
    wD: float = 0.5       # width of the drop switch (centred at 0)
    # branching coordinate
    kR: float = 6.0       # y curvature before the branching region
    kP: float = -10.0     # y curvature after it (< 0: VRI)
    xv: float = 0.8       # centre of the curvature switch
    wv: float = 0.35
    b_plus: float = 4.0
    b_minus: float = 4.0
    e: float = 0.2        # sextic confinement (product region)
    bR: float = 4.0       # quartic and sextic coefficients before the branching region
    eR: float = 0.2
    eps: float = 0.0      # tilt along u in the product region
    # valley bend (dynamic-matching geometry)
    A_c: float = 0.0
    xc: float = 0.3
    wc: float = 0.35
    # window for searches and plots
    x_min: float = -2.6
    x_max: float = 2.8
    y_min: float = -3.0
    y_max: float = 3.0


def _sw(x, x0, w):
    t = np.tanh((x - x0) / w)
    return 0.5 * (1 + t), 0.5 * (1 - t * t) / w


def V(x, y, p):
    sD, _ = _sw(x, 0.0, p.wD)
    s, _ = _sw(x, p.xv, p.wv)
    sc, _ = _sw(x, p.xc, p.wc)
    u = y - p.A_c * sc
    kap = p.kR + (p.kP - p.kR) * s
    bP = np.where(u > 0, p.b_plus, p.b_minus)
    b = p.bR + (bP - p.bR) * s
    e = p.eR + (p.e - p.eR) * s
    U = p.Vb * ((x / p.L) ** 2 - 1) ** 2 - p.Delta * sD
    return U + 0.5 * kap * u**2 + 0.25 * b * u**4 + e / 6 * u**6 + p.eps * s * u


def grad(x, y, p):
    sD, dsD = _sw(x, 0.0, p.wD)
    s, ds = _sw(x, p.xv, p.wv)
    sc, dsc = _sw(x, p.xc, p.wc)
    u = y - p.A_c * sc
    kap = p.kR + (p.kP - p.kR) * s
    bP = np.where(u > 0, p.b_plus, p.b_minus)
    b = p.bR + (bP - p.bR) * s
    e = p.eR + (p.e - p.eR) * s
    Fy = kap * u + b * u**3 + e * u**5 + p.eps * s
    dU = 4 * p.Vb * ((x / p.L) ** 2 - 1) * x / p.L**2 - p.Delta * dsD
    Fx = (dU + ds * (0.5 * (p.kP - p.kR) * u**2 + 0.25 * (bP - p.bR) * u**4
                     + (p.e - p.eR) / 6 * u**6 + p.eps * u)
          - p.A_c * dsc * Fy)
    return Fx, Fy


def force(q, p):
    """Force -grad V on coordinate-major positions q = [x, y] (shape (2, n))."""
    gx, gy = grad(q[0], q[1], p)
    return -np.array([gx, gy])


def hessian(xy, p, h=1e-5):
    H = np.zeros((2, 2))
    for i, d in enumerate(np.eye(2) * h):
        gp = np.array(grad(*(xy + d), p))
        gm = np.array(grad(*(xy - d), p))
        H[:, i] = (gp - gm) / (2 * h)
    return 0.5 * (H + H.T)


def _solve(z0, p):
    def f(z):
        g = np.array(grad(z[0], z[1], p))
        return float(g @ g)

    def df(z):
        g = np.array(grad(z[0], z[1], p))
        return 2 * hessian(z, p) @ g

    bounds = [(p.x_min, p.x_max), (p.y_min, p.y_max)]
    r = minimize(f, z0, jac=df, method="L-BFGS-B", bounds=bounds,
                 options=dict(ftol=1e-20, gtol=1e-12, maxiter=3000))
    return r.x, r.fun


def classify(xy, p):
    w, vecs = np.linalg.eigh(hessian(xy, p))
    neg, pos = int(np.sum(w < -1e-6)), int(np.sum(w > 1e-6))
    kind = {(0, 2): "min", (1, 1): "saddle", (2, 0): "max"}.get((neg, pos), "degenerate")
    return kind, w, vecs


def find_stationary(p, n_starts=400, seed=1, tol=1e-3):
    """Multistart minimisation of |grad V|^2; keeps true roots away from the box edge."""
    rng = np.random.default_rng(seed)
    seeds = np.column_stack([rng.uniform(p.x_min, p.x_max, n_starts),
                             rng.uniform(p.y_min, p.y_max, n_starts)])
    pts = []
    for s0 in seeds:
        z, f = _solve(s0, p)
        on_edge = (min(z[0] - p.x_min, p.x_max - z[0], z[1] - p.y_min, p.y_max - z[1]) < 1e-3)
        if f < 1e-14 and not on_edge:
            pts.append(z)
    pts = merge_points(pts, tol)
    out = []
    for z in pts:
        kind, w, vecs = classify(z, p)
        out.append(dict(xy=np.array(z), E=float(V(z[0], z[1], p)), kind=kind,
                        eig=w, vecs=vecs))
    order = {"min": 0, "saddle": 1, "max": 2, "degenerate": 3}
    out.sort(key=lambda d: (order[d["kind"]], d["xy"][0], d["xy"][1]))
    return out


def descend(z0, p, n=20000, lr=2e-3):
    z = np.array(z0, float)
    for _ in range(n):
        g = np.array(grad(z[0], z[1], p))
        z -= lr * g / max(1.0, np.linalg.norm(g) * lr / 0.01)   # cap the step at 0.01
        if np.linalg.norm(g) < 1e-7:
            break
    return z


def label_species(pts, p):
    """Name minima (R, I, P+, P-) and connect each saddle to the two minima its
    steepest-descent path (+/- unstable eigenvector) reaches."""
    mins = [d for d in pts if d["kind"] == "min"]
    sads = [d for d in pts if d["kind"] == "saddle"]
    for d in mins:
        x, y = d["xy"]
        if x < -0.3:
            d["name"] = "R"
        else:
            sc, _ = _sw(x, p.xc, p.wc)
            u = y - p.A_c * sc
            d["name"] = "I" if abs(u) < 0.3 else ("P+" if u > 0 else "P-")
    mxy = np.array([d["xy"] for d in mins])
    for d in sads:
        v = d["vecs"][:, 0]
        ends = []
        for sgn in (+1, -1):
            z = descend(d["xy"] + sgn * 0.02 * v, p)
            ends.append(mins[int(np.argmin(np.linalg.norm(mxy - z, axis=1)))]["name"])
        d["connects"] = tuple(ends)
        d["name"] = "TS(" + "-".join(sorted(ends)) + ")"
    return mins, sads


def basin_grid(p, mins, nx=221, ny=241):
    """Label every grid point by the minimum its steepest-descent path reaches."""
    X = np.linspace(p.x_min, p.x_max, nx)
    Y = np.linspace(p.y_min, p.y_max, ny)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    zx, zy = XX.copy(), YY.copy()
    for _ in range(6000):
        gx, gy = grad(zx, zy, p)
        gn = np.sqrt(gx**2 + gy**2) + 1e-12
        step = np.minimum(2e-3, 0.01 / gn)
        zx -= step * gx
        zy -= step * gy
    mxy = np.array([d["xy"] for d in mins])
    dist = (zx[..., None] - mxy[:, 0]) ** 2 + (zy[..., None] - mxy[:, 1]) ** 2
    return X, Y, np.argmin(dist, axis=-1)


def vri_location(p, x_grid=None):
    """x where the y-curvature along the valley floor changes sign (VRI) on the
    steepest-descent path from TS1 (returns None if it never does)."""
    pts = find_stationary(p)
    mins, sads = label_species(pts, p)
    ts1 = [d for d in sads if "R" in d["connects"]][0]
    v = ts1["vecs"][:, 0]
    v = v if v[0] > 0 else -v
    z = ts1["xy"] + 0.02 * v
    path = [z.copy()]
    for _ in range(40000):
        g = np.array(grad(z[0], z[1], p))
        gn = np.linalg.norm(g)
        if gn < 1e-6:
            break
        z -= min(2e-3, 0.005 / gn) * g
        path.append(z.copy())
    path = np.array(path)
    # curvature perpendicular to the path
    curv = []
    for i in range(1, len(path) - 1, 10):
        t = path[i + 1] - path[i - 1]
        t /= np.linalg.norm(t) + 1e-15
        nrm = np.array([-t[1], t[0]])
        curv.append((path[i], nrm @ hessian(path[i], p) @ nrm))
    for (z0, c0), (z1, c1) in zip(curv[:-1], curv[1:]):
        if c0 > 0 >= c1:
            return path, z0
    return path, None


def ridge_onset(p):
    """VRI estimate on the valley centre line u = 0: kappa(x) = 0 (None if kP > 0)."""
    if p.kP >= 0:
        return None
    s = p.kR / (p.kR - p.kP)
    x = p.xv + p.wv * np.arctanh(2 * s - 1)
    sc, _ = _sw(x, p.xc, p.wc)
    return np.array([x, p.A_c * sc])


def check_gradient(p, n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(p.x_min, p.x_max, n)
    y = rng.uniform(p.y_min, p.y_max, n)
    h = 1e-6
    gx, gy = grad(x, y, p)
    fx = (V(x + h, y, p) - V(x - h, y, p)) / (2 * h)
    fy = (V(x, y + h, p) - V(x, y - h, p)) / (2 * h)
    scale = 1 + np.abs(gx) + np.abs(gy)
    return float(np.max(np.abs(gx - fx) / scale)), float(np.max(np.abs(gy - fy) / scale))

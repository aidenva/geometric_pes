"""Stationary points of E_-: multistart |grad E_-|^2 minimization, Hessian
classification, reactant/product identification and steepest-descent basins."""
import numpy as np

from .model import P, E_pm, grad_Eminus


def hessian_numeric(xy, p=P, eps=1e-5):
    x, y = xy
    f0   = E_pm(x, y, p)[0]
    fx1  = E_pm(x+eps, y, p)[0]; fx2 = E_pm(x-eps, y, p)[0]
    fy1  = E_pm(x, y+eps, p)[0]; fy2 = E_pm(x, y-eps, p)[0]
    fxy1 = E_pm(x+eps, y+eps, p)[0]; fxy2 = E_pm(x+eps, y-eps, p)[0]
    fxy3 = E_pm(x-eps, y+eps, p)[0]; fxy4 = E_pm(x-eps, y-eps, p)[0]
    d2xx = (fx1 - 2*f0 + fx2) / (eps*eps)
    d2yy = (fy1 - 2*f0 + fy2) / (eps*eps)
    d2xy = (fxy1 - fxy2 - fxy3 + fxy4) / (4*eps*eps)
    return np.array([[d2xx, d2xy],[d2xy, d2yy]], float)

def _num_jac(fun, xy, p=P, eps=1e-6):
    f0 = fun(xy, p)
    J = np.zeros((2,2))
    for i, d in enumerate([np.array([eps,0.0]), np.array([0.0,eps])]):
        f1 = fun(xy + d, p)
        J[:, i] = (f1 - f0)/eps
    return J

def solve_stationary(xy0, p=P):
    try:
        from scipy.optimize import minimize
        def f(z):
            g = grad_Eminus(z, p)
            return float(g @ g)
        def df(z):
            g = grad_Eminus(z, p)
            J = _num_jac(grad_Eminus, z, p)
            return 2.0 * (J.T @ g)

        bounds = [(p.x_min, p.x_max), (p.y_min, p.y_max)]
        # L-BFGS-B respects simple bounds
        res = minimize(
            f, xy0, jac=df, method="L-BFGS-B",
            bounds=bounds,
            options=dict(ftol=1e-16, maxiter=3000)
        )
        # Clip just in case of tiny numeric drift
        z = np.array(res.x, dtype=float)
        z[0] = np.clip(z[0], p.x_min, p.x_max)
        z[1] = np.clip(z[1], p.y_min, p.y_max)
        return z
    except Exception:
        # Fallback: gradient descent with projection into the box every step
        xy = xy0.copy()
        lr = 0.02
        for _ in range(2000):
            g = grad_Eminus(xy, p)
            J = _num_jac(grad_Eminus, xy, p)
            xy -= 0.5 * (J.T @ g) * lr
            xy[0] = np.clip(xy[0], p.x_min, p.x_max)
            xy[1] = np.clip(xy[1], p.y_min, p.y_max)
        return xy

def merge_points(points, tol):
    out = []
    for q in points:
        placed = False
        for i, m in enumerate(out):
            if np.linalg.norm(q - m) < tol:
                out[i] = 0.5*(out[i] + q); placed = True; break
        if not placed:
            out.append(q)
    return out

def classify_point(xy, p=P):
    H = hessian_numeric(xy, p)   # Hessian of E_-(x,y)
    w = np.linalg.eigvalsh(H)
    neg = np.sum(w < -1e-6); pos = np.sum(w > 1e-6)
    if neg == 0 and pos >= 1: t = "min"
    elif neg == 1 and pos >= 1: t = "saddle"
    elif neg >= 2: t = "max"
    else: t = "degenerate"
    return t, w

def find_points(p=P):
    rng = np.random.default_rng(p.seed)
    seeds = np.column_stack([
        rng.uniform(p.x_min, p.x_max, p.n_starts),
        rng.uniform(p.y_min, p.y_max, p.n_starts)
    ])
    pts = [solve_stationary(s, p) for s in seeds]
    pts = merge_points(pts, p.tol_merge)
    pts = [solve_stationary(np.array(pt), p) for pt in pts]
    pts = merge_points(pts, p.tol_merge)
    return pts

def classify_all(pts, p=P):
    """Split stationary points into minima and saddles, each sorted by E_-."""
    mins, saddles = [], []
    for z in pts:
        t, _ = classify_point(z, p)
        if t == "min": mins.append(np.asarray(z))
        elif t == "saddle": saddles.append(np.asarray(z))
    key = lambda q: E_pm(q[0], q[1], p)[0]
    return sorted(mins, key=key), sorted(saddles, key=key)

def reactant_product(mins, p=P):
    """Reactant = minimum closest to diabatic well 1, product = closest to well 2."""
    i_r = int(np.argmin([np.hypot(z[0]-p.x1, z[1]-p.y1) for z in mins]))
    i_p = int(np.argmin([np.hypot(z[0]-p.x2, z[1]-p.y2) for z in mins]))
    return i_r, i_p

def basin_labels(XX, YY, mins, p=P, lr=0.02, n_iter=1500):
    """Label each grid point by the minimum its steepest-descent path on E_- reaches."""
    xy = np.array([XX.ravel(), YY.ravel()], dtype=float)
    for _ in range(n_iter):
        g = grad_Eminus(xy, p)
        xy -= lr * g / (1.0 + np.sqrt(g[0]**2 + g[1]**2))   # bounded step size
    M = np.array(mins)
    d2 = (xy[0][:, None] - M[:, 0])**2 + (xy[1][:, None] - M[:, 1])**2
    return np.argmin(d2, axis=1).reshape(XX.shape)

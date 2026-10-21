"""Two-state, two-coordinate diabatic model and its adiabatic surfaces.

Diabats V1, V2 (anisotropic harmonic wells + quartic stabilizer + bias),
coupling Delta (Gaussian or linear), adiabatic surfaces E_+/- from the 2x2
diagonalization, analytic gradients, eigenvectors and the exact 2x2
propagator. Units: Hartree atomic units, hbar = 1.
"""
from dataclasses import dataclass

import numpy as np

HBAR = 1.0


@dataclass
class Params:
    # Diabatic minima (centers of the two electronic states)
    x1: float = -1.1
    y1: float =  0.0
    x2: float =  0.0
    y2: float =  0.6

    # Harmonic curvatures (per-state, per-axis) for diabatic wells
    kx1: float = 1.0
    ky1: float = 5.0
    kx2: float = 5.0
    ky2: float = 1.0

    # Weak quartic stabilizer (keeps surfaces bounded far away)
    stab: float = 0.7   # multiplies (x^2 + y^2)^2

    # Coupling model: "gaussian" (avoided crossing) or "linear" (CI-like when Delta0=0)
    coupling: str = "linear"       # "gaussian" or "linear"
    Delta0: float = 0.05           # base coupling strength
    # Gaussian coupling widths (centered at the midpoint)
    wg_x: float = 0.3
    wg_y: float = 0.7
    # Linear coupling coefficients: Delta = Delta0 + cx*(x-xc) + cy*(y-yc)
    cx: float = 1.0
    cy: float = 1.0

    # Domain & grids (feel free to tighten for speed)
    # x_min: float = -2.2; x_max: float =  1.6
    # y_min: float = -1.6; y_max: float =  2.2

    # Visualization grid
    x_min: float = -1.6; x_max: float =  0.6
    y_min: float = -0.6; y_max: float =  1.6

    nx: int = 251       # 201–301 is plenty for interactive
    ny: int = 251

    # Stationary-point search
    n_starts: int = 200
    seed: int = 5
    tol_merge: float = 1e-3

    # Constant offsets of the diabatic wells (V_bias in the report)
    bias1: float = 0.0
    bias2: float = 0.0

    # Nuclear masses (Hartree a.u., hbar = 1). 1836 = proton mass; with m = 1
    # the zero-point energy (~1.6) exceeded every barrier on E_-.
    mass_x: float = 1836.0
    mass_y: float = 1836.0

    # Propagation box: larger than the plot window so that E_- on the edges
    # (~2.8 here) sits far above the packet energy. FFT boundaries are periodic.
    prop_x_min: float = -2.2; prop_x_max: float = 1.2
    prop_y_min: float = -1.2; prop_y_max: float = 1.8
    prop_nx: int = 256
    prop_ny: int = 256

    # Initial packet, centered on the reactant minimum of E_-
    init_state: str = "adiabatic"  # "adiabatic" (lower eigenvector) or "diabatic1"
    E_kin0: float = 0.3            # initial kinetic energy, aimed at the lowest saddle
    sigma_x: float = 0.0           # 0 -> ground-state width of the reactant well
    sigma_y: float = 0.0

    # Time stepping
    dt: float = 0.5
    t_final: float = 4000.0
    sample_every: int = 20


P = Params()


def V1(x, y, p=P):
    return 0.5*p.kx1*(x - p.x1)**2 + 0.5*p.ky1*(y - p.y1)**2 + p.stab*(x*x + y*y)**2 + p.bias1

def V2(x, y, p=P):
    return 0.5*p.kx2*(x - p.x2)**2 + 0.5*p.ky2*(y - p.y2)**2 + p.stab*(x*x + y*y)**2 + p.bias2

def coupling_Delta(x, y, p=P):
    xm, ym = 0.5*(p.x1 + p.x2), 0.5*(p.y1 + p.y2)
    if p.coupling == "gaussian":
        gx = (x - xm)/p.wg_x; gy = (y - ym)/p.wg_y
        return p.Delta0 * np.exp(-(gx*gx + gy*gy))
    else:  # "linear" (set Delta0=0 for a CI-like model)
        return p.Delta0 + p.cx*(x - xm) + p.cy*(y - ym)

def E_pm(x, y, p=P):
    v1 = V1(x, y, p)
    v2 = V2(x, y, p)
    d  = 0.5*(v1 - v2)
    D  = coupling_Delta(x, y, p)
    rad = np.sqrt(d*d + D*D)
    Em = 0.5*(v1 + v2) - rad   # lower adiabatic surface
    Ep = 0.5*(v1 + v2) + rad   # upper adiabatic surface
    return Em, Ep, v1, v2, D, d, rad

def grad_Eminus(xy, p=P):
    x, y = xy
    # ∂V_i
    dv1dx = p.kx1*(x - p.x1) + 4*p.stab*(x*x + y*y)*x
    dv1dy = p.ky1*(y - p.y1) + 4*p.stab*(x*x + y*y)*y
    dv2dx = p.kx2*(x - p.x2) + 4*p.stab*(x*x + y*y)*x
    dv2dy = p.ky2*(y - p.y2) + 4*p.stab*(x*x + y*y)*y

    xm, ym = 0.5*(p.x1 + p.x2), 0.5*(p.y1 + p.y2)
    if p.coupling == "gaussian":
        gx = (x - xm)/p.wg_x; gy = (y - ym)/p.wg_y
        g  = np.exp(-(gx*gx + gy*gy))
        D  = p.Delta0 * g
        dDdx = D * (-2*gx / p.wg_x)
        dDdy = D * (-2*gy / p.wg_y)
    else:
        D  = p.Delta0 + p.cx*(x - xm) + p.cy*(y - ym)
        dDdx, dDdy = p.cx, p.cy

    v1 = V1(x, y, p); v2 = V2(x, y, p)
    d  = 0.5*(v1 - v2)
    rad = np.sqrt(d*d + D*D)

    ddx = 0.5*(dv1dx - dv2dx)
    ddy = 0.5*(dv1dy - dv2dy)
    drdx = (d*ddx + D*dDdx) / (rad + 1e-14)
    drdy = (d*ddy + D*dDdy) / (rad + 1e-14)

    dEmdx = 0.5*(dv1dx + dv2dx) - drdx
    dEmdy = 0.5*(dv1dy + dv2dy) - drdy
    return np.array([dEmdx, dEmdy])


def surfaces_and_grads(x, y, p=P):
    """E_-, E_+, their analytic gradients, mixing angle theta, NAC = grad(theta),
    and the diabatic V1, V2, Delta."""
    Em, Ep, v1, v2, D, d, rad = E_pm(x, y, p)
    r2 = x * x + y * y
    dv1dx = p.kx1 * (x - p.x1) + 4 * p.stab * r2 * x
    dv1dy = p.ky1 * (y - p.y1) + 4 * p.stab * r2 * y
    dv2dx = p.kx2 * (x - p.x2) + 4 * p.stab * r2 * x
    dv2dy = p.ky2 * (y - p.y2) + 4 * p.stab * r2 * y
    xm, ym = 0.5 * (p.x1 + p.x2), 0.5 * (p.y1 + p.y2)
    if p.coupling == "gaussian":
        gx = (x - xm) / p.wg_x; gy = (y - ym) / p.wg_y
        dDdx = D * (-2 * gx / p.wg_x); dDdy = D * (-2 * gy / p.wg_y)
    else:
        dDdx = np.full_like(x, p.cx); dDdy = np.full_like(x, p.cy)
    ddx = 0.5 * (dv1dx - dv2dx); ddy = 0.5 * (dv1dy - dv2dy)
    rr = rad + 1e-14
    drx = (d * ddx + D * dDdx) / rr; dry = (d * ddy + D * dDdy) / rr
    mx = 0.5 * (dv1dx + dv2dx); my = 0.5 * (dv1dy + dv2dy)
    theta = 0.5 * np.arctan2(D, d)
    nacx = (d * dDdx - D * ddx) / (2 * rr * rr)
    nacy = (d * dDdy - D * ddy) / (2 * rr * rr)
    return dict(Em=Em, Ep=Ep, gm=(mx - drx, my - dry), gp=(mx + drx, my + dry),
                theta=theta, nac=(nacx, nacy), v1=v1, v2=v2, D=D)


def lower_eigvec(v1, v2, D, Em):
    """Lower adiabatic eigenvector, sign-smooth wherever state 1 is the lower diabat."""
    u1, u2 = v2 - Em, -D
    n = np.sqrt(u1*u1 + u2*u2) + 1e-300
    return u1/n, u2/n

def mixing_angle(v1, v2, D):
    """theta with upper = (cos, sin), lower = (-sin, cos); fine for populations."""
    return 0.5*np.arctan2(2*D, v1 - v2)


def propagator_2x2(v1, v2, D, dt, hbar=HBAR):
    """exp(-i H dt / hbar) for H = [[v1, D], [D, v2]], elementwise.

    Writing H = c I + [[a, b], [b, -a]] with Omega = sqrt(a^2 + b^2):
    U = e^{-i c dt} [cos(Omega dt) I - i sin(Omega dt)/Omega (H - c I)].
    Returns (U11, U22, U12); U21 = U12.
    """
    c = 0.5*(v1 + v2); a = 0.5*(v1 - v2); b = D
    Omega = np.sqrt(a*a + b*b) + 1e-16
    cd = np.cos(Omega*dt/hbar); sd = np.sin(Omega*dt/hbar)/Omega
    phase = np.exp(-1j*c*dt/hbar)
    return phase*(cd - 1j*sd*a), phase*(cd + 1j*sd*a), phase*(-1j*sd*b)

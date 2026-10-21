"""The decision at the electron-transfer junction: coupling cases, junction parameters, transverse
and thermal initial states of the branching mode, coherent superpositions, and the thermal sum.

params(d, cpl, M)       junction with the ridge a distance d past the seam; cpl in COUPLINGS
pset(...)               the same with the branching mode in transverse state n
grid(cpl, d, M)         the propagation grid with surfaces and basins (cached on disk)
thermal_init            classical initial conditions for a thermal branching mode (Wigner or Boltzmann)
packet_xy               a wavepacket that is a superposition of transverse states
thermal                 phi_1 of a thermal mixture from pure-state absorbed probabilities
"""
import os
import pickle
from dataclasses import replace

import numpy as np

from .surfaces.junction import J, Grid, ho_state
from .config import cache_dir

COUPLINGS = {"AC": dict(D0=0.5, c=0.0), "ACs": dict(D0=1.5, c=0.0),
             "CI": dict(D0=0.0, c=1.0, y_c=0.3)}

_G = {}


def grid(cpl, d, M):
    """Grid per (coupling, d, M), cached on disk (basin maps are slow; worker processes do not
    share the parent's memory)."""
    k = (cpl, d, M)
    if k not in _G:
        f = os.path.join(cache_dir(), "junction", f"{cpl}_d{d:g}_M{M:g}.pkl")
        if os.path.exists(f):
            with open(f, "rb") as fh:
                _G[k] = pickle.load(fh)
        else:
            _G[k] = Grid(pset(cpl, d, M))
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f + ".tmp", "wb") as fh:
                pickle.dump(_G[k], fh)
            os.replace(f + ".tmp", f)
    return _G[k]


def params(d, cpl, M):
    base = J()
    return replace(base, name=f"{cpl}_d{d:g}_M{M:g}", x_r=base.x_s + d, M=M, **COUPLINGS[cpl])


def pset(cpl, d, M, n=0, comp=False, **kw):
    return replace(params(d, cpl, M), transverse="ho", n_y=n, compensate=comp,
                   name=f"{cpl}_d{d:g}_M{M:g}_n{n}_{'c' if comp else 'u'}", **kw)


def regrid(G, p):
    """Grid object for p reusing the (expensive) surfaces/basins of G (same potential)."""
    H = object.__new__(type(G))
    H.__dict__.update(G.__dict__)
    H.p = p
    return H


def thermal_init(G, n, xy, how, rng, comp_n=None):
    pk = G.packet()
    p = G.p
    M, wy = p.M, pk["wy"]
    if how == "micro":
        nmax = p.E_kin0 / wy - 0.5
        nc = rng.uniform(-0.5, nmax, n)
        ekin = p.E_kin0 - nc * wy
        px0 = np.sqrt(2 * M * ekin)
    else:
        px0 = pk["px0"]
    x = pk["x0"] + rng.normal(0, pk["sx"], n)
    px = px0 + rng.normal(0, 1 / (2 * pk["sx"]), n)
    if how == "wigner":
        c = 1 / np.tanh(xy / 2)
        vy, vp = c / (2 * M * wy), M * wy * c / 2
    elif how == "boltz":
        kT = wy / xy
        vy, vp = kT / (M * wy**2), M * kT
    if how == "micro":
        Ey = (nc + 0.5) * wy
        ph = rng.uniform(0, 2 * np.pi, n)
        y = pk["y0"] + np.sqrt(2 * Ey / (M * wy**2)) * np.cos(ph)
        py = -np.sqrt(2 * M * Ey) * np.sin(ph)
    else:
        y = pk["y0"] + rng.normal(0, np.sqrt(vy), n)
        py = rng.normal(0, np.sqrt(vp), n)
    return x, y, px, py


def packet_xy(G, comps):
    """sum_k a_k chi_x^{(k)}(x) phi_{n_k}(y) for comps = [(a_k, n_k, ekin_k)]."""
    pk = G.packet()
    p = G.p
    psi = np.zeros(G.X.shape, complex)
    for a, n, ekin in comps:
        px = np.sqrt(2 * p.M * ekin)
        gx = np.exp(-((G.x - pk["x0"]) ** 2) / (4 * pk["sx"] ** 2) + 1j * px * (G.x - pk["x0"]))
        gx /= np.sqrt((np.abs(gx) ** 2).sum() * G.dx)
        psi += a * np.outer(gx, ho_state(G, n, pk["y0"], pk["sy"]))
    return psi


def thermal(n, Ap, Am, xy):
    """phi_1(T) from pure-n absorbed probabilities; parity sub-ensembles; tail weight."""
    assert np.all(n == np.arange(len(n))), "n must be contiguous from 0"
    pn = (1 - np.exp(-xy)) * np.exp(-xy * n)
    tail = 1 - pn.sum()
    phi = (pn * Ap).sum() / (pn * (Ap + Am)).sum()
    phi_naive = (pn * Ap / (Ap + Am)).sum() / pn.sum()
    ev, od = n % 2 == 0, n % 2 == 1
    pe = (pn[ev] * Ap[ev]).sum() / (pn[ev] * (Ap + Am)[ev]).sum()
    po = (pn[od] * Ap[od]).sum() / (pn[od] * (Ap + Am)[od]).sum()
    # effect of the odd/even alternation on the thermal answer: replace phi_1(n), n >= 1, by
    # the 3-point average of its neighbors (n = 0 kept: its partner is n = 1) and recompute
    ph = Ap / (Ap + Am)
    sm = ph.copy()
    sm[1:-1] = 0.25 * ph[:-2] + 0.5 * ph[1:-1] + 0.25 * ph[2:]
    Y = Ap + Am
    dpar = phi - (pn * Y * sm).sum() / (pn * Y).sum()
    return dict(phi=phi, phi_naive=phi_naive, phi_even=pe, phi_odd=po, parity=abs(po - pe), dpar=dpar,
                tail=tail, nbar=1 / np.expm1(xy), yld=(pn * (Ap + Am)).sum() / pn.sum())

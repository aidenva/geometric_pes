"""Split-operator quantum dynamics of the two-component diabatic wavepacket.

propagate() is the engine: Strang splitting T/2, V, T/2 with the kinetic step
via FFT and the potential step via the exact 2x2 exponential, starting from a
Gaussian on the reactant minimum of E_- (lower adiabatic state by default).
It returns diabatic, adiabatic and basin populations, the autocorrelation
C(t) = <psi0|psi(t)>, and energy/norm/edge checks.
"""
import dataclasses

import numpy as np

from .model import P, HBAR, lower_eigvec, mixing_angle, propagator_2x2
from .stationary import hessian_numeric
from .landscape import Landscape


def initial_packet(L, p=None):
    """Centre, widths and momentum of the initial Gaussian.

    Widths default to the ground-state widths of the reactant well (from the
    E_- Hessian); momentum points from the reactant minimum to the lowest
    saddle (or the product minimum) with kinetic energy p.E_kin0.
    """
    p = p or L.p
    H = hessian_numeric(L.r_min, p)
    sx = p.sigma_x or (HBAR**2 / (p.mass_x * H[0, 0]))**0.25
    sy = p.sigma_y or (HBAR**2 / (p.mass_y * H[1, 1]))**0.25
    target = L.saddles[0] if L.saddles else L.p_min
    u = (target - L.r_min) / np.linalg.norm(target - L.r_min)
    pmag = np.sqrt(2 * p.E_kin0 / (u[0]**2 / p.mass_x + u[1]**2 / p.mass_y))
    px0, py0 = pmag * u
    x0, y0 = L.r_min
    return dict(x0=x0, y0=y0, sx=sx, sy=sy, px0=px0, py0=py0)


def propagate(p=P, L=None, t_final=None, dt=None, sample_every=None, keep_psi=False,
              verbose=False, single_surface=False):
    """Propagate the wavepacket for parameters p on Landscape L.

    single_surface=True propagates on E_- alone (Born-Oppenheimer, no upper
    state, no geometric phase): implemented as v1 = E_-, D = 0, psi2 = 0.
    Returns a dict with time series t, P1, P2, Pminus, Pplus, P_react, P_prod,
    norm, P_edge, C, plus E0, sigE (energy spread), dE (energy drift) and
    averages over the second half.
    """
    L = L or Landscape(p)
    p = dataclasses.replace(p, **{k: v for k, v in dict(t_final=t_final, dt=dt,
                            sample_every=sample_every).items() if v is not None})
    Nx, Ny, dx, dy = len(L.x), len(L.y), L.dx, L.dy
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    T_k = HBAR**2 * (KX**2 / (2 * p.mass_x) + KY**2 / (2 * p.mass_y))
    v1, v2, D, Em = L.v1, L.v2, L.D, L.Em
    theta = mixing_angle(v1, v2, D)
    if single_surface:
        v1, v2, D = Em, Em + 10.0, np.zeros_like(Em)
        theta = np.full_like(Em, -np.pi / 2)      # lower state = component 1

    dt = p.dt
    U_T = np.exp(-1j * T_k * (dt / 2) / HBAR)
    U11, U22, U12 = propagator_2x2(v1, v2, D, dt)

    ip = initial_packet(L, p)
    XX, YY = L.XX, L.YY
    g = (np.exp(-0.5 * ((XX - ip["x0"]) / ip["sx"])**2 - 0.5 * ((YY - ip["y0"]) / ip["sy"])**2)
         * np.exp(1j * (ip["px0"] * (XX - ip["x0"]) + ip["py0"] * (YY - ip["y0"])) / HBAR))
    g /= np.sqrt(np.sum(np.abs(g)**2) * dx * dy)
    if single_surface:
        psi1, psi2 = g.astype(complex), np.zeros_like(g, dtype=complex)
    elif p.init_state == "adiabatic":
        e1, e2 = lower_eigvec(v1, v2, D, Em)
        psi1, psi2 = g * e1, g * e2
    else:  # "diabatic1"
        psi1, psi2 = g.astype(complex), np.zeros_like(g, dtype=complex)
    psi01, psi02 = psi1.copy(), psi2.copy()

    def energy(q1, q2):
        Tq = sum(np.sum(np.abs(np.fft.fft2(q))**2 * T_k) for q in (q1, q2)) / (Nx * Ny)
        Vq = np.sum(v1 * np.abs(q1)**2 + v2 * np.abs(q2)**2 + 2 * D * np.real(np.conj(q1) * q2))
        return float(np.real(Tq + Vq) * dx * dy)

    def energy_spread(q1, q2):
        # <H^2> - <H>^2 via H applied on the grid
        def Hq(r1, r2):
            T1 = np.fft.ifft2(np.fft.fft2(r1) * T_k); T2 = np.fft.ifft2(np.fft.fft2(r2) * T_k)
            return T1 + v1 * r1 + D * r2, T2 + v2 * r2 + D * r1
        h1, h2 = Hq(q1, q2)
        H1 = np.sum(np.conj(q1) * h1 + np.conj(q2) * h2).real * dx * dy
        H2 = np.sum(np.abs(h1)**2 + np.abs(h2)**2) * dx * dy
        return float(np.sqrt(max(H2 - H1**2, 0.0)))

    E0 = energy(psi1, psi2)
    sigE = energy_spread(psi1, psi2)
    # Strip of 8 grid points along every edge, used to detect wrap-around
    edge = np.zeros_like(XX, dtype=bool)
    edge[:8, :] = edge[-8:, :] = edge[:, :8] = edge[:, -8:] = True
    st, ct = np.sin(theta), np.cos(theta)
    prod, react = L.prod, L.react

    nsteps = int(round(p.t_final / dt))
    keys = ("t", "P1", "P2", "Pminus", "Pplus", "P_react", "P_prod", "norm", "P_edge", "C")
    hist = {k: [] for k in keys}
    for step in range(nsteps + 1):
        if step % p.sample_every == 0:
            r1, r2 = np.abs(psi1)**2, np.abs(psi2)**2
            rho = r1 + r2
            lower = np.abs(-st * psi1 + ct * psi2)**2
            hist["t"].append(step * dt)
            hist["P1"].append(r1.sum() * dx * dy)
            hist["P2"].append(r2.sum() * dx * dy)
            hist["Pminus"].append(lower.sum() * dx * dy)
            hist["Pplus"].append((rho - lower).sum() * dx * dy)
            hist["P_react"].append(rho[react].sum() * dx * dy)
            hist["P_prod"].append(rho[prod].sum() * dx * dy)
            hist["norm"].append(rho.sum() * dx * dy)
            hist["P_edge"].append(rho[edge].sum() * dx * dy)
            hist["C"].append(np.sum(np.conj(psi01) * psi1 + np.conj(psi02) * psi2) * dx * dy)
        if step == nsteps:
            break
        f1 = np.fft.ifft2(np.fft.fft2(psi1) * U_T); f2 = np.fft.ifft2(np.fft.fft2(psi2) * U_T)
        psi1, psi2 = U11 * f1 + U12 * f2, U12 * f1 + U22 * f2
        psi1 = np.fft.ifft2(np.fft.fft2(psi1) * U_T); psi2 = np.fft.ifft2(np.fft.fft2(psi2) * U_T)

    out = {k: np.array(v) for k, v in hist.items()}
    out["E0"] = E0
    out["sigE"] = sigE
    out["dE"] = energy(psi1, psi2) - E0
    half = out["t"] >= 0.5 * p.t_final
    out["Pprod_avg"] = float(out["P_prod"][half].mean())
    out["Preact_avg"] = float(out["P_react"][half].mean())
    out["Pminus_avg"] = float(out["Pminus"][half].mean())
    out["Pprod_max"] = float(out["P_prod"].max())
    out["packet"] = ip
    if keep_psi:
        out["psi0"] = (psi01, psi02); out["psi"] = (psi1, psi2)
    if verbose:
        print(f"<H>={E0:+.4f} sigE={sigE:.4f} dE={out['dE']:+.1e} "
              f"norm-1={out['norm'][-1]-1:+.1e} edge={out['P_edge'].max():.1e} "
              f"Pprod_avg={out['Pprod_avg']:.4f}")
    return out

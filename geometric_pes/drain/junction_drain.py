"""The electron-transfer junction (d = 2, M = 16) coupled to N harmonic drain modes, closed and
energy conserving. The adiabatic coupling is propagated on the lower surface; the conical
intersection with fewest-switches surface hopping (exact 2x2 electronic step, momentum rescaling).
Returns per-trajectory records in the same format as geometric_pes.drain.dynamics, so
geometric_pes.drain.metrics.summarize applies.
"""
from dataclasses import replace

import numpy as np

from ..integrators import caldeira_leggett_modes
from ..two_state.model import propagator_2x2
from ..surfaces.junction import Grid, adiabats, diabats, wigner
from ..junction_decision import params

WC = 0.5          # ohmic cutoff of the junction drain


def setup(cpl):
    p = replace(params(2.0, cpl, 16.0), transverse="ho", n_y=0)
    G = Grid(p)
    iR = G.names.index("R")
    r = G.mins[iR]
    ts1 = min(s["E"] for s in G.ts1)
    G.Rcut = r["E"] + 0.5 * (ts1 - r["E"])
    return p, G


def stat_fractions(G, E, N):
    out = []
    cell = G.dx * G.dy
    for e in np.atleast_1d(E):
        K = np.clip(e - G.Em, 0, None)
        lw = np.where(K > 0, N * np.log(np.maximum(K, 1e-300)), -np.inf)
        w = np.exp(lw - lw.max()) * cell
        b = np.array([w[G.lab == i].sum() for i in range(len(G.names))])
        out.append(b / b.sum())
    return np.array(out)


def run(cpl, N, gamma, theta, ntraj=3000, nT=60, seed=31, dt_frac=400, wc=WC):
    p, G = setup(cpl)
    M = p.M
    E_ts12 = G.ts12[0]["E"]
    iR, iPp, iPm = (G.names.index(k) for k in ("R", "P+", "P-"))
    dE_pp = E_ts12 - G.mins[iPm]["E"]
    kT = theta * dE_pp
    rng = np.random.default_rng(seed)
    x, y, px, py, qw = wigner(G, ntraj, rng, "wigner")     # n = 0: qw = 1 (positive Wigner)
    Nh = N // 2
    w, c = caldeira_leggett_modes(Nh, gamma, M, wc)
    w = np.concatenate([w, w]); c = np.concatenate([c, c])
    which = np.r_[np.zeros(Nh, int), np.ones(Nh, int)]
    w2 = (w**2)[:, None]; cc = c[:, None]
    sxy = np.array([x, y])
    Q = cc * sxy[which] / w2 + rng.standard_normal((N, ntraj)) * np.sqrt(kT / w2)
    P = rng.standard_normal((N, ntraj)) * np.sqrt(kT)
    fssh = cpl == "CI"
    state = np.zeros(ntraj, int)
    a0 = adiabats(x, y, p)
    c1 = (-np.sin(a0["theta"])).astype(complex); c2 = np.cos(a0["theta"]).astype(complex)

    def sys_force(x, y, st):
        a = adiabats(x, y, p)
        gx = np.where(st == 0, a["gm"][0], a["gp"][0]); gy = np.where(st == 0, a["gm"][1], a["gp"][1])
        return -gx, -gy, a

    def bath(x, y, Q):
        d = Q - cc * np.array([x, y])[which] / w2
        fb = cc * d
        return -w2 * d, fb[which == 0].sum(0), fb[which == 1].sum(0), 0.5 * (w2 * d * d).sum(0)

    def energy(x, y, px, py, Q, P, st):
        a = adiabats(x, y, p)
        es = np.where(st == 0, a["Em"], a["Ep"])
        return (px**2 + py**2) / (2 * M) + es + 0.5 * (P**2).sum(0) + bath(x, y, Q)[3]

    T_vib = 2 * np.pi / np.sqrt(p.kx / M)
    dt = T_vib / dt_frac
    t_final = nT * T_vib
    nsteps = int(round(t_final / dt))
    fx, fy, a = sys_force(x, y, state)
    fQ, bx, by, _ = bath(x, y, Q)
    fx, fy = fx + bx, fy + by
    E0 = energy(x, y, px, py, Q, P, state)
    lev_core = {nm: G.core[nm] for nm in ("P+", "P-")}
    first = -np.ones(ntraj, int); t_first = np.full(ntraj, np.nan)
    back = np.zeros(ntraj, bool); t_back = np.full(ntraj, np.nan)
    last = -np.ones(ntraj, int)
    nhop = 0
    for i in range(1, nsteps + 1):
        px = px + 0.5 * dt * fx; py = py + 0.5 * dt * fy; P = P + 0.5 * dt * fQ
        xo, yo = x, y
        x = x + dt * px / M; y = y + dt * py / M; Q = Q + dt * P
        if fssh:
            th_o = adiabats(xo, yo, p)["theta"]
            lo_o = np.abs(-np.sin(th_o) * c1 + np.cos(th_o) * c2) ** 2
            Pa_old = np.where(state == 0, lo_o, 1 - lo_o)
            v1, v2, D, *_ = diabats(0.5 * (x + xo), 0.5 * (y + yo), p)
            U11, U22, U12 = propagator_2x2(v1, v2, D, dt)
            c1, c2 = U11 * c1 + U12 * c2, U12 * c1 + U22 * c2
        gx, gy, a = sys_force(x, y, state)
        if fssh:
            lo = np.abs(-np.sin(a["theta"]) * c1 + np.cos(a["theta"]) * c2) ** 2
            Pa_new = np.where(state == 0, lo, 1 - lo)
            gprob = np.clip((Pa_old - Pa_new) / np.maximum(Pa_old, 1e-12), 0, 1)
            k = np.flatnonzero(rng.random(ntraj) < gprob)
            if len(k):
                dE = np.where(state[k] == 0, a["Ep"][k] - a["Em"][k], a["Em"][k] - a["Ep"][k])
                nx_, ny_ = a["nac"][0][k], a["nac"][1][k]
                nn = np.sqrt(nx_**2 + ny_**2) + 1e-300
                ux, uy = nx_ / nn, ny_ / nn
                pu = px[k] * ux + py[k] * uy
                disc = pu * pu - 2 * M * dE
                ok = disc >= 0
                al = pu - np.sign(pu) * np.sqrt(np.where(ok, disc, 0))
                kk = k[ok]
                px[kk] -= al[ok] * ux[ok]; py[kk] -= al[ok] * uy[ok]
                state[kk] = 1 - state[kk]
                nhop += int(ok.sum())
                gx, gy, a = sys_force(x, y, state)
        fQ, bx, by, _ = bath(x, y, Q)
        fx, fy = gx + bx, gy + by
        px = px + 0.5 * dt * fx; py = py + 0.5 * dt * fy; P = P + 0.5 * dt * fQ
        if i % 4 == 0:
            b = G.lookup(G.lab, x, y)
            inP = G.lookup(lev_core["P+"], x, y); inM = G.lookup(lev_core["P-"], x, y)
            inR = (b == iR) & (a["Em"] < G.Rcut)
            lab = np.where(inP, iPp, np.where(inM, iPm, np.where(inR, iR, -1)))
            last = np.where(lab >= 0, lab, last)
            newp = (first < 0) & (inP | inM)
            first[newp] = lab[newp]; t_first[newp] = i * dt
            nb = (first >= 0) & ~back & inR
            back[nb] = True; t_back[nb] = i * dt
    E1 = energy(x, y, px, py, Q, P, state)
    a = adiabats(x, y, p)
    e2d = (px**2 + py**2) / (2 * M) + np.where(state == 0, a["Em"], a["Ep"])
    Eg = np.linspace(E0.min(), E0.max() + 1e-9, 41)
    Fg = stat_fractions(G, Eg, N)
    Fi = np.column_stack([np.interp(E0, Eg, Fg[:, k]) for k in range(Fg.shape[1])])
    prm = dict(cpl=cpl, d=2.0, M=M, N=N, gamma=gamma, wc=wc, theta=theta, kT_m=kT, dE_pp=dE_pp,
               ntraj=ntraj, t_final=t_final, dt=dt, seed=seed, iR=iR, iPp=iPp, iPm=iPm,
               E_TS12=E_ts12, engine="FSSH" if fssh else "CBO", nhop=nhop, junction=p.__dict__)
    arr = dict(first=first, t_first=t_first, back=back, t_back=t_back, last=last, e2d=e2d,
               E=E0, dE=E1 - E0, stat=Fi, upper_end=(state == 1))
    return prm, arr

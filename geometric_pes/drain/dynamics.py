"""Classical dynamics of a branching surface coupled to a finite harmonic drain (velocity Verlet;
energy conserving). All runners return (params, arrays) with per-trajectory records.

run_rest       modes at rest; any surface, any spectrum (modes=(w, c, which)); records the
               decision, drain and exchange timescales, the return fraction, the exchange-mode
               energy, the hot exchange-coordinate time, and (record=True) every switch
run_thermal    thermal (classical Boltzmann) modes at drain temperature theta
run_qct_bose   Bose-occupied modes without zero-point energy (quasiclassical; the frozen limit)
run_rstart     start inside the reactant basin (uphill transfer: is the crossing initiated?)
"""
import numpy as np

from ..integrators import baoab_step, caldeira_leggett_modes
from ..surfaces.branching import V, grad
from ..surfaces.basins import Setup
from .ensembles import flux_sample, core_levels, rstart_sample
from .statistics import stat_fractions, stat_quantum, tst_tables
from ..config import surface as _surface


def _resolve(key, surface):
    """a Surface from an explicit object or a variant name in the config"""
    return surface if surface is not None else _surface(key)


def run_rest(key, N, eta, dE=1.0, ntraj=1000, t_final=400.0, dt=0.01, seed=31, wc=5.0,
        surface=None, modes=None, record=False, y_share=0.5):
    """surface: a Surface (default: config variant `key`); modes: (w, c, which) with which[j] = 0 for x,
    1 for y (default: ohmic, N/2 on each coordinate)"""
    p = _resolve(key, surface)
    S = Setup(p)
    ts12 = [s for s in S.sads if set(s["connects"]) == {"P+", "P-"}][0]["E"]
    E_xy = S.ts1["E"] + dE
    rng = np.random.default_rng(seed)
    x, y, vx, vy = flux_sample(S, E_xy, ntraj, rng)
    if modes is None:
        Nh = N // 2
        w, c = caldeira_leggett_modes(Nh, eta, 1.0, wc)
        w = np.concatenate([w, w]); c = np.concatenate([c, c])
        which = np.r_[np.zeros(Nh, int), np.ones(Nh, int)]
    else:
        w, c, which = modes
        N = len(w)
    w2 = (w**2)[:, None]; cc = c[:, None]
    sx = np.array([x, y])
    q = np.vstack([sx, cc * sx[which] / w2])            # modes at rest at their minimum
    v = np.vstack([np.array([vx, vy]), np.zeros((N, ntraj))])

    def force(q):
        gx, gy = grad(q[0], q[1], p)
        f = np.empty_like(q)
        f[0], f[1] = -gx, -gy
        d = q[2:] - cc * q[which] / w2
        f[2:] = -w2 * d
        fb = cc * d
        f[0] += fb[which == 0].sum(0)
        f[1] += fb[which == 1].sum(0)
        return f

    lev = core_levels(S)
    ts12_sad = [s for s in S.sads if set(s["connects"]) == {"P+", "P-"}][0]
    eT, kpm, kmp = tst_tables(S, ts12_sad)
    ksum = kpm + kmp
    n_ex = np.zeros(ntraj)                                # expected exchanges, TST, after arrival
    # exchange mode of each product well: the Hessian eigenvector most aligned with the direction
    # to TS12; its harmonic energy is compared with the well's barrier to TS12
    xm = {}
    for j in (S.iPp, S.iPm):
        mj = S.mins[j]
        d = ts12_sad["xy"] - mj["xy"]; d = d / np.linalg.norm(d)
        k = int(np.argmax(np.abs(mj["vecs"].T @ d)))
        xm[j] = (mj["xy"], mj["vecs"][:, k], float(mj["eig"][k]), ts12_sad["E"] - mj["E"])
    t_below_ex = np.full(ntraj, np.nan)                   # first time exchange-mode energy < barrier
    t_after_ex = np.zeros(ntraj); t_up_ex = np.zeros(ntraj)
    ev = []                                               # (traj, t, e2, Kx, Ky, E_q) at each switch
    # hot exchange-coordinate time after the first arrival: x, y energy above E_TS12 while at least
    # y_share (t_co05; default 0.5) or 30% (t_co03) of the x, y kinetic energy is in y
    t_co05 = np.zeros(ntraj); t_co03 = np.zeros(ntraj); t_abv = np.zeros(ntraj)
    ser = []                                              # every 1.0: e2, Kx, Ky, E_q, label
    f = force(q)
    first = -np.ones(ntraj, int); t_first = np.full(ntraj, np.nan)
    e2_first = np.full(ntraj, np.nan)
    last = -np.ones(ntraj, int); lastp = -np.ones(ntraj, int)
    n_switch = np.zeros(ntraj, int)
    t_lock = np.zeros(ntraj)                              # last time e2 >= E_TS12
    t_below = np.full(ntraj, np.nan)                      # first time e2 < E_TS12
    t_above_after = np.zeros(ntraj)                       # time with e2 >= E_TS12 after t_below
    e2_mean = []
    nsteps = int(round(t_final / dt))
    for i in range(1, nsteps + 1):
        q, v, f = baoab_step(q, v, f, force, 1.0, dt)
        if i % 5:
            continue
        t = i * dt
        e2 = V(q[0], q[1], p) + 0.5 * (v[0]**2 + v[1]**2)
        t_lock = np.where(e2 >= ts12, t, t_lock)
        t_above_after += 5 * dt * (np.isfinite(t_below) & (e2 >= ts12))
        kx_, ky_ = 0.5 * v[0]**2, 0.5 * v[1]**2
        hot = (first >= 0) & (e2 >= ts12)
        share = ky_ / np.maximum(kx_ + ky_, 1e-12)
        t_abv += 5 * dt * hot; t_co05 += 5 * dt * (hot & (share >= y_share)); t_co03 += 5 * dt * (hot & (share >= 0.3))
        n_ex += 5 * dt * (first >= 0) * np.interp(e2, eT, ksum, left=0.0, right=ksum[-1])
        nb = np.isnan(t_below) & (e2 < ts12)
        t_below[nb] = t
        e2_mean.append(e2.mean())
        b = S.basin(q[0], q[1])
        incore = V(q[0], q[1], p) < lev[b]
        last = np.where(incore, b, last)
        inprod = incore & np.isin(b, [S.iPp, S.iPm])
        newp = (first < 0) & inprod
        first[newp] = b[newp]; t_first[newp] = t; e2_first[newp] = e2[newp]
        # exchange-mode energy in the occupied product core (outside a core: counted as able)
        able = np.ones(ntraj, bool)
        Eq = np.full(ntraj, np.nan)
        for j, (xy0, u, lam, dV) in xm.items():
            sel = inprod & (b == j)
            if sel.any():
                qm = (q[0, sel] - xy0[0]) * u[0] + (q[1, sel] - xy0[1]) * u[1]
                vm = v[0, sel] * u[0] + v[1, sel] * u[1]
                Eq[sel] = (0.5 * vm**2 + 0.5 * lam * qm**2) / dV       # in units of the barrier
                able[sel] = Eq[sel] >= 1.0
        after = np.isfinite(t_below_ex)
        t_after_ex += 5 * dt * after; t_up_ex += 5 * dt * (after & able)
        nbx = ~after & (first >= 0) & ~able
        t_below_ex[nbx] = t
        sw = inprod & (lastp >= 0) & (b != lastp)
        if record:
            for k in np.nonzero(sw)[0]:
                ev.append((k, t, e2[k], 0.5 * v[0, k]**2, 0.5 * v[1, k]**2, Eq[k]))
            if i % 100 == 0:
                ser.append(np.array([e2, 0.5 * v[0]**2, 0.5 * v[1]**2, Eq,
                                     np.where(incore, b, -1).astype(float)], np.float32))
        n_switch += sw
        lastp = np.where(inprod, b, lastp)
    Fst = stat_fractions(S, [E_xy], N)[0]
    params = dict(key=key, N=N, eta=eta, dE=dE, ntraj=ntraj, t_final=t_final, dt=dt, seed=seed,
                  E_xy=E_xy, E_TS12=ts12, iR=S.iR, iPp=S.iPp, iPm=S.iPm)
    arrays = dict(first=first, t_first=t_first, e2_first=e2_first, last=last, n_switch=n_switch,
                  t_lock=t_lock, t_below=t_below, t_above_after=t_above_after, n_ex=n_ex,
                  t_below_ex=t_below_ex, t_after_ex=t_after_ex, t_up_ex=t_up_ex,
                  e2_mean=np.array(e2_mean), stat=Fst, t_co05=t_co05, t_co03=t_co03, t_abv=t_abv)
    if record:
        arrays["events"] = np.array(ev, float).reshape(-1, 6)
        arrays["series"] = np.array(ser, np.float32)      # (time, quantity, traj)
    return params, arrays


def run_thermal(key, N, eta, theta, init="boltz", hbar=1.0, ntraj=2000, t_final=400.0, wc=5.0, seed=21,
        dt=None):
    p = _surface(key)
    S = Setup(p)
    ts12 = [s for s in S.sads if set(s["connects"]) == {"P+", "P-"}][0]["E"]
    dE_pp = ts12 - S.mins[S.iPm]["E"]
    kT = theta * dE_pp
    E_xy = S.ts1["E"] + 1.0
    rng = np.random.default_rng(seed)
    x, y, vx, vy = flux_sample(S, E_xy, ntraj, rng)
    Nh = N // 2
    w, c = caldeira_leggett_modes(Nh, eta, 1.0, wc)
    w = np.concatenate([w, w]); c = np.concatenate([c, c])
    which = np.r_[np.zeros(Nh, int), np.ones(Nh, int)]
    w2 = (w**2)[:, None]
    cc = c[:, None]
    sx = np.array([x, y])
    if init == "wigner":
        x_ = hbar * w / 2
        ekT = np.where(x_ > 0, x_ / np.tanh(x_ / max(kT, 1e-300)), kT) if kT > 0 else x_
        ekT = ekT[:, None]
    else:
        ekT = np.full((N, 1), kT)
    disp = rng.standard_normal((N, ntraj)) * np.sqrt(ekT / w2)
    q = np.vstack([sx, cc * sx[which] / w2 + disp])
    v = np.vstack([np.array([vx, vy]), rng.standard_normal((N, ntraj)) * np.sqrt(ekT)])

    def force(q):
        gx, gy = grad(q[0], q[1], p)
        f = np.empty_like(q)
        f[0], f[1] = -gx, -gy
        d = q[2:] - cc * q[which] / w2
        f[2:] = -w2 * d
        fb = cc * d
        f[0] += fb[which == 0].sum(0)
        f[1] += fb[which == 1].sum(0)
        return f

    def energy(q, v):
        return (V(q[0], q[1], p) + 0.5 * (v**2).sum(0)
                + 0.5 * (w2 * (q[2:] - cc * q[which] / w2) ** 2).sum(0))

    lev = core_levels(S)
    dt = dt or 0.01
    f = force(q)
    E0 = energy(q, v)
    first = -np.ones(ntraj, int)
    t_first = np.full(ntraj, np.nan)
    back = np.zeros(ntraj, bool)
    t_back = np.full(ntraj, np.nan)
    last = -np.ones(ntraj, int)
    nsteps = int(round(t_final / dt))
    out_every = max(1, nsteps // 200)
    T, PP, PM, PR, TRAP = [], [], [], [], []
    for i in range(1, nsteps + 1):
        q, v, f = baoab_step(q, v, f, force, 1.0, dt)
        if i % 5 == 0:
            b = S.basin(q[0], q[1])
            incore = V(q[0], q[1], p) < lev[b]
            last = np.where(incore, b, last)
            newp = (first < 0) & incore & np.isin(b, [S.iPp, S.iPm])
            first[newp] = b[newp]
            t_first[newp] = i * dt
            nb = (first >= 0) & ~back & incore & (b == S.iR)
            back[nb] = True
            t_back[nb] = i * dt
        if i % out_every == 0:
            e2 = V(q[0], q[1], p) + 0.5 * (v[0]**2 + v[1]**2)
            T.append(i * dt)
            PP.append(np.mean(last == S.iPp)); PM.append(np.mean(last == S.iPm))
            PR.append(np.mean(last == S.iR)); TRAP.append(np.mean(e2 < ts12))
    E1 = energy(q, v)
    e2d = V(q[0], q[1], p) + 0.5 * (v[0]**2 + v[1]**2)
    # microcanonical reference at each trajectory's energy (interpolated on 41 energies)
    Eg = np.linspace(E0.min(), E0.max() + 1e-9, 41)
    Fg = stat_fractions(S, Eg, N)
    Fi = np.column_stack([np.interp(E0, Eg, Fg[:, k]) for k in range(Fg.shape[1])])
    params = dict(key=key, N=N, eta=eta, wc=wc, theta=theta, kT_m=kT, dE_pp=dE_pp, init=init,
                  hbar=hbar, ntraj=ntraj, t_final=t_final, dt=dt, seed=seed, E_xy=E_xy,
                  iR=S.iR, iPp=S.iPp, iPm=S.iPm, E_TS12=ts12)
    arrays = dict(first=first, t_first=t_first, back=back, t_back=t_back, last=last, e2d=e2d,
                  E=E0, dE=E1 - E0, stat=Fi, t=np.array(T), P_plus=np.array(PP),
                  P_minus=np.array(PM), P_R=np.array(PR), trapped=np.array(TRAP))
    return params, arrays


def run_qct_bose(key, N, eta, theta, init="bose", hbar=0.25, ntraj=2000, t_final=400.0, wc=5.0, seed=21,
        dt=0.01):
    p = _surface(key)
    S = Setup(p)
    ts12 = [s for s in S.sads if set(s["connects"]) == {"P+", "P-"}][0]["E"]
    dE_pp = ts12 - S.mins[S.iPm]["E"]
    kT = theta * dE_pp
    E_xy = S.ts1["E"] + 1.0
    rng = np.random.default_rng(seed)
    x, y, vx, vy = flux_sample(S, E_xy, ntraj, rng)
    Nh = N // 2
    w, c = caldeira_leggett_modes(Nh, eta, 1.0, wc)
    w = np.concatenate([w, w]); c = np.concatenate([c, c])
    which = np.r_[np.zeros(Nh, int), np.ones(Nh, int)]
    w2 = (w**2)[:, None]
    cc = c[:, None]
    sx = np.array([x, y])
    hw = hbar * w
    if init == "bose":
        if kT > 0:
            z = np.exp(-hw / kT)[:, None]
            u = rng.uniform(size=(N, ntraj))
            nk = np.where(z > 1e-300, np.floor(np.log1p(-u) / np.log(np.maximum(z, 1e-300))), 0)
        else:
            nk = np.zeros((N, ntraj))
        Ek = nk * hw[:, None]
        ph = rng.uniform(0, 2 * np.pi, size=(N, ntraj))
        disp = np.sqrt(2 * Ek) / w[:, None] * np.cos(ph)
        vm = -np.sqrt(2 * Ek) * np.sin(ph)
    elif init == "boltz":
        disp = rng.standard_normal((N, ntraj)) * np.sqrt(kT / w2)
        vm = rng.standard_normal((N, ntraj)) * np.sqrt(kT)
        nk = None
    else:
        raise ValueError(init)
    q = np.vstack([sx, cc * sx[which] / w2 + disp])
    v = np.vstack([np.array([vx, vy]), vm])

    def force(q):
        gx, gy = grad(q[0], q[1], p)
        f = np.empty_like(q)
        f[0], f[1] = -gx, -gy
        d = q[2:] - cc * q[which] / w2
        f[2:] = -w2 * d
        fb = cc * d
        f[0] += fb[which == 0].sum(0)
        f[1] += fb[which == 1].sum(0)
        return f

    def mode_energy(q, v):
        return 0.5 * v[2:] ** 2 + 0.5 * w2 * (q[2:] - cc * q[which] / w2) ** 2

    def energy(q, v):
        return V(q[0], q[1], p) + 0.5 * (v[:2] ** 2).sum(0) + mode_energy(q, v).sum(0)

    lev = core_levels(S)
    f = force(q)
    E0 = energy(q, v)
    first = -np.ones(ntraj, int)
    back = np.zeros(ntraj, bool)
    last = -np.ones(ntraj, int)
    nsteps = int(round(t_final / dt))
    out_every = max(1, nsteps // 200)
    T, PP, PM, PR, TRAP, NEG = [], [], [], [], [], []
    Em0 = mode_energy(q, v)
    for i in range(1, nsteps + 1):
        q, v, f = baoab_step(q, v, f, force, 1.0, dt)
        if i % 5 == 0:
            b = S.basin(q[0], q[1])
            incore = V(q[0], q[1], p) < lev[b]
            last = np.where(incore, b, last)
            newp = (first < 0) & incore & np.isin(b, [S.iPp, S.iPm])
            first[newp] = b[newp]
            nb = (first >= 0) & ~back & incore & (b == S.iR)
            back[nb] = True
        if i % out_every == 0:
            e2 = V(q[0], q[1], p) + 0.5 * (v[0]**2 + v[1]**2)
            T.append(i * dt)
            PP.append(np.mean(last == S.iPp)); PM.append(np.mean(last == S.iPm))
            PR.append(np.mean(last == S.iR)); TRAP.append(np.mean(e2 < ts12))
    E1 = energy(q, v)
    e2d = V(q[0], q[1], p) + 0.5 * (v[0]**2 + v[1]**2)
    Em1 = mode_energy(q, v)
    # final mode quantum numbers by action (energy / hbar w); quantum-number check
    n_final = Em1 / hw[:, None]
    # references at each trajectory's energy (interpolated on 41 energies)
    Eg = np.linspace(E0.min(), E0.max() + 1e-9, 41)
    Fg = stat_fractions(S, Eg, N)
    Fq = stat_quantum(S, Eg, hw)
    Fi = np.column_stack([np.interp(E0, Eg, Fg[:, k]) for k in range(Fg.shape[1])])
    Fqi = np.column_stack([np.interp(E0, Eg, Fq[:, k]) for k in range(Fq.shape[1])])
    params = dict(key=key, N=N, eta=eta, wc=wc, theta=theta, kT_m=kT, dE_pp=dE_pp, init=init,
                  hbar=hbar, ntraj=ntraj, t_final=t_final, dt=dt, seed=seed, E_xy=E_xy,
                  iR=S.iR, iPp=S.iPp, iPm=S.iPm, E_TS12=ts12)
    arrays = dict(first=first, back=back, last=last, e2d=e2d, E=E0, dE=E1 - E0, stat=Fi,
                  stat_q=Fqi, t=np.array(T), P_plus=np.array(PP), P_minus=np.array(PM),
                  P_R=np.array(PR), trapped=np.array(TRAP), Em0=Em0.sum(0), Em1=Em1.sum(0),
                  nbar_final=n_final.mean(1))
    return params, arrays


def run_rstart(key, N, eta, dE_above_ts1=1.0, ntraj=2000, t_final=400.0, wc=5.0, seed=21, dt=0.01):
    """Closed drain run (modes at rest) started microcanonically in R at E_TS1 + dE.
    The transfer is not presupposed: it competes with the drain. Records the same
    per-trajectory arrays as hmix2_drain.run (first product core, back to R, last core,
    final 2D energy) plus the first-core label including R."""
    Vf = V
    p = _surface(key)
    S = Setup(p)
    E_xy = S.ts1["E"] + dE_above_ts1
    rng = np.random.default_rng(seed)
    x, y, vx, vy = rstart_sample(S, E_xy, ntraj, rng)
    Ne = max(N, 2)
    Nh = Ne // 2
    w, c = caldeira_leggett_modes(Nh, eta if N > 0 else 1e-12, 1.0, wc)
    w = np.concatenate([w, w]); c = np.concatenate([c, c])
    which = np.r_[np.zeros(Nh, int), np.ones(Nh, int)]
    w2 = (w**2)[:, None]; cc = c[:, None]
    sx = np.array([x, y])
    q = np.vstack([sx, cc * sx[which] / w2])
    v = np.vstack([np.array([vx, vy]), np.zeros((Ne, ntraj))])

    def force(q):
        gx, gy = grad(q[0], q[1], p)
        f = np.empty_like(q); f[0], f[1] = -gx, -gy
        d = q[2:] - cc * q[which] / w2
        f[2:] = -w2 * d
        fb = cc * d
        f[0] += fb[which == 0].sum(0); f[1] += fb[which == 1].sum(0)
        return f

    def energy(q, v):
        return (Vf(q[0], q[1], p) + 0.5 * (v**2).sum(0)
                + 0.5 * (w2 * (q[2:] - cc * q[which] / w2) ** 2).sum(0))

    lev = core_levels(S)
    f = force(q); E0 = energy(q, v)
    first = -np.ones(ntraj, int); back = np.zeros(ntraj, bool); last = -np.ones(ntraj, int)
    t_cross = np.full(ntraj, np.nan)
    nsteps = int(round(t_final / dt)); out_every = max(1, nsteps // 200)
    T, PP, PR = [], [], []
    for i in range(1, nsteps + 1):
        q, v, f = baoab_step(q, v, f, force, 1.0, dt)
        if i % 5 == 0:
            b = S.basin(q[0], q[1])
            incore = Vf(q[0], q[1], p) < lev[b]
            last = np.where(incore, b, last)
            newp = (first < 0) & incore & np.isin(b, [S.iPp, S.iPm])
            first[newp] = b[newp]; t_cross[newp] = i * dt
            nb = (first >= 0) & ~back & incore & (b == S.iR)
            back[nb] = True
        if i % out_every == 0:
            T.append(i * dt)
            PP.append(np.isin(last, [S.iPp, S.iPm]).mean()); PR.append((last == S.iR).mean())
    E1 = energy(q, v)
    e2d = Vf(q[0], q[1], p) + 0.5 * (v[0]**2 + v[1]**2)
    stat = stat_fractions(S, E0, N)
    params = dict(key=key, N=N, eta=eta, wc=wc, theta=0.0, init="rstart", hbar=1.0, ntraj=ntraj,
                  t_final=t_final, dt=dt, seed=seed, E_xy=E_xy, dE_above_ts1=dE_above_ts1,
                  iR=S.iR, iPp=S.iPp, iPm=S.iPm,
                  E_TS12=[s["E"] for s in S.sads if set(s["connects"]) == {"P+", "P-"}][0])
    arrays = dict(first=first, back=back, last=last, e2d=e2d, E=E0, dE=E1 - E0, stat=stat,
                  t_cross=t_cross, t=np.array(T), P_P=np.array(PP), P_R=np.array(PR))
    return params, arrays

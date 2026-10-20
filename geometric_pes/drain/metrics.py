"""Directionality metrics from per-trajectory records.

summarize / ladder_level   phi_1, phi(tau_obs), memory m, inversion index, back-transfer B, level
metrics                    trajectory memory r, drain / exchange timescales, return fraction,
                           hot exchange-coordinate time
exchange_time              mean dwell time in a product core between switches (bare 2D run)
transfer_metrics           R -> P transfer metrics (uphill test)
"""
import numpy as np


def summarize(params, a, nboot=300, seed=0):
    """Ladder metrics with bootstrap errors (resampling trajectories)."""
    iR, iPp, iPm = params["iR"], params["iPp"], params["iPm"]
    n = len(a["first"])
    rng = np.random.default_rng(seed)

    def metrics(ix):
        first, last, back = a["first"][ix], a["last"][ix], a["back"][ix]
        st = a["stat"][ix].mean(0)
        got = first >= 0
        phi1 = np.mean(first[got] == iPp) if got.any() else np.nan
        Pp, Pm, PR = np.mean(last == iPp), np.mean(last == iPm), np.mean(last == iR)
        phi_end = Pp / max(Pp + Pm, 1e-12)
        phi_st = st[iPp] / (st[iPp] + st[iPm])
        B = np.mean(back[got]) if got.any() else np.nan
        den = phi1 - phi_st
        m = (phi_end - phi_st) / den if abs(den) > 1e-9 else np.nan
        iota_m, iota_p = Pm / st[iPm], Pp / st[iPp]
        trap = np.mean(a["e2d"][ix] < params["E_TS12"])
        return np.array([phi1, phi_end, phi_st, B, m, iota_m, iota_p, Pp, Pm, PR, trap, den,
                         st[iPp], st[iPm], st[iR]])
    names = ["phi1", "phi_end", "phi_stat", "B", "m", "iota_minus", "iota_plus", "P_plus",
             "P_minus", "P_R", "trapped", "phi1_minus_stat", "stat_plus", "stat_minus", "stat_R"]
    base = metrics(np.arange(n))
    boot = np.array([metrics(rng.integers(0, n, n)) for _ in range(nboot)])
    err = np.nanstd(boot, axis=0)
    out = {k: float(v) for k, v in zip(names, base)}
    out.update({k + "_err": float(e) for k, e in zip(names, err)})
    out["level"] = ladder_level(out)
    return out


def ladder_level(r, eps=0.05):
    """Highest level reached (D2 at most if not committed). m is meaningful only
    when the decision differs from statistics (|phi_1 - phi_stat| > 0.1)."""
    if not (r["B"] < eps):
        return "D2"
    if abs(r["phi1_minus_stat"]) < 0.1:
        return "D3*"            # committed; decision indistinguishable from statistics
    if not (r["m"] > 0.5):
        return "D3"
    # D5: the product with the smaller statistical share holds more population, iota >= 10
    small, big = (("P_minus", "stat_minus"), ("P_plus", "stat_plus"))
    if r["stat_minus"] > r["stat_plus"]:
        small, big = big, small
    iota = r[small[0]] / max(r[small[1]], 1e-300)
    if r[small[0]] > r[big[0]] and iota >= 10:
        return "D5"
    return "D4"


def metrics(params, a):
    iPp, iPm = params["iPp"], params["iPm"]
    got = a["first"] >= 0
    first, last = a["first"][got], a["last"][got]
    phi1 = np.mean(first == iPp)
    prod = np.isin(last, [iPp, iPm])
    Pp, Pm = np.mean(last == iPp), np.mean(last == iPm)
    phi_end = Pp / max(Pp + Pm, 1e-12)
    st = a["stat"]; phi_st = st[iPp] / (st[iPp] + st[iPm])
    den = phi1 - phi_st
    m = (phi_end - phi_st) / den if abs(den) >= 0.1 else np.nan
    kept = np.mean(last[prod] == first[prod])
    # trajectory-level memory: correlation between the first-entered and the final product
    # (independent of any statistical reference); undefined when the decision is lopsided
    f_, l_ = (first[prod] == iPp).astype(float), (last[prod] == iPp).astype(float)
    r = (np.mean((f_ - f_.mean()) * (l_ - l_.mean())) / (f_.std() * l_.std())
         if f_.std() >= 0.25 and l_.std() > 0 else np.nan)
    T = params["t_final"]
    tb = a["t_below"][got]
    frac_locked = np.mean(np.isfinite(tb))
    tau_drain = np.median(np.where(np.isfinite(tb), tb, np.inf)) if frac_locked >= 0.5 else np.nan
    fin = np.isfinite(tb)
    f_ret = (float(np.sum(a["t_above_after"][got][fin]) / np.sum(T - tb[fin])) if fin.any() else np.nan)
    if "t_below_ex" in a:
        tbx = a["t_below_ex"][got]
        tau_drain_ex = (np.median(np.where(np.isfinite(tbx), tbx, np.inf))
                        if np.mean(np.isfinite(tbx)) >= 0.5 else np.nan)
        den = a["t_after_ex"][got].sum()
        f_ret_ex = float(a["t_up_ex"][got].sum() / den) if den > 0 else np.nan
    else:
        tau_drain_ex = f_ret_ex = np.nan
    em = a["e2_mean"]; tt = 5 * params["dt"] * (1 + np.arange(len(em)))
    below = np.nonzero(em < params["E_TS12"])[0]
    tau_drain_mean = float(tt[below[0]]) if len(below) else np.nan
    tau_dec = np.nanmedian(a["t_first"][got])
    # energy removed from x, y by the time of the decision, relative to the excess over TS12
    ex = (params["E_xy"] - a["e2_first"][got]) / (params["E_xy"] - params["E_TS12"])
    return dict(phi1=phi1, phi_end=phi_end, phi_stat=phi_st, m=m, kept=kept, r=r,
                tau_dec=tau_dec, tau_drain=tau_drain, tau_drain_mean=tau_drain_mean, f_ret=f_ret,
                tau_drain_ex=tau_drain_ex, f_ret_ex=f_ret_ex,
                t_co05_mean=float(np.mean(a["t_co05"][a["first"] >= 0])) if "t_co05" in a else np.nan,
                t_co03_mean=float(np.mean(a["t_co03"][a["first"] >= 0])) if "t_co03" in a else np.nan,
                t_abv_mean=float(np.mean(a["t_abv"][a["first"] >= 0])) if "t_abv" in a else np.nan,
                n_ex=float(np.mean(a["n_ex"][got])) if "n_ex" in a else np.nan,
                r_pred=float(np.mean(np.exp(-a["n_ex"][got]))) if "n_ex" in a else np.nan,
                frac_locked=frac_locked,
                drained_at_decision=float(np.nanmedian(ex)), switches=float(a["n_switch"][got].mean()))


def exchange_time(params, a):
    """mean dwell time in a product core between switches, bare 2D dynamics"""
    got = a["first"] >= 0
    T = params["t_final"]
    time_in_products = np.sum(T - a["t_first"][got])
    return time_in_products / max(a["n_switch"][got].sum(), 1)


def transfer_metrics(params, a, ref2d, stat_key="stat", nboot=300, seed=0):
    iR, iPp, iPm = params["iR"], params["iPp"], params["iPm"]
    n = len(a["first"])
    rng = np.random.default_rng(seed)
    E_TS1 = params["E_xy"] - 1.0

    def m(ix):
        first, last, back = a["first"][ix], a["last"][ix], a["back"][ix]
        got = first >= 0
        Y = got.mean()
        PP = np.isin(last, [iPp, iPm]).mean()
        PR = (last == iR).mean()
        B = back[got].mean() if got.any() else np.nan
        if stat_key == "2d":
            sP, sR = ref2d["P_plus"] + ref2d["P_minus"], ref2d["R"]
        else:
            st = a[stat_key][ix].mean(0)
            sP, sR = st[iPp] + st[iPm], st[iR]
        mT = (PP - sP) / (Y - sP) if abs(Y - sP) > 1e-9 else np.nan
        trapP = np.mean((a["e2d"][ix] < E_TS1) & np.isin(last, [iPp, iPm]))
        return np.array([Y, PP, PR, B, sP, sR, mT, PP / max(sP, 1e-300), trapP])
    names = ["Y", "P_P", "P_R", "B_T", "stat_P", "stat_R", "m_T", "iota_P", "trapped_in_P"]
    base = m(np.arange(n))
    boot = np.array([m(rng.integers(0, n, n)) for _ in range(nboot)])
    out = {k: float(v) for k, v in zip(names, base)}
    out.update({k + "_err": float(e) for k, e in zip(names, np.nanstd(boot, axis=0))})
    out["inverted"] = bool(out["P_P"] > out["P_R"] and out["stat_P"] < out["stat_R"])
    lev = "T-D2"
    if out["B_T"] < 0.05:
        lev = "T-D3"
        if out["m_T"] > 0.5:
            lev = "T-D4"
            if out["inverted"] and out["iota_P"] >= 10:
                lev = "T-D5"
    out["level"] = lev
    return out

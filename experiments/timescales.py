#!/usr/bin/env python3
"""Decision, drain and exchange timescales on the four surfaces. For every (surface, N, eta): the
decision time, tau_drain (median first time the x, y energy falls below the product-product
saddle), tau_exchange (bare 2D dwell time between switches), chi = tau_drain / tau_exchange, the
trajectory memory r, the energy drained by the first arrival and the shift of the decision."""
import numpy as np

from _common import parse, settings, cached, pool_map, summary
from geometric_pes.config import surface
from geometric_pes.surfaces.basins import Setup
from geometric_pes.drain.dynamics import run_rest
from geometric_pes.drain.metrics import metrics, exchange_time
from geometric_pes.drain.statistics import p_return

EXP = "timescales"


def tag(key, N, eta):
    return f"{key}_bare" if N == 2 and eta < 1e-9 else f"{key}_N{N}_eta{eta:g}"


def jobs(s):
    return [(k, 2, 1e-12) for k in s["surfaces"]] + [(k, N, e) for k in s["surfaces"] for N in s["N"] for e in s["eta"]]


def job(args):
    (key, N, eta), s = args
    cached(EXP, tag(key, N, eta), s["_quick"], lambda: run_rest(
        key, N, eta, dE=s["start_above_ts1"], ntraj=s["ntraj"], t_final=s["t_obs"], dt=s["dt"], seed=s["seed"],
        wc=s["ohmic_cutoff"], y_share=s["hot_y_share"]))


def rows(s):
    bare = {}
    for k in s["surfaces"]:
        params, a = cached(EXP, tag(k, 2, 1e-12), s["_quick"], None)
        bare[k] = dict(tau_exchange=float(exchange_time(params, a)), **metrics(params, a))
    out = []
    for k in s["surfaces"]:
        for N in s["N"]:
            for e in s["eta"]:
                params, a = cached(EXP, tag(k, N, e), s["_quick"], None)
                m = metrics(params, a)
                tx = bare[k]["tau_exchange"]
                out.append(dict(key=k, N=N, eta=e, tau_exchange=tx,
                                chi=m["tau_drain"] / tx if np.isfinite(m["tau_drain"]) else float("inf"),
                                dphi1=m["phi1"] - bare[k]["phi1"],
                                p_ret=p_return(k, N, params["E_xy"], params["E_TS12"]), **m))
    return bare, out


if __name__ == "__main__":
    args = parse(__doc__); s = settings(EXP, args.quick)
    for k in s["surfaces"]:
        Setup(surface(k))
    pool_map(job, [(j, s) for j in jobs(s)], args.nproc)
    bare, out = rows(s)
    for r in out:
        print(r["key"], r["N"], r["eta"], f"r={r['r']:.2f} chi={r['chi']:.2f} drained@decision={r['drained_at_decision']:.2f} dphi1={r['dphi1']:+.3f}")
    summary(EXP, dict(no_drain=bare, runs=out), args.quick, s)

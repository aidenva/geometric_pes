#!/usr/bin/env python3
"""The geometric decision. Closed system (no drain) on surfaces with identical stationary points:
first-arrival split phi_1, the long-time split, and the surface's own (2D) statistics."""
from _common import parse, settings, cached, pool_map, summary
from geometric_pes.config import surface
from geometric_pes.surfaces.basins import Setup
from geometric_pes.drain.dynamics import run_rest
from geometric_pes.drain.metrics import metrics
from geometric_pes.drain.statistics import stat_fractions

EXP = "decision"


def job(args):
    key, s = args
    cached(EXP, f"{key}_bare", s["_quick"], lambda: run_rest(
        key, 2, 1e-12, dE=s["start_above_ts1"], ntraj=s["ntraj"], t_final=s["t_obs"], dt=s["dt"], seed=s["seed"]))


if __name__ == "__main__":
    args = parse(__doc__); s = settings(EXP, args.quick)
    for k in s["surfaces"]:
        Setup(surface(k))                                 # build basin caches before forking
    pool_map(job, [(k, s) for k in s["surfaces"]], args.nproc)
    out = {}
    for k in s["surfaces"]:
        params, a = cached(EXP, f"{k}_bare", s["_quick"], None)
        m = metrics(params, a)
        S = Setup(surface(k))
        st = stat_fractions(S, [params["E_xy"]], 0)[0]
        out[k] = dict(phi1=m["phi1"], phi_end=m["phi_end"],
                      phi_stat_2d=float(st[S.iPp] / (st[S.iPp] + st[S.iPm])), tau_decision=m["tau_dec"])
        print(k, {kk: round(v, 3) for kk, v in out[k].items()})
    summary(EXP, out, args.quick, s)

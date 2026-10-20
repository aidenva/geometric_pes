#!/usr/bin/env python3
"""The molecule's own modes keep the decision. Spectator modes at rest, N modes at coupling eta:
first-arrival split, kept split at tau_obs, the N-mode statistics, memory m, inversion index,
back-transfer and the directionality level."""
from _common import parse, settings, cached, pool_map, summary
from geometric_pes.config import surface
from geometric_pes.surfaces.basins import Setup
from geometric_pes.drain.dynamics import run_thermal
from geometric_pes.drain.metrics import summarize
from geometric_pes.drain.statistics import stat_fractions

EXP = "drain_modes"


def run_one(key, N, eta, s):
    if N == 0:                                            # bare 2D dynamics, 2D statistics
        params, a = run_thermal(key, 2, 1e-12, 0.0, ntraj=s["ntraj"], t_final=s["t_obs"], seed=s["seed"])
        a["stat"] = stat_fractions(Setup(surface(key)), a["E"], 0)
        params.update(N=0, eta=0.0)
        return params, a
    return run_thermal(key, N, eta, 0.0, ntraj=s["ntraj"], t_final=s["t_obs"], seed=s["seed"])


def jobs(s):
    return [(k, N, e) for k in s["surfaces"] for N in s["N"] for e in (s["eta"] if N else [0.0])]


def job(args):
    (key, N, eta), s = args
    cached(EXP, f"{key}_N{N}_eta{eta:g}", s["_quick"], lambda: run_one(key, N, eta, s))


if __name__ == "__main__":
    args = parse(__doc__); s = settings(EXP, args.quick)
    for k in s["surfaces"]:
        Setup(surface(k))
    pool_map(job, [(j, s) for j in jobs(s)], args.nproc)
    out = []
    for key, N, eta in jobs(s):
        params, a = cached(EXP, f"{key}_N{N}_eta{eta:g}", s["_quick"], None)
        r = summarize(params, a)
        out.append(dict(key=key, N=N, eta=eta, **r))
        print(key, N, eta, {k: round(r[k], 3) for k in ("phi1", "phi_end", "phi_stat", "m", "iota_minus", "B")}, r["level"])
    summary(EXP, out, args.quick, s)

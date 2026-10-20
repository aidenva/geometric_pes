#!/usr/bin/env python3
"""Thermal mixing of the drain. Classical Boltzmann modes at drain temperature theta =
k_B T_m / dE_PP: the rungs are lost from the top (inversion, then selection, then commitment)."""
from _common import parse, settings, cached, pool_map, summary
from geometric_pes.config import surface
from geometric_pes.surfaces.basins import Setup
from geometric_pes.drain.dynamics import run_thermal
from geometric_pes.drain.metrics import summarize

EXP = "thermal_drain"


def job(args):
    th, s = args
    cached(EXP, f"{s['surface']}_N{s['N']}_eta{s['eta']:g}_th{th:g}", s["_quick"], lambda: run_thermal(
        s["surface"], s["N"], s["eta"], th, ntraj=s["ntraj"], t_final=s["t_obs"], seed=s["seed"]))


if __name__ == "__main__":
    args = parse(__doc__); s = settings(EXP, args.quick)
    Setup(surface(s["surface"]))
    pool_map(job, [(th, s) for th in s["theta"]], args.nproc)
    out = []
    for th in s["theta"]:
        params, a = cached(EXP, f"{s['surface']}_N{s['N']}_eta{s['eta']:g}_th{th:g}", s["_quick"], None)
        r = summarize(params, a)
        out.append(dict(theta=th, **r))
        print(f"theta={th:<6g}", {k: round(r[k], 3) for k in ("phi_end", "m", "iota_minus", "B")}, r["level"])
    summary(EXP, out, args.quick, s)

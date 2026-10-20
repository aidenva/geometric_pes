#!/usr/bin/env python3
"""Uphill transfer. Started on TS1 heading uphill: does the drain hold an endothermic transfer
(commitment) against both the 2D and the many-mode statistics? Started in R: does it initiate it?"""
from _common import parse, settings, cached, pool_map, summary
from geometric_pes.config import surface
from geometric_pes.surfaces.basins import Setup
from geometric_pes.drain.dynamics import run_thermal, run_qct_bose, run_rstart
from geometric_pes.drain.metrics import summarize, transfer_metrics
from geometric_pes.drain.statistics import stat_fractions, reference_shares

EXP = "uphill"


def plan(s):
    J = []
    for k in s["surfaces"]:
        J.append(("cl", k, 0, 0.0, 0.0))
        J += [("cl", k, N, e, 0.0) for N in s["N"] for e in s["eta"]]
        J += [("cl", k, 64, 1.0, th) for th in s["theta"]]
        J += [("q", k, 64, 1.0, th) for th in s["quantum_theta"]]
        J += [("rs", k, N, e, dE) for dE in s["reactant_start_above_ts1"] for N, e in s["reactant_start_drains"]]
    return J


def tag(j):
    eng, k, N, e, x = j
    return f"{eng}_{k}_N{N}_eta{e:g}_{'dE' if eng == 'rs' else 'th'}{x:g}"


def run_one(j, s):
    eng, k, N, e, x = j
    kw = dict(ntraj=s["ntraj"], t_final=s["t_obs"], seed=s["seed"])
    if eng == "rs":
        return run_rstart(k, N, e, dE_above_ts1=x, **kw)
    if eng == "q":
        return run_qct_bose(k, N, e, x, hbar=s["quantum_hbar"], **kw)
    if N == 0:                                            # bare 2D dynamics
        params, a = run_thermal(k, 2, 1e-12, 0.0, **kw)
        a["stat"] = stat_fractions(Setup(surface(k)), a["E"], 0)
        params.update(N=0, eta=0.0)
        return params, a
    return run_thermal(k, N, e, x, **kw)


def job(args):
    j, s = args
    cached(EXP, tag(j), s["_quick"], lambda: run_one(j, s))


if __name__ == "__main__":
    args = parse(__doc__); s = settings(EXP, args.quick)
    refs = {}
    for k in s["surfaces"]:
        S = Setup(surface(k))
        refs[k] = reference_shares(S, S.ts1["E"] + s["start_above_ts1"])
    J = plan(s)
    pool_map(job, [(j, s) for j in J], args.nproc)
    rows = []
    for j in J:
        params, a = cached(EXP, tag(j), s["_quick"], None)
        ref2d = refs[j[1]]["N0"]
        row = dict(tag=tag(j), engine=j[0], key=j[1], N=j[2], eta=j[3], x=j[4],
                   transfer_2d=transfer_metrics(params, a, ref2d, "2d"),
                   transfer_many_mode=transfer_metrics(params, a, ref2d, "stat"))
        if "stat_q" in a:
            row["transfer_quantum"] = transfer_metrics(params, a, ref2d, "stat_q")
        if j[0] != "rs":
            row["selectivity"] = summarize(params, a)
        rows.append(row)
        t = row["transfer_2d"]
        print(row["tag"], f"P_P={t['P_P']:.3f} B={t['B_T']:.3f} level {t['level']}")
    summary(EXP, dict(refs=refs, runs=rows), args.quick, s)

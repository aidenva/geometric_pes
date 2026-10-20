#!/usr/bin/env python3
"""Quantum drain modes. QCT-Bose dynamics (Bose-occupied modes without zero-point energy: the
frozen limit, scored against both the classical and the quantum many-mode references), and the
Bloch-Redfield master-equation exchange time tau_x of the relaxed products."""
from _common import parse, settings, cached, pool_map, summary
from geometric_pes.config import surface
from geometric_pes.surfaces.basins import Setup
from geometric_pes.drain.dynamics import run_qct_bose
from geometric_pes.drain.metrics import summarize
from geometric_pes.drain.master_equation import redfield_case

EXP = "quantum_drain"


def tag(hb, th, s):
    return f"{s['surface']}_N{s['N']}_eta{s['eta']:g}_hb{hb:g}_th{th:g}"


def job(args):
    (hb, th), s = args
    cached(EXP, tag(hb, th, s), s["_quick"], lambda: run_qct_bose(
        s["surface"], s["N"], s["eta"], th, hbar=hb, ntraj=s["ntraj"], t_final=s["t_obs"], seed=s["seed"]))


if __name__ == "__main__":
    args = parse(__doc__); s = settings(EXP, args.quick)
    S = Setup(surface(s["surface"]))
    grid = [(hb, th) for hb in s["hbar"] for th in s["theta"]]
    pool_map(job, [(g, s) for g in grid], args.nproc)
    qct = []
    for hb, th in grid:
        params, a = cached(EXP, tag(hb, th, s), s["_quick"], None)
        aq = dict(a); aq["stat"] = a["stat_q"]
        r_cl, r_q = summarize(params, a), summarize(params, aq)
        qct.append(dict(hbar=hb, theta=th, classical_reference=r_cl, quantum_reference=r_q))
        print(f"QCT-Bose hbar={hb} theta={th}: phi_end={r_q['phi_end']:.3f} iota-(quantum)={r_q['iota_minus']:.2f} level {r_q['level']}")
    me = []
    E = S.ts1["E"] + s["start_above_ts1"]
    specs = s["master_equation"] if not s["_quick"] else [x for x in s["master_equation"] if x["hbar"] >= 0.5][:1]
    for m in specs:                                       # the expensive part: minutes per case
        me.append(redfield_case(S, m["hbar"], s["N"], s["eta"], E, m["delta"], ecut=m["ecut"],
                                tau_obs=s["t_obs"] if not s["_quick"] else 400.0))
    summary(EXP, dict(qct_bose=qct, master_equation=me), args.quick, s)

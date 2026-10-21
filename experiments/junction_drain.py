#!/usr/bin/env python3
"""The junction with N thermal drain modes. Is the decision made after an electron-transfer crossing
kept by the same drain, and which rung does thermal mixing remove first? The adiabatic coupling is
propagated on the lower surface, the conical intersection with surface hopping."""
from _common import parse, settings, cached, pool_map, summary
from geometric_pes.drain.junction_drain import run
from geometric_pes.drain.metrics import summarize

EXP = "junction_drain"


def tag(cpl, g, th, s):
    return f"{cpl}_N{s['N']}_g{g:g}_th{th:g}"


def job(args):
    (cpl, g, th), s = args
    cached(EXP, tag(cpl, g, th, s), s["_quick"], lambda: run(
        cpl, s["N"], g, th, ntraj=s["ntraj"], nT=s["periods"], seed=s["seed"],
        dt_frac=s["dt_fraction"][cpl], wc=s["cutoff"]))


if __name__ == "__main__":
    args = parse(__doc__); s = settings(EXP, args.quick)
    if args.quick:
        s.update(gamma=s["gamma"][:1], theta=[0.0, 0.2], periods=4)
    J = [(c, g, th) for c in s["couplings"] for g in s["gamma"] for th in s["theta"]]
    pool_map(job, [(j, s) for j in J], args.nproc)
    out = []
    for cpl, g, th in J:
        params, a = cached(EXP, tag(cpl, g, th, s), s["_quick"], None)
        r = summarize(params, a)
        out.append(dict(coupling=cpl, gamma=g, theta=th, **r))
        print(f"{cpl:3s} gamma={g:<5g} theta={th:<5g} phi1={r['phi1']:.3f} phi_end={r['phi_end']:.3f} "
              f"iota-={r['iota_minus']:.1f} B={r['B']:.3f} {r['level']}")
    summary(EXP, out, args.quick, s)

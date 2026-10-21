#!/usr/bin/env python3
"""The decision after an electron-transfer crossing (the junction). Four parts, run with --part:

  distance     first-arrival split and transfer yield against the seam-to-ridge distance d, for each
               coupling and mass: two-state quantum (Q2), quantum on the lower surface (QBO),
               classical on the lower surface (CBO) and surface hopping (FSSH)
  transverse   quantum-classical gap against the transverse quanta n of the branching mode
  thermal      thermal branching mode: pure-state quantum runs summed with Boltzmann weights,
               against classical Wigner trajectories at the same x_y = hbar w_y / k_B T
  coherent     superpositions (|n1> + e^{i alpha}|n2>)/sqrt 2 of transverse states
"""
import argparse

import numpy as np

from _common import settings, cached_json, pool_map, summary
from geometric_pes.junction_decision import params, pset, grid, regrid, thermal_init, packet_xy, thermal
from geometric_pes.surfaces.junction import quantum, classical
from dataclasses import replace

EXP = "junction_decision"


def period(p):
    return 2 * np.pi / np.sqrt(p.kx / p.M)


# ----------------------------------------------------------------- distance
def distance_job(args):
    (d, cpl, M), s = args

    def run():
        p = params(d, cpl, M)
        G = grid(cpl, d, M)
        T = period(p)
        out = dict(d=d, coupling=cpl, M=M)
        for eng in ("Q2", "QBO"):
            r = quantum(p, G, two_state=(eng == "Q2"), t_final=s["periods"] * T, W0=s["w0"])
            out[eng] = dict(phi1=float(r["phi1"]), yield_=float(r["yield_"]))
        for eng, mode in (("CBO", "BO"), ("FSSH", "fssh")):
            r = classical(p, G, n=s["distance_ntraj"], mode=mode, t_final=s["periods"] * T, W0=s["w0"])
            out[eng] = dict(phi1=r["phi1"], err=r["phi1_err"], yield_=r["yield_"])
        return out
    return cached_json(EXP, f"distance_{cpl}_d{d:g}_M{M:g}", s["_quick"], run)


# ----------------------------------------------------------------- transverse quanta
def transverse_job(args):
    (d, M, n), s = args

    def run():
        p = replace(params(d, "ACs", M), transverse="ho", n_y=n, name=f"ACs_d{d:g}_M{M:g}_n{n}")
        G = regrid(grid("ACs", d, M), p)
        T = period(p)
        q = quantum(p, G, two_state=False, t_final=s["periods"] * T, W0=s["w0"])
        c = classical(p, G, n=s["transverse_ntraj"] * (2 if n else 1), mode="BO", t_final=s["periods"] * T,
                      W0=s["w0"], how="wigner")
        return dict(d=d, M=M, n=n, QBO=float(q["phi1"]), CBO_wigner=c["phi1"], err=c["phi1_err"],
                    gap=float(q["phi1"]) - c["phi1"])
    return cached_json(EXP, f"transverse_d{d:g}_M{M:g}_n{n}", s["_quick"], run)


# ----------------------------------------------------------------- thermal branching mode
def pure_job(args):
    (cpl, d, M, n), s = args

    def run():
        G = regrid(grid(cpl, d, M), pset(cpl, d, M, n, False))
        q = quantum(G.p, G, two_state=(cpl == "CI"), t_final=s["periods"] * period(G.p), W0=s["w0"])
        return dict(n=n, A_plus=float(q["abs_p"][-1]), A_minus=float(q["abs_m"][-1]), phi1=float(q["phi1"]))
    return cached_json(EXP, f"pure_{cpl}_d{d:g}_M{M:g}_n{n}", s["_quick"], run)


def thermal_classical_job(args):
    (cpl, d, M, xy), s = args

    def run():
        G = regrid(grid(cpl, d, M), pset(cpl, d, M))
        rng = np.random.default_rng(s["thermal_seed"])
        ic = thermal_init(G, s["thermal_ntraj"], xy, "wigner", rng)
        c = classical(G.p, G, mode="BO", t_final=s["periods"] * period(G.p), W0=s["w0"], init=ic,
                      seed=s["thermal_seed"] + 1)
        return dict(xy=xy, phi1=c["phi1"], err=c["phi1_err"])
    return cached_json(EXP, f"thermalcl_{cpl}_d{d:g}_M{M:g}_xy{xy:g}", s["_quick"], run)


# ----------------------------------------------------------------- coherent superpositions
def coherent_job(args):
    (n1, n2, alpha), s = args
    c0 = s["coherent_case"]
    cpl, d, M = c0["coupling"], c0["d"], c0["M"]

    def run():
        G = regrid(grid(cpl, d, M), pset(cpl, d, M))
        p = G.p
        comps = [(1 / np.sqrt(2), n1, p.E_kin0), (np.exp(1j * alpha) / np.sqrt(2), n2, p.E_kin0)]
        q = quantum(p, G, two_state=(cpl == "CI"), t_final=s["periods"] * period(p), W0=s["w0"],
                    psi0=packet_xy(G, comps))
        return dict(n1=n1, n2=n2, alpha=alpha, phi1=float(q["phi1"]))
    return cached_json(EXP, f"coherent_{cpl}_{n1}{n2}_a{alpha:.4f}", s["_quick"], run)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", default="all", choices=["distance", "transverse", "thermal", "coherent", "all"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--nproc", type=int, default=None)
    args = ap.parse_args()
    s = settings("junction", args.quick)
    if args.quick:                                       # one small case of each part
        s.update(distances=[2.0], distance_couplings=["CI"], masses=[16.0], distance_ntraj=300,
                 transverse_cases=[[2.0, 16.0]], transverse_n=[0, 1], transverse_ntraj=300,
                 thermal_cases=[dict(coupling="ACs", d=2.0, M=16.0, nmax=3)], thermal_xy=[2.0, 5.0],
                 thermal_ntraj=300, coherent_pairs=[[0, 1]], coherent_phases=2, periods=4.0)
    parts = ["distance", "transverse", "thermal", "coherent"] if args.part == "all" else [args.part]
    out = {}
    cases = {(c, d, M) for c in s["distance_couplings"] for d in s["distances"] for M in s["masses"]} \
        if "distance" in parts else set()
    cases |= {("ACs", d, M) for d, M in s["transverse_cases"]} if "transverse" in parts else set()
    cases |= {(c["coupling"], c["d"], c["M"]) for c in s["thermal_cases"]} if "thermal" in parts else set()
    if "coherent" in parts:
        c0 = s["coherent_case"]; cases.add((c0["coupling"], c0["d"], c0["M"]))
    for c in cases:
        grid(*c)                                          # build grid caches before forking
    if "distance" in parts:
        J = [(d, c, M) for M in s["masses"] for c in s["distance_couplings"] for d in s["distances"]]
        out["distance"] = pool_map(distance_job, [(j, s) for j in J], args.nproc)
        for r in out["distance"]:
            print(f"{r['coupling']:3s} d={r['d']:+.1f} M={r['M']:g} " + " ".join(
                f"{e}={r[e]['phi1']:.3f}/{r[e]['yield_']:.3f}" for e in ("Q2", "QBO", "CBO", "FSSH")))
    if "transverse" in parts:
        J = [(d, M, n) for d, M in s["transverse_cases"] for n in s["transverse_n"]]
        out["transverse"] = pool_map(transverse_job, [(j, s) for j in J], args.nproc)
        for r in out["transverse"]:
            print(f"transverse d={r['d']:g} M={r['M']:g} n={r['n']}: gap {r['gap']:+.3f}")
    if "thermal" in parts:
        out["thermal"] = {}
        for c in s["thermal_cases"]:
            cpl, d, M = c["coupling"], c["d"], c["M"]
            pure = pool_map(pure_job, [((cpl, d, M, n), s) for n in range(c["nmax"] + 1)], args.nproc)
            cl = pool_map(thermal_classical_job, [((cpl, d, M, xy), s) for xy in s["thermal_xy"]], args.nproc)
            n = np.array([r["n"] for r in pure]); Ap = np.array([r["A_plus"] for r in pure])
            Am = np.array([r["A_minus"] for r in pure])
            rows = []
            for r in cl:
                t = thermal(n, Ap, Am, r["xy"])
                ok = t["tail"] < 5e-3                    # thermal tail beyond nmax: not converged
                rows.append(dict(xy=r["xy"], Q=t["phi"] if ok else None, C=r["phi1"], err=r["err"],
                                 gap=(t["phi"] - r["phi1"]) if ok else None, tail=t["tail"]))
                print(f"thermal {cpl} x_y={r['xy']:<4g} gap " + (f"{t['phi'] - r['phi1']:+.3f}" if ok else "(tail)"))
            out["thermal"][f"{cpl}_d{d:g}_M{M:g}"] = dict(pure=pure, rows=rows)
    if "coherent" in parts:
        al = np.arange(s["coherent_phases"]) * 2 * np.pi / s["coherent_phases"]
        J = [(n1, n2, float(a)) for n1, n2 in s["coherent_pairs"] for a in al]
        out["coherent"] = pool_map(coherent_job, [(j, s) for j in J], args.nproc)
        for r in out["coherent"]:
            print(f"coherent ({r['n1']},{r['n2']}) alpha={r['alpha']:.2f}: phi1={r['phi1']:.3f}")
    summary(EXP if args.part == "all" else f"{EXP}_{args.part}", out, args.quick, s)

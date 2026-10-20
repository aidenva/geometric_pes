"""Randomly drawn configurations for the out-of-sample test: surfaces (perturbations of surface A),
start energies, spectra, coupling patterns and drain sizes. The ranges are set in
config/experiments.toml [generalization]; test_configs(seed, prefix) is deterministic.
"""
from dataclasses import replace

import numpy as np

from .config import surface, section
from .surfaces.basins import Setup

def draw_surface(i, rng, g=None):
    g = g or section("generalization")
    p = replace(surface("A"), name=f"gen{i:03d}",
                kP=rng.uniform(*g["kP"]), b_plus=rng.uniform(*g["b_plus"]), b_minus=rng.uniform(*g["b_minus"]),
                eps=rng.uniform(*g["eps"]), A_c=rng.uniform(*g["A_c"]), Delta=rng.uniform(*g["Delta"]))
    return p, float(rng.uniform(*g["start_above_ts1"]))


def valid(p, dE):
    """three basins R, P+, P-; a P+/P- saddle; exchange possible at the start energy; real wells"""
    try:
        S = Setup(p)
    except Exception:
        return False
    ts12 = [s for s in S.sads if set(s["connects"]) == {"P+", "P-"}]
    if sorted(S.names) != ["P+", "P-", "R"] or not ts12:
        return False
    E = S.ts1["E"] + dE; e12 = ts12[0]["E"]
    depth = e12 - max(S.mins[S.iPp]["E"], S.mins[S.iPm]["E"])
    g = section("generalization")
    return E - e12 >= g["min_exchange_margin"] and depth >= g["min_well_depth"]


def test_configs(seed=2026, prefix="gen"):
    """parameters only (cheap and deterministic); validity is checked in the workers.
    seed 2026: the first (development) test set; seed 7: the fresh validation set"""
    g = section("generalization")
    rng = np.random.default_rng(seed)
    out = []
    for i in range(g["n_configs"]):
        p, dE = draw_surface(i, rng, g)
        p = replace(p, name=f"{prefix}{i:03d}")
        lo, hi = g["reorganization"]
        out.append(dict(i=i, surface=p, dE=dE, kind=str(rng.choice(g["spectra"])),
                        pattern=str(rng.choice(g["patterns"])), N=int(rng.choice(g["N"])),
                        Lam=float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),
                        wc=float(rng.choice(g["cutoff"])), seed=int(rng.integers(1e9))))
    return out

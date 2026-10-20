"""Shared helpers for the experiment drivers: config access, a process pool, raw-run caching and
summaries. Every driver accepts --quick (few trajectories, short runs) for a smoke test."""
import argparse
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometric_pes.config import load, section, results_dir           # noqa: E402
from geometric_pes.io import save_npz, load_npz, save_json             # noqa: E402

QUICK = dict(ntraj=60, t_obs=40.0, n_configs=6)


def parse(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--quick", action="store_true", help="smoke test: few trajectories, short runs")
    ap.add_argument("--nproc", type=int, default=None)
    return ap.parse_args()


def settings(name, quick):
    """the experiment's section merged with the shared [run] defaults (quick: scaled down)"""
    s = dict(load()["run"]); s.update(section(name))
    if quick:
        s["ntraj"] = min(s.get("ntraj", QUICK["ntraj"]), QUICK["ntraj"])
        s["t_obs"] = QUICK["t_obs"]
        if "n_configs" in s:
            s["n_configs"] = QUICK["n_configs"]
    s["_quick"] = quick
    return s


def raw_path(experiment, tag, quick):
    return os.path.join(results_dir("quick" if quick else "full", experiment), tag + ".npz")


def cached(experiment, tag, quick, fn):
    """run fn() -> (params, arrays) once and keep the raw arrays; return (params, arrays)"""
    path = raw_path(experiment, tag, quick)
    if not os.path.exists(path):
        params, arrays = fn()
        save_npz(path, params, **arrays)
    a, meta = load_npz(path)
    return meta["params"], a


def pool_map(fn, jobs, nproc):
    nproc = nproc or load()["run"]["nproc"]
    if nproc <= 1:
        return [fn(j) for j in jobs]
    with Pool(nproc) as pool:
        return pool.map(fn, jobs)


def summary(experiment, data, quick, settings_used):
    path = os.path.join(results_dir("quick" if quick else "full"), f"{experiment}.json")
    save_json(path, data, meta=dict(experiment=experiment, settings=settings_used))
    print("wrote", path)
    return path

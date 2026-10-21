#!/usr/bin/env python3
"""Edge cases of the a priori criterion. Every case it misclassifies in the generalization test is
rerun with full recording, together with the correctly classified case nearest in
(log chi_ex, log f_ret) with the same prediction, to see what actually happened to the decision:
when the switches occur, and where the energy sits (x or y) while it is above the saddle.
Run experiments/generalization.py first."""
import json
import os

import numpy as np

from _common import parse, settings, cached, pool_map, summary, results_dir
from geometric_pes.drain.dynamics import run_rest
from geometric_pes.drain.modes import modes
from geometric_pes.random_configs import test_configs

EXP = "edge_cases"
CRITERION = "chi_ex + return"


def select(gen):
    v = gen["criteria"][CRITERION]
    cut, fcut = v["cut"], v["f_cut"]
    rows = gen["rows"]["dev"] + gen["rows"]["fresh"]
    for r in rows:
        r["pred"] = bool(r["chi_ex"] < cut and r["f_ret"] < fcut)
        r["ok"] = r["pred"] == (r["r"] >= 0.5)
    bad = [r for r in rows if not r["ok"]]; good = [r for r in rows if r["ok"]]
    d = lambda a, b: np.hypot(np.log(a["chi_ex"] / b["chi_ex"]),
                              np.log(max(a["f_ret"], 1e-3) / max(b["f_ret"], 1e-3)))
    used, pairs = set(), []
    for r in bad:
        c = min((g for g in good if g["pred"] == r["pred"] and g["tag"] not in used), key=lambda g: d(r, g))
        used.add(c["tag"]); pairs.append((r, c))
    return pairs


def config_of(tag, gen_settings):
    which, i = tag[:-3], int(tag[-3:])
    seed = gen_settings["dev_seed"] if which == "dev" else gen_settings["fresh_seed"]
    return [c for c in test_configs(seed, which, i + 1) if c["i"] == i][0]


def job(args):
    tag, s, g = args
    c = config_of(tag, g)
    md = modes(c["kind"], c["pattern"], c["N"], c["Lam"], c["wc"], np.random.default_rng(c["seed"]))
    cached(EXP, tag, s["_quick"], lambda: run_rest(
        "A", c["N"], 0.0, dE=c["dE"], ntraj=s["ntraj"], t_final=s["t_obs"], seed=c["seed"] % 100000,
        surface=c["surface"], modes=md, record=True, y_share=s["hot_y_share"]))


def diagnose(p, a):
    iPp, iPm = p["iPp"], p["iPm"]
    got = (a["first"] >= 0) & np.isin(a["last"], [iPp, iPm])
    f = (a["first"][got] == iPp).astype(float); l = (a["last"][got] == iPp).astype(float)
    r = (np.mean((f - f.mean()) * (l - l.mean())) / (f.std() * l.std())
         if f.std() > 0 and l.std() > 0 else np.nan)
    ev = a["events"]; ser = a["series"]                    # ser: (time, [e2,Kx,Ky,Eq,lab], traj)
    tf = a["t_first"]
    T = p["t_final"]; E12 = p["E_TS12"]
    # time from first arrival to each switch, and the energy split at the switch
    k = ev[:, 0].astype(int); dt_sw = ev[:, 1] - tf[k]
    ky_share = ev[:, 4] / np.maximum(ev[:, 3] + ev[:, 4], 1e-12)
    eq_sw = ev[:, 5]
    flipped = got & (a["first"] != a["last"])              # trajectories that end in the other product
    first_sw = {}
    for kk, t in zip(k, ev[:, 1]):
        first_sw.setdefault(kk, t)
    fs_dt = np.array([first_sw[j] - tf[j] for j in np.nonzero(flipped)[0] if j in first_sw])
    # time above the saddle after the first drop below it, and the y share of the energy then
    tt = np.arange(1, ser.shape[0] + 1) * 1.0
    e2, kx, ky = ser[:, 0], ser[:, 1], ser[:, 2]
    tb = a["t_below"]
    after = tt[:, None] > np.where(np.isfinite(tb), tb, np.inf)[None, :]
    above = after & (e2 >= E12)
    t_above = above.sum() * 1.0
    n_late_sw = np.sum(ev[:, 1] > np.where(np.isfinite(tb[k]), tb[k], np.inf))
    return dict(r=float(r), n_traj=int(got.sum()),
                switches_per_traj=float(len(ev) / max(got.sum(), 1)),
                frac_ever_switch=float(np.mean(np.isin(np.nonzero(got)[0], k))),
                median_switch_delay=float(np.median(dt_sw)) if len(ev) else np.nan,
                frac_switch_within_5=float(np.mean(dt_sw < 5)) if len(ev) else np.nan,
                flipped=float(flipped[got].mean()),
                flips_first_switch_within_5=float(np.mean(fs_dt < 5)) if len(fs_dt) else np.nan,
                ky_share_at_switch=float(np.median(ky_share)) if len(ev) else np.nan,
                eq_at_switch=float(np.nanmedian(eq_sw)) if len(ev) else np.nan,
                time_above_after_drop=float(t_above / max(got.sum(), 1)),
                late_switches_per_time_above=float(n_late_sw / max(t_above, 1e-9)),
                ky_share_when_above=float(np.median((ky / np.maximum(kx + ky, 1e-12))[above])) if above.any() else np.nan)


if __name__ == "__main__":
    args = parse(__doc__); s = settings(EXP, args.quick)
    g = settings("generalization", args.quick)
    path = os.path.join(results_dir("quick" if args.quick else "full"), "generalization.json")
    gen = json.load(open(path)); gen = gen.get("data", gen)
    pairs = select(gen)
    tags = sorted({t for r, c in pairs for t in (r["tag"], c["tag"])})
    pool_map(job, [(t, s, g) for t in tags], args.nproc)
    groups = {"missed_survival": [], "control_for_missed": [], "false_survival": [], "control_for_false": []}
    for r, c in pairs:
        kind = "false_survival" if r["pred"] else "missed_survival"
        ctrl = "control_for_false" if r["pred"] else "control_for_missed"
        for grp, row in ((kind, r), (ctrl, c)):
            params, a = cached(EXP, row["tag"], s["_quick"], None)
            groups[grp].append(dict(tag=row["tag"], chi_ex=row["chi_ex"], f_ret=row["f_ret"], N=row["N"],
                                    pattern=row["pattern"], kind=row["kind"], **diagnose(params, a)))
    keys = ("r", "switches_per_traj", "time_above_after_drop", "late_switches_per_time_above",
            "ky_share_when_above", "ky_share_at_switch")
    for grp, rows in groups.items():
        if rows:
            print(f"{grp:20s}", {k: round(float(np.nanmedian([x[k] for x in rows])), 4) for k in keys})
    summary(EXP, groups, args.quick, s)

#!/usr/bin/env python3
"""Does the geometry-drain competition predict the outcome beyond the surfaces it was found on?

Training grid: the timescales runs (four surfaces, ohmic modes on x and y). Test sets: random
surfaces, spectra, coupling patterns and drain sizes (geometric_pes.random_configs); a first set
(dev_seed) to choose predictors and a fresh set (fresh_seed) used only for scoring. Every threshold
is fitted on the training grid by balanced accuracy. Predictors use the energy flow only:

  chi + return      measured tau_drain / tau_exchange, and return fraction f_ret
  chi_ex + return   a-priori friction on the exchange coordinate, and f_ret
  chi_ex            a-priori alone
  hot time          time after the decision with the x, y energy above TS12 and at least y_share
                    of the kinetic energy in y, over tau_exchange
  above time        the same without the condition on y
Outcome: the decision survives if the trajectory memory r >= survive_r.
"""
import numpy as np

from _common import parse, settings, cached, pool_map, summary, load
from geometric_pes.config import surface
from geometric_pes.surfaces.basins import Setup
from geometric_pes.drain.dynamics import run_rest
from geometric_pes.drain.metrics import metrics, exchange_time
from geometric_pes.drain.modes import modes, gamma_exchange, caldeira_leggett_modes
from geometric_pes.random_configs import test_configs, valid
import timescales

EXP = "generalization"


# ----------------------------------------------------------------- runs
def test_job(args):
    c, which, s = args
    if not valid(c["surface"], c["dE"]):
        return
    md = modes(c["kind"], c["pattern"], c["N"], c["Lam"], c["wc"], np.random.default_rng(c["seed"]))
    kw = dict(dE=c["dE"], ntraj=s["ntraj"], t_final=s["t_obs"], seed=c["seed"] % 100000, surface=c["surface"],
              y_share=s["hot_y_share"])
    cached(EXP, f"{which}{c['i']:03d}", s["_quick"], lambda: run_rest("A", c["N"], 0.0, modes=md, **kw))
    cached(EXP, f"{which}{c['i']:03d}_bare", s["_quick"], lambda: run_rest("A", 2, 1e-12, **kw))


def train_rows(s_ts):
    bare, rows = timescales.rows(s_ts)
    out = []
    for r in rows:
        Nh = r["N"] // 2
        w, c = caldeira_leggett_modes(Nh, r["eta"], 1.0, s_ts["ohmic_cutoff"])
        md = (np.concatenate([w, w]), np.concatenate([c, c]), np.r_[np.zeros(Nh, int), np.ones(Nh, int)])
        out.append(predictors(r, md, Setup(surface(r["key"])), r["tau_exchange"]))
    return out


def test_rows(which, seed, s):
    out = []
    for c in test_configs(seed, which):
        try:
            params, a = cached(EXP, f"{which}{c['i']:03d}", s["_quick"], None)
            pb, ab = cached(EXP, f"{which}{c['i']:03d}_bare", s["_quick"], None)
        except FileNotFoundError:
            continue                                      # invalid surface
        m = metrics(params, a)
        md = modes(c["kind"], c["pattern"], c["N"], c["Lam"], c["wc"], np.random.default_rng(c["seed"]))
        r = dict(tag=f"{which}{c['i']:03d}", kind=c["kind"], pattern=c["pattern"], N=c["N"], Lam=c["Lam"],
                 wc=c["wc"], dE=c["dE"], **m)
        out.append(predictors(r, md, Setup(c["surface"]), float(exchange_time(pb, ab))))
    return out


def predictors(r, md, S, tx):
    ge = gamma_exchange(*md, S)
    r = dict(r)
    r["tau_exchange"] = tx
    r["chi"] = r["tau_drain"] / tx if np.isfinite(r["tau_drain"]) else float("inf")
    r["chi_ex"] = 1.0 / (ge * tx) if ge > 0 else float("inf")
    r["hot"] = r["t_co05_mean"] / tx
    r["above"] = r["t_abv_mean"] / tx
    return r


# ----------------------------------------------------------------- scoring
CRITERIA = {
    "chi + return": ("chi", True),
    "chi_ex + return": ("chi_ex", True),
    "chi_ex": ("chi_ex", False),
    "hot exchange-coordinate time": ("hot", False),
    "time above the saddle": ("above", False),
}


def scores(rows, pred, rcut):
    y = np.array([r["r"] >= rcut for r in rows]); p = np.array([pred(r) for r in rows])
    if not len(rows):
        return {}
    tpr = (p & y).sum() / max(y.sum(), 1); tnr = (~p & ~y).sum() / max((~y).sum(), 1)
    return dict(n=len(rows), survive=int(y.sum()), accuracy=float(np.mean(p == y)), tpr=float(tpr),
                tnr=float(tnr), balanced=float((tpr + tnr) / 2), baseline=float(max(y.mean(), 1 - y.mean())))


def auc(rows, key, rcut):
    y = np.array([r["r"] >= rcut for r in rows]); s = -np.array([r[key] for r in rows], float)
    if y.all() or not y.any():
        return float("nan")
    pos, neg = s[y], s[~y]
    return float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))


def fit(rows, key, use_f, rcut):
    best = (-1.0, None, None)
    for cs in np.exp(np.linspace(np.log(1e-3), np.log(1e3), 300)):
        for fs in (np.linspace(0.02, 0.6, 59) if use_f else [np.inf]):
            b = scores(rows, lambda r: r[key] < cs and r["f_ret"] < fs, rcut)["balanced"]
            if b > best[0] + 1e-12:
                best = (b, float(cs), float(fs))
    return best[1], best[2]


def evaluate(sets, rcut):
    res = {}
    for name, (key, use_f) in CRITERIA.items():
        cs, fs = fit(sets["train"], key, use_f, rcut)
        pred = lambda r, cs=cs, fs=fs, key=key: r[key] < cs and r["f_ret"] < fs
        res[name] = dict(cut=cs, f_cut=fs if use_f else None,
                         **{k: dict(scores(v, pred, rcut), auc=auc(v, key, rcut)) for k, v in sets.items()})
    return res


if __name__ == "__main__":
    args = parse(__doc__); s = settings(EXP, args.quick)
    s_ts = dict(load()["run"]); s_ts.update(load()["timescales"])
    if args.quick:
        s_ts.update(ntraj=s["ntraj"], t_obs=s["t_obs"], surfaces=["A", "Aup1"], N=[16, 64], eta=[0.2, 1.0])
    s_ts["_quick"] = args.quick
    for k in s_ts["surfaces"]:
        Setup(surface(k))
    pool_map(timescales.job, [(j, s_ts) for j in timescales.jobs(s_ts)], args.nproc)    # training grid
    sets = {}
    for which, seed in (("dev", s["dev_seed"]), ("fresh", s["fresh_seed"])):
        cfgs = test_configs(seed, which)
        pool_map(test_job, [(c, which, s) for c in cfgs], args.nproc)
    sets["train"] = [r for r in train_rows(s_ts) if np.isfinite(r["r"])]
    for which, seed in (("dev", s["dev_seed"]), ("fresh", s["fresh_seed"])):
        sets[which] = [r for r in test_rows(which, seed, s) if np.isfinite(r["r"])]
    res = evaluate(sets, s["survive_r"])
    for name, v in res.items():
        print(f"{name}: cut {v['cut']:.3g}" + (f", f_ret < {v['f_cut']:.2f}" if v["f_cut"] else ""))
        for k in ("train", "dev", "fresh"):
            x = v[k]
            if x.get("n"):
                print(f"   {k:5s} n={x['n']:3d} survive={x['survive']:3d} balanced={x['balanced']:.2f} "
                      f"TPR={x['tpr']:.2f} TNR={x['tnr']:.2f} AUC={x['auc']:.2f}")
    summary(EXP, dict(criteria=res, rows=sets), args.quick, s)

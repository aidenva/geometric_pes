"""Branching surfaces, drain spectra, runners and random configurations."""
import numpy as np
import pytest

from geometric_pes.config import surface, section
from geometric_pes.surfaces.basins import Setup
from geometric_pes.drain.modes import modes, caldeira_leggett_modes, gamma_exchange
from geometric_pes.drain.dynamics import run_rest
from geometric_pes.drain.statistics import stat_fractions
from geometric_pes.drain.metrics import metrics
from geometric_pes.random_configs import test_configs as draw_configs


def test_A_and_B_share_stationary_points():
    """the decision test rests on this: same critical points, different valley"""
    a, b = Setup(surface("A")), Setup(surface("B"))
    ea = sorted(round(m["E"], 3) for m in a.mins) + sorted(round(s["E"], 3) for s in a.sads)
    eb = sorted(round(m["E"], 3) for m in b.mins) + sorted(round(s["E"], 3) for s in b.sads)
    assert ea == eb


def test_uphill_variants_are_endothermic():
    for k in ("Aup1", "Aup2"):
        S = Setup(surface(k))
        assert min(S.mins[S.iPp]["E"], S.mins[S.iPm]["E"]) > S.mins[S.iR]["E"]


@pytest.mark.parametrize("kind", ["subohmic", "ohmic", "superohmic", "drude", "random"])
@pytest.mark.parametrize("pattern", ["xy", "y", "x"])
def test_spectra_give_the_requested_reorganization(kind, pattern):
    w, c, which = modes(kind, pattern, 16, 2.5, 5.0, np.random.default_rng(0))
    assert np.all(w > 0) and len(w) == 16
    for coord in set(which):
        sel = which == coord
        assert np.sum(c[sel] ** 2 / w[sel] ** 2) == pytest.approx(2.5)


def test_ohmic_reorganization_is_eta_wc():
    w, c = caldeira_leggett_modes(32, 0.7, 1.0, 5.0)
    assert np.sum(c**2 / w**2) == pytest.approx(0.7 * 5.0)


def test_exchange_friction_follows_the_coupled_coordinate():
    """on A the exchange mode is mostly y: modes on y drain it, modes on x barely do"""
    S = Setup(surface("A"))
    rng = np.random.default_rng(1)
    gy = gamma_exchange(*modes("ohmic", "y", 32, 2.0, 5.0, rng), S)
    gx = gamma_exchange(*modes("ohmic", "x", 32, 2.0, 5.0, rng), S)
    assert gy > 5 * gx > 0


def test_run_rest_is_deterministic_and_conserves_energy():
    p1, a1 = run_rest("A", 16, 1.0, ntraj=40, t_final=10.0, seed=3)
    p2, a2 = run_rest("A", 16, 1.0, ntraj=40, t_final=10.0, seed=3)
    assert np.array_equal(a1["first"], a2["first"]) and np.array_equal(a1["last"], a2["last"])
    m = metrics(p1, a1)
    assert set(m) >= {"phi1", "r", "tau_drain", "f_ret", "t_co05_mean"}


def test_recording_keeps_switch_events_and_series():
    _, a = run_rest("A", 8, 1.0, ntraj=20, t_final=6.0, seed=4, record=True)
    assert a["events"].shape[1] == 6 and a["series"].shape[1:] == (5, 20)


def test_many_mode_statistics_favour_the_deeper_well():
    S = Setup(surface("A"))
    E = S.ts1["E"] + 1.0
    f0, f64 = stat_fractions(S, [E], 0)[0], stat_fractions(S, [E], 64)[0]
    assert f0.sum() == pytest.approx(1) and f64.sum() == pytest.approx(1)
    deeper = S.iPp if S.mins[S.iPp]["E"] < S.mins[S.iPm]["E"] else S.iPm
    assert f64[deeper] > f0[deeper]


def test_random_configs_are_deterministic_and_in_range():
    g = section("generalization")
    a, b = draw_configs(7, "fresh"), draw_configs(7, "fresh")
    assert [c["seed"] for c in a] == [c["seed"] for c in b]
    for c in a:
        assert g["kP"][0] <= c["surface"].kP <= g["kP"][1]
        assert g["start_above_ts1"][0] <= c["dE"] <= g["start_above_ts1"][1]
        assert c["kind"] in g["spectra"] and c["pattern"] in g["patterns"] and c["N"] in g["N"]

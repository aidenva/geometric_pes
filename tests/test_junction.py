"""The electron-transfer junction: cases, initial states, thermal sums and the drain runner."""
import numpy as np
import pytest

from geometric_pes.junction_decision import params, pset, thermal, COUPLINGS
from geometric_pes.drain.junction_drain import run


def test_ridge_sits_downstream_of_the_seam():
    for cpl in COUPLINGS:
        for d in (0.5, 2.0):
            p = params(d, cpl, 16.0)
            assert p.x_r - p.x_s == pytest.approx(d)


def test_transverse_state_is_set():
    p = pset("ACs", 2.0, 16.0, n=3)
    assert p.n_y == 3 and p.transverse == "ho"


def test_thermal_sum_limits():
    """cold limit: the n = 0 result; the weights are Boltzmann in n"""
    n = np.arange(30)
    Ap = np.where(n % 2 == 0, 0.6, 0.4); Am = 1 - Ap
    assert thermal(n, Ap, Am, 50.0)["phi"] == pytest.approx(0.6)
    t = thermal(n, Ap, Am, 1.0)
    assert 0.4 < t["phi"] < 0.6 and t["tail"] < 1e-10


def test_junction_drain_runs_and_conserves_energy():
    prm, a = run("ACs", 8, 0.05, 0.0, ntraj=20, nT=1, seed=1)
    assert np.max(np.abs(a["dE"])) < 0.05
    assert set(a) >= {"first", "last", "stat", "e2d"}

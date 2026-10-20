"""Stationary points of the default surface and conservation laws of the propagator."""
import dataclasses

import numpy as np
import pytest

from geometric_pes.two_state import P, E_pm, Landscape, propagate, p_stat


@pytest.fixture(scope="module")
def default_landscape():
    return Landscape(P)


def test_default_stationary_points(default_landscape):
    L = default_landscape
    assert len(L.mins) == 2 and len(L.saddles) == 2
    assert L.E_r == pytest.approx(0.103, abs=1e-3)
    assert L.E_p == pytest.approx(-0.381, abs=1e-3)
    assert sorted(L.E_saddles) == pytest.approx([0.302, 0.360], abs=1e-3)


def test_conical_intersection_present():
    # linear coupling: the two surfaces touch near (-0.25, -0.05) (WORKFLOW.md Step 1)
    X, Y = np.meshgrid(np.linspace(-0.4, -0.1, 301), np.linspace(-0.2, 0.1, 301))
    Em, Ep, *_ = E_pm(X, Y, P)
    assert (Ep - Em).min() < 5e-3


def test_propagation_conserves_norm_and_energy():
    p = dataclasses.replace(P, prop_nx=128, prop_ny=128, t_final=200.0, sample_every=40)
    h = propagate(p)
    np.testing.assert_allclose(h["norm"], 1.0, atol=1e-10)
    assert abs(h["dE"]) < 1e-3
    assert h["P_edge"].max() < 1e-6
    np.testing.assert_allclose(h["P1"] + h["P2"], h["norm"], atol=1e-12)
    np.testing.assert_allclose(h["Pminus"] + h["Pplus"], h["norm"], atol=1e-12)
    assert abs(h["C"][0] - 1.0) < 1e-10


@pytest.mark.slow
def test_microcanonical_baseline_reproduces_step2(default_landscape):
    # WORKFLOW.md Step 2 / foundations track: P_stat = 0.718 at E = 0.455
    assert p_stat(default_landscape, 0.455) == pytest.approx(0.718, abs=0.005)

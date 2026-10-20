"""Integrators reproduce the right ensembles."""
import numpy as np
import pytest

from geometric_pes.integrators import baoab_step, brownian_step, thermostat, caldeira_leggett_modes

M, W = 2.0, 1.5                       # harmonic oscillator mass and frequency
force = lambda q: -M * W**2 * q


def _energy_error(dt, t=20.0):
    q, p = np.array([[1.0]]), np.array([[0.0]])
    f = force(q)
    E0 = 0.5 * p**2 / M + 0.5 * M * W**2 * q**2
    err = 0.0
    for _ in range(int(round(t / dt))):
        q, p, f = baoab_step(q, p, f, force, M, dt)
        err = max(err, abs((0.5 * p**2 / M + 0.5 * M * W**2 * q**2 - E0).item()))
    return err


def test_baoab_without_friction_is_symplectic_second_order():
    e1, e2 = _energy_error(0.02), _energy_error(0.01)
    assert e1 < 1e-3
    assert e1 / e2 == pytest.approx(4.0, rel=0.1)


def test_baoab_samples_boltzmann():
    kT, n = 0.3, 20000
    rng = np.random.default_rng(0)
    q, p = np.zeros((1, n)), np.zeros((1, n))
    f = force(q)
    for _ in range(3000):
        q, p, f = baoab_step(q, p, f, force, M, 0.05, gamma=2.0, kT=kT, rng=rng)
    assert np.mean(q**2) == pytest.approx(kT / (M * W**2), rel=0.03)
    assert np.mean(p**2) == pytest.approx(M * kT, rel=0.03)


def test_brownian_samples_boltzmann():
    kT, n, gamma = 0.3, 20000, 20.0
    rng = np.random.default_rng(1)
    q = np.zeros(n)
    for _ in range(3000):
        q = brownian_step(q, force(q), M, gamma, kT, 0.01, rng)
    assert np.mean(q**2) == pytest.approx(kT / (M * W**2), rel=0.04)


def test_thermostat_rows_leave_other_dofs_alone():
    rng = np.random.default_rng(2)
    p = np.ones((3, 10))
    m = np.array([[1.0], [2.0], [3.0]])
    out = thermostat(p, m, 0.5, 1.0, rng, rows=[1])
    np.testing.assert_array_equal(out[[0, 2]], p[[0, 2]])
    assert not np.allclose(out[1], p[1])


def test_caldeira_leggett_reorganization_sum():
    # sum_j c_j^2 / w_j^2 = (2/pi) int J(w)/w dw = eta wc for the ohmic density
    gamma, M_, wc = 1e-3, 1836.0, 0.05
    w, c = caldeira_leggett_modes(64, gamma, M_, wc)
    assert np.all(w > 0) and np.all(np.diff(w) < 0)
    assert np.sum(c**2 / w**2) == pytest.approx(M_ * gamma * wc, rel=1e-12)

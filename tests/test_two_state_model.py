"""Model invariants: analytic gradients, eigenvectors and the 2x2 propagator."""
import dataclasses

import numpy as np
import pytest
from scipy.linalg import expm

from geometric_pes.two_state.model import (P, E_pm, grad_Eminus, surfaces_and_grads, lower_eigvec,
                          propagator_2x2)

COUPLINGS = [P, dataclasses.replace(P, coupling="gaussian", Delta0=0.2)]


def _points(p, n=200, seed=0, min_gap=0.05):
    rng = np.random.default_rng(seed)
    x = rng.uniform(p.x_min, p.x_max, n)
    y = rng.uniform(p.y_min, p.y_max, n)
    Em, Ep, *_ = E_pm(x, y, p)
    keep = Ep - Em > min_gap          # stay away from the conical intersection cusp
    return x[keep], y[keep]


def _fd(fun, x, y, h=1e-6):
    return ((fun(x + h, y) - fun(x - h, y)) / (2 * h),
            (fun(x, y + h) - fun(x, y - h)) / (2 * h))


@pytest.mark.parametrize("p", COUPLINGS, ids=["linear", "gaussian"])
def test_grad_Eminus_matches_finite_differences(p):
    x, y = _points(p)
    gx, gy = grad_Eminus(np.array([x, y]), p)
    fx, fy = _fd(lambda a, b: E_pm(a, b, p)[0], x, y)
    np.testing.assert_allclose(gx, fx, atol=1e-6)
    np.testing.assert_allclose(gy, fy, atol=1e-6)


@pytest.mark.parametrize("p", COUPLINGS, ids=["linear", "gaussian"])
def test_surfaces_and_grads(p):
    x, y = _points(p)
    s = surfaces_and_grads(x, y, p)
    gx, gy = grad_Eminus(np.array([x, y]), p)
    np.testing.assert_allclose(s["gm"][0], gx, atol=1e-12)
    np.testing.assert_allclose(s["gm"][1], gy, atol=1e-12)
    fx, fy = _fd(lambda a, b: E_pm(a, b, p)[1], x, y)
    np.testing.assert_allclose(s["gp"][0], fx, atol=1e-6)
    np.testing.assert_allclose(s["gp"][1], fy, atol=1e-6)
    # NAC = grad(theta); test away from the arctan2 branch cut (d < 0, D ~ 0)
    ok = surfaces_and_grads(x, y, p)["v1"] > surfaces_and_grads(x, y, p)["v2"]
    theta = lambda a, b: surfaces_and_grads(a, b, p)["theta"]
    tx, ty = _fd(theta, x[ok], y[ok])
    np.testing.assert_allclose(s["nac"][0][ok], tx, atol=1e-5)
    np.testing.assert_allclose(s["nac"][1][ok], ty, atol=1e-5)


def test_lower_eigvec_is_normalized_eigenvector():
    x, y = _points(P)
    Em, Ep, v1, v2, D, *_ = E_pm(x, y, P)
    u1, u2 = lower_eigvec(v1, v2, D, Em)
    np.testing.assert_allclose(u1**2 + u2**2, 1.0, atol=1e-12)
    np.testing.assert_allclose(v1 * u1 + D * u2, Em * u1, atol=1e-10)
    np.testing.assert_allclose(D * u1 + v2 * u2, Em * u2, atol=1e-10)


def test_propagator_2x2_is_exact_and_unitary():
    rng = np.random.default_rng(1)
    v1, v2, D = rng.normal(size=(3, 50))
    dt = 0.7
    U11, U22, U12 = propagator_2x2(v1, v2, D, dt)
    for i in range(50):
        U = np.array([[U11[i], U12[i]], [U12[i], U22[i]]])
        H = np.array([[v1[i], D[i]], [D[i], v2[i]]])
        np.testing.assert_allclose(U, expm(-1j * H * dt), atol=1e-12)
        np.testing.assert_allclose(U.conj().T @ U, np.eye(2), atol=1e-12)

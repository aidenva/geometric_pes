"""Vectorized classical trajectory building blocks shared by every track.

All functions work on arrays of any layout; positions q, momenta p and forces
f share one shape, and the mass m broadcasts against it (e.g. shape (d, 1)
for coordinate-major arrays of shape (d, n), or (1, d) / scalar for (n, d)).
Random numbers are drawn with the shape of the array they act on, so a
caller's layout fixes the order of the draws.

Integrators
-----------
* baoab_step   - Langevin BAOAB (Leimkuhler & Matthews 2013): half kick,
                 half drift, exact Ornstein-Uhlenbeck step, half drift, half
                 kick. Samples the Maxwell distribution exactly; reduces to
                 velocity Verlet at gamma = 0. Two hooks let surface-hopping
                 schemes act between the drift and the final kick.
* brownian_step - overdamped Brownian dynamics (Euler-Maruyama), valid for
                 gamma >> vibrational frequencies.

Baths and hopping
-----------------
* caldeira_leggett_modes - discretized ohmic bath (frequencies, couplings).
* fermi, cme_hop         - classical-master-equation hops between a
                           molecular level and a metal (Fermi-function rates).
"""
import numpy as np


# ---------------------------------------------------------------- Langevin

def ou_coefficients(gamma, kT, dt):
    """c1 = exp(-gamma dt), c2 = sqrt((1 - c1^2) kT) for the exact OU step."""
    c1 = np.exp(-gamma * dt)
    return c1, np.sqrt((1 - c1**2) * kT)


def kick(p, f, dt):
    """B: half kick, p + (dt/2) f."""
    return p + 0.5 * dt * f


def drift(q, p, m, dt):
    """A: half drift, q + (dt/2) p / m."""
    return q + 0.5 * dt * p / m


def thermostat(p, m, c1, c2, rng, rows=None):
    """O: exact Ornstein-Uhlenbeck step on momenta, p -> c1 p + c2 sqrt(m) xi.

    rows: optional index (along axis 0) of the degrees of freedom that feel
    friction; the others are left unchanged. Noise is drawn with the shape of
    the thermostatted block, rows in order.
    """
    if rows is None:
        return c1 * p + c2 * np.sqrt(m) * rng.standard_normal(p.shape)
    p = p.copy()
    mr = m[rows] if np.ndim(m) else m
    p[rows] = c1 * p[rows] + c2 * np.sqrt(mr) * rng.standard_normal(p[rows].shape)
    return p


def baoab_step(q, p, f, force, m, dt, gamma=0.0, kT=0.0, rng=None, rows=None,
               before_force=None, after_force=None):
    """One BAOAB step. Returns (q, p, f) with f the force at the new q.

    force(q) -> f. before_force(q, p, q_old) -> p runs after the second drift
    (e.g. an electronic hop that changes the force field); after_force(q, p, f)
    -> (p, f) runs after the new force is known (e.g. fewest-switches hops that
    need the new surfaces and rescale p).
    """
    q_old = q
    p = kick(p, f, dt)
    q = drift(q, p, m, dt)
    if gamma > 0:
        c1, c2 = ou_coefficients(gamma, kT, dt)
        p = thermostat(p, m, c1, c2, rng, rows)
    q = drift(q, p, m, dt)
    if before_force is not None:
        p = before_force(q, p, q_old)
    f = force(q)
    if after_force is not None:
        p, f = after_force(q, p, f)
    p = kick(p, f, dt)
    return q, p, f


def brownian_step(q, f, m, gamma, kT, dt, rng):
    """Overdamped Euler-Maruyama step: q + f dt/(m gamma) + sqrt(2 kT dt/(m gamma)) xi."""
    return q + f * (dt / (m * gamma)) + np.sqrt(2 * kT * dt / (m * gamma)) * rng.standard_normal(q.shape)


# ---------------------------------------------------------------- explicit bath

def caldeira_leggett_modes(N, gamma, M, wc):
    """Ohmic spectral density J(w) = (pi/2) eta w exp(-w/wc), eta = M gamma,
    discretized as w_j = -wc ln((j - 1/2)/N), c_j = w_j sqrt(eta wc / N).
    Bath modes have unit mass; the system-bath potential (with counterterm) is
    sum_j w_j^2/2 (q_j - c_j y / w_j^2)^2."""
    j = np.arange(1, N + 1)
    w = -wc * np.log((j - 0.5) / N)
    c = w * np.sqrt(M * gamma * wc / N)
    return w, c

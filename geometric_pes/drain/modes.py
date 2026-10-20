"""Drain spectra: discretized spectral densities, coupling patterns, and the a-priori friction on
the exchange coordinate.

The ohmic default (caldeira_leggett_modes, re-exported from geometric_pes.integrators) is
w_j = -wc ln((j - 1/2)/N), c_j = w_j sqrt(eta wc / N). discretize/modes cover sub-ohmic, ohmic,
super-ohmic, Drude-Lorentz and random sparse spectra at a given reorganization Lambda.
"""
import numpy as np

from ..integrators import caldeira_leggett_modes  # noqa: F401  (re-export)


def discretize(kind, N, Lam, wc, rng):
    """N modes whose frequencies are quantiles of rho(w) ~ J(w)/w and whose couplings give the
    reorganization Lambda = sum c^2 / w^2 on the coordinate they couple to (per coordinate)."""
    if kind == "random":
        w = np.sort(np.exp(rng.uniform(np.log(0.2), np.log(3 * wc), N)))
    else:
        g = np.linspace(1e-4, 20 * wc, 200001)
        if kind == "subohmic":
            rho = g ** -0.5 * np.exp(-g / wc)
        elif kind == "ohmic":
            rho = np.exp(-g / wc)
        elif kind == "superohmic":
            rho = g * np.exp(-g / wc)
        elif kind == "drude":
            rho = wc / (g**2 + wc**2)
        cdf = np.cumsum(rho); cdf /= cdf[-1]
        w = np.interp((np.arange(N) + 0.5) / N, cdf, g)
    return w


def modes(kind, pattern, N, Lam, wc, rng):
    if pattern == "xy":
        wh = discretize(kind, N // 2, Lam, wc, rng)
        w = np.concatenate([wh, wh]); which = np.r_[np.zeros(N // 2, int), np.ones(N // 2, int)]
        c = w * np.sqrt(Lam / (N // 2))
    else:
        w = discretize(kind, N, Lam, wc, rng)
        which = np.full(N, 1 if pattern == "y" else 0)
        c = w * np.sqrt(Lam / N)
    return w, c, which


def exchange_modes(S):
    """for each product well: the Hessian eigenvector most aligned with the direction to TS12,
    and its frequency"""
    t12 = [s for s in S.sads if set(s["connects"]) == {"P+", "P-"}][0]
    out = []
    for j in (S.iPp, S.iPm):
        mj = S.mins[j]
        d = t12["xy"] - mj["xy"]; d = d / np.linalg.norm(d)
        k = int(np.argmax(np.abs(mj["vecs"].T @ d)))
        out.append((mj["vecs"][:, k], float(np.sqrt(max(mj["eig"][k], 1e-6)))))
    return out


def gamma_exchange(w, c, which, S, frac=0.25):
    """a-priori drain rate of the exchange mode: the Markovian friction J_ex(w)/w at the mode's
    frequency, with J_ex the spectral density projected on the mode, sum_j (c_j u_a(j))^2 / w_j
    (pi/2) delta(w - w_j), each delta broadened to a Gaussian of width frac * w_ex; averaged over
    the two wells"""
    g = []
    for u, wex in exchange_modes(S):
        lam = (c * u[which]) ** 2 / w**2
        sig = frac * wex
        g.append(np.pi / 2 * np.sum(lam * w * np.exp(-0.5 * ((w - wex) / sig) ** 2))
                 / (np.sqrt(2 * np.pi) * sig))
    return float(np.mean(g))


def mode_freqs(N, eta, wc=5.0):
    Nh = N // 2
    w, _ = caldeira_leggett_modes(Nh, eta, 1.0, wc)
    return np.concatenate([w, w])

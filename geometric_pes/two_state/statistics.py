"""Statistical baselines on E_-: microcanonical and canonical basin populations.

All functions take a Landscape (anything with Em, prod and react arrays on the
propagation grid).
"""
import numpy as np


def p_stat(L, E, N=0):
    """Classical microcanonical P_prod at energy E on E_-.

    In 2D, integrating out the momenta gives a configurational density
    proportional to the area where E_- < E (mass independent). With N extra
    harmonic bath modes coupled bilinearly to y (Caldeira-Leggett form with
    counterterm), integrating out the bath coordinates and all momenta gives
    weight (E - E_-)^N. Returns P_prod / (P_prod + P_react + others).
    """
    E = np.atleast_1d(np.asarray(E, float))
    out = []
    for e in E:
        w = np.clip(e - L.Em, 0.0, None)
        w = (w > 0).astype(float) if N == 0 else w**N
        tot = w.sum()
        out.append(w[L.prod].sum() / tot if tot > 0 else np.nan)
    out = np.array(out)
    return out[0] if out.size == 1 else out


def p_react_stat(L, E, N=0):
    w = np.clip(E - L.Em, 0.0, None)
    w = (w > 0).astype(float) if N == 0 else w**N
    return w[L.react].sum() / w.sum()


def p_react_canon(L, kT):
    """Classical canonical (Boltzmann, configurational) P_react on E_- at temperature kT."""
    w = np.exp(-(L.Em - L.Em.min()) / kT)
    return float(w[L.react].sum() / w.sum())

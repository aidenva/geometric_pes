"""The two-state, two-coordinate funnel: harmonic diabats coupled through a tunable electronic
interaction, their adiabatic surfaces, and closed-system dynamics on them (the project's starting
model; it calibrates the bottom of the directionality ladder).

Modules
-------
model       Params, diabats, coupling, adiabatic surfaces, gradients, 2x2 propagator
stationary  stationary points of E_-, Hessian classification, basins
landscape   Landscape: stationary points + propagation grid + basin labels
quantum     split-operator wavepacket dynamics
statistics  microcanonical / canonical basin-population baselines
"""
from .model import (P, HBAR, Params, V1, V2, coupling_Delta, E_pm, grad_Eminus,
                    surfaces_and_grads, lower_eigvec, mixing_angle, propagator_2x2)
from .stationary import (hessian_numeric, solve_stationary, merge_points, classify_point,
                         find_points, classify_all, reactant_product, basin_labels)
from .landscape import Landscape
from .quantum import initial_packet, propagate
from .statistics import p_stat, p_react_stat, p_react_canon

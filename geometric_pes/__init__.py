"""geometric_pes: geometric product decisions on bifurcating potential energy surfaces, and the
finite internal drains that keep or lose them.

Subpackages
-----------
surfaces     branching surfaces (VRI), basin bookkeeping, the two-state ET junction
drain        start ensembles, drain spectra, statistical references, dynamics, metrics,
             quantum master equation
two_state    the original two-state, two-coordinate funnel model
Modules
-------
config       experiment parameters (config/experiments.toml) and surface variants
integrators  BAOAB / Brownian integrators and the Caldeira-Leggett ohmic discretization
io           save results with their parameters and code version
random_configs  random surfaces, spectra and couplings for the out-of-sample test
"""
__version__ = "1.0.0"

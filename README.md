# geometric_pes

Model Hamiltonians and simulation code for *Reversibility, Relaxation, and Product Selectivity on
a Bifurcating Potential Energy Surface* (L. M. Le).

After a valley–ridge inflection, the shape of the potential energy surface decides which product
forms. This package simulates that decision on model surfaces with two reacting coordinates,
coupled to a finite, energy-conserving set of harmonic modes (an idealized model of intramolecular
vibrational energy redistribution), and measures whether the decision is kept or relaxes toward
the statistics.

## Install

```bash
pip install -e ".[dev]"
```

Python 3.11 or later; the only dependencies are NumPy and SciPy.

## Run

Every experiment reads its parameters from `config/experiments.toml` and writes a JSON summary to
`results/full/` (or `results/quick/` with `--quick`, a smoke test with few trajectories):

```bash
python experiments/decision.py --quick
```

```bash
python experiments/timescales.py
```

```bash
python experiments/generalization.py
```

`generalization.py` reuses the `timescales.py` runs as its training grid, and `edge_cases.py`
reads the `generalization.py` summary; the other drivers are independent. Full runs use the
worker count in `[run] nproc` (override with `--nproc`). The largest, `generalization.py`, is
about an hour on seven cores; `quantum_drain.py` spends most of its time in the master equation.

Raw per-run arrays are cached under `results/`, so a rerun only computes what is missing. Basin
maps of the surfaces are cached under `results/cache/`.

## Layout

See [WORKFLOW.md](WORKFLOW.md) for the module layout, which experiment calls which utility, and
what each runner measures.

## Tests

```bash
python -m pytest -q
```

## Using the library directly

```python
from geometric_pes.config import surface
from geometric_pes.drain.dynamics import run_rest
from geometric_pes.drain.metrics import metrics

params, arrays = run_rest("A", N=64, eta=1.0, ntraj=500, t_final=400.0)
print(metrics(params, arrays)["r"])          # memory of the first decision
```

Any `Surface` can be passed with `surface=`, and any discretized spectrum with
`modes=(w, c, which)` (see `geometric_pes.drain.modes`).

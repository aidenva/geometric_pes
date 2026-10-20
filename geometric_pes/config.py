"""Load config/experiments.toml (or $GEOMETRIC_PES_CONFIG) and build surfaces from it."""
import os
import tomllib
from dataclasses import replace
from functools import lru_cache

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT = os.path.join(os.path.dirname(_HERE), "config", "experiments.toml")


@lru_cache(maxsize=1)
def load(path=None):
    path = path or os.environ.get("GEOMETRIC_PES_CONFIG", _DEFAULT)
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg["_path"] = os.path.abspath(path)
    return cfg


def section(name):
    return load()[name]


def _root():
    return os.path.dirname(os.path.dirname(load()["_path"]))


def results_dir(*parts):
    d = os.path.join(_root(), load()["paths"]["results"], *parts)
    os.makedirs(d, exist_ok=True)
    return d


def cache_dir():
    d = os.path.join(_root(), load()["paths"]["cache"])
    os.makedirs(d, exist_ok=True)
    return d


@lru_cache(maxsize=None)
def surface(name):
    """the Surface named `name` in [surface.*], built on the dataclass defaults or on its base"""
    from .surfaces.branching import Surface
    spec = dict(load()["surface"][name])
    base = spec.pop("base", None)
    p = surface(base) if base else Surface()
    return replace(p, **spec)

"""Save results together with the parameters (and code version) that produced them.

Every .npz written with save_npz carries a `meta` entry (JSON) with the
parameter dataclass or dict, its type name, the git commit and whether the
working tree was dirty, and a timestamp. load_npz returns (arrays, meta).
"""
import dataclasses
import datetime
import json
import os
import subprocess

import numpy as np


def params_dict(params):
    if params is None:
        return None
    if dataclasses.is_dataclass(params):
        return dataclasses.asdict(params)
    return dict(params)


def git_version(path=None):
    """(short commit, dirty) of the repository containing path, or (None, None)."""
    cwd = os.path.dirname(os.path.abspath(path or __file__))
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=cwd,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return commit, dirty
    except Exception:
        return None, None


def make_meta(params=None, **extra):
    commit, dirty = git_version()
    meta = dict(params=params_dict(params),
                params_type=type(params).__name__ if params is not None else None,
                git_commit=commit, git_dirty=dirty,
                created=datetime.datetime.now().isoformat(timespec="seconds"))
    meta.update(extra)
    return meta


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def save_npz(path, params=None, meta=None, **arrays):
    """np.savez(path, **arrays) plus a `meta` JSON entry describing params."""
    m = make_meta(params, **(meta or {}))
    np.savez(path, meta=np.array(json.dumps(m, default=_default)), **arrays)


def load_npz(path):
    """Return (dict of arrays, meta dict or None)."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"])) if "meta" in z.files else None
    return arrays, meta


def save_json(path, data, params=None, meta=None):
    """Write {"meta": ..., "data": data} as JSON."""
    with open(path, "w") as f:
        json.dump(dict(meta=make_meta(params, **(meta or {})), data=data), f,
                  indent=1, default=_default)


def load_json(path):
    with open(path) as f:
        d = json.load(f)
    return d.get("data"), d.get("meta")

"""Stationary points, propagation grid and E_- basin labels for one parameter set."""
import hashlib
import json
import os

import numpy as np

from .model import P, E_pm
from .stationary import find_points, classify_all, reactant_product, basin_labels

# Fields that change the surface, the grid or the stationary-point search, and
# therefore the basin labels (used as the cache key)
SURFACE_FIELDS = ("x1 y1 x2 y2 kx1 ky1 kx2 ky2 stab coupling Delta0 wg_x wg_y cx cy "
                  "bias1 bias2 prop_x_min prop_x_max prop_y_min prop_y_max "
                  "prop_nx prop_ny x_min x_max y_min y_max n_starts seed").split()


def surface_key(p):
    d = {f: getattr(p, f) for f in SURFACE_FIELDS}
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


class Landscape:
    """Stationary points, propagation grid and basin labels for parameters p.

    cache_dir: if given, basin labels (the slow part) are cached there as
    labels_<surface_key>.npy.
    """

    def __init__(self, p=P, pts=None, cache_dir=None):
        self.p = p
        self.pts = find_points(p) if pts is None else pts
        self.mins, self.saddles = classify_all(self.pts, p)
        self.i_r, self.i_p = reactant_product(self.mins, p)
        self.r_min, self.p_min = self.mins[self.i_r], self.mins[self.i_p]
        self.E_r = float(E_pm(*self.r_min, p)[0])
        self.E_p = float(E_pm(*self.p_min, p)[0])
        self.E_saddles = [float(E_pm(*s, p)[0]) for s in self.saddles]

        Nx, Ny = p.prop_nx, p.prop_ny
        self.x = np.linspace(p.prop_x_min, p.prop_x_max, Nx, endpoint=False)
        self.y = np.linspace(p.prop_y_min, p.prop_y_max, Ny, endpoint=False)
        self.dx, self.dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
        self.XX, self.YY = np.meshgrid(self.x, self.y, indexing="xy")
        self.Em, self.Ep, self.v1, self.v2, self.D, *_ = E_pm(self.XX, self.YY, p)

        f = os.path.join(cache_dir, f"labels_{surface_key(p)}.npy") if cache_dir else None
        if f and os.path.exists(f):
            self.labels = np.load(f)
        else:
            self.labels = basin_labels(self.XX, self.YY, self.mins, p)
            if f:
                os.makedirs(cache_dir, exist_ok=True)
                tmp = f"{f}.{os.getpid()}.npy"
                np.save(tmp, self.labels)
                os.replace(tmp, f)          # atomic: parallel workers may race here
        self.prod = self.labels == self.i_p
        self.react = self.labels == self.i_r

    def label_at(self, x, y):
        """Basin label (index into mins) at arbitrary points, nearest grid node."""
        i = np.clip(np.rint((x - self.x[0]) / self.dx).astype(int), 0, len(self.x) - 1)
        j = np.clip(np.rint((y - self.y[0]) / self.dy).astype(int), 0, len(self.y) - 1)
        return self.labels[j, i]

    def summary(self):
        return (f"react min E={self.E_r:+.3f} at ({self.r_min[0]:+.3f},{self.r_min[1]:+.3f}); "
                f"prod min E={self.E_p:+.3f} at ({self.p_min[0]:+.3f},{self.p_min[1]:+.3f}); "
                f"saddles {[round(e, 3) for e in self.E_saddles]}; n_min={len(self.mins)}")

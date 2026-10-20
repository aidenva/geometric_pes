"""Basin bookkeeping for a branching surface: stationary points and species labels, the
transition-state direction, commitment thresholds, and a steepest-descent basin map (cached on disk,
keyed by the surface name, under geometric_pes.config.cache_dir()).
"""
import os

import numpy as np

from .branching import V, find_stationary, label_species, basin_grid
from ..config import cache_dir

COMMIT_MARGIN = 1.0      # commitment threshold below the lowest exit saddle, in units of kT


class Setup:
    def __init__(self, p, kT=1.0, commit_margin=COMMIT_MARGIN):
        self.p, self.kT = p, kT
        mins, sads = label_species(find_stationary(p), p)
        self.mins, self.sads = mins, sads
        self.names = [d["name"] for d in mins]
        self.mxy = np.array([d["xy"] for d in mins])
        ts1 = [s for s in sads if "R" in s["connects"]][0]
        self.ts1 = ts1
        n = ts1["vecs"][:, 0]
        self.n = n if n[0] > 0 else -n            # points to the product side
        self.t = np.array([-self.n[1], self.n[0]])
        # commitment thresholds: lowest saddle out of each basin minus 1 kT
        self.Ecut = np.empty(len(mins))
        for i, nm in enumerate(self.names):
            out = [s["E"] for s in sads if nm in s["connects"]]
            self.Ecut[i] = min(out) - commit_margin * kT
        self.absorbing = np.array([nm in ("R", "P+", "P-") for nm in self.names])
        # steepest-descent basin map (cached): a nearest-minimum rule would put the
        # product side just past TS1 into R
        cache = os.path.join(cache_dir(), f"basins_{p.name}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            X, Y, lab = z["X"], z["Y"], z["lab"]
        else:
            X, Y, lab = basin_grid(p, mins)
            os.makedirs(cache_dir(), exist_ok=True)
            np.savez_compressed(cache, X=X, Y=Y, lab=lab)
        self.gx0, self.gdx, self.gy0, self.gdy = X[0], X[1] - X[0], Y[0], Y[1] - Y[0]
        self.lab = lab
        self.iR = self.names.index("R")
        self.iPp, self.iPm = self.names.index("P+"), self.names.index("P-")

    def sample_start(self, N, rng):
        s = np.linspace(-2.5, 2.5, 5001)
        pts = self.ts1["xy"][:, None] + np.outer(self.t, s)
        w = np.exp(-(V(pts[0], pts[1], self.p) - self.ts1["E"]) / self.kT)
        cdf = np.cumsum(w); cdf /= cdf[-1]
        sv = np.interp(rng.uniform(size=N), cdf, s)
        x = self.ts1["xy"][0] + sv * self.t[0]
        y = self.ts1["xy"][1] + sv * self.t[1]
        vn = np.sqrt(-2 * self.kT * np.log(rng.uniform(size=N)))     # flux-weighted
        vt = np.sqrt(self.kT) * rng.standard_normal(N)
        return x, y, vn * self.n[0] + vt * self.t[0], vn * self.n[1] + vt * self.t[1]

    def side(self, x, y):
        return (x - self.ts1["xy"][0]) * self.n[0] + (y - self.ts1["xy"][1]) * self.n[1] > 0

    def basin(self, x, y):
        nx, ny = self.lab.shape
        i = np.clip(np.rint((x - self.gx0) / self.gdx).astype(int), 0, nx - 1)
        k = np.clip(np.rint((y - self.gy0) / self.gdy).astype(int), 0, ny - 1)
        return self.lab[i, k]

    def committed(self, x, y, vx, vy):
        """index of the basin a walker is committed to, -1 if none."""
        j = self.basin(x, y)
        E = V(x, y, self.p) + 0.5 * (vx * vx + vy * vy)
        ok = (E < self.Ecut[j]) & self.absorbing[j]
        return np.where(ok, j, -1)

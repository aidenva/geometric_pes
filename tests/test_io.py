"""Results round-trip with the parameters that produced them."""
import dataclasses

import numpy as np

from geometric_pes.two_state import P
from geometric_pes.io import save_npz, load_npz, save_json, load_json


def test_npz_roundtrip_keeps_params(tmp_path):
    p = dataclasses.replace(P, bias2=-0.2)
    f = tmp_path / "r.npz"
    save_npz(f, params=p, t=np.arange(3.0), P_prod=np.array([0.1, 0.2, 0.3]))
    arrays, meta = load_npz(f)
    np.testing.assert_array_equal(arrays["t"], np.arange(3.0))
    assert meta["params"]["bias2"] == -0.2
    assert meta["params_type"] == "Params"
    assert dataclasses.replace(P, **meta["params"]) == p
    assert "git_commit" in meta and "created" in meta


def test_json_roundtrip_with_numpy_values(tmp_path):
    f = tmp_path / "r.json"
    save_json(f, {"k": np.float64(1.5), "a": np.arange(2)}, params={"gamma": 0.1})
    data, meta = load_json(f)
    assert data == {"k": 1.5, "a": [0, 1]}
    assert meta["params"] == {"gamma": 0.1}

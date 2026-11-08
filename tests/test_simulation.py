import numpy as np
from mcq_profiles.simulation import simulate, centred_shift


def test_shapes_and_flags():
    d = simulate(500, None, np.zeros(8), rng=0, dur=dict(alpha=3, beta=0, s=0.5), limit=30, missing_rate=0.1)
    assert d["Y"].shape == (500, 8) and d["synthetic"]
    assert np.all(d["logT"] <= np.log(30) + 1e-12)
    assert 0.05 < np.isnan(d["Y"]).mean() < 0.15


def test_centred_shift_is_centred():
    S = centred_shift(20, 3, 1.0, rng=0)
    assert np.allclose(S.mean(1), 0) and np.allclose(S[0], 0)


def test_null_has_single_class():
    d = simulate(100, None, np.zeros(5), rng=1)
    assert (d["cls"] == 0).all()

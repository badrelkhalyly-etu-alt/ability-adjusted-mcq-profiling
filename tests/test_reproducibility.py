import numpy as np
from mcq_profiles.simulation import simulate
from mcq_profiles.latent_classes import MixIRT


def test_same_seed_same_result():
    d = simulate(800, None, np.linspace(-1, 1, 10), rng=3)
    r = [MixIRT(2, "2pl", class_specific="b", n_starts=2, max_iter=100, random_state=5).fit(d["Y"]).result_.loglik for _ in range(2)]
    assert r[0] == r[1]


def test_simulation_seeded():
    a = simulate(100, None, np.zeros(5), rng=9)["Y"]
    b = simulate(100, None, np.zeros(5), rng=9)["Y"]
    assert np.array_equal(a, b)

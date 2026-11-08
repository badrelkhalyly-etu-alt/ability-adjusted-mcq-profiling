import numpy as np
from sklearn.metrics import adjusted_rand_score
from mcq_profiles.simulation import simulate, centred_shift
from mcq_profiles.latent_classes import MixIRT, LCA


def test_ability_only_mixture_shares_items():
    rng = np.random.default_rng(0)
    b = np.linspace(-2, 2, 12)
    d = simulate(800, None, b, rng=1)
    m = MixIRT(2, "2pl", class_specific=None, n_starts=2, max_iter=200).fit(d["Y"])
    assert m.b.shape[0] == 1  # item parameters tied across classes


def test_positive_control_selects_class_model():
    rng = np.random.default_rng(1)
    J = 20
    b = np.linspace(-2, 1.5, J)
    a = rng.uniform(0.8, 1.6, J)
    S = centred_shift(J, 2, 1.5, rng=3)
    d = simulate(3000, a, b, rng=4, n_classes=2, class_b_shift=S)
    m0 = MixIRT(1, "2pl").fit(d["Y"])
    m1 = MixIRT(2, "2pl", class_specific="b", n_starts=2, max_iter=400, tol=1e-8).fit(d["Y"])
    assert m1.bic(3000) < m0.bic(3000)
    pc = m1.posterior(d["Y"])[0]
    assert adjusted_rand_score(d["cls"], pc.argmax(1)) > 0.3
    est = m1.b[1] - m1.b[0]
    true = S[1] - S[0]
    assert max(np.corrcoef(est, true)[0, 1], np.corrcoef(-est, true)[0, 1]) > 0.9


def test_identification_constraints():
    rng = np.random.default_rng(2)
    b = np.linspace(-2, 2, 10)
    d = simulate(1000, None, b, rng=5)
    m = MixIRT(2, "2pl", class_specific="b", n_starts=1, max_iter=100).fit(d["Y"])
    assert np.allclose(m.b.mean(1), m.b.mean(1)[0], atol=1e-8)  # equal mean difficulty
    pi = np.exp(m.logpi[0])
    mean = (pi * m.mu).sum()
    var = (pi * (m.sigma ** 2 + m.mu ** 2)).sum() - mean ** 2
    assert abs(mean) < 1e-6 and abs(var - 1) < 1e-6


def test_lca_recovers_two_classes():
    rng = np.random.default_rng(3)
    p = np.array([[0.2] * 10, [0.8] * 10])
    c = rng.integers(0, 2, 2000)
    Y = (rng.random((2000, 10)) < p[c]).astype(float)
    l = LCA(2, n_starts=3).fit(Y)
    assert adjusted_rand_score(c, l.posterior(Y)[0].argmax(1)) > 0.8

"""Flexible duration model, sampling weights and class ordering of duration parameters."""
import numpy as np
from scipy.optimize import approx_fprime

from mcq_profiles.latent_classes import MixIRT, MixIRTDurPoly, ContinuousIRT, _poly_affine
from mcq_profiles.simulation import simulate


def _data(n=500, seed=3):
    rng = np.random.default_rng(seed)
    J = 10
    d = simulate(n, rng.uniform(0.8, 1.5, J), np.linspace(-1.5, 1.5, J), rng=seed,
                 dur=dict(alpha=np.log(20), beta=0.2, s=0.4), group_props=[0.5, 0.5])
    return d["Y"], d["logT"], np.zeros(n, bool), d["group"]


def test_poly_affine():
    c = np.array([1.0, -2.0, 0.5, 0.3])
    t = np.linspace(-2, 2, 7)
    assert np.allclose(np.polynomial.polynomial.polyval(0.4 + 1.7 * t, c),
                       np.polynomial.polynomial.polyval(t, _poly_affine(c, 0.4, 1.7)))


def test_linear_poly_matches_m2():
    Y, logT, cens, g = _data()
    kw = dict(n_nodes=21, max_iter=300, tol=1e-9, group_effects=True)
    m2 = MixIRT(1, "2pl", duration=True, **kw).fit(Y, logT, cens, g)
    mp = MixIRTDurPoly(1, "2pl", deg=1, het=False, **kw).fit(Y, logT, cens, g)
    assert mp.result_.n_params == m2.result_.n_params
    assert abs(mp.result_.loglik - m2.result_.loglik) < 0.5


def test_identify_and_gradient():
    Y, logT, cens, g = _data()
    cens = np.zeros(len(Y), bool); cens[:20] = True
    m = MixIRTDurPoly(2, "2pl", class_specific="b", deg=3, het=True, n_nodes=21, max_iter=30, n_starts=1,
                      group_effects=True).fit(Y, logT, cens, g)
    ll0 = m.loglik(Y, logT, cens, g)
    m.mu = m.mu + np.array([0.3, -0.2]); m.b = m.b + np.array([[0.3], [-0.2]])
    for k in range(2):
        m.C[k] = _poly_affine(m.C[k], -np.array([0.3, -0.2])[k], 1.0)
        m.logs[k] -= m.delta[k] * np.array([0.3, -0.2])[k]
    ll1 = m.loglik(Y, logT, cens, g)
    m._identify()
    assert abs(m.loglik(Y, logT, cens, g) - ll1) < 1e-6 * abs(ll1)
    assert abs(ll1 - ll0) < 1e-6 * abs(ll0)
    # analytic gradient of the duration M-step objective
    data = m._prep(Y, logT, cens, g)
    post, _ = m._estep(*data)
    captured = {}
    import scipy.optimize as so
    orig = so.minimize
    def spy(f, v0, **kw):
        captured["f"], captured["v0"] = f, v0
        return orig(f, v0, **dict(kw, options={"maxiter": 1}))
    import mcq_profiles.latent_classes as lc
    lc.minimize = spy
    try:
        m._mstep_duration(post, data[2], data[3], data[4])
    finally:
        lc.minimize = orig
    f, v0 = captured["f"], captured["v0"]
    num = approx_fprime(v0, lambda v: f(v)[0], 1e-6)
    assert np.allclose(f(v0)[1], num, rtol=1e-3, atol=1e-2)


def test_weights_of_ones_equal_unweighted():
    Y, logT, cens, g = _data(300)
    a = MixIRT(1, "2pl", n_nodes=21, max_iter=200).fit(Y)
    b = MixIRT(1, "2pl", n_nodes=21, max_iter=200).fit(Y, weights=np.full(len(Y), 3.0))
    assert np.allclose(a.b, b.b) and abs(a.result_.loglik - b.result_.loglik) < 1e-6
    c = ContinuousIRT(guessing=True, max_iter=50).fit(Y)
    d = ContinuousIRT(guessing=True, max_iter=50).fit(Y, weights=np.ones(len(Y)))
    assert np.allclose(c.c, d.c)


def test_order_classes_moves_duration_params():
    Y, logT, cens, g = _data()
    m = MixIRT(2, "2pl", class_specific="b", duration=True, n_nodes=21, max_iter=40, n_starts=1).fit(Y, logT, cens, g)
    ll = m.loglik(Y, logT, cens, g)
    m.mu = m.mu[::-1].copy(); m.sigma = m.sigma[::-1].copy(); m.b = m.b[::-1].copy(); m.logpi = m.logpi[:, ::-1].copy()
    m.alpha, m.beta, m.logs = m.alpha[::-1].copy(), m.beta[::-1].copy(), m.logs[::-1].copy()
    m._order_classes()
    assert abs(m.loglik(Y, logT, cens, g) - ll) < 1e-6 * abs(ll)

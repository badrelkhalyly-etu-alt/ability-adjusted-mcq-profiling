import numpy as np
import pytest
from scipy.special import expit

from mcq_profiles.latent_classes import ContinuousIRT
from mcq_profiles.temporal import (ContinuousTimeModel, RegimeTimeModel, _logpdf_cdf, _logpdf_cdf_fast, design,
                                   item_loglik_grid, spline_basis)


def _item_model(J=20, seed=0):
    rng = np.random.default_rng(seed)
    m = ContinuousIRT(dims=1, guessing=True)
    m.A = rng.uniform(0.8, 1.8, (J, 1))
    m.d = rng.normal(0, 1, J)
    m.c = rng.uniform(0.05, 0.2, J)
    return m


def _responses(m, theta, rng):
    P = m.c + (1 - m.c) * expit(theta[:, None] * m.A[:, 0] + m.d)
    return (rng.random(P.shape) < P).astype(float)


@pytest.mark.parametrize("family,shape", [("normal", []), ("skewnormal", [2.0]), ("t", [np.log(3.0)]), ("shash", [0.4, np.log(0.7)])])
def test_densities_integrate_to_one_and_fast_equals_full(family, shape):
    z = np.linspace(-60, 60, 60001)
    lp, cdf = _logpdf_cdf(z, family, np.array(shape))
    assert np.trapezoid(np.exp(lp), z) == pytest.approx(1.0, abs=2e-3)
    assert np.allclose(lp, _logpdf_cdf_fast(z, family, np.array(shape)))
    assert cdf[0] < 1e-3 and cdf[-1] > 1 - 1e-3


def test_spline_basis_shape_and_partition():
    x = np.linspace(-3, 3, 41)
    B = spline_basis(x, df=5)
    assert B.shape == (41, 4)
    assert design(x, "cubic").shape == (41, 3)


def test_continuous_model_recovers_cubic_heteroscedastic_curve():
    rng = np.random.default_rng(1)
    m = _item_model()
    th = rng.normal(size=4000)
    Y = _responses(m, th, rng)
    g = rng.integers(0, 2, 4000)
    v = 0.1 - 0.4 * th + 0.1 * th ** 2 + 0.3 * g + np.exp(-1 + 0.2 * th) * rng.normal(size=4000)
    LY = item_loglik_grid(m, Y)
    f = ContinuousTimeModel("normal", "cubic", "linear").fit(LY, v, g, m.X[:, 0])
    a0, bm, l0, bs, sh, ga = f._unpack(f.p)
    assert a0 == pytest.approx(0.1, abs=0.05) and bm[0] == pytest.approx(-0.4, abs=0.06)
    assert l0 == pytest.approx(-1, abs=0.06) and bs[0] == pytest.approx(0.2, abs=0.06)
    assert ga[1] == pytest.approx(0.3, abs=0.04)
    pit = f.pit(LY, v, g)
    assert abs(np.mean((pit > 0.05) & (pit < 0.95)) - 0.9) < 0.03          # calibrated 90 % interval


def test_regimes_found_when_present_and_not_preferred_when_absent():
    rng = np.random.default_rng(2)
    m = _item_model(seed=2)
    n = 4000
    th = rng.normal(size=n)
    Y = _responses(m, th, rng)
    g = np.zeros(n, int)
    LY = item_loglik_grid(m, Y)
    x = m.X[:, 0]
    # (a) two well-separated regimes
    k = rng.random(n) < 0.3
    v = -0.3 * th + np.where(k, 1.2, 0.0) + 0.3 * rng.normal(size=n)
    r2 = RegimeTimeModel(2, n_starts=3, seed=0).fit(LY, v, g, x)
    c = ContinuousTimeModel("shash", "spline", "spline").fit(LY, v, g, x)
    assert r2.bic() < c.bic() - 50
    s = r2.summary()
    assert s["intercepts"][1] - s["intercepts"][0] == pytest.approx(1.2, abs=0.1)
    assert s["prevalence_marginal"][1] == pytest.approx(0.3, abs=0.05)
    # (b) a single skewed continuous distribution: the flexible continuous model should win on BIC
    from scipy.stats import skewnorm
    v2 = -0.3 * th + 0.5 * skewnorm.rvs(4, size=n, random_state=3)
    r2b = RegimeTimeModel(2, n_starts=3, seed=0).fit(LY, v2, g, x)
    cb = ContinuousTimeModel("skewnormal", "spline", "spline").fit(LY, v2, g, x)
    assert cb.bic() < r2b.bic()


def test_regime_order_and_posterior():
    rng = np.random.default_rng(4)
    m = _item_model(seed=4)
    n = 3000
    th = rng.normal(size=n)
    Y = _responses(m, th, rng)
    LY = item_loglik_grid(m, Y)
    k = rng.random(n) < 0.5
    v = np.where(k, 1.0, -1.0) + 0.3 * rng.normal(size=n)
    r = RegimeTimeModel(2, n_starts=2, seed=1).fit(LY, v, np.zeros(n, int), m.X[:, 0])
    a = r.summary()["intercepts"]
    assert a[0] < a[1]
    post = r.regime_posterior(LY, v, np.zeros(n, int))
    assert np.allclose(post.sum(1), 1)
    assert ((post[:, 1] > 0.5) == k).mean() > 0.97

import itertools

import numpy as np
import pytest
from scipy.special import expit

from mcq_profiles.continuous_map import (ContinuousIRTDur, interval_from_distribution, masked_score_distribution,
                                         poisson_binomial, relative_time_index, separation_strata, time_intensities)


def _sim(n=3000, J=20, seed=0, G=2):
    rng = np.random.default_rng(seed)
    th = rng.normal(size=n)
    a = rng.uniform(0.8, 1.8, J)
    b = rng.normal(0, 1, J)
    c = rng.uniform(0.05, 0.25, J)
    P = c + (1 - c) * expit(a * (th[:, None] - b))
    Y = (rng.random((n, J)) < P).astype(float)
    g = rng.integers(0, G, n)
    C = np.array([0.2, -0.3, 0.1, 0.05])
    mean = C @ (th[None, :] ** np.arange(4)[:, None]) + np.array([0.0, 0.4])[g]
    s = np.exp(-1.0 + 0.2 * th)
    v = mean + s * rng.normal(size=n)
    return Y, v, g, th, C


def test_relative_time_index_handles_missing_and_min_items():
    L = np.log(np.array([[10, 20, 30, np.nan], [np.nan, np.nan, 5, np.nan], [10, 20, 30, 40.]]))
    lam = time_intensities(L)
    v, se, n = relative_time_index(L, lam, min_items=3)
    assert n.tolist() == [3, 1, 4]
    assert np.isnan(v[1]) and np.isnan(se[1])
    assert np.isfinite(v[[0, 2]]).all() and (se[[0, 2]] > 0).all()


def test_poisson_binomial_matches_enumeration():
    p = np.array([0.2, 0.7, 0.5])
    d = poisson_binomial(p[None, :])[0]
    exp = np.zeros(4)
    for y in itertools.product([0, 1], repeat=3):
        exp[sum(y)] += np.prod([pi if yi else 1 - pi for pi, yi in zip(p, y)])
    assert np.allclose(d, exp)
    assert np.isclose(d.sum(), 1)


def test_interval_from_distribution():
    d = np.array([[0.04, 0.5, 0.42, 0.04]])
    lo, hi = interval_from_distribution(d, 0.9)
    assert (lo[0], hi[0]) == (1, 2)


def test_masked_score_distribution_ignores_unobserved_items():
    Pn = np.array([[0.5, 0.5], [0.9, 0.9]])
    post = np.array([[1.0, 0.0]])
    dist = masked_score_distribution(Pn, post, np.array([[True, False]]))
    assert np.allclose(dist[0], [0.5, 0.5, 0.0])


def test_joint_model_recovers_duration_curve_and_time_sharpens_theta():
    Y, v, g, th, C = _sim()
    m = ContinuousIRTDur(guessing=True, max_iter=300, tol=1e-7).fit(Y, v, None, g)
    assert np.allclose(m.C, C, atol=0.08)
    assert abs(m.delta - 0.2) < 0.08 and abs(m.ls0 + 1.0) < 0.08
    assert abs(m.gamma[1] - 0.4) < 0.05
    _, eap_r, sd_r, _ = m.posterior(Y[:, :8], None, None, g) if False else m.posterior(np.where(np.arange(Y.shape[1]) < 8, Y, np.nan), None, None, g)
    _, eap_j, sd_j, _ = m.posterior(np.where(np.arange(Y.shape[1]) < 8, Y, np.nan), v, None, g)
    assert sd_j.mean() < sd_r.mean()                     # informative time narrows the ability posterior
    assert np.corrcoef(eap_j, th)[0, 1] > np.corrcoef(eap_r, th)[0, 1]


def test_coordinates_are_standardised_and_intervals_cover():
    Y, v, g, th, C = _sim(seed=1)
    m = ContinuousIRTDur(guessing=True, max_iter=300, tol=1e-7).fit(Y, v, None, g)
    co = m.coordinates(Y, v, np.zeros_like(v), g, level=0.9)
    assert abs(co.rel_time.mean()) < 0.1 and 0.85 < co.rel_time.std() < 1.15
    cover = ((th >= co.theta_lo) & (th <= co.theta_hi)).mean()
    assert 0.86 < cover < 0.94                            # nominal 90 % ability interval


def test_separation_strata():
    assert separation_strata(0.8) == pytest.approx((4 * 2 + 1) / 3)

import numpy as np
from mcq_profiles.simulation import simulate
from mcq_profiles.latent_classes import MixIRT


def _data(n=3000, seed=0, **kw):
    rng = np.random.default_rng(seed)
    J = 15
    b = np.linspace(-2, 1.5, J)
    a = rng.uniform(0.7, 1.8, J)
    return a, b, simulate(n, a, b, rng=seed + 1, **kw)


def test_2pl_recovery():
    a, b, d = _data()
    m = MixIRT(1, "2pl").fit(d["Y"])
    assert np.corrcoef(m.a[0], a)[0, 1] > 0.9
    assert np.sqrt(((m.b[0] - b) ** 2).mean()) < 0.15


def test_rasch_runs_and_monotone_em():
    _, b, d = _data(n=1500)
    m = MixIRT(1, "rasch").fit(d["Y"])
    h = np.array(m.result_.history)
    assert np.all(np.diff(h) > -1e-6)
    assert np.allclose(m.a, 1.0)


def test_duration_censored_recovery():
    _, _, d = _data(dur=dict(alpha=np.log(17), beta=-0.15, s=0.4), limit=25)
    assert d["cens"].mean() > 0.05
    m = MixIRT(1, "2pl", duration=True).fit(d["Y"], d["logT"], d["cens"])
    assert abs(m.alpha[0] - np.log(17)) < 0.05
    assert abs(m.beta[0] + 0.15) < 0.05
    assert abs(np.exp(m.logs[0]) - 0.4) < 0.03


def test_missing_responses_and_durations():
    _, _, d = _data(n=1000, missing_rate=0.2, dur=dict(alpha=3.0, beta=0.1, s=0.3))
    logT = d["logT"].copy()
    logT[:100] = np.nan
    m = MixIRT(1, "2pl", duration=True).fit(d["Y"], logT)
    assert np.isfinite(m.result_.loglik)

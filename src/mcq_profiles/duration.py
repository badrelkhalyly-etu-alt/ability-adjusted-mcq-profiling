"""Duration-model helpers. The conditional duration likelihood itself lives in
``latent_classes.MixIRT`` (log T | theta, c ~ N(alpha_c + gamma_g + beta_c theta, s_c^2),
right-censored when a limit is supplied). Total completion time is NOT item-level speed."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def lognormal_marginal_loglik(logT_train, logT_test, cens_test=None, limit=None):
    """Held-out log density of log T under an intercept-only (no ability) normal model."""
    tr = logT_train[~np.isnan(logT_train)]
    m, s = tr.mean(), tr.std()
    te = logT_test[~np.isnan(logT_test)]
    return norm.logpdf(te, m, s)


def conditional_duration_density(model, Y, logT, group=None):
    """Held-out log density of log T given responses: log p(T, Y) - log p(Y)."""
    from .latent_classes import MixIRT
    assert isinstance(model, MixIRT) and model.duration
    full = model.loglik(Y, logT, None, group, per_person=True)
    model.duration = False
    try:
        resp = model.loglik(Y, None, None, group, per_person=True)
    finally:
        model.duration = True
    return full - resp

"""Homogeneous IRT convenience layer on top of the MixIRT engine (K = 1)."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from .latent_classes import MixIRT


def fit_rasch(Y, **kw):
    return MixIRT(1, "rasch", **kw).fit(Y)


def fit_2pl(Y, logT=None, cens=None, group=None, duration=False, **kw):
    return MixIRT(1, "2pl", duration=duration, group_effects=group is not None and duration, **kw).fit(Y, logT, cens, group)


def icc(a, b, theta):
    """Item characteristic curves, shape (len(theta), J)."""
    return expit(np.outer(theta, np.ones_like(b)) * a - a * b)


def item_information(a, b, theta):
    p = icc(a, b, theta)
    return a ** 2 * p * (1 - p)


def expected_probs(model, eap):
    """P(Y=1 | EAP theta) under a K=1 model (for residual / person-fit diagnostics)."""
    a = model.a[0]
    b = model.b[0]
    return expit(a[None, :] * (eap[:, None] - b[None, :]))

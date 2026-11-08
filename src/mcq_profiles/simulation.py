"""Data-generating processes for null (continuous) and positive-control (mixture) studies.

All generators return a dict with keys
    Y (N x J float, NaN = not administered), logT (N,), cens (N, bool), group (N,),
    theta (N,), cls (N,) true class (all 0 under the null).
Synthetic data are labelled ``synthetic=True`` and must never be reported as PISA
or FSBM results.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit


def _theta(rng, n, dist="normal", shape=None):
    if dist == "normal":
        return rng.standard_normal(n)
    if dist == "skew":  # standardised skew-normal-like via sinh-arcsinh
        e, d = shape if shape is not None else (-0.5, 1.0)
        z = np.sinh(d * np.arcsinh(rng.standard_normal(n)) - e)
        return (z - z.mean()) / z.std()
    if dist == "empirical":  # resample from supplied values, add small jitter
        v = np.asarray(shape)
        z = rng.choice(v, n) + rng.normal(0, 0.05, n)
        return (z - z.mean()) / z.std()
    raise ValueError(dist)


def simulate(n=2000, a=None, b=None, *, rng=None, theta_dist="normal", theta_shape=None,
             n_classes=1, class_props=None, class_b_shift=None, class_mu=None,
             dur=None, limit=None, group_props=None, group_theta_shift=None, group_dur_shift=None,
             class_group_dep=0.0, missing_rate=0.0):
    """Generic generator.

    dur = dict(alpha=..., beta=..., s=..., class_alpha=[...]) for log T = alpha + alpha_c + gamma_g
          + beta * theta + s * eps.  limit: right-censoring point on the minute scale.
    class_b_shift: (K, J) array added to b for class k (pattern differences, should be
          centred over items to avoid confounding with ability).
    class_group_dep: in [0,1); larger values tie classes to groups (confounding).
    """
    rng = np.random.default_rng(rng)
    a = np.ones_like(b) if a is None else np.asarray(a, float)
    b = np.asarray(b, float)
    J = len(b)
    G = 1 if group_props is None else len(group_props)
    group = rng.choice(G, n, p=group_props) if G > 1 else np.zeros(n, int)
    K = n_classes
    if K > 1:
        props = np.full(K, 1 / K) if class_props is None else np.asarray(class_props)
        cls = rng.choice(K, n, p=props)
        if class_group_dep > 0 and G > 1:
            sw = rng.random(n) < class_group_dep
            cls[sw] = group[sw] % K
    else:
        cls = np.zeros(n, int)
    theta = _theta(rng, n, theta_dist, theta_shape)
    if class_mu is not None:
        theta = theta + np.asarray(class_mu)[cls]
    if group_theta_shift is not None:
        theta = theta + np.asarray(group_theta_shift)[group]
    bb = b[None, :] + (np.asarray(class_b_shift)[cls] if class_b_shift is not None else 0.0)
    p = expit(a[None, :] * (theta[:, None] - bb))
    Y = (rng.random((n, J)) < p).astype(float)
    if missing_rate > 0:
        Y[rng.random((n, J)) < missing_rate] = np.nan
    logT = np.zeros(n)
    cens = np.zeros(n, bool)
    if dur is not None:
        alpha_c = np.asarray(dur.get("class_alpha", np.zeros(K)))[cls]
        beta_c = np.asarray(dur.get("class_beta", np.full(K, dur.get("beta", 0.0))))[cls]
        gam = np.asarray(group_dur_shift)[group] if group_dur_shift is not None else 0.0
        logT = dur["alpha"] + alpha_c + gam + beta_c * theta + dur["s"] * rng.standard_normal(n)
        if limit is not None:
            L = np.log(limit)
            cens = logT >= L
            logT = np.minimum(logT, L)
    return dict(Y=Y, logT=logT, cens=cens, group=group, theta=theta, cls=cls, synthetic=True)


def centred_shift(J, K, magnitude, rng=None, n_items_affected=None):
    """Class-specific difficulty shifts, centred over items within each class."""
    rng = np.random.default_rng(rng)
    S = np.zeros((K, J))
    m = J // 2 if n_items_affected is None else n_items_affected
    for k in range(1, K):
        idx = rng.choice(J, m, replace=False)
        S[k, idx] = magnitude * rng.choice([-1, 1], m)
        S[k] -= S[k].mean()
    return S

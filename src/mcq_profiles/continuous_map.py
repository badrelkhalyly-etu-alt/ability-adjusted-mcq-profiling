"""Uncertainty-aware continuous representation of examinees: ability x relative time.

Two coordinates per student, both with an uncertainty:

* **ability** ``theta``: posterior mean and SD under a unidimensional IRT model with item-specific lower
  asymptotes (G3PL, ``ContinuousIRT(guessing=True)``), computed from the item responses only;
* **relative time** ``rel_time``: how much longer (positive) or shorter (negative) the student took than
  expected for a student of the same ability, on the same booklet, *given the items the student actually
  reached and their time intensities*. It is a standardised residual, not a psychological speed: it mixes
  working pace with checking, rapid guessing, pauses and anything else that changes time.

The relative-time index is built in two steps:

1. item-intensity-adjusted log time ``v_i = mean_j (log t_ij - lambda_j)`` over the timed items the student
   reached (``lambda_j`` = mean log time of item j among training students); its sampling SE is the
   within-student SD of these residuals over sqrt(number of items);
2. a joint model (``ContinuousIRTDur``) for responses and ``v``: ``v | theta, g ~ N(m(theta) + gamma_g,
   s(theta)^2)`` with a cubic ``m`` and log-linear ``s`` in theta. The relative-time coordinate is
   ``(v - m(theta) - gamma_g) / s(theta)`` averaged over the ability posterior.

The same joint model also predicts unseen item responses from observed responses *and* time, which is how
the representation is compared with discrete profiles out of sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from .latent_classes import ContinuousIRT, FitResult, LOG2PI


# --------------------------------------------------------------------------- relative-time index
def time_intensities(log_rt: np.ndarray) -> np.ndarray:
    """lambda_j: mean log item time over students with a positive time on item j."""
    return np.nanmean(log_rt, 0)


def relative_time_index(log_rt: np.ndarray, lam: np.ndarray, min_items: int = 3):
    """Item-intensity-adjusted mean log time over reached, timed items.

    Returns (v, se, n_items). ``v`` is NaN for students with fewer than ``min_items`` timed items."""
    R = log_rt - lam[None, :]
    n = (~np.isnan(R)).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.nanmean(np.where(n[:, None] > 0, R, 0.0), 1)
        sd = np.nanstd(np.where(n[:, None] > 1, R, 0.0), 1, ddof=1)
    v = np.where(n >= min_items, v, np.nan)
    pooled = np.nanmedian(sd[n >= min_items]) if (n >= min_items).any() else np.nan
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, pooled)
    se = np.where(n >= min_items, sd / np.sqrt(np.maximum(n, 1)), np.nan)
    return v, se, n


# --------------------------------------------------------------------------- joint model
class ContinuousIRTDur(ContinuousIRT):
    """Unidimensional IRT (optionally with guessing) jointly with a continuous time variable ``v``:
    v | theta, group g ~ N(sum_p C_p theta^p + gamma_g, exp(ls0 + delta * theta)^2).
    theta ~ N(0, 1) (fixed quadrature), so no rescaling is needed. ``gamma`` of the first group is 0."""

    def __init__(self, guessing=True, deg=3, het=True, **kw):
        super().__init__(dims=1, guessing=guessing, **kw)
        self.deg, self.het = deg, het
        self.duration = True
        self.x = self.X[:, 0]

    # -- duration part
    def _powers(self):
        return self.x[None, :] ** np.arange(self.deg + 1)[:, None]            # (deg+1, Q)

    def _mean_ls(self, C, ls0, delta, gamma, group):
        mean = (C @ self._powers())[None, :] + gamma[group][:, None]       # N,Q
        ls = ls0 + (delta * self.x if self.het else 0.0 * self.x)         # Q
        return mean, ls

    def _dur_ll(self, v, group):
        mean, ls = self._mean_ls(self.C, self.ls0, self.delta, self.gamma, group)
        z = (np.nan_to_num(v)[:, None] - mean) / np.exp(ls)[None, :]
        ll = -0.5 * z ** 2 - ls[None, :] - 0.5 * LOG2PI
        ll[np.isnan(v)] = 0.0
        return ll

    def _estep_joint(self, Yf, M, v, group):
        p = np.clip(self._p(), 1e-12, 1 - 1e-12)
        ll = Yf @ np.log(p).T + (M - Yf) @ np.log1p(-p).T + self.logw[None, :]
        if v is not None:
            ll = ll + self._dur_ll(v, group)
        lli = logsumexp(ll, 1)
        return np.exp(ll - lli[:, None]), lli

    def _mstep_duration(self, post, v, group):
        G, D = self.G, self.deg + 1
        obs = ~np.isnan(v)
        post = post * obs[:, None]
        vv = np.nan_to_num(v)
        X = self._powers()

        def unpack(t):
            C = t[:D]
            ls0 = t[D]
            delta = t[D + 1] if self.het else 0.0
            o = D + (2 if self.het else 1)
            ga = np.concatenate([[0.0], t[o:]]) if G > 1 else np.zeros(1)
            return C, ls0, delta, ga

        def f(t):
            C, ls0, delta, ga = unpack(t)
            mean, ls = self._mean_ls(C, ls0, delta, ga, group)
            s = np.exp(ls)[None, :]
            z = (vv[:, None] - mean) / s
            ll = -0.5 * z ** 2 - ls[None, :]
            gm = post * z / s                      # d ll / d mean
            gl = post * (z ** 2 - 1.0)             # d ll / d ls
            grad = [X @ gm.sum(0), [gl.sum()]]
            if self.het:
                grad.append([(gl * self.x[None, :]).sum()])
            if G > 1:
                gg = np.zeros(G)
                np.add.at(gg, group, gm.sum(1))
                grad.append(gg[1:])
            return -(post * ll).sum(), -np.concatenate(grad)

        t0 = np.concatenate([self.C, [self.ls0]] + ([[self.delta]] if self.het else []) + ([self.gamma[1:]] if G > 1 else []))
        r = minimize(f, t0, jac=True, method="L-BFGS-B", options={"maxiter": 200})
        if r.fun <= f(t0)[0]:
            self.C, self.ls0, self.delta, self.gamma = unpack(r.x)
            self.ls0 = max(self.ls0, -6.0)

    # -- fitting
    def fit(self, Y, v=None, cens=None, group=None, weights=None):
        """``v``: continuous time variable (NaN allowed); ``group``: integer booklet codes."""
        Y = np.asarray(Y, float)
        N, J = Y.shape
        group = np.zeros(N, int) if group is None else np.asarray(group, int)
        self.G = int(group.max()) + 1
        w = None if weights is None else np.asarray(weights, float) / np.mean(weights)
        M = (~np.isnan(Y)).astype(float)
        Yf = np.nan_to_num(Y)
        pj = (Yf.sum(0) + .5) / (M.sum(0) + 1)
        self.A = np.ones((J, 1))
        self.d = np.log(pj / (1 - pj))
        self.c = np.full(J, 0.15) if self.guessing else np.zeros(J)
        vv = None if v is None else np.asarray(v, float)
        ok = np.isfinite(vv) if vv is not None else None
        self.C = np.zeros(self.deg + 1)
        self.C[0] = np.nanmean(vv) if vv is not None else 0.0
        self.ls0 = np.log(np.nanstd(vv)) if vv is not None else 0.0
        self.delta = 0.0
        self.gamma = np.zeros(self.G)
        if vv is not None and self.G > 1:
            gm = pd.Series(vv[ok]).groupby(group[ok]).mean()
            for g_, m_ in gm.items():
                self.gamma[g_] = m_ - gm.iloc[0]
            self.C[0] = gm.iloc[0]
        hist, prev, conv = [], -np.inf, False
        for it in range(self.max_iter):
            post, lli = self._estep_joint(Yf, M, vv, group)
            if w is not None:
                post, lli = post * w[:, None], lli * w
            obj = lli.sum() - self._penalty()
            hist.append(obj)
            if abs(obj - prev) < self.tol * abs(obj):
                conv = True
                break
            prev = obj
            R = post.T @ Yf
            Nn = post.T @ M
            for j in range(J):
                self._mstep_item(j, R[:, j], Nn[:, j])
            if vv is not None:
                self._mstep_duration(post, vv, group)
        self.result_ = FitResult(lli.sum(), obj, it, conv, self.n_params(J), hist)
        self.start_logliks = [lli.sum()]
        return self

    def n_params(self, J, G=None):
        G = self.G if G is None else G
        return super().n_params(J) + (self.deg + 1) + 1 + (1 if self.het else 0) + (G - 1)

    def loglik(self, Y, v=None, cens=None, group=None, per_person=False):
        Y = np.asarray(Y, float)
        group = np.zeros(len(Y), int) if group is None else np.asarray(group, int)
        lli = self._estep_joint(np.nan_to_num(Y), (~np.isnan(Y)).astype(float), v, group)[1]
        return lli if per_person else lli.sum()

    def posterior(self, Y, v=None, cens=None, group=None):
        """Posterior over the theta grid given responses and (optionally) the time variable."""
        Y = np.asarray(Y, float)
        group = np.zeros(len(Y), int) if group is None else np.asarray(group, int)
        group = np.clip(group, 0, self.G - 1)
        post, _ = self._estep_joint(np.nan_to_num(Y), (~np.isnan(Y)).astype(float), v, group)
        eap = post @ self.x
        psd = np.sqrt(np.maximum(post @ self.x ** 2 - eap ** 2, 0))
        return np.ones((len(Y), 1)), eap, psd, post

    # -- representation
    def coordinates(self, Y, v, v_se, group, level=0.9):
        """Ability (responses only) and relative time, each with posterior SD and a central interval."""
        from scipy.stats import norm
        group = np.clip(np.asarray(group, int), 0, self.G - 1)
        _, th, th_sd, post = self.posterior(Y, None, None, group)          # responses only
        mean, ls = self._mean_ls(self.C, self.ls0, self.delta, self.gamma, group)
        s = np.exp(ls)[None, :]
        r = (np.nan_to_num(v)[:, None] - mean) / s                         # N,Q
        rel = (post * r).sum(1)
        rel_var_theta = np.maximum((post * r ** 2).sum(1) - rel ** 2, 0)
        s_i = (post * s).sum(1)
        rel_sd = np.sqrt(rel_var_theta + (np.nan_to_num(v_se) / s_i) ** 2)
        rel = np.where(np.isnan(v), np.nan, rel)
        rel_sd = np.where(np.isnan(v), np.nan, rel_sd)
        zq = norm.ppf(0.5 + level / 2)
        return pd.DataFrame(dict(theta=th, theta_sd=th_sd, theta_lo=th - zq * th_sd, theta_hi=th + zq * th_sd,
                                 rel_time=rel, rel_time_sd=rel_sd, rel_time_lo=rel - zq * rel_sd,
                                 rel_time_hi=rel + zq * rel_sd))


# --------------------------------------------------------------------------- predictive score intervals
def poisson_binomial(P: np.ndarray) -> np.ndarray:
    """Distribution of the number of successes for independent Bernoulli(P[..., j]); returns (..., J+1)."""
    shape = P.shape[:-1]
    J = P.shape[-1]
    dist = np.zeros(shape + (J + 1,))
    dist[..., 0] = 1.0
    for j in range(J):
        p = P[..., j][..., None]
        new = dist * (1 - p)
        new[..., 1:] += dist[..., :-1] * p
        dist = new
    return dist


def masked_score_distribution(P_nodes: np.ndarray, post: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Predictive distribution of the number correct on a student's observed masked items.

    P_nodes: (Q, m) item probabilities at the quadrature nodes; post: (N, Q) posterior;
    observed: (N, m) bool. Unobserved items get probability 0 (they do not count)."""
    P = np.where(observed[:, None, :], P_nodes[None, :, :], 0.0)       # N,Q,m
    return np.einsum("nq,nqk->nk", post, poisson_binomial(P))


def interval_from_distribution(dist: np.ndarray, level: float = 0.9):
    """Equal-tailed central interval [lo, hi] of discrete distributions on 0..K (rows)."""
    cdf = np.cumsum(dist, 1)
    a = (1 - level) / 2
    lo = (cdf < a - 1e-12).sum(1)
    hi = (cdf < 1 - a - 1e-12).sum(1)
    return lo, hi


# --------------------------------------------------------------------------- reliability / separation
def separation_strata(reliability: float) -> float:
    """Number of statistically distinct strata supported by a reliability (Wright & Masters, 1982)."""
    g = np.sqrt(reliability / (1 - reliability))
    return (4 * g + 1) / 3

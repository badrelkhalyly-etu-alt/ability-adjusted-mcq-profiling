"""Latent time regimes versus flexible continuous time distributions (option C).

All models describe a time variable ``v`` (total log time, or the item-intensity-adjusted relative log time of
``continuous_map.relative_time_index``) *conditional on ability theta*, where theta is integrated over the same
quadrature grid and the same response likelihood ``LY[n, q] = log p(Y_n | theta_q) + log w_q`` for every model.
The response part is fixed (item parameters estimated once on the training responses), so log-likelihoods,
BIC values and held-out conditional log-densities ``log p(v | Y)`` of different time models are strictly
comparable: same observations, same conditioning information, same response model.

Continuous families (no latent classes):
    location: linear, cubic or natural-spline function of theta (+ booklet intercepts)
    log-scale: constant, linear or spline function of theta (conditional variance)
    residual: normal, skew-normal, Student-t, or sinh-arcsinh (SHASH; skew and tail weight)
Latent-regime alternative:
    K regimes; regime k has its own intercept, linear slope and log-scale; curvature (theta^2, theta^3),
    the theta-slope of the log-scale and booklet intercepts are shared; regime prevalence is a logistic
    function of theta, so regimes are *not* ability groups by construction. Regimes are ordered by intercept
    (identification of label switching); prevalences are bounded away from 0 by the softmax.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.special import gammaln, log_ndtr, logsumexp, ndtr
from scipy.stats import t as student_t

LOG2PI = np.log(2 * np.pi)


# --------------------------------------------------------------------------- response part
def item_loglik_grid(model, Y):
    """N x Q matrix log p(Y_n | theta_q) + log prior weight, for a fitted ``ContinuousIRT`` (dims=1)."""
    Y = np.asarray(Y, float)
    M = (~np.isnan(Y)).astype(float)
    Yf = np.nan_to_num(Y)
    p = np.clip(model._p(), 1e-12, 1 - 1e-12)
    return Yf @ np.log(p).T + (M - Yf) @ np.log1p(-p).T + model.logw[None, :]


# --------------------------------------------------------------------------- bases
def spline_basis(x, df=5, lo=-3.0, hi=3.0):
    """Cubic B-spline basis (no intercept column) on [lo, hi]; x outside is clamped (linear tails avoided)."""
    xc = np.clip(x, lo, hi)
    k = 3
    inner = np.linspace(lo, hi, df - k + 1)[1:-1] if df > k + 1 else np.array([])
    t = np.concatenate([[lo] * (k + 1), inner, [hi] * (k + 1)])
    nb = len(t) - k - 1
    B = np.column_stack([BSpline(t, np.eye(nb)[i], k, extrapolate=False)(xc) for i in range(nb)])
    B = np.nan_to_num(B)
    return B[:, 1:]                                               # drop first column: intercept is separate


def design(x, kind, df=5):
    if kind == "const":
        return np.zeros((len(x), 0))
    if kind == "linear":
        return x[:, None]
    if kind == "cubic":
        return np.column_stack([x, x ** 2, x ** 3])
    if kind == "spline":
        return spline_basis(x, df)
    raise ValueError(kind)


# --------------------------------------------------------------------------- residual families
def _logpdf_cdf(z, family, shape):
    """log density and cdf of the standardised residual z for each family (density of v = mean + s z
    is this log density minus log s)."""
    if family == "normal":
        return -0.5 * z ** 2 - 0.5 * LOG2PI, ndtr(z)
    if family == "skewnormal":
        a = shape[0]
        lp = np.log(2) - 0.5 * z ** 2 - 0.5 * LOG2PI + log_ndtr(a * z)
        from scipy.stats import skewnorm
        return lp, skewnorm.cdf(z, a)
    if family == "t":
        nu = 2.0 + np.exp(shape[0])
        lp = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
              - (nu + 1) / 2 * np.log1p(z ** 2 / nu))
        return lp, student_t.cdf(z, nu)
    if family == "shash":                                   # Jones & Pewsey (2009) sinh-arcsinh
        eps, delta = shape[0], np.exp(shape[1])
        s_ = np.sinh(delta * np.arcsinh(z) - eps)
        c_ = np.cosh(delta * np.arcsinh(z) - eps)
        lp = np.log(delta) + np.log(c_) - 0.5 * np.log1p(z ** 2) - 0.5 * s_ ** 2 - 0.5 * LOG2PI
        return lp, ndtr(s_)
    raise ValueError(family)


N_SHAPE = {"normal": 0, "skewnormal": 1, "t": 1, "shash": 2}


# --------------------------------------------------------------------------- autodiff objective
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
_LOG2PI_T = float(LOG2PI)


def _t_logpdf(z, family, sh):
    if family == "normal":
        return -0.5 * z ** 2 - 0.5 * _LOG2PI_T
    if family == "skewnormal":
        return np.log(2) - 0.5 * z ** 2 - 0.5 * _LOG2PI_T + torch.special.log_ndtr(sh[0] * z)
    if family == "t":
        nu = 2.0 + torch.exp(sh[0])
        return (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * np.pi)
                - (nu + 1) / 2 * torch.log1p(z ** 2 / nu))
    if family == "shash":
        delta = torch.exp(sh[1])
        h = delta * torch.asinh(z) - sh[0]
        logcosh = torch.abs(h) + torch.log1p(torch.exp(-2 * torch.abs(h))) - np.log(2)
        return torch.log(delta) + logcosh - 0.5 * torch.log1p(z ** 2) - 0.5 * torch.sinh(h) ** 2 - 0.5 * _LOG2PI_T
    raise ValueError(family)


def _minimize_torch(fun_t, p0, maxiter):
    """L-BFGS-B on a torch scalar function of a parameter vector (exact gradients by autodiff)."""
    def f(p):
        pt = torch.tensor(p, requires_grad=True)
        val = fun_t(pt)
        val.backward()
        v = float(val.detach())
        g = pt.grad.detach().numpy().copy()
        if not np.isfinite(v):
            return 1e300, np.zeros_like(p)
        return v, g
    return minimize(f, p0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))


# --------------------------------------------------------------------------- continuous model
@dataclass
class FitInfo:
    loglik: float
    n_params: int
    converged: bool
    n_obs: int


class ContinuousTimeModel:
    """v | theta, g ~ location(theta) + gamma_g + exp(logscale(theta)) * Z,  Z ~ family (standardised)."""

    def __init__(self, family="normal", loc="cubic", scale="linear", df=5, maxiter=500):
        self.family, self.loc, self.scale, self.df, self.maxiter = family, loc, scale, df, maxiter
        self.name = f"{family}|{loc}|{scale}"

    def _setup(self, x, G):
        self.x = x
        self.Bm = design(x, self.loc, self.df)
        self.Bs = design(x, self.scale, max(3, self.df - 2))
        self.G = G
        self.sizes = [1, self.Bm.shape[1], 1, self.Bs.shape[1], N_SHAPE[self.family], G - 1]

    def _unpack(self, p):
        out, o = [], 0
        for s in self.sizes:
            out.append(p[o:o + s])
            o += s
        a0, bm, l0, bs, sh, ga = out
        return a0[0], bm, l0[0], bs, sh, np.concatenate([[0.0], ga])

    def _logpdf_grid(self, p, v, g, with_cdf=False):
        a0, bm, l0, bs, sh, ga = self._unpack(p)
        mean = a0 + self.Bm @ bm                       # Q
        ls = l0 + self.Bs @ bs                          # Q
        z = (v[:, None] - mean[None, :] - ga[g][:, None]) / np.exp(ls)[None, :]
        lp, cdf = _logpdf_cdf(z, self.family, sh) if with_cdf else (_logpdf_cdf_fast(z, self.family, sh), None)
        return lp - ls[None, :], cdf

    def _negll(self, p, LY, v, g, w):
        lp, _ = self._logpdf_grid(p, v, g)
        ll = logsumexp(LY + lp, 1) - logsumexp(LY, 1)
        return -(w * ll).sum() if w is not None else -ll.sum()

    def _start(self, v, g):
        p0 = np.zeros(sum(self.sizes))
        p0[0] = np.nanmean(v)
        p0[1 + self.sizes[1]] = np.log(np.nanstd(v))
        if self.family == "t":
            p0[2 + self.sizes[1] + self.sizes[3]] = np.log(8.0)
        return p0

    def fit(self, LY, v, g, x, weights=None, start=None):
        """LY: N x Q response log-likelihood grid (+ log prior); v: time variable (NaN rows are dropped);
        g: booklet codes; x: quadrature nodes."""
        ok = np.isfinite(v)
        self._setup(x, int(np.max(g)) + 1)
        w = None if weights is None else np.asarray(weights, float)[ok] / np.mean(np.asarray(weights, float)[ok])
        p0 = self._start(v[ok], g[ok]) if start is None else start
        LYt = torch.tensor(LY[ok])
        base = torch.logsumexp(LYt, 1)
        vt, gt = torch.tensor(v[ok]), torch.tensor(g[ok])
        Bm, Bs = torch.tensor(self.Bm), torch.tensor(self.Bs)
        wt = None if w is None else torch.tensor(w)
        idx = np.cumsum([0] + self.sizes)

        def fun(pt):
            a0, bm, l0, bs = pt[0], pt[idx[1]:idx[2]], pt[idx[2]], pt[idx[3]:idx[4]]
            sh = pt[idx[4]:idx[5]]
            ga = torch.cat([torch.zeros(1), pt[idx[5]:idx[6]]])
            mean = a0 + Bm @ bm
            ls = l0 + Bs @ bs
            z = (vt[:, None] - mean[None, :] - ga[gt][:, None]) / torch.exp(ls)[None, :]
            lp = _t_logpdf(z, self.family, sh) - ls[None, :]
            ll = torch.logsumexp(LYt + lp, 1) - base
            return -(ll * wt).sum() if wt is not None else -ll.sum()

        r = _minimize_torch(fun, p0, self.maxiter)
        self.p, self.info = r.x, FitInfo(-r.fun, len(r.x), bool(r.success), int(ok.sum()))
        return self

    def bic(self):
        return -2 * self.info.loglik + self.info.n_params * np.log(self.info.n_obs)

    def cond_loglik(self, LY, v, g):
        """Per-person log p(v | Y) (NaN where v is missing)."""
        out = np.full(len(v), np.nan)
        ok = np.isfinite(v)
        lp, _ = self._logpdf_grid(self.p, v[ok], np.clip(g[ok], 0, self.G - 1))
        out[ok] = logsumexp(LY[ok] + lp, 1) - logsumexp(LY[ok], 1)
        return out

    def pit(self, LY, v, g):
        """Probability integral transform F(v | Y) for calibration checks."""
        out = np.full(len(v), np.nan)
        ok = np.isfinite(v)
        _, cdf = self._logpdf_grid(self.p, v[ok], np.clip(g[ok], 0, self.G - 1), with_cdf=True)
        post = np.exp(LY[ok] - logsumexp(LY[ok], 1)[:, None])
        out[ok] = (post * cdf).sum(1)
        return out

    def sample(self, theta, g, rng):
        """Draw v given theta (any values) and booklet."""
        a0, bm, l0, bs, sh, ga = self._unpack(self.p)
        Bm = design(theta, self.loc, self.df)
        Bs = design(theta, self.scale, max(3, self.df - 2))
        mean = a0 + Bm @ bm + ga[g]
        s = np.exp(l0 + Bs @ bs)
        return mean + s * _draw(self.family, sh, len(theta), rng)


def _logpdf_cdf_fast(z, family, shape):
    """Log density only (no cdf), used inside the optimiser."""
    if family == "normal":
        return -0.5 * z ** 2 - 0.5 * LOG2PI
    if family == "skewnormal":
        return np.log(2) - 0.5 * z ** 2 - 0.5 * LOG2PI + log_ndtr(shape[0] * z)
    if family == "t":
        nu = 2.0 + np.exp(shape[0])
        return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
                - (nu + 1) / 2 * np.log1p(z ** 2 / nu))
    if family == "shash":
        eps, delta = shape[0], np.exp(shape[1])
        h = delta * np.arcsinh(z) - eps
        return np.log(delta) + np.log(np.cosh(h)) - 0.5 * np.log1p(z ** 2) - 0.5 * np.sinh(h) ** 2 - 0.5 * LOG2PI
    raise ValueError(family)


def _draw(family, shape, n, rng):
    if family == "normal":
        return rng.normal(size=n)
    if family == "skewnormal":
        from scipy.stats import skewnorm
        return skewnorm.rvs(shape[0], size=n, random_state=rng)
    if family == "t":
        return rng.standard_t(2.0 + np.exp(shape[0]), size=n)
    if family == "shash":
        eps, delta = shape[0], np.exp(shape[1])
        return np.sinh((np.arcsinh(rng.normal(size=n)) + eps) / delta)
    raise ValueError(family)


# --------------------------------------------------------------------------- latent regimes
class RegimeTimeModel:
    """K latent time regimes (normal within regime), conditional on theta; see module docstring."""

    def __init__(self, K=2, n_starts=5, seed=0, maxiter=800):
        self.K, self.n_starts, self.seed, self.maxiter = K, n_starts, seed, maxiter
        self.name = f"regimes|K={K}"

    def _setup(self, x, G):
        self.x, self.G = x, G
        K = self.K
        # a_k, b_k, l_k (K each), shared c2, c3, d, prevalence e_k, f_k (K-1 each), gamma (G-1)
        self.sizes = [K, K, K, 1, 1, 1, K - 1, K - 1, G - 1]

    def _unpack(self, p):
        out, o = [], 0
        for s in self.sizes:
            out.append(p[o:o + s])
            o += s
        a, b, l, c2, c3, d, e, f, ga = out
        return a, b, l, c2[0], c3[0], d[0], np.concatenate([[0.0], e]), np.concatenate([[0.0], f]), np.concatenate([[0.0], ga])

    def _logpdf_grid(self, p, v, g, with_cdf=False, per_regime=False):
        a, b, l, c2, c3, d, e, f, ga = self._unpack(p)
        x = self.x
        mean = a[:, None] + b[:, None] * x[None] + c2 * x[None] ** 2 + c3 * x[None] ** 3          # K,Q
        ls = l[:, None] + d * x[None]                                                              # K,Q
        lpi = e[:, None] + f[:, None] * x[None]
        lpi = lpi - logsumexp(lpi, 0)[None]                                                       # K,Q
        z = (v[:, None, None] - mean[None] - ga[g][:, None, None]) / np.exp(ls)[None]             # N,K,Q
        lk = lpi[None] - 0.5 * z ** 2 - 0.5 * LOG2PI - ls[None]
        if per_regime:
            return lk
        lp = logsumexp(lk, 1)
        cdf = (np.exp(lpi)[None] * ndtr(z)).sum(1) if with_cdf else None
        return lp, cdf

    def _negll(self, p, LY, v, g, w):
        lp, _ = self._logpdf_grid(p, v, g)
        ll = logsumexp(LY + lp, 1) - logsumexp(LY, 1)
        return -(w * ll).sum() if w is not None else -ll.sum()

    def fit(self, LY, v, g, x, weights=None):
        ok = np.isfinite(v)
        self._setup(x, int(np.max(g)) + 1)
        w = None if weights is None else np.asarray(weights, float)[ok] / np.mean(np.asarray(weights, float)[ok])
        LYt = torch.tensor(LY[ok])
        base = torch.logsumexp(LYt, 1)
        vt, gt = torch.tensor(v[ok]), torch.tensor(g[ok])
        xt = torch.tensor(x)
        wt = None if w is None else torch.tensor(w)
        idx = np.cumsum([0] + self.sizes)
        K = self.K

        def fun(pt):
            a, b, l = pt[0:K], pt[K:2 * K], pt[2 * K:3 * K]
            c2, c3, d = pt[3 * K], pt[3 * K + 1], pt[3 * K + 2]
            e = torch.cat([torch.zeros(1), pt[idx[6]:idx[7]]])
            f = torch.cat([torch.zeros(1), pt[idx[7]:idx[8]]])
            ga = torch.cat([torch.zeros(1), pt[idx[8]:idx[9]]])
            mean = a[:, None] + b[:, None] * xt[None] + c2 * xt[None] ** 2 + c3 * xt[None] ** 3
            ls = l[:, None] + d * xt[None]
            lpi = e[:, None] + f[:, None] * xt[None]
            lpi = lpi - torch.logsumexp(lpi, 0)[None]
            z = (vt[:, None, None] - mean[None] - ga[gt][:, None, None]) / torch.exp(ls)[None]
            lk = lpi[None] - 0.5 * z ** 2 - 0.5 * _LOG2PI_T - ls[None]
            lp = torch.logsumexp(lk, 1)
            ll = torch.logsumexp(LYt + lp, 1) - base
            return -(ll * wt).sum() if wt is not None else -ll.sum()

        rng = np.random.default_rng(self.seed)
        m, s = np.nanmean(v[ok]), np.nanstd(v[ok])
        best = None
        self.start_logliks = []
        for st in range(self.n_starts):
            p0 = np.zeros(sum(self.sizes))
            K = self.K
            p0[:K] = np.sort(m + s * rng.normal(0, 0.8, K))
            p0[K:2 * K] = rng.normal(0, 0.1, K)
            p0[2 * K:3 * K] = np.log(s) - 0.3 + rng.normal(0, 0.1, K)
            r = _minimize_torch(fun, p0, self.maxiter)
            self.start_logliks.append(-r.fun)
            if best is None or r.fun < best.fun:
                best = r
        self.p = self._order(best.x)
        self.info = FitInfo(-best.fun, len(best.x), bool(best.success), int(ok.sum()))
        return self

    def _order(self, p):
        """Order regimes by intercept a_k (label-switching identification); prevalence parameters are
        re-expressed relative to the new first regime."""
        a, b, l, c2, c3, d, e, f, ga = self._unpack(p)
        o = np.argsort(a)
        e, f = e[o] - e[o][0], f[o] - f[o][0]
        return np.concatenate([a[o], b[o], l[o], [c2, c3, d], e[1:], f[1:], ga[1:]])

    def bic(self):
        return -2 * self.info.loglik + self.info.n_params * np.log(self.info.n_obs)

    cond_loglik = ContinuousTimeModel.cond_loglik
    pit = ContinuousTimeModel.pit

    def regime_posterior(self, LY, v, g):
        """N x K posterior regime probabilities given responses and time."""
        ok = np.isfinite(v)
        out = np.full((len(v), self.K), np.nan)
        lk = self._logpdf_grid(self.p, v[ok], np.clip(g[ok], 0, self.G - 1), per_regime=True)   # N,K,Q
        j = LY[ok][:, None, :] + lk
        r = logsumexp(j, 2)
        out[ok] = np.exp(r - logsumexp(r, 1)[:, None])
        return out

    def summary(self):
        a, b, l, c2, c3, d, e, f, ga = self._unpack(self.p)
        lpi = e[:, None] + f[:, None] * self.x[None]
        pi = np.exp(lpi - logsumexp(lpi, 0)[None])
        w0 = np.exp(-0.5 * self.x ** 2)
        w0 /= w0.sum()
        return dict(intercepts=a.tolist(), slopes=b.tolist(), log_scales=l.tolist(),
                    prevalence_at_theta0=pi[:, np.argmin(np.abs(self.x))].tolist(),
                    prevalence_marginal=(pi @ w0).tolist())

    def sample(self, theta, g, rng):
        a, b, l, c2, c3, d, e, f, ga = self._unpack(self.p)
        lpi = e[:, None] + f[:, None] * theta[None]
        pi = np.exp(lpi - logsumexp(lpi, 0)[None])                       # K,N
        u = rng.random(len(theta))
        k = (u[None] > np.cumsum(pi, 0)).sum(0).clip(max=self.K - 1)
        mean = a[k] + b[k] * theta + c2 * theta ** 2 + c3 * theta ** 3 + ga[g]
        return mean + np.exp(l[k] + d * theta) * rng.normal(size=len(theta)), k


CONTINUOUS_SPECS = [
    ("normal", "linear", "const"),      # M2 analogue
    ("normal", "cubic", "linear"),      # the cubic heteroscedastic competitor already in the paper
    ("normal", "spline", "spline"),     # GAM-type: spline mean and spline log-variance
    ("skewnormal", "spline", "spline"),
    ("t", "spline", "spline"),
    ("shash", "spline", "spline"),
]

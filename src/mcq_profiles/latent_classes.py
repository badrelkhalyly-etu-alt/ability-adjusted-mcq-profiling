"""Mixture IRT engine (EM on a class x quadrature-node grid) and plain LCA.

One estimator covers the model ladder of the protocol:

* ``MixIRT(n_classes=1)``                       homogeneous Rasch / 2PL (M0)
* ``MixIRT(K, class_specific=None)``            classes differ in ability only
                                                (flexible continuous ability; "type A")
* ``MixIRT(K, class_specific="b")``             class-specific item difficulties
                                                after ability adjustment ("type B", M1)
* ``MixIRT(..., duration=True)``                + conditional log-normal total-time
                                                model, right-censored at a limit (M2/M3)
* ``MixIRT(..., group_effects=True)``           + group-specific class prevalence and
                                                group duration intercepts (M4)

Model (person i, item j, class c, ability theta):
    P(Y_ij = 1 | theta, c) = sigmoid(a_j(c) * (theta - b_cj))
    theta | c ~ N(mu_c, sigma_c^2)
    log T_i | theta, c, g ~ N(alpha_c + gamma_g + beta_c * theta, s_c^2)   (censored at L)
    P(C_i = c | g) = pi_gc

Identification (all are exact reparametrisations, applied after every M-step):
    * mixture distribution of theta standardised to mean 0, variance 1
      (Rasch: mean 0 only, discriminations fixed at 1);
    * with class-specific b: mean_j b_cj equal across classes, so mu_c carries
      every uniform shift and b_cj - mean_c b_cj carries only *pattern* differences;
    * gamma for the first group fixed at 0.
Estimation: generalized EM, one damped Newton step per item per iteration,
ridge penalty ``ridge`` on class deviations of b, weak log-normal prior on a.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.special import expit, log_expit, logsumexp, log_ndtr
from scipy.optimize import minimize
from numpy.polynomial import polynomial as P

LOG2PI = np.log(2 * np.pi)


def _nodes(n_nodes: int, lim: float = 5.5):
    x = np.linspace(-lim, lim, n_nodes)
    return x


def _normal_logw(x, mu, sigma):
    """log weights of N(mu, sigma) on grid x, normalised (shape K x Q)."""
    lw = -0.5 * ((x[None, :] - mu[:, None]) / sigma[:, None]) ** 2
    return lw - logsumexp(lw, axis=1, keepdims=True)


@dataclass
class FitResult:
    loglik: float
    penalized: float
    n_iter: int
    converged: bool
    n_params: int
    history: list = field(default_factory=list)


class MixIRT:
    def __init__(self, n_classes=1, model="2pl", class_specific=None, duration=False,
                 group_effects=False, n_nodes=41, ridge=1.0, a_prior_sd=1.0,
                 max_iter=2000, tol=1e-7, n_starts=5, random_state=0, verbose=False):
        assert model in ("rasch", "2pl")
        assert class_specific in (None, "b", "ab")
        if n_classes == 1:
            class_specific = None
        self.K = n_classes
        self.model = model
        self.cs = class_specific
        self.duration = duration
        self.group_effects = group_effects
        self.x = _nodes(n_nodes)
        self.ridge = ridge
        self.a_prior_sd = a_prior_sd
        self.max_iter = max_iter
        self.tol = tol
        self.n_starts = n_starts
        self.rs = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ data
    @staticmethod
    def _prep(Y, logT=None, cens=None, group=None):
        Y = np.asarray(Y, dtype=float)
        M = ~np.isnan(Y)
        Yf = np.where(M, Y, 0.0)
        n = Y.shape[0]
        if group is None:
            group = np.zeros(n, dtype=int)
        if logT is None:
            logT = np.zeros(n)
        if cens is None:
            cens = np.zeros(n, dtype=bool)
        logT = np.asarray(logT, float)
        cens = np.asarray(cens, bool) & ~np.isnan(logT)
        return Yf, M.astype(float), logT, cens, np.asarray(group, int)

    # --------------------------------------------------------------- params
    def n_params(self, J, G):
        K = self.K
        p = J  # b
        if self.model == "2pl":
            p += J  # a
            if self.cs == "ab":
                p += (K - 1) * J
        if self.cs in ("b", "ab"):
            p += (K - 1) * (J - 1)
        p += 2 * K - (2 if self.model == "2pl" else 1)  # mu, sigma minus standardisation
        p += (K - 1) * (G if self.group_effects else 1)
        if self.duration:
            p += 3 * K + ((G - 1) if self.group_effects else 0)
        return p

    def _init(self, Yf, M, logT, cens, group, rng):
        K, J = self.K, Yf.shape[1]
        G = self.G
        pj = (Yf.sum(0) + 0.5) / (M.sum(0) + 1.0)
        b0 = -np.log(pj / (1 - pj)) / 1.0
        self.a = np.ones((K, J)) if self.cs == "ab" else np.ones((1, J))
        if self.model == "2pl":
            self.a = self.a * 1.2
        self.b = np.repeat(b0[None, :], K if self.cs else 1, 0)
        if K > 1:
            if self.cs in ("b", "ab"):
                self.b = self.b + rng.normal(0, 0.5, size=(K, J))
            self.mu = np.sort(rng.normal(0, 0.7, size=K))
        else:
            self.mu = np.zeros(1)
        self.sigma = np.ones(K) * (1.0 if K == 1 else 0.8)
        self.logpi = np.log(np.full((G, K), 1.0 / K))
        if self.duration:
            obs = ~cens & ~np.isnan(logT)
            m, s = logT[obs].mean(), logT[obs].std() + 1e-3
            self.alpha = np.full(K, m) + (rng.normal(0, 0.1 * s, K) if K > 1 else 0)
            self.beta = np.zeros(K)
            self.logs = np.full(K, np.log(s))
            self.gamma = np.zeros(G)

    # ---------------------------------------------------------------- E-step
    def _item_logp(self):
        a = self.a if self.a.shape[0] == self.K else np.repeat(self.a, self.K, 0)
        b = self.b if self.b.shape[0] == self.K else np.repeat(self.b, self.K, 0)
        eta = a[:, None, :] * (self.x[None, :, None] - b[:, None, :])  # K,Q,J
        return log_expit(eta), log_expit(-eta)

    def _dur_ll(self, logT, cens, group):
        """log-density of log T on the grid: N x K x Q."""
        mean = (self.alpha[None, :, None] + self.gamma[group][:, None, None]
                + self.beta[None, :, None] * self.x[None, None, :])
        s = np.exp(self.logs)[None, :, None]
        miss = np.isnan(logT)
        z = (np.nan_to_num(logT)[:, None, None] - mean) / s
        ll = -0.5 * z ** 2 - np.log(s) - 0.5 * LOG2PI
        if cens.any():
            ll[cens] = log_ndtr(-z[cens])
        ll[miss] = 0.0  # duration unobserved: contributes nothing
        return ll, z

    def _joint(self, Yf, M, logT, cens, group):
        lp, lq = self._item_logp()
        N, K, Q = Yf.shape[0], self.K, len(self.x)
        ll = np.empty((N, K, Q))
        for c in range(K):
            ll[:, c, :] = Yf @ lp[c].T + (M - Yf) @ lq[c].T
        lw = _normal_logw(self.x, self.mu, self.sigma)  # K,Q
        ll += lw[None] + self.logpi[group][:, :, None]
        if self.duration:
            ll += self._dur_ll(logT, cens, group)[0]
        return ll

    def _estep(self, *data):
        ll = self._joint(*data)
        N = ll.shape[0]
        flat = ll.reshape(N, -1)
        lli = logsumexp(flat, axis=1)
        post = np.exp(flat - lli[:, None]).reshape(ll.shape)
        return post, lli

    # ---------------------------------------------------------------- M-step
    def _penalty(self):
        pen = 0.0
        if self.K > 1 and self.cs in ("b", "ab"):
            pen += 0.5 * self.ridge * ((self.b - self.b.mean(0)) ** 2).sum()
        if self.model == "2pl":
            pen += 0.5 * (np.log(self.a) ** 2).sum() / self.a_prior_sd ** 2
        return pen

    def _mstep_items(self, post, Yf, M):
        J, x = Yf.shape[1], self.x
        R = np.einsum("nkq,nj->kqj", post, Yf)
        Nn = np.einsum("nkq,nj->kqj", post, M)
        K = self.b.shape[0]  # number of distinct item-parameter sets
        if K == 1:  # item parameters shared by all classes: pool expected counts
            R, Nn = R.sum(0, keepdims=True), Nn.sum(0, keepdims=True)

        def obj_item(a, b):  # a: (K or 1, J) b: (K,J) -> (J,) log-lik + log-prior
            aa = a if a.shape[0] == K else np.repeat(a, K, 0)
            eta = aa[:, None, :] * (x[None, :, None] - b[:, None, :])
            v = (R * log_expit(eta) + (Nn - R) * log_expit(-eta)).sum((0, 1))
            if K > 1 and self.cs in ("b", "ab"):
                v -= 0.5 * self.ridge * ((b - b.mean(0)) ** 2).sum(0)
            if self.model == "2pl":
                v -= 0.5 * (np.log(a) ** 2).sum(0) / self.a_prior_sd ** 2
            return v

        a, b = self.a.copy(), self.b.copy()
        aa = a if a.shape[0] == K else np.repeat(a, K, 0)
        eta = aa[:, None, :] * (x[None, :, None] - b[:, None, :])
        p = expit(eta)
        res = R - Nn * p                      # K,Q,J
        W = Nn * p * (1 - p)
        d = x[None, :, None] - b[:, None, :]   # K,Q,J
        # parameter vector per item: [a-params..., b_1..b_K]
        free_a = self.model == "2pl"
        na = (K if self.cs == "ab" else 1) if free_a else 0
        P = na + K
        g = np.zeros((J, P))
        H = np.zeros((J, P, P))
        # b blocks
        g[:, na:] = (-(aa[:, None, :]) * res).sum(1).T
        for c in range(K):
            H[:, na + c, na + c] = -(W[c] * aa[c][None, :] ** 2).sum(0)
        if free_a:
            if na == 1:
                g[:, 0] = (res * d).sum((0, 1))
                H[:, 0, 0] = -(W * d ** 2).sum((0, 1))
                for c in range(K):
                    h = (W[c] * aa[c][None, :] * d[c] - res[c]).sum(0)
                    H[:, 0, na + c] = h
                    H[:, na + c, 0] = h
                la = np.log(a[0])
                g[:, 0] += -la / (a[0] * self.a_prior_sd ** 2)
                H[:, 0, 0] += -1.0 / (a[0] ** 2 * self.a_prior_sd ** 2)
            else:
                for c in range(K):
                    g[:, c] = (res[c] * d[c]).sum(0)
                    H[:, c, c] = -(W[c] * d[c] ** 2).sum(0)
                    h = (W[c] * aa[c][None, :] * d[c] - res[c]).sum(0)
                    H[:, c, na + c] = h
                    H[:, na + c, c] = h
                    g[:, c] += -np.log(a[c]) / (a[c] * self.a_prior_sd ** 2)
                    H[:, c, c] += -1.0 / (a[c] ** 2 * self.a_prior_sd ** 2)
        if K > 1 and self.cs in ("b", "ab"):
            dev = b - b.mean(0)
            g[:, na:] += -self.ridge * dev.T
            H[:, na:, na:] += -self.ridge * (np.eye(K) - 1.0 / K)[None]
        # damped Newton with step halving (per item)
        A = -H + 1e-6 * np.eye(P)[None]
        lam = np.maximum(1e-8, -np.linalg.eigvalsh(A).min(1) + 1e-4)
        A = A + np.where(np.linalg.eigvalsh(A).min(1) <= 0, lam, 0)[:, None, None] * np.eye(P)[None]
        step = np.linalg.solve(A, g[..., None])[..., 0]
        step = np.clip(step, -1.0, 1.0)
        f0 = obj_item(a, b)
        t = np.ones(J)
        for _ in range(12):
            na_, nb_ = a.copy(), b.copy()
            if free_a:
                na_ = np.maximum(a + (t * step[:, :na].T), 0.05)
            nb_ = b + t * step[:, na:].T
            f1 = obj_item(na_, nb_)
            ok = f1 >= f0 - 1e-10
            if ok.all():
                break
            t = np.where(ok, t, t * 0.5)
        a_new = np.where(ok[None, :], na_, a) if free_a else a
        b_new = np.where(ok[None, :], nb_, b)
        self.a, self.b = a_new, b_new

    def _mstep_theta(self, post):
        wk = post.sum(2)              # N,K posterior class
        nq = post.sum(0)              # K,Q
        nk = nq.sum(1) + 1e-12
        mu = (nq * self.x[None]).sum(1) / nk
        var = (nq * (self.x[None] - mu[:, None]) ** 2).sum(1) / nk
        self.mu = mu
        self.sigma = np.sqrt(np.maximum(var, 0.05 ** 2))
        return wk

    def _mstep_pi(self, wk, group):
        G, K = self.G, self.K
        if self.group_effects:
            cnt = np.zeros((G, K))
            np.add.at(cnt, group, wk)
        else:
            cnt = np.repeat(wk.sum(0, keepdims=True), G, 0)
        cnt += 1e-3
        self.logpi = np.log(cnt / cnt.sum(1, keepdims=True))

    def _mstep_duration(self, post, logT, cens, group):
        K, G = self.K, self.G
        ge = self.group_effects and G > 1

        def unpack(v):
            al, be, ls = v[:K], v[K:2 * K], v[2 * K:3 * K]
            ga = np.concatenate([[0.0], v[3 * K:]]) if ge else np.zeros(G)
            return al, be, ls, ga

        x = self.x
        obs = ~np.isnan(logT)
        post = post * obs[:, None, None]
        logT = np.nan_to_num(logT)

        def f(v):
            al, be, ls, ga = unpack(v)
            mean = al[None, :, None] + ga[group][:, None, None] + be[None, :, None] * x[None, None, :]
            s = np.exp(ls)[None, :, None]
            z = (logT[:, None, None] - mean) / s
            ll = -0.5 * z ** 2 - np.log(s)
            dz = -z  # dl/dz for observed
            dls = np.full_like(z, -1.0)
            if cens.any():
                zc = z[cens]
                lc = log_ndtr(-zc)
                ll[cens] = lc
                hz = np.exp(-0.5 * zc ** 2 - 0.5 * LOG2PI - lc)  # phi(z)/Phi(-z)
                dz[cens] = -hz
                dls[cens] = 0.0
            wl = post * ll
            gz = post * dz  # dl/dz weighted
            # dz/dalpha = -1/s ; dz/dbeta = -x/s ; dz/dlogs = -z (plus explicit -1 for obs)
            ga_ = -(gz / s)
            g_al = ga_.sum((0, 2))
            g_be = (ga_ * x[None, None, :]).sum((0, 2))
            g_ls = (gz * (-z) + post * dls).sum((0, 2))
            grad = [g_al, g_be, g_ls]
            if ge:
                gg = np.zeros(G)
                np.add.at(gg, group, ga_.sum((1, 2)))
                grad.append(gg[1:])
            return -wl.sum(), -np.concatenate(grad)

        v0 = np.concatenate([self.alpha, self.beta, self.logs] + ([self.gamma[1:]] if ge else []))
        r = minimize(f, v0, jac=True, method="L-BFGS-B", options={"maxiter": 50})
        al, be, ls, ga = unpack(r.x)
        self.alpha, self.beta, self.logs, self.gamma = al, be, np.maximum(ls, -5), ga

    # -------------------------------------------------------- identification
    def _identify(self):
        K = self.K
        # (1) equal mean difficulty across classes (class-specific b only)
        if K > 1 and self.cs in ("b", "ab"):
            aa = self.a if self.a.shape[0] == K else np.repeat(self.a, K, 0)
            target = self.b.mean()
            d = target - self.b.mean(1)            # shift class c by d_c
            self.b = self.b + d[:, None]
            self.mu = self.mu + d
            if self.duration:
                self.alpha = self.alpha - self.beta * d
        # (2) standardise mixture theta distribution
        pi = np.exp(self.logpi).mean(0)
        m = (pi * self.mu).sum()
        v = (pi * (self.sigma ** 2 + self.mu ** 2)).sum() - m ** 2
        s = np.sqrt(v) if self.model == "2pl" else 1.0
        self.mu = (self.mu - m) / s
        self.sigma = self.sigma / s
        self.b = (self.b - m) / s
        if self.model == "2pl":
            self.a = self.a * s
        if self.duration:
            self.alpha = self.alpha + self.beta * m
            self.beta = self.beta * s

    # ------------------------------------------------------------------- fit
    def fit(self, Y, logT=None, cens=None, group=None, weights=None):
        """weights: optional sampling weights (pseudo-likelihood; rescaled to mean 1)."""
        data = self._prep(Y, logT, cens, group)
        Yf, M, logT, cens, group = data
        w = None if weights is None else np.asarray(weights, float) / np.mean(weights)
        self.G = int(group.max()) + 1
        J = Yf.shape[1]
        rng = np.random.default_rng(self.rs)
        best = None
        self.start_logliks = []
        for s in range(self.n_starts if self.K > 1 else 1):
            self._init(Yf, M, logT, cens, group, rng)
            hist = []
            prev = -np.inf
            conv = False
            for it in range(self.max_iter):
                post, lli = self._estep(*data)
                if w is not None:
                    post, lli = post * w[:, None, None], lli * w
                obj = lli.sum() - self._penalty()
                hist.append(obj)
                if abs(obj - prev) < self.tol * abs(obj):
                    conv = True
                    break
                prev = obj
                self._mstep_items(post, Yf, M)
                wk = self._mstep_theta(post)
                self._mstep_pi(wk, group)
                if self.duration:
                    self._mstep_duration(post, logT, cens, group)
                self._identify()
            ll = lli.sum()
            self.start_logliks.append(ll)
            if self.verbose:
                print(f"start {s}: ll={ll:.3f} it={it} conv={conv}")
            if best is None or obj > best[0]:
                best = (obj, self._get_state(), FitResult(ll, obj, it, conv, self.n_params(J, self.G), hist))
        self._set_state(best[1])
        self.result_ = best[2]
        self._order_classes()
        return self

    def _get_state(self):
        keys = ["a", "b", "mu", "sigma", "logpi"] + (["alpha", "beta", "logs", "gamma"] if self.duration else [])
        return {k: np.copy(getattr(self, k)) for k in keys}

    def _set_state(self, st):
        for k, v in st.items():
            setattr(self, k, v)

    def _order_classes(self):
        """Label-switching convention: order classes by ability mean."""
        o = np.argsort(self.mu)
        self.mu, self.sigma = self.mu[o], self.sigma[o]
        if self.b.shape[0] == self.K:
            self.b = self.b[o]
        if self.a.shape[0] == self.K:
            self.a = self.a[o]
        self.logpi = self.logpi[:, o]
        if self.duration:
            self.alpha, self.beta, self.logs = self.alpha[o], self.beta[o], self.logs[o]

    # ------------------------------------------------------------- inference
    def loglik(self, Y, logT=None, cens=None, group=None, per_person=False):
        data = self._prep(Y, logT, cens, group)
        _, lli = self._estep(*data)
        return lli if per_person else lli.sum()

    def posterior(self, Y, logT=None, cens=None, group=None):
        """Returns (class posterior N x K, EAP theta, posterior SD theta, joint N x K x Q)."""
        data = self._prep(Y, logT, cens, group)
        post, _ = self._estep(*data)
        pc = post.sum(2)
        pt = post.sum(1)
        eap = pt @ self.x
        psd = np.sqrt(np.maximum(pt @ self.x ** 2 - eap ** 2, 0))
        return pc, eap, psd, post

    def predict_items(self, post):
        """P(Y_ij = 1) for all items from a joint posterior N x K x Q."""
        lp, _ = self._item_logp()
        p = np.exp(lp)  # K,Q,J
        return np.einsum("nkq,kqj->nj", post, p)

    def bic(self, n):
        return -2 * self.result_.loglik + self.result_.n_params * np.log(n)

    def aic(self):
        return -2 * self.result_.loglik + 2 * self.result_.n_params

    def entropy_r2(self, pc):
        K = pc.shape[1]
        if K == 1:
            return np.nan
        e = -(pc * np.log(np.clip(pc, 1e-300, 1))).sum()
        return 1 - e / (pc.shape[0] * np.log(K))


class LCA:
    """Unrestricted latent class analysis for binary items with missing data (EM)."""

    def __init__(self, n_classes=2, n_starts=10, max_iter=2000, tol=1e-8, random_state=0, prior=0.5):
        self.K, self.n_starts, self.max_iter, self.tol, self.rs, self.prior = n_classes, n_starts, max_iter, tol, random_state, prior

    def _ll(self, Yf, M):
        lp, lq = np.log(self.p), np.log1p(-self.p)
        return Yf @ lp.T + (M - Yf) @ lq.T + np.log(self.pi)[None]

    def fit(self, Y):
        Y = np.asarray(Y, float)
        M = (~np.isnan(Y)).astype(float)
        Yf = np.nan_to_num(Y)
        rng = np.random.default_rng(self.rs)
        best = None
        for s in range(self.n_starts):
            self.pi = np.full(self.K, 1 / self.K)
            self.p = rng.uniform(0.2, 0.8, size=(self.K, Y.shape[1]))
            prev = -np.inf
            for it in range(self.max_iter):
                L = self._ll(Yf, M)
                lli = logsumexp(L, 1)
                ll = lli.sum()
                if abs(ll - prev) < self.tol * abs(ll):
                    break
                prev = ll
                r = np.exp(L - lli[:, None])
                self.pi = (r.sum(0) + 1e-6) / (r.sum() + 1e-6 * self.K)
                self.p = (r.T @ Yf + self.prior) / (r.T @ M + 2 * self.prior)
            if best is None or ll > best[0]:
                best = (ll, self.pi.copy(), self.p.copy(), it)
        self.loglik_, self.pi, self.p, self.n_iter_ = best
        o = np.argsort(self.p.mean(1))
        self.pi, self.p = self.pi[o], self.p[o]
        self.n_params_ = self.K * Y.shape[1] + self.K - 1
        return self

    def posterior(self, Y):
        Y = np.asarray(Y, float)
        M = (~np.isnan(Y)).astype(float)
        L = self._ll(np.nan_to_num(Y), M)
        lli = logsumexp(L, 1)
        return np.exp(L - lli[:, None]), lli

    def loglik(self, Y, per_person=False):
        lli = self.posterior(Y)[1]
        return lli if per_person else lli.sum()

    def predict_items(self, r):
        return r @ self.p

    def bic(self, n):
        return -2 * self.loglik_ + self.n_params_ * np.log(n)


class ContinuousIRT:
    """Continuous competitors to latent classes (exploratory E3b):

    * dims=2: compensatory two-dimensional 2PL, theta ~ N(0, I), first item loads on dim 1 only
      (rotational identification). P = sigmoid(a1 th1 + a2 th2 + d).
    * guessing=True (dims=1): 2PL with lower asymptote, P = c + (1-c) sigmoid(a(th - b)),
      prior c ~ Beta(2, 8) (mean 0.2) and log a ~ N(0, 1).
    EM with fixed product quadrature; per-item L-BFGS in the M-step.
    The interface (fit / loglik / posterior / predict_items / result_) mirrors MixIRT.
    """

    def __init__(self, dims=1, guessing=False, n_nodes=None, max_iter=1500, tol=1e-8, verbose=False):
        self.dims, self.guessing, self.max_iter, self.tol, self.verbose = dims, guessing, max_iter, tol, verbose
        q = n_nodes or (41 if dims == 1 else 21)
        g = np.linspace(-5 if dims == 1 else -4, 5 if dims == 1 else 4, q)
        if dims == 1:
            self.X = g[:, None]
        else:
            a, b = np.meshgrid(g, g, indexing="ij")
            self.X = np.column_stack([a.ravel(), b.ravel()])
        lw = -0.5 * (self.X ** 2).sum(1)
        self.logw = lw - logsumexp(lw)
        self.K = 1
        self.duration = False

    def _p(self, A=None, d=None, c=None):
        A = self.A if A is None else A
        d = self.d if d is None else d
        c = self.c if c is None else c
        eta = self.X @ A.T + d[None, :]
        p = expit(eta)
        return c[None, :] + (1 - c[None, :]) * p

    def _estep(self, Yf, M):
        p = np.clip(self._p(), 1e-12, 1 - 1e-12)
        ll = Yf @ np.log(p).T + (M - Yf) @ np.log1p(-p).T + self.logw[None, :]
        lli = logsumexp(ll, 1)
        return np.exp(ll - lli[:, None]), lli

    def _penalty(self):
        pen = 0.5 * (self.A ** 2).sum() / 4.0
        if self.guessing:
            pen -= ((2 - 1) * np.log(self.c) + (8 - 1) * np.log1p(-self.c)).sum()
        return pen

    def fit(self, Y, logT=None, cens=None, group=None, weights=None):
        w = None if weights is None else np.asarray(weights, float) / np.mean(weights)
        Y = np.asarray(Y, float)
        M = (~np.isnan(Y)).astype(float)
        Yf = np.nan_to_num(Y)
        J, D = Y.shape[1], self.dims
        pj = (Yf.sum(0) + .5) / (M.sum(0) + 1)
        self.A = np.full((J, D), 1.0)
        if D == 2:
            self.A[:, 1] = 0.3
            self.A[0, 1] = 0.0
        self.d = np.log(pj / (1 - pj))
        self.c = np.full(J, 0.15) if self.guessing else np.zeros(J)
        hist, prev, conv = [], -np.inf, False
        for it in range(self.max_iter):
            post, lli = self._estep(Yf, M)
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
        self.result_ = FitResult(lli.sum(), obj, it, conv, self.n_params(J), hist)
        self.start_logliks = [lli.sum()]
        return self

    def _mstep_item(self, j, r, n):
        D, X = self.dims, self.X
        fix2 = D == 2 and j == 0

        def unpack(v):
            a = v[:D].copy()
            if fix2:
                a[1] = 0.0
            dd = v[D]
            c = expit(v[D + 1]) if self.guessing else 0.0
            return a, dd, c

        def f(v):
            a, dd, c = unpack(v)
            s = expit(X @ a + dd)
            p = np.clip(c + (1 - c) * s, 1e-12, 1 - 1e-12)
            ll = (r * np.log(p) + (n - r) * np.log1p(-p)).sum() - 0.5 * (a ** 2).sum() / 4.0
            if self.guessing:
                ll += (2 - 1) * np.log(c) + (8 - 1) * np.log1p(-c)
            # gradient
            w = (r / p - (n - r) / (1 - p)) * (1 - c) * s * (1 - s)
            ga = X.T @ w - a / 4.0
            if fix2:
                ga[1] = 0.0
            g = [ga, [w.sum()]]
            if self.guessing:
                gc = ((r / p - (n - r) / (1 - p)) * (1 - s)).sum() * c * (1 - c) + (2 - 1) * (1 - c) - (8 - 1) * c
                g.append([gc])
            return -ll, -np.concatenate(g)

        v0 = np.concatenate([self.A[j], [self.d[j]]] + ([[np.log(self.c[j] / (1 - self.c[j]))]] if self.guessing else []))
        res = minimize(f, v0, jac=True, method="L-BFGS-B", options={"maxiter": 20})
        if res.fun <= f(v0)[0]:
            a, dd, c = unpack(res.x)
            self.A[j], self.d[j] = a, dd
            if self.guessing:
                self.c[j] = c

    def n_params(self, J):
        return J * (self.dims + 1) - (1 if self.dims == 2 else 0) + (J if self.guessing else 0)

    def loglik(self, Y, logT=None, cens=None, group=None, per_person=False):
        Y = np.asarray(Y, float)
        lli = self._estep(np.nan_to_num(Y), (~np.isnan(Y)).astype(float))[1]
        return lli if per_person else lli.sum()

    def posterior(self, Y, logT=None, cens=None, group=None):
        Y = np.asarray(Y, float)
        post, _ = self._estep(np.nan_to_num(Y), (~np.isnan(Y)).astype(float))
        eap = post @ self.X[:, 0]
        return np.ones((len(Y), 1)), eap, None, post

    def predict_items(self, post):
        return post @ self._p()

    def bic(self, n):
        return -2 * self.result_.loglik + self.result_.n_params * np.log(n)


def _poly_affine(c, a0, a1):
    """Coefficients (ascending) of t -> P(a0 + a1 t), where P has ascending coefficients c."""
    out = np.zeros(len(c))
    base = np.array([1.0])
    for p, cp in enumerate(c):
        out[:len(base)] += cp * base
        base = P.polymul(base, [a0, a1])[:len(c)]
    return out


class MixIRTDurPoly(MixIRT):
    """MixIRT with a flexible duration model (exploratory competitor to duration classes):
    log T | theta, class k, group g ~ N(alpha_k + gamma_g + sum_p beta_kp theta^p, exp(logs_k + delta_k theta)^2),
    i.e. a polynomial of degree ``deg`` in theta and (``het=True``) a log-linear residual SD.
    With n_classes=1 this is a *continuous* model with a non-linear speed-ability relation.
    """

    def __init__(self, *args, deg=3, het=True, **kw):
        kw["duration"] = True
        super().__init__(*args, **kw)
        self.deg, self.het = deg, het

    def n_params(self, J, G):
        return super().n_params(J, G) + self.K * (self.deg - 1) + (self.K if self.het else 0)

    def _init(self, Yf, M, logT, cens, group, rng):
        super()._init(Yf, M, logT, cens, group, rng)
        self.C = np.zeros((self.K, self.deg + 1))
        self.C[:, 0] = self.alpha
        self.delta = np.zeros(self.K)

    def _powers(self):
        return self.x[None, :] ** np.arange(self.deg + 1)[:, None]  # (deg+1, Q)

    def _mean_logs(self, C, logs, delta, gamma, group):
        mean = (C @ self._powers())[None] + gamma[group][:, None, None]       # N,K,Q
        ls = logs[:, None] + (delta[:, None] * self.x[None] if self.het else 0.0)  # K,Q
        return mean, np.broadcast_to(ls, (self.K, len(self.x)))[None]

    def _dur_ll(self, logT, cens, group):
        mean, ls = self._mean_logs(self.C, self.logs, self.delta, self.gamma, group)
        s = np.exp(ls)
        miss = np.isnan(logT)
        z = (np.nan_to_num(logT)[:, None, None] - mean) / s
        ll = -0.5 * z ** 2 - ls - 0.5 * LOG2PI
        if cens.any():
            ll[cens] = log_ndtr(-z[cens])
        ll[miss] = 0.0
        return ll, z

    def _mstep_duration(self, post, logT, cens, group):
        K, G, D = self.K, self.G, self.deg + 1
        ge = self.group_effects and G > 1
        X = self._powers()
        obs = ~np.isnan(logT)
        post = post * obs[:, None, None]
        logT = np.nan_to_num(logT)
        nC = K * D

        def unpack(v):
            C = v[:nC].reshape(K, D)
            logs = v[nC:nC + K]
            o = nC + K
            delta = v[o:o + K] if self.het else np.zeros(K)
            o += K if self.het else 0
            ga = np.concatenate([[0.0], v[o:]]) if ge else np.zeros(G)
            return C, logs, delta, ga

        def f(v):
            C, logs, delta, ga = unpack(v)
            mean, ls = self._mean_logs(C, logs, delta, ga, group)
            s = np.exp(ls)
            z = (logT[:, None, None] - mean) / s
            ll = -0.5 * z ** 2 - ls
            dmean = z / s
            dls = z ** 2 - 1.0
            if cens.any():
                zc = z[cens]
                lc = log_ndtr(-zc)
                ll[cens] = lc
                h = np.exp(-0.5 * zc ** 2 - 0.5 * LOG2PI - lc)
                dmean[cens] = h / s[0][None]
                dls[cens] = h * zc
            gm, gl = post * dmean, post * dls
            gmk = gm.sum(0)                      # K,Q
            grad = [(gmk @ X.T).ravel(), gl.sum((0, 2))]
            if self.het:
                grad.append((gl * self.x[None, None, :]).sum((0, 2)))
            if ge:
                gg = np.zeros(G)
                np.add.at(gg, group, gm.sum((1, 2)))
                grad.append(gg[1:])
            return -(post * ll).sum(), -np.concatenate(grad)

        v0 = np.concatenate([self.C.ravel(), self.logs] + ([self.delta] if self.het else []) + ([self.gamma[1:]] if ge else []))
        r = minimize(f, v0, jac=True, method="L-BFGS-B", options={"maxiter": 100})
        C, logs, delta, ga = unpack(r.x)
        self.C, self.logs, self.delta, self.gamma = C, np.maximum(logs, -5), delta, ga
        self.alpha, self.beta = C[:, 0], C[:, 1]

    def _identify(self):
        K = self.K
        if K > 1 and self.cs in ("b", "ab"):
            d = self.b.mean() - self.b.mean(1)
            self.b = self.b + d[:, None]
            self.mu = self.mu + d
            for k in range(K):   # theta_new = theta_old + d_k
                self.C[k] = _poly_affine(self.C[k], -d[k], 1.0)
                self.logs[k] = self.logs[k] - self.delta[k] * d[k]
        pi = np.exp(self.logpi).mean(0)
        m = (pi * self.mu).sum()
        v = (pi * (self.sigma ** 2 + self.mu ** 2)).sum() - m ** 2
        s = np.sqrt(v) if self.model == "2pl" else 1.0
        self.mu = (self.mu - m) / s
        self.sigma = self.sigma / s
        self.b = (self.b - m) / s
        if self.model == "2pl":
            self.a = self.a * s
        for k in range(K):       # theta_old = m + s theta_new
            self.C[k] = _poly_affine(self.C[k], m, s)
        self.logs = self.logs + self.delta * m
        self.delta = self.delta * s
        self.alpha, self.beta = self.C[:, 0], self.C[:, 1]

    def _get_state(self):
        st = super()._get_state()
        st.update(C=np.copy(self.C), delta=np.copy(self.delta))
        return st

    def _order_classes(self):
        o = np.argsort(self.mu)
        super()._order_classes()
        self.C, self.delta = self.C[o], self.delta[o]

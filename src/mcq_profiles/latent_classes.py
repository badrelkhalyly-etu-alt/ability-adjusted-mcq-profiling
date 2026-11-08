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
        return Yf, M.astype(float), np.asarray(logT, float), np.asarray(cens, bool), np.asarray(group, int)

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
            obs = ~cens
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
        z = (logT[:, None, None] - mean) / s
        ll = -0.5 * z ** 2 - np.log(s) - 0.5 * LOG2PI
        if cens.any():
            ll[cens] = log_ndtr(-z[cens])
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
    def fit(self, Y, logT=None, cens=None, group=None):
        data = self._prep(Y, logT, cens, group)
        Yf, M, logT, cens, group = data
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

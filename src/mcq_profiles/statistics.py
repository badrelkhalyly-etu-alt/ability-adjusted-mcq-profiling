"""Statistical utilities: association, intervals, matching, person fit, local dependence, DIF."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.special import expit
from sklearn.metrics import adjusted_rand_score


def cramers_v(a, b):
    tab = pd.crosstab(np.asarray(a), np.asarray(b)).to_numpy()
    if min(tab.shape) < 2:
        return 0.0
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def match_labels(ref, other, K=None):
    """Permute labels of ``other`` to best agree with ``ref`` (Hungarian on contingency)."""
    K = K or int(max(ref.max(), other.max()) + 1)
    C = np.zeros((K, K))
    np.add.at(C, (ref, other), 1)
    r, c = linear_sum_assignment(-C)
    perm = np.empty(K, int)
    perm[c] = r
    return perm[other], perm


def matched_ari(a, b):
    return adjusted_rand_score(a, b)


def hopkins(X, m=None, rng=None):
    """Hopkins statistic (≈0.5 for uniform/unstructured, →1 for clustered)."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(rng)
    X = np.asarray(X, float)
    n = len(X)
    m = m or max(10, n // 10)
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    idx = rng.choice(n, m, replace=False)
    w = nn.kneighbors(X[idx], 2)[0][:, 1]
    U = rng.uniform(X.min(0), X.max(0), size=(m, X.shape[1]))
    u = nn.kneighbors(U, 1)[0][:, 0]
    return float(u.sum() / (u.sum() + w.sum()))


def kr20(Y):
    Y = np.asarray(Y, float)
    ok = ~np.isnan(Y).any(1)
    Y = Y[ok]
    k = Y.shape[1]
    p = Y.mean(0)
    return float(k / (k - 1) * (1 - (p * (1 - p)).sum() / Y.sum(1).var(ddof=1)))


def lz_person_fit(Y, P):
    """Standardised log-likelihood person-fit l_z (Drasgow et al., 1985), missing ignored."""
    M = ~np.isnan(Y)
    Yf = np.nan_to_num(Y)
    P = np.clip(P, 1e-6, 1 - 1e-6)
    l0 = np.where(M, Yf * np.log(P) + (1 - Yf) * np.log(1 - P), 0).sum(1)
    E = np.where(M, P * np.log(P) + (1 - P) * np.log(1 - P), 0).sum(1)
    V = np.where(M, P * (1 - P) * np.log(P / (1 - P)) ** 2, 0).sum(1)
    return (l0 - E) / np.sqrt(V)


def yen_q3(Y, P):
    R = Y - P
    df = pd.DataFrame(R)
    Q = df.corr(min_periods=50).to_numpy(copy=True)
    np.fill_diagonal(Q, np.nan)
    return Q


def bh_fdr(p, q=0.05):
    p = np.asarray(p)
    n = len(p)
    o = np.argsort(p)
    thresh = q * (np.arange(1, n + 1)) / n
    ok = p[o] <= thresh
    k = np.max(np.flatnonzero(ok)) + 1 if ok.any() else 0
    rej = np.zeros(n, bool)
    rej[o[:k]] = True
    return rej


def logistic_dif(Y, group, n_strata=None):
    """Swaminathan-Rogers logistic-regression DIF with rest-score matching.

    For each item j: M0 y ~ rest; M1 + group; M2 + group + rest:group.
    Returns DataFrame with LR chi2 (2 df), p, and ΔNagelkerke R² (M2 vs M0).
    """
    import statsmodels.api as sm
    Y = np.asarray(Y, float)
    g = np.asarray(group, float)
    rows = []
    for j in range(Y.shape[1]):
        ok = ~np.isnan(Y[:, j]) & ~np.isnan(g)
        others = np.delete(Y, j, 1)
        rest = np.nanmean(others, 1)[ok] * others.shape[1]
        y = Y[ok, j]
        gg = g[ok]
        rest = (rest - rest.mean()) / rest.std()
        X0 = sm.add_constant(np.column_stack([rest]))
        X2 = sm.add_constant(np.column_stack([rest, gg, rest * gg]))
        m0 = sm.Logit(y, X0).fit(disp=0)
        m2 = sm.Logit(y, X2).fit(disp=0)
        lr = 2 * (m2.llf - m0.llf)
        n = len(y)

        def nag(m):
            cs = 1 - np.exp(2 * (m.llnull - m.llf) / n)
            return cs / (1 - np.exp(2 * m.llnull / n))

        rows.append(dict(item=j, lr=lr, p=stats.chi2.sf(lr, 2), dR2=nag(m2) - nag(m0),
                         uniform_coef=m2.params[2], nonuniform_coef=m2.params[3], n=n))
    out = pd.DataFrame(rows)
    out["reject_bh"] = bh_fdr(out.p.values)
    # Jodoin & Gierl (2001) effect classes: A < .035 <= B < .070 <= C
    out["effect_class"] = pd.cut(out.dR2, [-np.inf, 0.035, 0.070, np.inf], labels=["A", "B", "C"])
    return out


def paired_bootstrap_ci(d, B=2000, seed=0, alpha=0.05):
    """Percentile CI for the mean of per-student paired differences d."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))

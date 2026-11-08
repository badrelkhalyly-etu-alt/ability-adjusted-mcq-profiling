"""Reproduction of the source study's score-time clustering pipeline (Safeh) and variants."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def features(score, minutes, log_time=False):
    t = np.log(minutes) if log_time else np.asarray(minutes, float)
    X = np.column_stack([score, t])
    ok = ~np.isnan(X).any(1)
    return X, ok


def kmeans_fit(X, k, seed=42, n_init=50):
    Z = StandardScaler().fit_transform(X)
    km = KMeans(k, n_init=n_init, random_state=seed).fit(Z)
    return km, Z


def indices(Z, labels, sample=None, seed=0):
    lab = np.asarray(labels)
    ok = lab >= 0
    if len(np.unique(lab[ok])) < 2:
        return dict(silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
    return dict(silhouette=float(silhouette_score(Z[ok], lab[ok], sample_size=sample, random_state=seed)),
                davies_bouldin=float(davies_bouldin_score(Z[ok], lab[ok])),
                calinski_harabasz=float(calinski_harabasz_score(Z[ok], lab[ok])))


def k_curve(X, ks, seed=42):
    out = []
    for k in ks:
        km, Z = kmeans_fit(X, k, seed)
        out.append(dict(k=k, inertia=km.inertia_, **indices(Z, km.labels_)))
    return out


def cv_silhouette(X, k=4, folds=5, seed=42):
    """Source study's 5-fold held-out silhouette."""
    Z = StandardScaler().fit_transform(X)
    res = []
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(Z):
        km = KMeans(k, n_init=50, random_state=seed).fit(Z[tr])
        lab = km.predict(Z[te])
        res.append(silhouette_score(Z[te], lab))
    return res


def bootstrap_stability(X, k=4, B=100, seed=0):
    """Hennig-style: ARI between full-data labels and bootstrap-model labels on all points,
    plus cluster-wise Jaccard (mean over bootstraps)."""
    rng = np.random.default_rng(seed)
    km, Z = kmeans_fit(X, k)
    ref = km.labels_
    aris, jac = [], np.zeros((B, k))
    for b in range(B):
        idx = rng.integers(0, len(Z), len(Z))
        kb = KMeans(k, n_init=10, random_state=int(rng.integers(1e9))).fit(Z[idx])
        lab = kb.predict(Z)
        aris.append(adjusted_rand_score(ref, lab))
        for c in range(k):
            A = ref == c
            jac[b, c] = max(((A & (lab == d)).sum() / (A | (lab == d)).sum()) for d in range(k))
    return np.array(aris), jac.mean(0)


def gmm_bic(X, ks, seed=0):
    Z = StandardScaler().fit_transform(X)
    return [dict(k=k, bic=GaussianMixture(k, covariance_type="full", n_init=5, random_state=seed).fit(Z).bic(Z)) for k in ks]


def agglomerative(X, k=4):
    Z = StandardScaler().fit_transform(X)
    lab = AgglomerativeClustering(k, linkage="ward").fit_predict(Z)
    return lab, Z


def dbscan_grid(X, eps_grid, ms_grid):
    Z = StandardScaler().fit_transform(X)
    rows = []
    for e in eps_grid:
        for m in ms_grid:
            lab = DBSCAN(eps=e, min_samples=m).fit_predict(Z)
            ncl = len(set(lab) - {-1})
            rows.append(dict(eps=e, min_samples=m, n_clusters=ncl, noise=float((lab == -1).mean()),
                             **(indices(Z, lab) if ncl >= 2 else {})))
    return rows


def hdbscan(X, min_cluster_size=50):
    Z = StandardScaler().fit_transform(X)
    lab = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(Z)
    return lab, Z


class ScoreTimeProfilePredictor:
    """Predict masked items from K-means score-time profiles (source-study representation).

    Profiles are fitted on training students' (proportion correct on unmasked items, time).
    Each profile carries training item p-values; a test student is assigned to the nearest
    centroid using only unmasked information.
    """

    def __init__(self, k=4, seed=42):
        self.k, self.seed = k, seed

    def fit(self, Y, minutes, mask):
        s = np.nanmean(np.where(mask[None, :], np.nan, Y), 1)
        X = np.column_stack([s, np.log(minutes)])
        ok = ~np.isnan(X).any(1)
        self.sc = StandardScaler().fit(X[ok])
        self.km = KMeans(self.k, n_init=50, random_state=self.seed).fit(self.sc.transform(X[ok]))
        lab = self.km.labels_
        Yo = Y[ok]
        self.p = np.array([(np.nansum(Yo[lab == c], 0) + 0.5) / ((~np.isnan(Yo[lab == c])).sum(0) + 1) for c in range(self.k)])
        self.p_all = (np.nansum(Yo, 0) + 0.5) / ((~np.isnan(Yo)).sum(0) + 1)
        return self

    def predict(self, Y, minutes, mask):
        s = np.nanmean(np.where(mask[None, :], np.nan, Y), 1)
        X = np.column_stack([s, np.log(minutes)])
        ok = ~np.isnan(X).any(1)
        P = np.tile(self.p_all, (len(Y), 1))
        lab = self.km.predict(self.sc.transform(X[ok]))
        P[ok] = self.p[lab]
        return P[:, mask]

"""Proper scoring rules, calibration and model-selection helpers."""
from __future__ import annotations

import numpy as np


def logloss_matrix(Y, P):
    """Per-student mean log loss over observed entries (NaN in Y ignored)."""
    M = ~np.isnan(Y)
    P = np.clip(P, 1e-9, 1 - 1e-9)
    L = np.where(M, -(np.nan_to_num(Y) * np.log(P) + (1 - np.nan_to_num(Y)) * np.log(1 - P)), 0.0)
    return L.sum(1), M.sum(1)


def brier_matrix(Y, P):
    M = ~np.isnan(Y)
    B = np.where(M, (np.nan_to_num(Y) - P) ** 2, 0.0)
    return B.sum(1), M.sum(1)


def calibration_table(Y, P, bins=10):
    M = ~np.isnan(Y)
    y, p = Y[M], P[M]
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        s = idx == b
        if s.sum():
            rows.append((b, s.sum(), p[s].mean(), y[s].mean()))
    rows = np.array(rows)
    ece = float((rows[:, 1] * np.abs(rows[:, 2] - rows[:, 3])).sum() / rows[:, 1].sum())
    return rows, ece


def predict_masked(model, Y_test, mask, logT=None, cens=None, group=None):
    """Predict masked item columns from unmasked responses (+ optional duration)."""
    Yin = Y_test.copy()
    Yin[:, mask] = np.nan
    if hasattr(model, "x"):  # MixIRT
        _, _, _, post = model.posterior(Yin, logT, cens, group)
        P = model.predict_items(post)
    else:  # LCA-like
        r, _ = model.posterior(Yin)
        P = model.predict_items(r)
    return P[:, mask]


def selection_summary(models: dict, n: int):
    """BIC/AIC table for fitted MixIRT/LCA objects."""
    rows = []
    for name, m in models.items():
        ll = m.result_.loglik if hasattr(m, "result_") else m.loglik_
        k = m.result_.n_params if hasattr(m, "result_") else m.n_params_
        conv = m.result_.converged if hasattr(m, "result_") else True
        rows.append(dict(model=name, loglik=ll, n_params=k, AIC=-2 * ll + 2 * k,
                         BIC=-2 * ll + k * np.log(n), converged=conv))
    return rows

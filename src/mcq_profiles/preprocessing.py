"""Splits, item masks and duration variants (protocol sections 1-2)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def train_test_split_stratified(df: pd.DataFrame, seed: int, test_fraction: float = 0.3, stratify: str = "CNT"):
    rng = np.random.default_rng(seed)
    test = np.zeros(len(df), bool)
    for _, idx in df.groupby(stratify).indices.items():
        idx = np.asarray(idx)
        k = int(round(test_fraction * len(idx)))
        test[rng.choice(idx, k, replace=False)] = True
    return ~test, test


def item_mask(J: int, fraction: float, seed: int) -> np.ndarray:
    """Boolean vector of masked item positions (fixed across students)."""
    rng = np.random.default_rng(seed)
    m = np.zeros(J, bool)
    m[rng.choice(J, int(round(fraction * J)), replace=False)] = True
    return m


def log_duration(minutes, trim=None, censor=None):
    """log-minutes with optional trimming (quantile pair -> NaN outside) and censoring flag."""
    t = np.asarray(minutes, float).copy()
    t[t <= 0] = np.nan
    flag = np.zeros(len(t), bool)
    if trim is not None:
        lo, hi = np.nanquantile(t, trim)
        t[(t < lo) | (t > hi)] = np.nan
    if censor is not None:
        flag = t >= censor
        t = np.where(flag, censor, t)
    return np.log(t), flag & ~np.isnan(t)


def unmasked_duration(rt_seconds: np.ndarray, raw_missing: np.ndarray, mask: np.ndarray):
    """Total minutes on unmasked items only (not-reached items count 0)."""
    T = np.where(np.isnan(rt_seconds) & raw_missing, 0.0, rt_seconds)[:, ~mask]
    out = np.nansum(T, 1) / 60.0
    out[np.isnan(T).all(1)] = np.nan
    return out


def group_codes(series: pd.Series):
    cats = sorted(series.dropna().unique())
    return series.map({c: i for i, c in enumerate(cats)}).to_numpy(int), cats

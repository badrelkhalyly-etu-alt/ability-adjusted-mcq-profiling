"""Back-end of the descriptive ability x relative-time dashboard (no Streamlit dependency, so it is testable).

Input: a table with item responses ``I<j>`` (0/1, empty = not reached or not administered), item times in
seconds ``RT<j>`` (same j) and optionally a ``booklet`` column. Output: per-student coordinates with
uncertainty intervals and summary indicators. The dashboard never assigns psychological labels, profiles or
recommendations; it reports two measured quantities and how precisely they are known.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from .continuous_map import ContinuousIRTDur, relative_time_index, separation_strata, time_intensities

LIMITATIONS = [
    "Ability is estimated from the item responses of this test only; its interval shows the measurement uncertainty.",
    "Relative time compares a student's time with the time expected for a student of the same estimated ability, "
    "on the same booklet and the same items reached. It is not a measure of speed, effort, anxiety or engagement: "
    "it mixes working pace, checking, pauses, distraction and rapid guessing.",
    "Overlapping intervals mean the data cannot distinguish the two students on that quantity.",
    "The map describes one test sitting. It must not be used alone for grading, placement, diagnosis or any other "
    "high-stakes decision about an individual.",
    "No grouping into 'profiles' is shown: on PISA 2025, discrete score-time profiles were not distinct from a "
    "continuum, lost predictive information, and about 30 % of students would change profile on a parallel test.",
]


def simulate_demo(n=1500, J=20, seed=0):
    """Synthetic data set (no real student data): responses with guessing, item times with intensities."""
    rng = np.random.default_rng(seed)
    th = rng.normal(size=n)
    a = rng.uniform(0.8, 1.8, J)
    b = rng.normal(0, 1, J)
    c = rng.uniform(0.05, 0.25, J)
    P = c + (1 - c) * expit(a * (th[:, None] - b))
    Y = (rng.random((n, J)) < P).astype(float)
    lam = rng.normal(4.0, 0.4, J)                                    # log seconds
    pace = 0.4 * rng.normal(size=n) - 0.15 * th
    LT = lam[None, :] + pace[:, None] + 0.5 * rng.normal(size=(n, J))
    RT = np.exp(LT)
    reached = np.arange(J)[None, :] < rng.integers(int(0.7 * J), J + 1, n)[:, None]
    Y[~reached] = np.nan
    RT[~reached] = np.nan
    df = pd.DataFrame(Y, columns=[f"I{j + 1}" for j in range(J)])
    for j in range(J):
        df[f"RT{j + 1}"] = RT[:, j]
    df["booklet"] = rng.integers(1, 3, n)
    df.insert(0, "student_id", [f"S{i:05d}" for i in range(n)])
    return df


def parse_table(df: pd.DataFrame):
    items = sorted([c for c in df.columns if c.startswith("I") and c[1:].isdigit()], key=lambda c: int(c[1:]))
    if not items:
        raise ValueError("no item columns named I1, I2, ...")
    Y = df[items].to_numpy(float)
    if not np.isin(Y[~np.isnan(Y)], [0, 1]).all():
        raise ValueError("item responses must be 0, 1 or empty")
    rt_cols = [f"RT{c[1:]}" for c in items]
    missing = [c for c in rt_cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing time columns: {missing[:5]}")
    RT = df[rt_cols].to_numpy(float)
    g = pd.factorize(df["booklet"])[0] if "booklet" in df.columns else np.zeros(len(df), int)
    ids = df["student_id"].astype(str).to_numpy() if "student_id" in df.columns else np.array([str(i) for i in range(len(df))])
    return ids, Y, RT, g


def compute_map(df: pd.DataFrame, level=0.9, max_iter=500):
    """Fit the joint model on the table and return (per-student coordinates, summary indicators)."""
    ids, Y, RT, g = parse_table(df)
    LT = np.where((RT > 0) & ~np.isnan(Y), np.log(np.where(RT > 0, RT, 1.0)), np.nan)
    lam = time_intensities(LT)
    v, v_se, n_items = relative_time_index(LT, lam)
    m = ContinuousIRTDur(guessing=True, deg=3, het=True, max_iter=max_iter, tol=1e-7).fit(Y, v, None, g)
    co = m.coordinates(Y, v, v_se, g, level=level)
    co.insert(0, "student_id", ids)
    co["items_answered"] = (~np.isnan(Y)).sum(1)
    co["items_timed"] = n_items
    rel = float(np.var(co.theta) / (np.var(co.theta) + np.mean(co.theta_sd ** 2)))
    med = float(np.median(co.theta))
    ok = np.isfinite(co.rel_time)
    summary = dict(n_students=len(co), n_items=Y.shape[1], level=level, ability_reliability=rel,
                   separation_strata=float(separation_strata(rel)),
                   mean_ability_interval_width=float((co.theta_hi - co.theta_lo).mean()),
                   share_ability_interval_excludes_median=float(((co.theta_lo > med) | (co.theta_hi < med)).mean()),
                   share_relative_time_interval_excludes_0=float(((co.rel_time_lo > 0) | (co.rel_time_hi < 0))[ok].mean()),
                   share_relative_time_unavailable=float((~ok).mean()), converged=bool(m.result_.converged))
    return co, summary


def distinguishable(co: pd.DataFrame, i: int, j: int, quantity="theta"):
    """True when the two students' intervals on ``quantity`` do not overlap."""
    lo, hi = f"{quantity}_lo", f"{quantity}_hi"
    a, b = co.iloc[i], co.iloc[j]
    return bool(a[hi] < b[lo] or b[hi] < a[lo])

"""Data ingestion for the PISA 2025 public-use files (CY09_MS_*).

The analytic unit is a *fixed test form*: the set of students on the linear (non-adaptive)
mathematics path who received exactly the same item set. This mirrors the source study
(one fixed set of ~20 items) while avoiding the item-selection dependence of the
adaptive path.

Raw codes (scored variables): '0','1','2' credit, 'r' not reached, 't' technical issue,
'7' not administered. They arrive here from ``scripts/extract_items.py`` as
0/1/2, -8 (not reached), -9 (technical), NaN (not administered).
Timing variables (``...TT``) are in milliseconds; codes 99999997-99999999 are
'Not applicable/Invalid/No response'.

Schema of the analytic frame (one row per student):
    student_id_anonymous, CNT, CNTSCHID, OECD, BOOKID, LANGTEST_COG,
    I01..IJJ (0/1 or NaN), RT01..RTJJ (seconds), total_duration_minutes,
    final_score (sum of administered item scores), n_not_reached, n_technical,
    n_polytomous_full_credit
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed")
SENTINEL = 99999990.0


def load_extracts(processed=PROCESSED):
    cog = pd.read_parquet(processed / "cog_linear_items.parquet")
    pro = pd.read_parquet(processed / "process_linear_items.parquet")
    return cog, pro


def form_catalogue(cog, path="MPATH", prefix=("CM",), max_book=12):
    """Return list of (n_students, item_tuple) for fixed forms, most frequent first."""
    sub = cog[(cog[path] == 1) & (cog.BOOKID <= max_book)]
    items = [c for c in cog.columns if c.endswith("S") and c.startswith(prefix)]
    pres = sub[items].notna().to_numpy()
    keys = [tuple(np.flatnonzero(r)) for r in pres]
    vc = pd.Series(keys, index=sub.index).value_counts()
    return [(int(n), tuple(items[i] for i in k)) for k, n in vc.items()], sub, pd.Series(keys, index=sub.index)


def build_form(cog, pro, rank=0, path="MPATH", prefix=("CM",), not_reached="missing", polytomous="full"):
    """Analytic frame for the ``rank``-th most frequent fixed form.

    not_reached: 'missing' (PISA scaling convention) or 'incorrect' (sensitivity).
    polytomous: 'full' -> 1 only for the maximum code; 'any' -> 1 for any credit.
    """
    cat, sub, keys = form_catalogue(cog, path, prefix)
    n, items = cat[rank]
    target = tuple(np.flatnonzero(np.isin([c for c in cog.columns if c.endswith("S") and c.startswith(prefix)], items)))
    f = sub.loc[keys.map(lambda k: k == target)].copy()
    assert len(f) == n, (len(f), n)
    raw = f[list(items)].to_numpy(float, copy=True)
    nr = (raw == -8).sum(1)
    tech = (raw == -9).sum(1)
    maxcode = np.nanmax(np.where(raw >= 0, raw, np.nan), axis=0)
    Y = raw.copy()
    Y[raw == -9] = np.nan
    Y[raw == -8] = np.nan if not_reached == "missing" else 0.0
    poly = np.flatnonzero(maxcode > 1)
    for j in poly:
        col = Y[:, j]
        if polytomous == "full":
            Y[:, j] = np.where(np.isnan(col), np.nan, (col >= maxcode[j]).astype(float))
        else:
            Y[:, j] = np.where(np.isnan(col), np.nan, (col >= 1).astype(float))
    J = len(items)
    out = pd.DataFrame({
        "student_id_anonymous": [hashlib.sha256(f"{a}|{b}".encode()).hexdigest()[:16] for a, b in zip(f.CNT, f.CNTSTUID)],
        "CNT": f.CNT.values, "CNTSCHID": f.CNTSCHID.values, "CNTSTUID": f.CNTSTUID.values,
        "OECD": f.OECD.values, "BOOKID": f.BOOKID.values, "LANGTEST_COG": f.LANGTEST_COG.values,
    })
    icols = [f"I{j + 1:02d}" for j in range(J)]
    out[icols] = Y
    tt = [i[:-1] + "TT" for i in items]
    m = f[["CNT", "CNTSTUID"]].merge(pro[["CNT", "CNTSTUID"] + [t for t in tt if t in pro.columns]], how="left")
    T = m.reindex(columns=tt).to_numpy(float, copy=True)
    T[T >= SENTINEL] = np.nan
    T = T / 1000.0  # seconds
    rcols = [f"RT{j + 1:02d}" for j in range(J)]
    out[rcols] = T
    # Not-reached items were never visited: their time is 0, not missing.
    T_nr0 = np.where(raw == -8, np.nan_to_num(T, nan=0.0), T)
    miss_t = np.isnan(T_nr0)
    out["n_time_missing"] = miss_t.sum(1)
    out["total_duration_minutes"] = np.where(miss_t.all(1), np.nan, np.nansum(T_nr0, 1) / 60.0)
    out["final_score"] = np.nansum(Y, 1)
    out["n_administered_scored"] = (~np.isnan(Y)).sum(1)
    out["n_not_reached"] = nr
    out["n_technical"] = tech
    meta = dict(form_rank=rank, path=path, n=int(n), items=list(items), item_map=dict(zip(icols, items)),
                polytomous_items=[items[j] for j in poly], not_reached=not_reached, polytomous_rule=polytomous)
    return out, meta


def fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def item_matrix(df):
    return df[[c for c in df.columns if c.startswith("I") and c[1:].isdigit()]].to_numpy(float)


def rt_matrix(df):
    return df[[c for c in df.columns if c.startswith("RT") and c[2:].isdigit()]].to_numpy(float)


def save_form(df, meta, name, processed=PROCESSED):
    df.to_parquet(processed / f"{name}.parquet")
    meta = dict(meta, fingerprint=fingerprint(df))
    (processed / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return meta

"""Phase 3: continuous, uncertainty-aware representation (CTR) versus discrete score-time profiles.

  dev      Form A training split, internal 70/30 split (exploratory; the Form A test split is never used).
  confirm  Forms B and C: fit on the training split, evaluate once on the untouched test split.

Representation (fixed before confirmation, see reports/protocol_continuous_temporal.md):
  CTR = ContinuousIRTDur(guessing=True, deg=3, het=True) on responses and the item-intensity-adjusted
  relative log time v (unmasked items only). Coordinates: ability (responses only) and relative time.

Usage: python scripts/run_continuous_eval.py --part dev|confirm [--form math_formB]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from run_experiments import load_form, arrays, CKPT, RES, CFG, mk, MODEL_SPECS  # noqa: E402
from mcq_profiles import baseline as bl  # noqa: E402
from mcq_profiles.data import rt_matrix  # noqa: E402
from mcq_profiles.evaluation import logloss_matrix, brier_matrix, calibration_table  # noqa: E402
from mcq_profiles.latent_classes import ContinuousIRT  # noqa: E402
from mcq_profiles.preprocessing import item_mask, unmasked_duration, log_duration  # noqa: E402
from mcq_profiles.statistics import paired_bootstrap_ci, match_labels  # noqa: E402
from mcq_profiles.continuous_map import (ContinuousIRTDur, relative_time_index, time_intensities,  # noqa: E402
                                         masked_score_distribution, interval_from_distribution, poisson_binomial,
                                         separation_strata)

N_JOBS = int(os.environ.get("AALBP_N_JOBS", 5))
OUT = RES / "continuous"
OUT.mkdir(exist_ok=True)
PAIRS = [("CTR", "KMeans4_score_time"), ("CTR", "CTR_kmeans4"), ("CTR", "CTR_grid2x2"), ("CTR", "G3PL"),
         ("CTR", "CTR_no_time"), ("CTR", "M3_3_time"), ("CTR", "MIRT2G"), ("CTR", "M1_3"), ("G3PL", "KMeans4_score_time")]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(obj, name):
    def conv(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    (OUT / name).write_text(json.dumps(obj, indent=1, default=conv), encoding="utf8")


def log_item_times(df, Y):
    RT = rt_matrix(df)
    return np.where((RT > 0) & ~np.isnan(Y), np.log(np.where(RT > 0, RT, 1.0)), np.nan)


class GroupPredictor:
    """Discretised representation: groups defined on 2-D coordinates; each group predicts with its training
    item p-values (as the K-means score-time predictor does)."""

    def __init__(self, kind):
        self.kind = kind

    def fit(self, Xc, Y):
        ok = np.isfinite(Xc).all(1)
        self.sc = StandardScaler().fit(Xc[ok])
        Z = self.sc.transform(Xc[ok])
        if self.kind == "kmeans4":
            self.km = KMeans(4, n_init=50, random_state=42).fit(Z)
            lab = self.km.labels_
        else:                                    # 2 x 2 median grid
            self.med = np.median(Z, 0)
            lab = self._grid(Z)
        Yo = Y[ok]
        self.p = np.array([(np.nansum(Yo[lab == c], 0) + .5) / ((~np.isnan(Yo[lab == c])).sum(0) + 1) for c in range(4)])
        self.p_all = (np.nansum(Yo, 0) + .5) / ((~np.isnan(Yo)).sum(0) + 1)
        return self

    def _grid(self, Z):
        return (Z[:, 0] > self.med[0]).astype(int) * 2 + (Z[:, 1] > self.med[1]).astype(int)

    def labels(self, Xc):
        ok = np.isfinite(Xc).all(1)
        lab = np.full(len(Xc), -1)
        Z = self.sc.transform(Xc[ok])
        lab[ok] = self.km.predict(Z) if self.kind == "kmeans4" else self._grid(Z)
        return lab

    def predict(self, Xc):
        lab = self.labels(Xc)
        P = np.tile(self.p_all, (len(Xc), 1))
        P[lab >= 0] = self.p[lab[lab >= 0]]
        return P


def _mask_run(seed, form, part, fit, ev, with_classes):
    import torch
    torch.set_num_threads(1)
    df, _ = load_form(form)
    Y, logT, _, g, _ = arrays(df)
    J = Y.shape[1]
    mask = item_mask(J, CFG["masking"]["fraction"], seed)
    RT = rt_matrix(df)
    mins_u = unmasked_duration(RT, np.isnan(Y), mask)
    logTu, _ = log_duration(mins_u)
    LT = log_item_times(df, Y)
    LT[:, mask] = np.nan
    lam = time_intensities(LT[fit])
    v, v_se, _ = relative_time_index(LT, lam)
    Yf, Ye = Y[fit], Y[ev]
    Yin = Ye.copy()
    Yin[:, mask] = np.nan
    target = Ye[:, mask]
    preds, post_store = {}, {}
    preds["Marginal"] = np.tile(np.nanmean(Yf, 0)[mask], (len(Ye), 1))
    kp = bl.ScoreTimeProfilePredictor(4).fit(Yf, mins_u[fit], mask)
    preds["KMeans4_score_time"] = kp.predict(Ye, mins_u[ev], mask)
    g3 = ContinuousIRT(dims=1, guessing=True).fit(Yf)
    pg = g3.posterior(Yin)[3]
    preds["G3PL"] = g3.predict_items(pg)[:, mask]
    post_store["G3PL"] = (pg, g3._p()[:, mask])
    ctr = ContinuousIRTDur(guessing=True, deg=3, het=True).fit(Yf, v[fit], None, g[fit])
    pc = ctr.posterior(Yin, v[ev], None, g[ev])[3]
    preds["CTR"] = ctr.predict_items(pc)[:, mask]
    post_store["CTR"] = (pc, ctr._p()[:, mask])
    pn = ctr.posterior(Yin, None, None, g[ev])[3]
    preds["CTR_no_time"] = ctr.predict_items(pn)[:, mask]
    # discretised versions of the same continuous information (unmasked items only, for both groups)
    Yfin = Yf.copy()
    Yfin[:, mask] = np.nan
    cf = ctr.coordinates(Yfin, v[fit], v_se[fit], g[fit])
    ce = ctr.coordinates(Yin, v[ev], v_se[ev], g[ev])
    Xf, Xe = cf[["theta", "rel_time"]].to_numpy(), ce[["theta", "rel_time"]].to_numpy()
    for kind in ["kmeans4", "grid2x2"]:
        gp = GroupPredictor(kind).fit(Xf, Yf)
        preds[f"CTR_{kind}"] = gp.predict(Xe)[:, mask]
    m0 = mk(1).fit(Yf, None, None, g[fit])
    preds["M0"] = m0.predict_items(m0.posterior(Yin, None, None, g[ev])[3])[:, mask]
    if with_classes:
        ck = pickle.load(open(CKPT / f"{form}_train_models.pkl", "rb"))
        preds["M1_3"] = ck["models"]["M1_3"].predict_items(ck["models"]["M1_3"].posterior(Yin, None, None, g[ev])[3])[:, mask]
        comp = pickle.load(open(CKPT / f"{form}_competitors.pkl", "rb"))
        preds["MIRT2G"] = comp["MIRT2G"].predict_items(comp["MIRT2G"].posterior(Yin)[3])[:, mask]
        sp = MODEL_SPECS["M3_3"]
        m33 = mk(sp["K"], sp.get("cs"), True, sp.get("group_effects", False)).fit(Yf, logTu[fit], None, g[fit])
        preds["M3_3_time"] = m33.predict_items(m33.posterior(Yin, logTu[ev], None, g[ev])[3])[:, mask]
    rows = {}
    for k, P in preds.items():
        ll, nobs = logloss_matrix(target, P)
        br, _ = brier_matrix(target, P)
        cal, ece = calibration_table(target, P)
        rows[k] = dict(ll=ll, br=br, nobs=nobs, ece=ece)
    # predictive intervals for the number correct on the observed masked items (C3)
    obs = ~np.isnan(target)
    y_sum = np.nansum(target, 1)
    inter = {}
    for k in ["CTR", "G3PL"]:
        post, Pn = post_store[k]
        dist = masked_score_distribution(Pn, post, obs)
        inter[k] = dist
    for k in ["KMeans4_score_time", "CTR_kmeans4"]:
        inter[k] = poisson_binomial(np.where(obs, preds[k], 0.0))
    iv = {}
    for k, dist in inter.items():
        lo, hi = interval_from_distribution(dist, 0.9)
        cover = (y_sum >= lo) & (y_sum <= hi)
        lscore = np.log(np.clip(dist[np.arange(len(y_sum)), y_sum.astype(int)], 1e-12, 1))
        cdf = np.cumsum(dist, 1)
        n_ = np.arange(len(y_sum))
        expected = cdf[n_, hi] - np.where(lo > 0, cdf[n_, np.maximum(lo - 1, 0)], 0.0)   # model mass inside [lo, hi]
        iv[k] = dict(cover=cover, width=hi - lo, logscore=lscore, expected=expected)
    return seed, rows, iv, obs.sum(1)


def split_half(form, fit, ev):
    """Stability of each representation across two disjoint halves of the items (no masking)."""
    df, _ = load_form(form)
    Y, _, _, g, _ = arrays(df)
    J = Y.shape[1]
    LT = log_item_times(df, Y)
    RT = rt_matrix(df)
    res = {}
    halves = [np.arange(J) % 2 == 0, np.arange(J) % 2 == 1]
    coords, labs, qlabs = [], [], []
    for h in halves:
        Yh = np.where(h[None, :], Y, np.nan)
        LTh = np.where(h[None, :], LT, np.nan)
        lam = time_intensities(LTh[fit])
        v, v_se, _ = relative_time_index(LTh, lam)
        ctr = ContinuousIRTDur(guessing=True, deg=3, het=True).fit(Yh[fit][:, h], v[fit], None, g[fit])
        c = ctr.coordinates(Yh[ev][:, h], v[ev], v_se[ev], g[ev])
        coords.append(c)
        # K-means score-time profiles on this half (fit on fit set, applied to eval set)
        mins = np.nansum(np.where(h[None, :] & (RT > 0), RT, 0.0), 1) / 60
        sc = np.nanmean(Yh, 1)
        X = np.column_stack([sc, np.log(np.where(mins > 0, mins, np.nan))])
        okf = np.isfinite(X[fit]).all(1)
        s = StandardScaler().fit(X[fit][okf])
        km = KMeans(4, n_init=50, random_state=42).fit(s.transform(X[fit][okf]))
        Xe = X[ev]
        lab = np.full(len(Xe), -1)
        oke = np.isfinite(Xe).all(1)
        lab[oke] = km.predict(s.transform(Xe[oke]))
        labs.append((lab, km.cluster_centers_))
        # best possible 4-group discretisation of the continuous coordinates: quartiles of ability
        qlabs.append(np.digitize(c.theta, np.quantile(c.theta, [.25, .5, .75])))
    a, b = coords
    ok = np.isfinite(a.rel_time) & np.isfinite(b.rel_time)
    r_th = np.corrcoef(a.theta, b.theta)[0, 1]
    r_rt = np.corrcoef(a.rel_time[ok], b.rel_time[ok])[0, 1]
    (la, ca), (lb, cb) = labs
    okl = (la >= 0) & (lb >= 0)
    # match the second half's cluster labels to the first half's by centroid proximity
    from scipy.optimize import linear_sum_assignment
    cost = ((ca[:, None, :] - cb[None, :, :]) ** 2).sum(2)
    r_, c_ = linear_sum_assignment(cost)
    perm = np.empty(4, int)
    perm[c_] = r_
    lb2 = np.where(lb >= 0, perm[np.clip(lb, 0, 3)], -1)
    res = dict(theta_r=r_th, theta_sb=2 * r_th / (1 + r_th), rel_time_r=r_rt, rel_time_sb=2 * r_rt / (1 + r_rt),
               kmeans_kappa=cohen_kappa_score(la[okl], lb2[okl]), kmeans_agreement=float((la[okl] == lb2[okl]).mean()),
               theta_quartile_kappa=cohen_kappa_score(qlabs[0], qlabs[1]), n=int(len(a)))
    return res


def effort_and_dashboard(form, fit, ev):
    """Out-of-sample prediction of self-reported effort, and descriptive dashboard indicators (full test)."""
    df, _ = load_form(form)
    Y, _, _, g, _ = arrays(df)
    LT = log_item_times(df, Y)
    lam = time_intensities(LT[fit])
    v, v_se, n_it = relative_time_index(LT, lam)
    ctr = ContinuousIRTDur(guessing=True, deg=3, het=True).fit(Y[fit], v[fit], None, g[fit])
    co = ctr.coordinates(Y, v, v_se, g)
    ext = pd.read_parquet(ROOT / "data" / "processed" / "student_external.parquet")
    d = df[["CNT", "CNTSTUID", "BOOKID", "final_score", "total_duration_minutes"]].merge(
        ext[["CNT", "CNTSTUID", "EFFORT1", "W_FSTUWT"]], how="left")
    X = np.column_stack([df.final_score.to_numpy(float), df.total_duration_minutes.to_numpy(float)])
    okx = np.isfinite(X).all(1) & (X[:, 1] > 0)
    s = StandardScaler().fit(X[fit & okx])
    km = KMeans(4, n_init=50, random_state=42).fit(s.transform(X[fit & okx]))
    lab = np.full(len(df), -1)
    lab[okx] = km.predict(s.transform(X[okx]))
    th, rt = co.theta.to_numpy(), co.rel_time.to_numpy()
    use = np.isfinite(d.EFFORT1.to_numpy()) & np.isfinite(rt) & (lab >= 0)
    B = pd.get_dummies(d.BOOKID.astype(str), drop_first=True).to_numpy(float)
    P = np.eye(4)[np.clip(lab, 0, 3)][:, 1:]
    C = np.column_stack([th, rt, th ** 2, rt ** 2, th * rt])
    y = d.EFFORT1.to_numpy(float)
    out = {}
    for nm, Xd in {"profiles": np.column_stack([B, P]), "continuous": np.column_stack([B, C]),
                   "both": np.column_stack([B, P, C])}.items():
        A = np.column_stack([np.ones(len(y)), Xd])
        a_, b_ = fit & use, ev & use
        beta = np.linalg.lstsq(A[a_], y[a_], rcond=None)[0]
        pred = A[b_] @ beta
        out[nm] = float(1 - np.sum((y[b_] - pred) ** 2) / np.sum((y[b_] - y[b_].mean()) ** 2))
        out[nm + "_resid"] = (y[b_] - pred)
    dse = out["profiles_resid"] ** 2 - out["continuous_resid"] ** 2
    eff = dict(r2_test={k: out[k] for k in ["profiles", "continuous", "both"]}, n=int((ev & use).sum()),
               mse_gain_continuous_vs_profiles=paired_bootstrap_ci(dse, 2000, seed=21))
    # dashboard indicators on the evaluation students
    e = co[ev]
    med = np.median(e.theta)
    rel_ev = float(np.var(e.theta) / (np.var(e.theta) + np.mean(e.theta_sd ** 2)))
    dash = dict(theta_interval_width_mean=float((e.theta_hi - e.theta_lo).mean()),
                share_theta_interval_excludes_median=float(((e.theta_lo > med) | (e.theta_hi < med)).mean()),
                share_rel_time_interval_excludes_0=float(((e.rel_time_lo > 0) | (e.rel_time_hi < 0))[np.isfinite(e.rel_time)].mean()),
                eap_reliability=rel_ev, separation_strata=float(separation_strata(rel_ev)),
                corr_theta_rel_time=float(np.corrcoef(e.theta[np.isfinite(e.rel_time)], e.rel_time[np.isfinite(e.rel_time)])[0, 1]),
                share_rel_time_missing=float(np.isnan(e.rel_time).mean()),
                between_profile_share_of_theta_variance=float(
                    pd.Series(th[ev & (lab >= 0)]).groupby(lab[ev & (lab >= 0)]).transform("mean").var() / np.var(th[ev & (lab >= 0)])),
                between_profile_share_of_rel_time_variance=float(
                    pd.Series(rt[ev & (lab >= 0) & np.isfinite(rt)]).groupby(lab[ev & (lab >= 0) & np.isfinite(rt)]).transform("mean").var()
                    / np.nanvar(rt[ev & (lab >= 0) & np.isfinite(rt)])))
    return eff, dash


def run(form, part):
    if part == "dev":
        from run_temporal import dev_split
        fit, ev = dev_split(form)
        with_classes = False
    else:
        ck = pickle.load(open(CKPT / f"{form}_train_models.pkl", "rb"))
        fit, ev = ck["tr"], ck["te"]
        with_classes = True
    seeds = CFG["masking"]["seeds"]
    out = Parallel(n_jobs=min(N_JOBS, len(seeds)))(delayed(_mask_run)(s, form, part, fit, ev, with_classes) for s in seeds)
    models = list(out[0][1])
    LL = {m: sum(o[1][m]["ll"] for o in out) for m in models}
    BR = {m: sum(o[1][m]["br"] for o in out) for m in models}
    NO = sum(o[1][models[0]]["nobs"] for o in out)
    ok = NO > 0
    df, _ = load_form(form)
    ext = pd.read_parquet(ROOT / "data" / "processed" / "student_external.parquet")
    w = df[["CNT", "CNTSTUID"]].merge(ext[["CNT", "CNTSTUID", "W_FSTUWT"]], how="left").W_FSTUWT.to_numpy(float)[ev]
    w = np.where(np.isnan(w), np.nanmedian(w), w)[ok]
    per = {m: LL[m][ok] / NO[ok] for m in models}
    summ = pd.DataFrame([dict(model=m, logloss=LL[m][ok].sum() / NO[ok].sum(), brier=BR[m][ok].sum() / NO[ok].sum(),
                              ece=np.mean([o[1][m]["ece"] for o in out]),
                              logloss_weighted=float(np.average(per[m], weights=w))) for m in models]).sort_values("logloss")
    cmp = []
    for a_, b_ in PAIRS:
        if a_ in per and b_ in per:
            d = per[b_] - per[a_]                      # > 0: a_ better
            mean, lo, hi = paired_bootstrap_ci(d, 2000, seed=9001)
            rng = np.random.default_rng(9002)
            bs = [np.average(d[i], weights=w[i]) for i in (rng.integers(0, len(d), len(d)) for _ in range(1000))]
            cmp.append(dict(better=a_, than=b_, gain=mean, ci_lo=lo, ci_hi=hi, gain_weighted=float(np.average(d, weights=w)),
                            ci_lo_weighted=float(np.quantile(bs, .025)), ci_hi_weighted=float(np.quantile(bs, .975))))
    cmp = pd.DataFrame(cmp)
    iv = {}
    for k in out[0][2]:
        cov = np.concatenate([o[2][k]["cover"][o[3] > 0] for o in out])
        wid = np.concatenate([o[2][k]["width"][o[3] > 0] for o in out])
        ls = np.concatenate([o[2][k]["logscore"][o[3] > 0] for o in out])
        ex = np.concatenate([o[2][k]["expected"][o[3] > 0] for o in out])
        iv[k] = dict(coverage90=float(cov.mean()), expected_coverage=float(ex.mean()),
                     coverage_minus_expected=float(cov.mean() - ex.mean()), width_mean=float(wid.mean()),
                     logscore_mean=float(ls.mean()))
    stab = split_half(form, fit, ev)
    eff, dash = effort_and_dashboard(form, fit, ev)
    summ.to_csv(OUT / f"{part}_{form}_scores.csv", index=False)
    cmp.to_csv(OUT / f"{part}_{form}_paired.csv", index=False)
    jdump(dict(form=form, part=part, n_eval=int(ok.sum()), intervals=iv, split_half=stab, effort=eff, dashboard=dash),
          f"{part}_{form}_summary.json")
    log(summ.to_string(), "\n", cmp.to_string(), "\n", iv, "\n", stab, "\n", eff, "\n", dash)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["dev", "confirm"])
    ap.add_argument("--form", default="math_formA")
    a = ap.parse_args()
    run("math_formA" if a.part == "dev" else a.form, a.part)

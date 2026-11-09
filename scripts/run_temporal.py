"""Option C: latent time regimes versus flexible continuous time distributions.

Parts (see reports/protocol_continuous_temporal.md):
  dev      Form A training split only, internal 70/30 split (exploratory development).
  null     Null simulations from continuous fits on the Form A development split (false regime discoveries)
           and a positive control with true regimes (power).
  boot     Bootstrap stability of the regime model on the Form A development split.
  confirm  Forms B and C: fit on the training split, evaluate once on the untouched test split.

Usage: python scripts/run_temporal.py --part dev|null|boot|confirm [--form math_formB]
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from run_experiments import load_form, arrays, CKPT, RES  # noqa: E402
from mcq_profiles.data import rt_matrix  # noqa: E402
from mcq_profiles.latent_classes import ContinuousIRT  # noqa: E402
from mcq_profiles.preprocessing import train_test_split_stratified  # noqa: E402
from mcq_profiles.continuous_map import relative_time_index, time_intensities  # noqa: E402
from mcq_profiles.temporal import (CONTINUOUS_SPECS, ContinuousTimeModel, RegimeTimeModel,  # noqa: E402
                                   item_loglik_grid)
from mcq_profiles.statistics import paired_bootstrap_ci  # noqa: E402

N_JOBS = int(os.environ.get("AALBP_N_JOBS", 12))
OUT = RES / "temporal"
OUT.mkdir(exist_ok=True)
VARS = ["v_relative", "log_total_time"]          # primary: item-adjusted relative log time


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(obj, name):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    (OUT / name).write_text(json.dumps(obj, indent=1, default=conv), encoding="utf8")


# --------------------------------------------------------------------------- data
def prep(form, fit, ev):
    """Responses, booklets, weights and the two time variables. Item time intensities come from ``fit``."""
    df, _ = load_form(form)
    Y, logT, _, g, _ = arrays(df)
    RT = rt_matrix(df)
    LT = np.where((RT > 0) & ~np.isnan(Y), np.log(np.where(RT > 0, RT, 1.0)), np.nan)
    lam = time_intensities(LT[fit])
    v, v_se, n_it = relative_time_index(LT, lam)
    ext = pd.read_parquet(ROOT / "data" / "processed" / "student_external.parquet")
    w = df[["CNT", "CNTSTUID"]].merge(ext[["CNT", "CNTSTUID", "W_FSTUWT"]], how="left").W_FSTUWT.to_numpy(float)
    w = np.where(np.isnan(w), np.nanmedian(w), w)
    item = ContinuousIRT(dims=1, guessing=True).fit(Y[fit])
    LY = item_loglik_grid(item, Y)
    return dict(df=df, Y=Y, g=g, w=w, x=item.X[:, 0], LY=LY, item=item, fit=fit, ev=ev,
                var={"v_relative": v, "log_total_time": logT}, v_se=v_se, n_items=n_it)


def make(name):
    if name.startswith("regimes"):
        return RegimeTimeModel(int(name.split("=")[1]), n_starts=5, seed=7)
    fam, loc, sc = name.split("|")
    return ContinuousTimeModel(fam, loc, sc)


MODEL_NAMES = ["|".join(s) for s in CONTINUOUS_SPECS] + ["regimes|K=2", "regimes|K=3"]


def _fit_one(name, LY, v, g, x, w):
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    m = make(name).fit(LY, v, g, x, weights=w)
    m.secs = time.time() - t0
    return name, m


def fit_all(D, idx, var, weights=False, names=MODEL_NAMES):
    v = D["var"][var]
    w = D["w"][idx] if weights else None
    res = Parallel(n_jobs=min(N_JOBS, len(names)))(
        delayed(_fit_one)(nm, D["LY"][idx], v[idx], D["g"][idx], D["x"], w) for nm in names)
    return dict(res)


def evaluate(models, D, idx, var):
    v, LY, g, w = D["var"][var][idx], D["LY"][idx], D["g"][idx], D["w"][idx]
    rows, per = [], {}
    for nm, m in models.items():
        cl = m.cond_loglik(LY, v, g)
        pit = m.pit(LY, v, g)
        ok = np.isfinite(cl)
        per[nm] = cl
        pk = pit[np.isfinite(pit)]
        rows.append(dict(model=nm, train_loglik=m.info.loglik, k=m.info.n_params, train_bic=m.bic(),
                         converged=m.info.converged, heldout_ll=float(cl[ok].mean()),
                         heldout_ll_weighted=float(np.average(cl[ok], weights=w[ok])),
                         pit_cover90=float(((pk > 0.05) & (pk < 0.95)).mean()),
                         pit_ks=float(np.max(np.abs(np.sort(pk) - (np.arange(1, len(pk) + 1) - 0.5) / len(pk)))),
                         n_eval=int(ok.sum()), secs=getattr(m, "secs", np.nan)))
    return pd.DataFrame(rows), per


def compare(tab, per, w=None):
    """Best regime vs best continuous model (both selected by training BIC); paired bootstrap on held-out ll."""
    cont = tab[~tab.model.str.startswith("regimes")].sort_values("train_bic").iloc[0].model
    reg = tab[tab.model.str.startswith("regimes")].sort_values("train_bic").iloc[0].model
    d = per[reg] - per[cont]
    ok = np.isfinite(d)
    mean, lo, hi = paired_bootstrap_ci(d[ok], 2000, seed=11)
    out = dict(best_continuous=cont, best_regime=reg,
               dBIC_cont_minus_regime=float(tab.set_index("model").train_bic[cont] - tab.set_index("model").train_bic[reg]),
               heldout_gain_regime_vs_cont=[mean, lo, hi])
    if w is not None:
        rng = np.random.default_rng(12)
        dd, ww = d[ok], w[ok]
        bs = [np.average(dd[i], weights=ww[i]) for i in (rng.integers(0, len(dd), len(dd)) for _ in range(2000))]
        out["heldout_gain_weighted"] = [float(np.average(dd, weights=ww)), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
    return out


# --------------------------------------------------------------------------- parts
def dev_split(form="math_formA"):
    ck = pickle.load(open(CKPT / f"{form}_train_models.pkl", "rb"))
    df, _ = load_form(form)
    tr = ck["tr"]
    sub = df[tr].reset_index()
    a, b = train_test_split_stratified(sub, seed=11, test_fraction=0.3)
    fit = np.zeros(len(df), bool)
    ev = np.zeros(len(df), bool)
    fit[sub["index"][a]] = True
    ev[sub["index"][b]] = True
    assert not (fit & ck["te"]).any() and not (ev & ck["te"]).any()        # the Form A test split is never used
    return fit, ev


def part_dev():
    form = "math_formA"
    fit, ev = dev_split(form)
    D = prep(form, fit, ev)
    summary = {}
    for var in VARS:
        log("dev", var)
        models = fit_all(D, fit, var)
        tab, per = evaluate(models, D, ev, var)
        tab.to_csv(OUT / f"dev_{form}_{var}.csv", index=False)
        summary[var] = compare(tab, per, D["w"][ev])
        summary[var]["regime_summary"] = {nm: m.summary() for nm, m in models.items() if nm.startswith("regimes")}
        # weighted fits (pseudo-likelihood) of the two compared models
        wm = fit_all(D, fit, var, weights=True, names=[summary[var]["best_continuous"], summary[var]["best_regime"]])
        wt, wp = evaluate(wm, D, ev, var)
        summary[var]["weighted_fit"] = dict(table=wt.to_dict("records"),
                                            heldout_gain=compare(wt, wp)["heldout_gain_regime_vs_cont"])
        pickle.dump(models, open(CKPT / f"temporal_dev_{var}.pkl", "wb"))
        log(tab[["model", "k", "train_bic", "heldout_ll", "pit_cover90", "pit_ks", "secs"]].to_string())
        log(summary[var])
    summary["n_fit"], summary["n_eval"] = int(fit.sum()), int(ev.sum())
    jdump(summary, f"dev_{form}_summary.json")


def _null_rep(kind, r, gen, item, D, fit, var, seed):
    """One replication: theta ~ N(0,1), responses from the fitted G3PL with a real missingness pattern,
    time from the generator; fit all models; return BICs and held-out ll on a simulated validation half."""
    import torch
    torch.set_num_threads(1)
    from scipy.special import expit
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(fit)
    n = len(idx)
    src = rng.choice(idx, 2 * n, replace=True)
    th = rng.normal(size=2 * n)
    P = item.c + (1 - item.c) * expit(th[:, None] * item.A[:, 0] + item.d)
    Y = (rng.random(P.shape) < P).astype(float)
    Y[np.isnan(D["Y"][src])] = np.nan
    g = D["g"][src]
    v = gen.sample(th, g, rng)
    if isinstance(v, tuple):
        v = v[0]
    v[np.isnan(D["var"][var][src])] = np.nan
    LY = item_loglik_grid(item, Y)
    a, b = np.arange(n), np.arange(n, 2 * n)
    out = {}
    for nm in MODEL_NAMES:
        m = make(nm).fit(LY[a], v[a], g[a], D["x"])
        cl = m.cond_loglik(LY[b], v[b], g[b])
        out[nm] = dict(bic=m.bic(), heldout=float(np.nanmean(cl)))
    return kind, r, out


def part_null(R=20):
    form = "math_formA"
    fit, ev = dev_split(form)
    D = prep(form, fit, ev)
    var = "v_relative"
    models = pickle.load(open(CKPT / f"temporal_dev_{var}.pkl", "rb"))
    gens = {"shash_spline": models["shash|spline|spline"], "t_spline": models["t|spline|spline"],
            "skewnormal_spline": models["skewnormal|spline|spline"], "normal_cubic_het": models["normal|cubic|linear"],
            "positive_regimes2": models["regimes|K=2"]}
    jobs = [(k, r, gens[k]) for k in gens for r in range(R)]
    res = Parallel(n_jobs=N_JOBS)(delayed(_null_rep)(k, r, gm, D["item"], D, fit, var, 1000 * i + r)
                                  for i, (k, r, gm) in enumerate(jobs))
    rows = []
    for kind, r, out in res:
        tab = pd.DataFrame([dict(model=nm, **o) for nm, o in out.items()])
        cont = tab[~tab.model.str.startswith("regimes")].sort_values("bic").iloc[0]
        reg = tab[tab.model.str.startswith("regimes")].sort_values("bic").iloc[0]
        rows.append(dict(generator=kind, rep=r, best_continuous=cont.model, best_regime=reg.model,
                         dBIC_cont_minus_regime=cont.bic - reg.bic, heldout_gain_regime=reg.heldout - cont.heldout,
                         regime_wins_bic=bool(reg.bic < cont.bic)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "null_simulations.csv", index=False)
    s = t.groupby("generator").agg(R=("rep", "size"), regime_wins_bic=("regime_wins_bic", "mean"),
                                    dBIC_q95=("dBIC_cont_minus_regime", lambda x: np.quantile(x, .95)),
                                    dBIC_median=("dBIC_cont_minus_regime", "median"),
                                    heldout_gain_median=("heldout_gain_regime", "median"),
                                    heldout_gain_q95=("heldout_gain_regime", lambda x: np.quantile(x, .95)))
    s.to_csv(OUT / "null_summary.csv")
    log(s.to_string())


def part_boot(B=20):
    form = "math_formA"
    fit, ev = dev_split(form)
    D = prep(form, fit, ev)
    var = "v_relative"
    ref = pickle.load(open(CKPT / f"temporal_dev_{var}.pkl", "rb"))
    summ = json.loads((OUT / f"dev_{form}_summary.json").read_text())
    nm = summ[var]["best_regime"]
    idx = np.flatnonzero(fit)

    def one(b):
        import torch
        torch.set_num_threads(1)
        rng = np.random.default_rng(500 + b)
        s = rng.choice(idx, len(idx), replace=True)
        m = make(nm).fit(D["LY"][s], D["var"][var][s], D["g"][s], D["x"])
        post = m.regime_posterior(D["LY"][idx], D["var"][var][idx], D["g"][idx])
        return m.summary(), np.nanargmax(np.nan_to_num(post, nan=-1), 1)
    res = Parallel(n_jobs=N_JOBS)(delayed(one)(b) for b in range(B))
    from sklearn.metrics import adjusted_rand_score
    rp = ref[nm].regime_posterior(D["LY"][idx], D["var"][var][idx], D["g"][idx])
    lab0 = np.nanargmax(np.nan_to_num(rp, nan=-1), 1)
    ok = np.isfinite(D["var"][var][idx])
    aris = [adjusted_rand_score(lab0[ok], lab[ok]) for _, lab in res]
    gaps = [np.diff(s["intercepts"]).tolist() for s, _ in res]
    prev = [s["prevalence_marginal"] for s, _ in res]
    out = dict(model=nm, B=B, ari_vs_full=dict(median=float(np.median(aris)), q05=float(np.quantile(aris, .05)), values=aris),
               intercept_gaps=dict(mean=np.mean(gaps, 0), q025=np.quantile(gaps, .025, 0), q975=np.quantile(gaps, .975, 0)),
               prevalence=dict(mean=np.mean(prev, 0), sd=np.std(prev, 0)), reference=ref[nm].summary())
    jdump(out, "bootstrap_stability.json")
    log(out["ari_vs_full"]["median"], out["intercept_gaps"], out["prevalence"])


def part_confirm(form):
    ck = pickle.load(open(CKPT / f"{form}_train_models.pkl", "rb"))
    tr, te = ck["tr"], ck["te"]
    D = prep(form, tr, te)
    summary = {}
    for var in VARS:
        log("confirm", form, var)
        models = fit_all(D, tr, var)
        tab, per = evaluate(models, D, te, var)
        tab.to_csv(OUT / f"confirm_{form}_{var}.csv", index=False)
        summary[var] = compare(tab, per, D["w"][te])
        summary[var]["regime_summary"] = {nm: m.summary() for nm, m in models.items() if nm.startswith("regimes")}
        log(tab[["model", "k", "train_bic", "heldout_ll", "pit_cover90", "pit_ks"]].to_string())
        log(summary[var])
    summary["n_train"], summary["n_test"] = int(tr.sum()), int(te.sum())
    jdump(summary, f"confirm_{form}_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["dev", "null", "boot", "confirm"])
    ap.add_argument("--form", default="math_formB")
    a = ap.parse_args()
    {"dev": part_dev, "null": part_null, "boot": part_boot}.get(a.part, lambda: part_confirm(a.form))()

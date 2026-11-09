"""Experiment stages E1, E5-E10 and H5 (imported by run_experiments.py)."""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from run_experiments import (CFG, CKPT, RES, N_JOBS, log, save_json, load_form, arrays, mk, MODEL_SPECS,
                             fit_models, stage_fit)
from mcq_profiles import baseline as bl
from mcq_profiles import statistics as st
from mcq_profiles.data import rt_matrix
from mcq_profiles.evaluation import logloss_matrix, brier_matrix, calibration_table
from mcq_profiles.latent_classes import MixIRT, LCA
from mcq_profiles.preprocessing import item_mask, unmasked_duration, log_duration
from mcq_profiles.simulation import simulate, centred_shift, _theta


def load_ckpt(form):
    with open(CKPT / f"{form}_train_models.pkl", "rb") as f:
        return pickle.load(f)


def bic(m, n):
    return -2 * m.result_.loglik + m.result_.n_params * np.log(n)


def best_continuous(models, n):
    c = {k: bic(models[k], n) for k in ("M0", "A2", "A3")}
    return min(c, key=c.get), c


# ------------------------------------------------------------------ E1
def _copula_null_silhouette(X, rho_s, seed, k=4):
    """Silhouette of K-means(k) on data with the same marginals and Spearman correlation
    but no cluster structure (Gaussian copula)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    r = 2 * np.sin(np.pi * rho_s / 6)
    z = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], n)
    u = stats.norm.cdf(z)
    Xs = np.column_stack([np.quantile(X[:, j], u[:, j], method="inverted_cdf") for j in range(2)])
    Xs = Xs + rng.uniform(-1e-6, 1e-6, Xs.shape)
    km, Z = bl.kmeans_fit(Xs, k, seed=seed, n_init=10)
    return silhouette_score(Z, km.labels_, sample_size=3000, random_state=seed)


def stage_baseline(form, demo=False):
    df, meta = load_form(form, demo)
    X, ok = bl.features(df.final_score.to_numpy(float), df.total_duration_minutes.to_numpy(float))
    n_missing_time = int((~ok).sum())
    nonpos = ok & (X[:, 1] <= 0)
    ok = ok & ~nonpos   # zero total time (all items not reached) is not a valid duration; reported, not silent
    Xo = X[ok]
    out = dict(form=form, n_total=len(df), n_used=int(ok.sum()), n_excluded_no_time=n_missing_time,
               n_excluded_nonpositive_time=int(nonpos.sum()))
    out["pearson_r"], out["pearson_p"] = stats.pearsonr(Xo[:, 0], Xo[:, 1])
    out["spearman_r"] = stats.spearmanr(Xo[:, 0], Xo[:, 1])[0]
    # quadratic shape of time-score relation
    t = (Xo[:, 1] - Xo[:, 1].mean()) / Xo[:, 1].std()
    beta = np.polyfit(t, Xo[:, 0], 2)
    out["score_on_time_quadratic"] = beta
    out["k_curve"] = bl.k_curve(Xo, CFG["models"]["kmeans_k"])
    km, Z = bl.kmeans_fit(Xo, 4)
    lab = km.labels_
    out["k4"] = bl.indices(Z, lab)
    cent = pd.DataFrame(Xo).groupby(lab).agg(["mean", "size"])
    out["k4_profiles"] = [dict(cluster=int(c), n=int(cent.loc[c, (0, "size")]), score=float(cent.loc[c, (0, "mean")]),
                               minutes=float(cent.loc[c, (1, "mean")])) for c in cent.index]
    out["k4_cv_silhouette"] = bl.cv_silhouette(Xo, 4)
    aris, jac = bl.bootstrap_stability(Xo, 4, B=100 if not demo else 10)
    out["k4_bootstrap_ari"] = dict(mean=aris.mean(), q025=np.quantile(aris, .025), q975=np.quantile(aris, .975))
    out["k4_jaccard"] = jac
    kml, Zl = bl.kmeans_fit(np.column_stack([Xo[:, 0], np.log(Xo[:, 1])]), 4)
    out["k4_logtime"] = bl.indices(Zl, kml.labels_)
    out["k4_logtime_ari_vs_raw"] = adjusted_rand_score(lab, kml.labels_)
    out["gmm_bic"] = bl.gmm_bic(Xo, range(1, 9))
    agl, Za = bl.agglomerative(Xo, 4)
    out["agglomerative"] = dict(**bl.indices(Za, agl), ari_vs_kmeans=adjusted_rand_score(lab, agl))
    out["dbscan"] = bl.dbscan_grid(Xo, [0.1, 0.2, 0.3, 0.4, 0.5], [5, 10, 20])
    hl, Zh = bl.hdbscan(Xo, 50)
    out["hdbscan"] = dict(n_clusters=len(set(hl) - {-1}), noise=float((hl == -1).mean()))
    out["hopkins"] = st.hopkins(Z, rng=0)
    # null silhouette distribution (no clusters, same marginals and rank correlation)
    R = 100 if not demo else 5
    sil0 = Parallel(n_jobs=N_JOBS)(delayed(_copula_null_silhouette)(Xo, out["spearman_r"], s) for s in range(R))
    out["null_silhouette"] = dict(R=R, mean=np.mean(sil0), q95=np.quantile(sil0, .95), max=np.max(sil0),
                                  p_value=float((np.sum(np.array(sil0) >= out["k4"]["silhouette"]) + 1) / (R + 1)),
                                  values=sil0)
    out["cramers_v_oecd"] = st.cramers_v(lab, df.OECD.to_numpy()[ok])
    out["cramers_v_book"] = st.cramers_v(lab, df.BOOKID.to_numpy()[ok])
    near = (Xo[:, 1] >= 55).mean()
    out["share_ge_55min"], out["share_gt_60min"] = near, (Xo[:, 1] > 60).mean()
    out["kr20"] = st.kr20(df[[c for c in df.columns if c.startswith("I") and c[1:].isdigit()]].to_numpy(float))
    full = np.full(len(df), -1)
    full[ok] = lab
    np.save(CKPT / f"{form}_kmeans_labels.npy", full)
    save_json(out, f"{form}_E1_baseline.json")
    log("E1", {k: out[k] for k in ["n_used", "pearson_r", "spearman_r", "k4", "hopkins", "cramers_v_oecd"]})
    log("E1 null silhouette", {k: out["null_silhouette"][k] for k in ["mean", "q95", "p_value"]})
    return out


def _cached(path, fn, *args):
    """Run fn(*args) once; persist the result so an interrupted stage can resume."""
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    out = fn(*args)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    tmp.replace(path)
    return out


# ------------------------------------------------------------------ E5
def _null_rep(r, gen, Ytr, logTtr, gtr, cfg, seed, demo):
    rng = np.random.default_rng(seed + r)
    n, J = Ytr.shape
    m0 = gen["cont"]
    pi = np.exp(m0.logpi[0])
    if cfg == "fitted":
        c = rng.choice(m0.K, n, p=pi)
        theta = m0.mu[c] + m0.sigma[c] * rng.standard_normal(n)
    else:
        theta = _theta(rng, n, "skew", (-0.6, 1.0))
    a, b = m0.a[0], m0.b[0]
    Y = (rng.random((n, J)) < expit(a[None] * (theta[:, None] - b[None]))).astype(float)
    rows = rng.permutation(n)
    Y[np.isnan(Ytr[rows])] = np.nan
    g = gtr[rows]
    d = gen["dur"]
    th_std = (theta - theta.mean()) / theta.std()
    logT = d.alpha[0] + d.gamma[g] + d.beta[0] * th_std + np.exp(d.logs[0]) * rng.standard_normal(n)
    logT[np.isnan(logTtr[rows])] = np.nan
    names = ["M0", "A2", "M1_2", "M1_3", "M2", "M3_2"]
    res = {}
    fits = {}
    for nm in names:
        sp = MODEL_SPECS[nm]
        m = mk(sp["K"], sp.get("cs"), sp.get("duration", False), sp.get("group_effects", False), starts=3, demo=demo)
        dur = sp.get("duration", False)
        m.fit(Y, logT if dur else None, None, g)
        fits[nm] = m
        res[nm] = dict(ll=m.result_.loglik, k=m.result_.n_params, conv=m.result_.converged, bic=bic(m, n))
    cont = min(res["M0"]["bic"], res["A2"]["bic"])
    mix = min(res["M1_2"]["bic"], res["M1_3"]["bic"])
    res["dBIC_resp"] = cont - mix                       # >0 => class model preferred
    res["dBIC_dur"] = res["M2"]["bic"] - res["M3_2"]["bic"]
    pc = fits["M1_2"].posterior(Y)[0]
    res["M1_2_min_prop"] = float(pc.mean(0).min())
    res["M1_2_entropy"] = float(fits["M1_2"].entropy_r2(pc))
    # LCA enumeration
    lb = []
    for K in range(1, 6):
        l = LCA(K, n_starts=4, random_state=K).fit(Y)
        lb.append(l.bic(n))
    res["lca_K"] = int(np.argmin(lb) + 1)
    # source-study pipeline: K-means k=4 silhouette on (score, time)
    score = np.nansum(Y, 1)
    ok = ~np.isnan(logT)
    km, Z = bl.kmeans_fit(np.column_stack([score[ok], np.exp(logT[ok])]), 4, n_init=10)
    res["kmeans4_silhouette"] = float(silhouette_score(Z, km.labels_, sample_size=3000, random_state=0))
    res["cfg"], res["rep"] = cfg, r
    return res


def stage_null(form, demo=False):
    ck = load_ckpt(form)
    df, _ = load_form(form, demo)
    Y, logT, cens, g, _ = arrays(df)
    tr = ck["tr"]
    n = tr.sum()
    best, bics = best_continuous(ck["models"], n)
    gen = dict(cont=ck["models"][best], dur=ck["models"]["M2"])
    R = CFG["null_sim"]["R"] if not demo else 2
    log("E5 generator", best, bics)
    jobs = [(r, cfg) for cfg in CFG["null_sim"]["configs"] for r in range(R)]
    cdir = CKPT / f"{form}_E5_reps"
    cdir.mkdir(exist_ok=True)
    out = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(_cached)(cdir / f"{cfg}_{r}.pkl", _null_rep, r, gen, Y[tr], logT[tr], g[tr], cfg, 7001, demo)
                                             for r, cfg in jobs)
    tab = pd.DataFrame([{**{k: v for k, v in o.items() if not isinstance(v, dict)},
                         **{f"{m}_{k}": v for m in ["M0", "A2", "M1_2", "M1_3", "M2", "M3_2"] for k, v in o[m].items()}} for o in out])
    tab.to_csv(RES / "tables" / f"{form}_E5_null_reps.csv", index=False)
    summ = {}
    for cfg, t in tab.groupby("cfg"):
        k = int((t.dBIC_resp > 0).sum())
        kd = int((t.dBIC_dur > 0).sum())
        summ[cfg] = dict(R=len(t), fp_resp=k / len(t), fp_resp_ci=st.clopper_pearson(k, len(t)),
                         fp_dur=kd / len(t), fp_dur_ci=st.clopper_pearson(kd, len(t)),
                         dBIC_resp_q95=float(np.quantile(t.dBIC_resp, .95)), dBIC_dur_q95=float(np.quantile(t.dBIC_dur, .95)),
                         lca_K_counts=t.lca_K.value_counts().to_dict(),
                         kmeans4_sil_mean=float(t.kmeans4_silhouette.mean()), kmeans4_sil_q95=float(t.kmeans4_silhouette.quantile(.95)),
                         nonconverged=int((~t[[c for c in t.columns if c.endswith("_conv")]].astype(bool)).sum().sum()))
    save_json(dict(generator=best, generator_bics=bics, summary=summ), f"{form}_E5_null.json")
    log("E5", summ)


# ------------------------------------------------------------------ E6
def _pos_rep(cell, r, a, b, demo):
    mag, prop, n, dur = cell
    seed = 8001 + hash((mag, prop, n, dur, r)) % 100000
    J = len(b)
    S = centred_shift(J, 2, mag, rng=seed)
    d = simulate(n, a, b, rng=seed + 1, n_classes=2, class_props=[1 - prop, prop], class_b_shift=S,
                 dur=dict(alpha=np.log(30), beta=0.2, s=0.45, class_alpha=[0, 0.3 if dur else 0]))
    t0 = time.time()
    fits = {}
    for nm in ["M0", "A2", "M1_2", "M1_3"] + (["M2", "M3_2"] if dur else []):
        sp = MODEL_SPECS[nm]
        m = mk(sp["K"], sp.get("cs"), sp.get("duration", False), False, starts=3, demo=demo)
        m.fit(d["Y"], d["logT"] if sp.get("duration") else None, None, None)
        fits[nm] = m
    B = {k: bic(m, n) for k, m in fits.items()}
    sel = min(["M0", "A2", "M1_2", "M1_3"], key=B.get)
    pc = fits["M1_2"].posterior(d["Y"])[0]
    lab = pc.argmax(1)
    ari = adjusted_rand_score(d["cls"], lab)
    # b-deviation recovery (sign/label aligned)
    est = fits["M1_2"].b[1] - fits["M1_2"].b[0]
    true = S[1] - S[0]
    rm = min(np.sqrt(((est - true) ** 2).mean()), np.sqrt(((-est - true) ** 2).mean()))
    # calibration of assignment: mean max posterior vs accuracy
    lab_al, _ = st.match_labels(d["cls"], lab, 2)
    acc = float((lab_al == d["cls"]).mean())
    res = dict(mag=mag, prop=prop, n=n, dur=dur, rep=r, selected=sel, power=sel.startswith("M1"),
               K_hat=int(sel[-1]) if sel.startswith("M1") else 1, ari=ari, rmse_bdev=rm,
               mean_maxpost=float(pc.max(1).mean()), accuracy=acc, secs=time.time() - t0)
    if dur:
        pc3 = fits["M3_2"].posterior(d["Y"], d["logT"])[0]
        res["ari_with_duration"] = adjusted_rand_score(d["cls"], pc3.argmax(1))
        res["dBIC_dur"] = B["M2"] - B["M3_2"]
    return res


def stage_positive(form, demo=False):
    ck = load_ckpt(form)
    a, b = ck["models"]["M0"].a[0], ck["models"]["M0"].b[0]
    P = CFG["positive_sim"]
    R = P["R"] if not demo else 1
    cells = [(m, p, n, False) for m in P["magnitude"] for p in P["proportion"] for n in P["n"]]
    cells += [(1.0, p, n, True) for p in P["proportion"] for n in P["n"]]
    jobs = [(c, r) for c in cells for r in range(R)]
    cdir = CKPT / f"{form}_E6_reps"
    cdir.mkdir(exist_ok=True)
    out = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(_cached)(cdir / ("_".join(map(str, c)) + f"_{r}.pkl"), _pos_rep, c, r, a, b, demo)
                                             for c, r in jobs)
    tab = pd.DataFrame(out)
    tab.to_csv(RES / "tables" / f"{form}_E6_positive_reps.csv", index=False)
    g = tab.groupby(["mag", "prop", "n", "dur"])
    summ = g.agg(power=("power", "mean"), K2=("K_hat", lambda s: (s == 2).mean()), ari=("ari", "mean"),
                 rmse_bdev=("rmse_bdev", "mean"), accuracy=("accuracy", "mean"), mean_maxpost=("mean_maxpost", "mean"),
                 secs=("secs", "mean")).reset_index()
    if "ari_with_duration" in tab:
        summ = summ.merge(g.agg(ari_dur=("ari_with_duration", "mean")).reset_index(), how="left")
    summ.to_csv(RES / "tables" / f"{form}_E6_positive_summary.csv", index=False)
    log("E6\n" + summ.to_string())


# ------------------------------------------------------------------ E10
def _e10_mask(seed, form, df, Y, logT, g, tr, te, models, lcas, demo):
    J = Y.shape[1]
    mask = item_mask(J, CFG["masking"]["fraction"], seed)
    RT = rt_matrix(df)
    raw_nr = np.isnan(Y)  # not reached / missing responses -> time 0 if time missing
    mins_u = unmasked_duration(RT, raw_nr, mask)
    logTu, _ = log_duration(mins_u)
    Ytr = Y[tr]
    # duration models refit on unmasked time (train students, all items' responses)
    dur_models = {}
    for nm in ["M2", "M3_2", "M3_3"]:
        sp = MODEL_SPECS[nm]
        m = mk(sp["K"], sp.get("cs"), True, sp.get("group_effects", False), demo=demo)
        m.fit(Ytr, logTu[tr], None, g[tr])
        dur_models[nm] = m
    # Exploratory ablation (deviation log, 2026-11-08): item-level speed instead of total time.
    # van der Linden log-normal: log t_ij = lambda_j - tau_i + e; lambda_j from train students,
    # tau_hat_i = mean over the student's unmasked, timed items of (lambda_j - log t_ij).
    LT = np.log(np.where(RT > 0, RT, np.nan))
    LT[:, mask] = np.nan
    lam = np.nanmean(LT[tr], 0)
    tau = np.nanmean(lam[None, :] - LT, 1)
    tau[np.isnan(LT).all(1)] = np.nan
    for nm in ["M2", "M3_2"]:
        sp = MODEL_SPECS[nm]
        m = mk(sp["K"], sp.get("cs"), True, sp.get("group_effects", False), demo=demo)
        m.fit(Ytr, -tau[tr], None, g[tr])   # -tau so that larger = slower, same orientation as log T
        dur_models[nm + "_speed"] = m
    speed_var = {k: (-tau if k.endswith("_speed") else logTu) for k in dur_models}
    Yte = Y[te]
    target = Yte[:, mask]
    preds = {}
    p_marg = np.nanmean(Ytr, 0)[mask]
    preds["Marginal"] = np.tile(p_marg, (len(Yte), 1))
    kp = bl.ScoreTimeProfilePredictor(4).fit(Ytr, mins_u[tr], mask)
    preds["KMeans4_score_time"] = kp.predict(Yte, mins_u[te], mask)
    nb = len(Ytr)
    lbest = min(lcas, key=lambda k: lcas[k].bic(nb))
    Yin = Yte.copy()
    Yin[:, mask] = np.nan
    r, _ = lcas[lbest].posterior(Yin)
    preds[f"{lbest}"] = (r @ lcas[lbest].p)[:, mask]
    for nm in ["M0", "A2", "A3", "M1_2", "M1_3"]:
        post = models[nm].posterior(Yin, None, None, g[te])[3]
        preds[nm] = models[nm].predict_items(post)[:, mask]
    cp = CKPT / f"{form}_competitors.pkl"
    if cp.exists():  # exploratory continuous competitors (E3b)
        for nm, m in pickle.load(open(cp, "rb")).items():
            if nm == "C2PL":
                continue
            post = m.posterior(Yin)[3]
            preds[nm] = m.predict_items(post)[:, mask]
    for nm, m in dur_models.items():
        post = m.posterior(Yin, speed_var[nm][te], None, g[te])[3]
        preds[nm if nm.endswith("_speed") else nm + "_time"] = m.predict_items(post)[:, mask]
    rows = {}
    for k, P in preds.items():
        ll, nobs = logloss_matrix(target, P)
        br, _ = brier_matrix(target, P)
        _, ece = calibration_table(target, P)
        cal, _ = calibration_table(target, P)
        rows[k] = dict(ll=ll, br=br, nobs=nobs, ece=ece, cal=cal)
    # duration density on test (unmasked time) : M2 conditional vs intercept-only
    from mcq_profiles.duration import lognormal_marginal_loglik
    m2 = dur_models["M2"]
    full = m2.loglik(Yin, logTu[te], None, g[te], per_person=True)
    m2.duration = False
    resp = m2.loglik(Yin, None, None, g[te], per_person=True)
    m2.duration = True
    ok = ~np.isnan(logTu[te])
    dens_cond = (full - resp)[ok]
    dens_marg = lognormal_marginal_loglik(logTu[tr], logTu[te])
    return seed, mask, rows, dict(cond=float(dens_cond.mean()), marg=float(dens_marg.mean()), n=int(ok.sum()))


def stage_e10(form, demo=False):
    ck = load_ckpt(form)
    df, _ = load_form(form, demo)
    Y, logT, cens, g, _ = arrays(df)
    tr, te = ck["tr"], ck["te"]
    seeds = CFG["masking"]["seeds"] if not demo else CFG["masking"]["seeds"][:1]
    out = Parallel(n_jobs=len(seeds))(delayed(_e10_mask)(s, form, df, Y, logT, g, tr, te, ck["models"], ck["lcas"], demo) for s in seeds)
    models = list(out[0][2])
    # per student: sum over masks of log loss / sum of obs
    LL = {m: sum(o[2][m]["ll"] for o in out) for m in models}
    NO = sum(o[2][models[0]]["nobs"] for o in out)
    BR = {m: sum(o[2][m]["br"] for o in out) for m in models}
    ok = NO > 0
    summ = []
    for m in models:
        summ.append(dict(model=m, logloss=LL[m][ok].sum() / NO[ok].sum(), brier=BR[m][ok].sum() / NO[ok].sum(),
                         ece=np.mean([o[2][m]["ece"] for o in out])))
    summ = pd.DataFrame(summ).sort_values("logloss")
    # paired comparisons (per-student mean log loss per item)
    per = {m: LL[m][ok] / NO[ok] for m in models}
    cmp = []
    pairs = [("M0", "Marginal"), ("M0", "KMeans4_score_time"), ("M1_2", "M0"), ("M1_3", "M0"), ("A2", "M0"),
             ("M1_2", "A2"), ("M1_3", "A2"), ("M2_time", "M0"), ("M3_2_time", "M1_2"), ("M3_3_time", "M1_3"),
             ("M3_2_time", "M2_time"), ("M0", [k for k in models if k.startswith("LCA")][0]),
             ("M2_speed", "M0"), ("M2_speed", "M2_time"), ("M3_2_speed", "M2_speed")]
    pairs += [(x, y) for x, y in [("MIRT2", "M0"), ("M1_3", "MIRT2"), ("M1_2", "MIRT2"), ("G3PL", "M0"), ("M1_2", "G3PL"), ("M1_3", "G3PL"),
                                   ("MIRT2G", "M0"), ("M1_3", "MIRT2G")]
              if x in models and y in models]
    for a_, b_ in pairs:
        d = per[b_] - per[a_]   # >0 => a_ better
        mean, lo, hi = st.paired_bootstrap_ci(d, CFG["decision"]["bootstrap_ci"] if not demo else 200, seed=9001)
        cmp.append(dict(better=a_, than=b_, gain_nats_per_item=mean, ci_lo=lo, ci_hi=hi))
    cmp = pd.DataFrame(cmp)
    summ.to_csv(RES / "tables" / f"{form}_E10_test_scores.csv", index=False)
    cmp.to_csv(RES / "tables" / f"{form}_E10_paired.csv", index=False)
    dens = pd.DataFrame([dict(seed=o[0], **o[3]) for o in out])
    dens.to_csv(RES / "tables" / f"{form}_E10_duration_density.csv", index=False)
    # calibration curve data for figure (first mask, key models)
    save_json(dict(masks=[o[1] for o in out], n_test=int(ok.sum()),
                   calibration={m: out[0][2][m]["cal"] for m in models}), f"{form}_E10_meta.json")
    log("E10\n" + summ.to_string() + "\n" + cmp.to_string() + "\n" + dens.to_string())


# ------------------------------------------------------------------ E7
def _boot_fit(bix, Y, g, seed, demo):
    rng = np.random.default_rng(seed + bix)
    idx = rng.integers(0, len(Y), len(Y))
    m = mk(2, "b", starts=3, demo=demo).fit(Y[idx], None, None, g[idx])
    pc = m.posterior(Y, None, None, g)[0]
    return pc.argmax(1), (m.b[1] - m.b[0]), m.mu, np.exp(m.logpi[0])


def stage_stability(form, demo=False):
    ck = load_ckpt(form)
    df, _ = load_form(form, demo)
    Y, logT, cens, g, _ = arrays(df)
    tr = ck["tr"]
    ref = ck["models"]["M1_2"]
    pc = ref.posterior(Y[tr], None, None, g[tr])[0]
    lab0 = pc.argmax(1)
    dev0 = ref.b[1] - ref.b[0]
    B = CFG["stability"]["B"] if not demo else 3
    out = Parallel(n_jobs=N_JOBS)(delayed(_boot_fit)(b, Y[tr], g[tr], 9001, demo) for b in range(B))
    aris, cors = [], []
    for lab, dev, mu, pi in out:
        aris.append(adjusted_rand_score(lab0, lab))
        cors.append(abs(np.corrcoef(dev0, dev)[0, 1]))
    devs = []
    for lab, dev, mu, pi in out:
        devs.append(dev if np.corrcoef(dev0, dev)[0, 1] >= 0 else -dev)  # align label orientation
    res = dict(B=B, ari_median=np.median(aris), ari_q=np.quantile(aris, [.025, .975]), dev_corr_median=np.median(cors),
               dev_corr_q=np.quantile(cors, [.025, .975]), aris=aris, cors=cors, dev_ref=dev0, devs=np.array(devs),
               mu_ref=ref.mu, props_ref=pc.mean(0), b_ref=ref.b, a_ref=ref.a)
    # multiple initialisations on full train
    lls = ck["models"]["M1_2"].start_logliks
    res["start_logliks_M1_2"] = lls
    res["start_logliks_M1_3"] = ck["models"]["M1_3"].start_logliks
    save_json(res, f"{form}_E7_stability.json")
    log("E7", {k: res[k] for k in ["ari_median", "ari_q", "dev_corr_median", "dev_corr_q"]})


# ------------------------------------------------------------------ E8
def stage_e8(form, demo=False):
    """Within-score heterogeneity: is the spread of class posteriors at a fixed score larger
    than under a continuous null with the same items?"""
    ck = load_ckpt(form)
    df, _ = load_form(form, demo)
    Y, logT, cens, g, _ = arrays(df)
    tr = ck["tr"]
    Ytr = Y[tr]
    m0, m1 = ck["models"]["M0"], ck["models"]["M1_2"]
    from mcq_profiles.irt import expected_probs
    _, eap, psd, _ = m0.posterior(Ytr)
    lz = st.lz_person_fit(Ytr, expected_probs(m0, eap))
    pc = m1.posterior(Ytr)[0][:, 1]
    full = ~np.isnan(Ytr).any(1)
    score = np.nansum(Ytr, 1)
    # null: simulate from M0 with same missingness, refit M1_2, same summaries
    rng = np.random.default_rng(42)
    th = rng.standard_normal(len(Ytr))
    Ys = (rng.random(Ytr.shape) < expit(m0.a[0] * (th[:, None] - m0.b[0]))).astype(float)
    Ys[np.isnan(Ytr)] = np.nan
    m1s = mk(2, "b", starts=3, demo=demo).fit(Ys)
    pcs = m1s.posterior(Ys)[0][:, 1]
    _, eaps, _, _ = m0.posterior(Ys)
    lzs = st.lz_person_fit(Ys, expected_probs(m0, eaps))
    rows = []
    for s in range(int(np.nanmax(score)) + 1):
        a_ = full & (score == s)
        b_ = (~np.isnan(Ys).any(1)) & (np.nansum(Ys, 1) == s)
        if a_.sum() >= 100 and b_.sum() >= 100:
            rows.append(dict(score=s, n=int(a_.sum()), sd_classpost=pc[a_].std(), sd_classpost_null=pcs[b_].std(),
                             frac_confident=float(((pc[a_] > .8) | (pc[a_] < .2)).mean()),
                             frac_confident_null=float(((pcs[b_] > .8) | (pcs[b_] < .2)).mean()),
                             mean_lz=lz[a_].mean(), sd_lz=lz[a_].std(), sd_lz_null=lzs[b_].std(),
                             frac_lz_lt_m2=float((lz[a_] < -1.96).mean()), frac_lz_lt_m2_null=float((lzs[b_] < -1.96).mean())))
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "tables" / f"{form}_E8_within_score.csv", index=False)
    save_json(dict(lz_mean=float(np.nanmean(lz)), lz_sd=float(np.nanstd(lz)), frac_lz_lt_m196=float(np.nanmean(lz < -1.96)),
                   frac_lz_lt_m196_null=float(np.nanmean(lzs < -1.96))), f"{form}_E8_personfit.json")
    np.save(CKPT / f"{form}_E8_arrays.npy", np.column_stack([score, pc, lz, eap, full]))
    log("E8\n" + tab.round(3).to_string())


# ------------------------------------------------------------------ E9
def stage_dif(form, demo=False):
    df, _ = load_form(form, demo)
    Y, logT, cens, g, _ = arrays(df)
    ext = pd.read_parquet("data/processed/student_external.parquet") if not demo else None
    res = {}
    oecd = df.OECD.to_numpy(float)
    d1 = st.logistic_dif(Y, oecd)
    d1.to_csv(RES / "tables" / f"{form}_E9_dif_oecd.csv", index=False)
    res["oecd"] = dict(n_flag_bh=int(d1.reject_bh.sum()), n_BC=int((d1.effect_class != "A").sum()), max_dR2=float(d1.dR2.max()))
    if ext is not None:
        m = df[["CNT", "CNTSTUID"]].merge(ext, how="left")
        sex = (m.ST004D01T == 2).astype(float).where(m.ST004D01T.notna())
        d2 = st.logistic_dif(Y, sex.to_numpy())
        d2.to_csv(RES / "tables" / f"{form}_E9_dif_gender.csv", index=False)
        res["gender"] = dict(n_flag_bh=int(d2.reject_bh.sum()), n_BC=int((d2.effect_class != "A").sum()), max_dR2=float(d2.dR2.max()))
    # class enumeration within OECD / non-OECD (response models)
    for lab, sel in [("oecd", oecd == 1), ("non_oecd", oecd == 0)]:
        n = sel.sum()
        ms, _ = fit_models(Y[sel], None, None, g[sel], ["M0", "A2", "M1_2", "M1_3"], demo, tag=f"[{lab}] ")
        b = {k: bic(m, n) for k, m in ms.items()}
        res[f"enum_{lab}"] = dict(n=int(n), bic=b, selected=min(b, key=b.get))
    save_json(res, f"{form}_E9_dif.json")
    log("E9", res)


# ------------------------------------------------------------------ H5
def stage_external(form, demo=False):
    import statsmodels.formula.api as smf
    ck = load_ckpt(form)
    df, _ = load_form(form, demo)
    Y, logT, cens, g, _ = arrays(df)
    ext = pd.read_parquet("data/processed/student_external.parquet")
    d = df[["CNT", "CNTSTUID", "CNTSCHID", "BOOKID", "final_score", "total_duration_minutes"]].merge(ext, how="left")
    m0, m1 = ck["models"]["M0"], ck["models"]["M1_2"]
    _, eap, _, _ = m0.posterior(Y, None, None, g)
    d["eap"] = eap
    d["pc2"] = m1.posterior(Y, None, None, g)[0][:, 1]
    if "M1_3" in ck["models"]:
        p3 = ck["models"]["M1_3"].posterior(Y, None, None, g)[0]
        d["pc3_1"], d["pc3_2"] = p3[:, 1], p3[:, 2]
    km = np.load(CKPT / f"{form}_kmeans_labels.npy")
    d["km"] = km
    d["logtime"] = np.log(d.total_duration_minutes)
    d = d[(d.km >= 0) & d.EFFORT1.notna() & d.logtime.notna()].copy()
    d["km"] = d.km.astype("category")
    # students without a school id form their own cluster (no silent drop)
    d["school"] = pd.factorize(d.CNTSCHID.fillna("__nosch__" + d.CNTSTUID.astype(str)))[0]
    d = d.dropna(subset=[c for c in ["eap", "final_score", "logtime", "pc2", "pc3_1", "pc3_2", "BOOKID"] if c in d]).copy()
    base = "EFFORT1 ~ eap + final_score + logtime + I(logtime**2) + C(BOOKID)"
    res = {}
    fits = {}
    for nm, f in [("base", base), ("kmeans", base + " + C(km)"), ("class2", base + " + pc2"),
                  ("class3", base + " + pc3_1 + pc3_2")]:
        fits[nm] = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": d.school})
        res[nm] = dict(r2=fits[nm].rsquared, n=int(fits[nm].nobs))
    for nm in ["kmeans", "class2", "class3"]:
        extra = [p for p in fits[nm].params.index if p not in fits["base"].params.index]
        R = np.zeros((len(extra), len(fits[nm].params)))
        for i, p in enumerate(extra):
            R[i, list(fits[nm].params.index).index(p)] = 1
        w = fits[nm].wald_test(R, scalar=True)
        res[nm].update(dR2=fits[nm].rsquared - fits["base"].rsquared, wald_F=float(w.statistic), wald_p=float(w.pvalue),
                       coefs={p: (float(fits[nm].params[p]), float(fits[nm].bse[p])) for p in extra})
    # raw (unadjusted) association of K-means profiles with effort, as the source study would read it
    res["kmeans_effort_means"] = d.groupby("km", observed=True).EFFORT1.mean().to_dict()
    res["base_coefs"] = {k: (float(v), float(fits["base"].bse[k])) for k, v in fits["base"].params.items() if not k.startswith("C(")}
    save_json(res, f"{form}_H5_external.json")
    log("H5", {k: (v.get("dR2"), v.get("wald_p")) for k, v in res.items() if isinstance(v, dict) and "r2" in v})


# ------------------------------------------------------------------ replication
def stage_replication(form, demo=False):
    for f in CFG["data"]["replications"]:
        stage_fit(f, demo)


# ------------------------------------------------------------------ E3b (exploratory, logged deviation)
def _mdim_rep(r, gen, Ytr, seed, demo):
    """Multidimensional continuous null: data from the fitted 2D 2PL; does the class rule fire?"""
    rng = np.random.default_rng(seed + r)
    n, J = Ytr.shape
    th = rng.standard_normal((n, 2))
    P = expit(th @ gen.A.T + gen.d[None, :])
    Y = (rng.random((n, J)) < P).astype(float)
    Y[np.isnan(Ytr[rng.permutation(n)])] = np.nan
    res = {}
    for nm in ["M0", "A2", "M1_2", "M1_3"]:
        sp = MODEL_SPECS[nm]
        m = mk(sp["K"], sp.get("cs"), False, False, starts=3, demo=demo).fit(Y)
        res[nm] = bic(m, n)
    from mcq_profiles.latent_classes import ContinuousIRT
    m2 = ContinuousIRT(2, max_iter=600 if not demo else 100).fit(Y)
    res["MIRT2"] = m2.bic(n)
    res["dBIC_M1_vs_1Dcont"] = min(res["M0"], res["A2"]) - min(res["M1_2"], res["M1_3"])
    res["dBIC_M1_vs_MIRT2"] = res["MIRT2"] - min(res["M1_2"], res["M1_3"])
    res["rep"] = r
    return res


def stage_e3b(form, demo=False):
    from mcq_profiles.latent_classes import ContinuousIRT
    from mcq_profiles.irt import expected_probs
    ck = load_ckpt(form)
    df, _ = load_form(form, demo)
    Y, logT, cens, g, _ = arrays(df)
    tr = ck["tr"]
    Ytr = Y[tr]
    n = tr.sum()
    comps = {}
    for nm, kw in [("MIRT2", dict(dims=2)), ("G3PL", dict(dims=1, guessing=True)), ("C2PL", dict(dims=1))]:
        t0 = time.time()
        comps[nm] = ContinuousIRT(**kw, max_iter=1500 if not demo else 100).fit(Ytr)
        log(f"E3b {nm}: ll={comps[nm].result_.loglik:.2f} k={comps[nm].result_.n_params} conv={comps[nm].result_.converged} {time.time()-t0:.0f}s")
    with open(CKPT / f"{form}_competitors.pkl", "wb") as f:
        pickle.dump(comps, f)
    B = {k: bic(m, n) for k, m in ck["models"].items() if not m.duration}
    B.update({k: m.bic(n) for k, m in comps.items()})
    # local dependence under M0
    m0 = ck["models"]["M0"]
    eap = m0.posterior(Ytr)[1]
    Q = st.yen_q3(Ytr, expected_probs(m0, eap))
    Qc = Q - np.nanmean(Q)
    meta = __import__("json").loads((Path("data/processed") / f"{form}.json").read_text()) if not demo else {"items": list(range(Ytr.shape[1]))}
    iu = np.triu_indices_from(Q, 1)
    ld = [dict(i=str(meta["items"][i]), j=str(meta["items"][j]), q3=float(Q[i, j]), q3c=float(Qc[i, j]))
          for i, j in zip(*iu) if abs(Qc[i, j]) > 0.2]
    # how the M1_2 class contrast relates to item properties
    m1 = ck["models"]["M1_2"]
    dev = m1.b[1] - m1.b[0]
    rel = dict(corr_dev_b=float(np.corrcoef(dev, m0.b[0])[0, 1]), corr_dev_a=float(np.corrcoef(dev, m0.a[0])[0, 1]),
               corr_dev_mirt_a2=float(np.corrcoef(dev, comps["MIRT2"].A[:, 1])[0, 1]),
               corr_dev_c=float(np.corrcoef(dev, comps["G3PL"].c)[0, 1]))
    # multidimensional null
    R = 20 if not demo else 2
    out = Parallel(n_jobs=N_JOBS)(delayed(_mdim_rep)(r, comps["MIRT2"], Ytr, 5001, demo) for r in range(R))
    tab = pd.DataFrame(out)
    tab.to_csv(RES / "tables" / f"{form}_E3b_mdim_null.csv", index=False)
    k = int((tab.dBIC_M1_vs_1Dcont > 0).sum())
    res = dict(bic_train=B, local_dependence_pairs=ld, class_contrast_vs_item_properties=rel,
               mirt2_loadings=comps["MIRT2"].A, g3pl_c=comps["G3PL"].c,
               mdim_null=dict(R=R, fp_M1_over_1D=k / R, fp_ci=st.clopper_pearson(k, R),
                              dBIC_M1_vs_1Dcont_median=float(tab.dBIC_M1_vs_1Dcont.median()),
                              dBIC_M1_vs_1Dcont_q95=float(tab.dBIC_M1_vs_1Dcont.quantile(.95)),
                              dBIC_M1_vs_MIRT2_median=float(tab.dBIC_M1_vs_MIRT2.median())))
    save_json(res, f"{form}_E3b_competitors.json")
    log("E3b BIC", {k: round(v) for k, v in sorted(B.items(), key=lambda x: x[1])})
    log("E3b", rel, res["mdim_null"], "LD pairs", ld)


def _guess_rep(r, gen, Ytr, seed, demo):
    """Guessing null: data from the fitted 2PL-with-asymptote (no classes)."""
    from mcq_profiles.latent_classes import ContinuousIRT
    rng = np.random.default_rng(seed + r)
    n, J = Ytr.shape
    th = rng.standard_normal(n)
    P = gen.c[None, :] + (1 - gen.c[None, :]) * expit(th[:, None] * gen.A[:, 0][None, :] + gen.d[None, :])
    Y = (rng.random((n, J)) < P).astype(float)
    Y[np.isnan(Ytr[rng.permutation(n)])] = np.nan
    res = {}
    for nm in ["M0", "A2", "M1_2", "M1_3"]:
        sp = MODEL_SPECS[nm]
        res[nm] = bic(mk(sp["K"], sp.get("cs"), False, False, starts=3, demo=demo).fit(Y), n)
    res["G3PL"] = ContinuousIRT(1, guessing=True, max_iter=600 if not demo else 100).fit(Y).bic(n)
    res["dBIC_M1_vs_1Dcont"] = min(res["M0"], res["A2"]) - min(res["M1_2"], res["M1_3"])
    res["dBIC_M1_vs_G3PL"] = res["G3PL"] - min(res["M1_2"], res["M1_3"])
    res["rep"] = r
    return res


def stage_e3c(form, demo=False):
    ck = load_ckpt(form)
    df, _ = load_form(form, demo)
    Y = arrays(df)[0]
    Ytr = Y[ck["tr"]]
    comps = pickle.load(open(CKPT / f"{form}_competitors.pkl", "rb"))
    R = 20 if not demo else 2
    out = Parallel(n_jobs=N_JOBS)(delayed(_guess_rep)(r, comps["G3PL"], Ytr, 6001, demo) for r in range(R))
    tab = pd.DataFrame(out)
    tab.to_csv(RES / "tables" / f"{form}_E3c_guess_null.csv", index=False)
    k = int((tab.dBIC_M1_vs_1Dcont > 0).sum())
    kg = int((tab.dBIC_M1_vs_G3PL > 0).sum())
    res = dict(R=R, fp_M1_over_1D=k / R, fp_ci=st.clopper_pearson(k, R),
               dBIC_M1_vs_1Dcont_median=float(tab.dBIC_M1_vs_1Dcont.median()),
               dBIC_M1_vs_1Dcont_q95=float(tab.dBIC_M1_vs_1Dcont.quantile(.95)),
               M1_beats_G3PL=kg / R, dBIC_M1_vs_G3PL_median=float(tab.dBIC_M1_vs_G3PL.median()))
    save_json(res, f"{form}_E3c_guess_null.json")
    log("E3c", res)


# ------------------------------------------------------------------ contingency tables (descriptive)
def _profile_names(df, km):
    """Name K-means profiles by centroid position relative to sample tertiles of score and time."""
    ok = km >= 0
    sq = np.quantile(df.final_score[ok], [1 / 3, 2 / 3])
    tq = np.quantile(df.total_duration_minutes[ok], [1 / 3, 2 / 3])
    lvl = lambda v, q, w: w[int(np.searchsorted(q, v))]
    names = {}
    for k in sorted(set(km[ok])):
        s = df.final_score[km == k].mean()
        t = df.total_duration_minutes[km == k].mean()
        names[k] = (f"P{k + 1}: {lvl(s, sq, ['low', 'medium', 'high'])} score, "
                    f"{lvl(t, tq, ['short', 'medium', 'long'])} time ({s:.1f} pts, {t:.0f} min)")
    return names


def _ctab(a, b, la, lb):
    t = pd.crosstab(pd.Series(a, name=la), pd.Series(b, name=lb))
    chi2, p, dof, _ = stats.chi2_contingency(t)
    return t, dict(chi2=float(chi2), dof=int(dof), p=float(p), cramers_v=float(st.cramers_v(a, b)), n=int(t.to_numpy().sum()))


def stage_contingency(form, demo=False):
    ck = load_ckpt(form)
    df, _ = load_form(form, demo)
    Y, logT, cens, g, _ = arrays(df)
    km = np.load(CKPT / f"{form}_kmeans_labels.npy")
    names = _profile_names(df, km)
    M = ck["models"]
    cls2 = M["M1_2"].posterior(Y, None, None, g)[0].argmax(1) + 1
    cls3 = M["M1_3"].posterior(Y, None, None, g)[0].argmax(1) + 1
    ok = km >= 0
    prof = np.array([names.get(k, "") for k in km])
    band = pd.qcut(df.final_score, 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"], duplicates="drop").astype(str).to_numpy()
    oecd = np.where(df.OECD.to_numpy() == 1, "OECD", "non-OECD")
    book = df.BOOKID.astype(int).astype(str).to_numpy()
    pairs = {
        "profile_x_M1_3": (prof[ok], cls3[ok], "K-means profile", "M1_3 class"),
        "profile_x_M1_2": (prof[ok], cls2[ok], "K-means profile", "M1_2 class"),
        "M1_3_x_scoreband": (cls3, band, "M1_3 class", "score quartile"),
        "profile_x_oecd": (prof[ok], oecd[ok], "K-means profile", "OECD"),
        "profile_x_booklet": (prof[ok], book[ok], "K-means profile", "booklet"),
        "M1_3_x_oecd": (cls3, oecd, "M1_3 class", "OECD"),
    }
    out, tex = {}, []
    for key, (a, b, la, lb) in pairs.items():
        t, s = _ctab(a, b, la, lb)
        t.to_csv(RES / "tables" / f"{form}_ct_{key}.csv")
        (t.div(t.sum(1), axis=0) * 100).round(1).to_csv(RES / "tables" / f"{form}_ct_{key}_rowpct.csv")
        out[key] = dict(stats=s, counts=t.to_dict())
        cells = t.astype(int).astype(str) + " (" + (t.div(t.sum(1), axis=0) * 100).round(1).astype(str) + r"\%)"
        cells.columns = [str(c) for c in cells.columns]
        cap = (rf"{la} $\times$ {lb}, Form~A; counts (row \%). $\chi^2({s['dof']})={s['chi2']:.1f}$, "
               rf"Cram\'er's $V={s['cramers_v']:.2f}$, $n={s['n']}$.")
        tex.append(cells.to_latex(caption=cap.replace("_", r"\_"), label="tab:ct_" + key.replace("_", ""), escape=False)
                   .replace("M1_", r"M1\_")
                   .replace(r"\begin{tabular}", r"\resizebox{\textwidth}{!}{\begin{tabular}")
                   .replace(r"\end{tabular}", r"\end{tabular}}"))
        log("contingency", key, s)
    # descriptive mean ability / time per M1_3 class
    eap = M["M0"].posterior(Y, None, None, g)[1]
    desc = pd.DataFrame(dict(cls=cls3, eap=eap, score=df.final_score, minutes=df.total_duration_minutes)).groupby("cls").agg(
        n=("eap", "size"), eap=("eap", "mean"), score=("score", "mean"), minutes=("minutes", "median"))
    desc.to_csv(RES / "tables" / f"{form}_M1_3_class_profile.csv")
    out["M1_3_class_profile"] = desc.reset_index().to_dict(orient="records")
    out["profile_names"] = {int(k): v for k, v in names.items()}
    save_json(out, f"{form}_contingency.json")
    (RES.parent / "manuscript" / "tables" / f"{form}_contingency.tex").write_text("\n\n".join(tex), encoding="utf-8")


# ------------------------------------------------------------------ continuous competitors on replication forms (exploratory, deviation log)
def _fit_comp(nm, kw, Ytr, demo):
    from mcq_profiles.latent_classes import ContinuousIRT
    t0 = time.time()
    m = ContinuousIRT(**kw, max_iter=1500 if not demo else 100).fit(Ytr)
    return nm, m, time.time() - t0


def stage_competitors(form, demo=False):
    ck = load_ckpt(form)
    df, _ = load_form(form, demo)
    Y, *_ = arrays(df)
    tr = ck["tr"]
    Ytr = Y[tr]
    n = tr.sum()
    specs = [("MIRT2", dict(dims=2)), ("G3PL", dict(dims=1, guessing=True)), ("MIRT2G", dict(dims=2, guessing=True))]
    out = Parallel(n_jobs=3)(delayed(_fit_comp)(nm, kw, Ytr, demo) for nm, kw in specs)
    comps = {nm: m for nm, m, _ in out}
    for nm, m, s in out:
        log(f"{form} {nm}: ll={m.result_.loglik:.2f} k={m.result_.n_params} conv={m.result_.converged} {s:.0f}s")
    with open(CKPT / f"{form}_competitors.pkl", "wb") as f:
        pickle.dump(comps, f)
    B = {k: float(bic(m, n)) for k, m in ck["models"].items() if not m.duration}
    B.update({k: float(m.bic(n)) for k, m in comps.items()})
    best = min(B, key=B.get)
    res = dict(n_train=int(n), bic_train=B, best=best,
               dBIC_bestM1_vs_A=min(B["A2"], B["A3"], B["M0"]) - min(B["M1_2"], B["M1_3"]),
               dBIC_bestM1_vs_G3PL=B["G3PL"] - min(B["M1_2"], B["M1_3"]),
               dBIC_bestM1_vs_bestcontinuous=min(B[k] for k in ["M0", "A2", "A3", "MIRT2", "G3PL", "MIRT2G"]) - min(B["M1_2"], B["M1_3"]))
    save_json(res, f"{form}_competitors.json")
    log(form, "competitors", res)

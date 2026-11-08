"""Run the pre-registered experiments (see reports/experimental_protocol.md).

Usage:  python scripts/run_experiments.py --stage fit|baseline|null|positive|e10|stability|e8|dif|external|replication|all
        [--form math_formA] [--demo]   (demo = synthetic data, tiny settings; for CI/Colab smoke tests)
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mcq_profiles import baseline as bl  # noqa: E402
from mcq_profiles import statistics as st  # noqa: E402
from mcq_profiles.data import item_matrix, rt_matrix  # noqa: E402
from mcq_profiles.evaluation import logloss_matrix, brier_matrix, calibration_table, predict_masked, selection_summary  # noqa: E402
from mcq_profiles.latent_classes import MixIRT, LCA  # noqa: E402
from mcq_profiles.preprocessing import train_test_split_stratified, item_mask, log_duration, unmasked_duration, group_codes  # noqa: E402
from mcq_profiles.simulation import simulate, centred_shift  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())
RES = ROOT / "results"
CKPT = RES / "checkpoints"
for d in [RES / "metrics", RES / "tables", RES / "figures", CKPT]:
    d.mkdir(parents=True, exist_ok=True)
N_JOBS = 14


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save_json(obj, name):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    (RES / "metrics" / name).write_text(json.dumps(obj, indent=2, default=conv))


# ---------------------------------------------------------------------- data
def load_form(name, demo=False):
    if demo:
        rng = np.random.default_rng(0)
        J = 12
        b = np.linspace(-1.5, 1.5, J)
        d = simulate(600, rng.uniform(0.8, 1.6, J), b, rng=1, dur=dict(alpha=np.log(20), beta=0.2, s=0.4),
                     group_props=[0.5, 0.5])
        df = pd.DataFrame(d["Y"], columns=[f"I{j + 1:02d}" for j in range(J)])
        df["total_duration_minutes"] = np.exp(d["logT"])
        df["final_score"] = np.nansum(d["Y"], 1)
        df["CNT"] = np.where(d["group"] == 0, "AAA", "BBB")
        df["BOOKID"] = d["group"] + 1
        df["OECD"] = d["group"]
        df["CNTSCHID"] = np.arange(len(df)) // 20
        df["CNTSTUID"] = np.arange(len(df))
        for j in range(J):
            df[f"RT{j + 1:02d}"] = np.exp(d["logT"]) * 60 / J
        df["n_not_reached"] = 0
        return df, dict(synthetic=True, fingerprint="synthetic-demo")
    df = pd.read_parquet(ROOT / "data/processed" / f"{name}.parquet")
    meta = json.loads((ROOT / "data/processed" / f"{name}.json").read_text())
    return df, meta


def arrays(df, not_reached=None):
    Y = item_matrix(df)
    logT, cens = log_duration(df.total_duration_minutes.to_numpy())
    g, books = group_codes(df.BOOKID)
    return Y, logT, cens, g, books


def mk(K, cs=None, duration=False, group_effects=False, starts=None, demo=False):
    m = CFG["models"]
    return MixIRT(K, "2pl", class_specific=cs, duration=duration, group_effects=group_effects,
                  n_nodes=m["n_nodes"] if not demo else 21, ridge=m["ridge"],
                  n_starts=starts or (m["n_starts"] if not demo else 2),
                  max_iter=m["max_iter"] if not demo else 200, tol=m["tol"] if not demo else 1e-6, random_state=K * 100 + 7)


MODEL_SPECS = {
    "M0": dict(K=1), "A2": dict(K=2), "A3": dict(K=3),
    "M1_2": dict(K=2, cs="b"), "M1_3": dict(K=3, cs="b"),
    "M2": dict(K=1, duration=True, group_effects=True),
    "M3_2": dict(K=2, cs="b", duration=True, group_effects=False),
    "M3_3": dict(K=3, cs="b", duration=True, group_effects=False),
    "M4_2": dict(K=2, cs="b", duration=True, group_effects=True),
    "M4_3": dict(K=3, cs="b", duration=True, group_effects=True),
}


def _fit_one(name, spec, Y, logT, cens, g, demo):
    t = time.time()
    m = mk(spec["K"], spec.get("cs"), spec.get("duration", False), spec.get("group_effects", False), demo=demo)
    dur = spec.get("duration", False)
    m.fit(Y, logT if dur else None, cens if dur else None, g)
    return name, m, time.time() - t


def fit_models(Y, logT, cens, g, names, demo=False, tag=""):
    out = Parallel(n_jobs=min(N_JOBS, len(names)))(delayed(_fit_one)(n, MODEL_SPECS[n], Y, logT, cens, g, demo) for n in names)
    models, secs = {}, {}
    for n, m, s in out:
        models[n], secs[n] = m, s
        log(f"{tag}{n}: ll={m.result_.loglik:.2f} k={m.result_.n_params} it={m.result_.n_iter} conv={m.result_.converged} "
            f"starts={np.round(m.start_logliks, 1).tolist()} {s:.0f}s")
    return models, secs


# ------------------------------------------------------------------- stages
def stage_fit(form, demo=False):
    """E2-E4: model ladder on the training split."""
    df, meta = load_form(form, demo)
    tr, te = train_test_split_stratified(df, CFG["split"]["seed"], CFG["split"]["test_fraction"])
    Y, logT, cens, g, books = arrays(df)
    Ytr = Y[tr]
    names = list(MODEL_SPECS)
    models, secs = fit_models(Ytr, logT[tr], cens[tr], g[tr], names, demo)
    lcas = {}
    for K in CFG["models"]["K_lca"]:
        lcas[f"LCA_{K}"] = LCA(K, n_starts=10 if not demo else 3, random_state=K).fit(Ytr)
    n = tr.sum()
    rows = selection_summary(models, n) + selection_summary({k: _LCAWrap(v) for k, v in lcas.items()}, n)
    for r in rows:
        r["family"] = "duration" if r["model"][:2] in ("M2", "M3", "M4") else "responses"
        r["secs"] = secs.get(r["model"], np.nan)
    tab = pd.DataFrame(rows)
    # response-only comparable likelihood for duration models (marginal of responses)
    for nme in ["M2", "M3_2", "M3_3", "M4_2", "M4_3"]:
        m = models[nme]
        m.duration = False
        tab.loc[tab.model == nme, "loglik_responses_only"] = m.loglik(Ytr, None, None, g[tr])
        m.duration = True
    # class diagnostics
    diag = {}
    for nme, m in models.items():
        if m.K > 1:
            dur = m.duration
            pc = m.posterior(Ytr, logT[tr] if dur else None, cens[tr] if dur else None, g[tr])[0]
            diag[nme] = dict(props=pc.mean(0), min_prop=float(pc.mean(0).min()), entropy=float(m.entropy_r2(pc)),
                             mu=m.mu, sigma=m.sigma, b=m.b, a=m.a,
                             alpha=getattr(m, "alpha", None), beta=getattr(m, "beta", None), s=np.exp(m.logs) if dur else None,
                             gamma=getattr(m, "gamma", None) if dur else None,
                             n_starts_at_best=int(np.sum(np.abs(np.array(m.start_logliks) - max(m.start_logliks)) < 0.5)))
        else:
            diag[nme] = dict(a=m.a, b=m.b, mu=m.mu, sigma=m.sigma,
                             alpha=getattr(m, "alpha", None) if m.duration else None,
                             beta=getattr(m, "beta", None) if m.duration else None,
                             s=np.exp(m.logs) if m.duration else None, gamma=m.gamma if m.duration else None)
    for k, v in lcas.items():
        r, _ = v.posterior(Ytr)
        diag[k] = dict(props=r.mean(0), entropy=float(1 + (r * np.log(np.clip(r, 1e-300, 1))).sum() / (len(r) * np.log(v.K))) if v.K > 1 else None)
    tab.to_csv(RES / "tables" / f"{form}_model_selection_train.csv", index=False)
    save_json(dict(form=form, fingerprint=meta.get("fingerprint"), n_train=int(n), n_test=int(te.sum()),
                   books=[float(b) for b in books], diagnostics=diag), f"{form}_fit_diagnostics.json")
    with open(CKPT / f"{form}_train_models.pkl", "wb") as f:
        pickle.dump(dict(models=models, lcas=lcas, tr=tr, te=te), f)
    log("\n" + tab.sort_values("BIC").to_string())
    return tab


class _LCAWrap:
    def __init__(self, l):
        self.loglik_, self.n_params_ = l.loglik_, l.n_params_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--form", default=CFG["data"]["primary"])
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        a.form = "demo"  # synthetic outputs never share names with real-data outputs
        CFG["data"]["replications"] = ["demo"]
    import importlib
    stages = importlib.import_module("experiment_stages") if a.stage not in ("fit",) else None
    if a.stage == "fit":
        stage_fit(a.form, a.demo)
    else:
        getattr(stages, f"stage_{a.stage}")(a.form, a.demo)


if __name__ == "__main__":
    main()

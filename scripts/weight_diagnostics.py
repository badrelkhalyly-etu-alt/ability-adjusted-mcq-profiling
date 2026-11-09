"""Sampling-weight diagnostics for the robustness stage: Kish effective n, country weight shares, ESS-rescaled BIC.
Writes results/metrics/weight_diagnostics.json. Run after `run_experiments.py --stage robustness`."""
import os, pathlib; os.chdir(pathlib.Path(__file__).resolve().parents[1])
import sys, json, numpy as np, pandas as pd
sys.path[:0] = ["scripts", "src"]
from experiment_stages import load_ckpt, load_form, RES
ext = pd.read_parquet(RES.parent / "data" / "processed" / "student_external.parquet")
out = {}
for f in "ABC":
    form = f"math_form{f}"
    ck = load_ckpt(form); df, meta = load_form(form, False); tr = ck["tr"]
    m = df[["CNT", "CNTSTUID"]].merge(ext[["CNT", "CNTSTUID", "W_FSTUWT"]], how="left")[tr]
    w = m.W_FSTUWT.to_numpy(float); n = len(w)
    ess = w.sum()**2 / (w**2).sum()
    sh = m.groupby("CNT").W_FSTUWT.sum().sort_values(ascending=False) / w.sum()
    cnt_n = m.CNT.value_counts()
    top = [(c, round(float(sh[c]), 3), int(cnt_n[c])) for c in sh.index[:3]]
    rob = pd.read_csv(RES / "tables" / f"{form}_robustness.csv")
    W = rob[rob.variant == "weighted"].set_index("model")
    r = ess / n
    def bic(mod): return r * (-2 * W.loc[mod, "loglik"]) + W.loc[mod, "k"] * np.log(ess)
    best_m1 = min(["M1_2", "M1_3"], key=bic)
    out[form] = dict(n=n, n_countries=int(m.CNT.nunique()), kish_ess=round(float(ess), 1),
                     ess_ratio=round(float(r), 3), max_over_median_w=round(float(w.max() / np.median(w)), 1),
                     top3_country_weight_share=top,
                     ess_scaled_BIC={k: round(float(bic(k)), 1) for k in W.index},
                     ess_scaled_dBIC_G3PL_minus_bestM1=round(float(bic("G3PL") - bic(best_m1)), 1),
                     ess_scaled_dBIC_MIRT2G_minus_bestM1=round(float(bic("MIRT2G") - bic(best_m1)), 1))
print(json.dumps(out, indent=1))
json.dump(out, open(RES / "metrics" / "weight_diagnostics.json", "w"), indent=1)

"""Descriptive table of the three linear mathematics forms (manuscript Table tab:data).

Reads data/processed/math_form{A,B,C}.parquet (not distributed) and writes
results/tables/data_descriptives.csv and manuscript/tables/tab_data.tex.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rows = []
for f in "ABC":
    p = ROOT / f"data/processed/math_form{f}.parquet"
    if not p.exists():
        continue
    d = pd.read_parquet(p)
    meta = __import__("json").loads(p.with_suffix(".json").read_text())
    items = [c for c in d.columns if c.startswith("I") and c[1:].isdigit()]
    Y = d[items].to_numpy(dtype=float, copy=True)
    t = d["total_duration_minutes"].to_numpy(dtype=float, copy=True)
    no_time = int(np.isnan(t).sum())
    zero_time = int((t <= 0).sum())
    tt = t[t > 0]
    rows.append(dict(
        form=f, n=len(d), items=len(items), polytomous_items=len(meta.get("polytomous_items", [])), countries=d.CNT.nunique(), schools=d.CNTSCHID.nunique(),
        oecd_share=float(d.OECD.mean()), booklets=d.BOOKID.nunique(),
        mean_pcorrect=float(np.nanmean(Y)), score_mean=float(np.nansum(Y, 1).mean()),
        pct_missing_resp=float(np.isnan(Y).mean() * 100), pct_any_not_reached=float((d.n_not_reached > 0).mean() * 100),
        time_median=float(np.median(tt)), time_q1=float(np.quantile(tt, .25)), time_q3=float(np.quantile(tt, .75)),
        no_time=no_time, zero_time=zero_time))
tab = pd.DataFrame(rows)
tab.to_csv(ROOT / "results/tables/data_descriptives.csv", index=False)

lab = [("Students", "n", "{:,.0f}"), ("Items", "items", "{:.0f}"), ("of which polytomous", "polytomous_items", "{:.0f}"), ("Countries/economies", "countries", "{:.0f}"),
       ("Schools", "schools", "{:,.0f}"), ("Booklets", "booklets", "{:.0f}"), ("OECD share", "oecd_share", "{:.2f}"),
       ("Mean proportion correct", "mean_pcorrect", "{:.3f}"), ("Mean score", "score_mean", "{:.2f}"),
       ("Missing responses (\\%)", "pct_missing_resp", "{:.2f}"),
       ("Students with $\\ge$1 not-reached item (\\%)", "pct_any_not_reached", "{:.1f}"),
       ("Total time, median (IQR), min", None, None),
       ("No timing recorded", "no_time", "{:.0f}"), ("Total time $=0$", "zero_time", "{:.0f}")]
cols = list(tab.form)
L = ["\\begin{table}[t]\\centering\\small",
     "\\caption{Linear-path PISA 2025 mathematics forms. Form~A is confirmatory; B and C are replications. "
     "Polytomous items scored 1 for full credit; not-reached and technical codes treated as missing.}\\label{tab:data}",
     "\\begin{tabular}{l" + "r" * len(cols) + "}\\toprule",
     " & " + " & ".join(f"Form {c}" for c in cols) + "\\\\\\midrule"]
for name, key, fmt in lab:
    if key is None:
        vals = [f"{r.time_median:.1f} ({r.time_q1:.1f}--{r.time_q3:.1f})" for r in tab.itertuples()]
    else:
        vals = [fmt.format(v).replace(",", "{,}") for v in tab[key]]
    L.append(name + " & " + " & ".join(vals) + "\\\\")
L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
(ROOT / "manuscript/tables/tab_data.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
print(tab.T)

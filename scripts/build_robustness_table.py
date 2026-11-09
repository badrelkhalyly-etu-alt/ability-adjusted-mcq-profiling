"""Summarise the robustness stage (not-reached scoring, sampling weights, cubic duration model) for the three forms.

Writes results/tables/robustness_summary.csv and manuscript/tables/tab_robustness.tex.
Sign convention: positive = the continuous (one-class) model has the lower BIC."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T, M = ROOT / "results" / "tables", ROOT / "results" / "metrics"
FORMS = ["math_formA", "math_formB", "math_formC"]
CLASS = ["M1_2", "M1_3"]
CONT = ["M0", "A2", "G3PL", "MIRT2", "MIRT2G"]
wd = json.loads((M / "weight_diagnostics.json").read_text())

rows = {}
for f in FORMS:
    tab = pd.read_csv(T / f"{f}_robustness.csv")
    rb = json.loads((M / f"{f}_robustness.json").read_text())
    d = rb["duration"]
    out = {}
    for v, lab in [("reference", "resp_reference"), ("nr_incorrect", "resp_nr_incorrect"), ("weighted", "resp_weighted_nominal")]:
        b = tab[tab.variant == v].set_index("model").BIC
        out[lab] = b[CLASS].min() - b[CONT].min()
    e = wd[f]["ess_scaled_BIC"]
    out["resp_weighted_ess"] = min(e[m] for m in CLASS) - min(e[m] for m in CONT)
    out["dur_M3_2_vs_M2_linear"] = -d["dBIC_M2linear_minus_M3_2linear"]
    out["dur_M3_2_vs_M2_cubic"] = -d["dBIC_M2poly_minus_M3_2poly"]
    out["dur_M3_3_vs_M2_cubic"] = -d["dBIC_M2poly_minus_M3_3poly"]
    out["dur_M2_cubic_vs_linear"] = -d["dBIC_M2linear_minus_M2poly"]
    out["dev_corr_nr_incorrect"] = rb["nr_incorrect_M1_3_dev_corr_with_reference"]
    out["dev_corr_weighted"] = rb["weighted_M1_3_dev_corr_with_reference"]
    out["min_class_prop_weighted"] = min(rb["weighted_M1_3_class_props"])
    out["kish_ess"] = wd[f]["kish_ess"]
    rows[f] = out
summ = pd.DataFrame(rows)
summ.index.name = "quantity"
summ.to_csv(T / "robustness_summary.csv", float_format="%.4f")

def i(x): return f"{x:,.0f}".replace(",", "{,}").replace("-", "$-$")
def r2(x): return f"{x:.2f}".replace("-", "$-$")
spec = [
    (r"\multicolumn{4}{l}{\emph{Responses: best class minus best continuous model}}", None, None),
    ("Reference (not reached = missing)", "resp_reference", i),
    ("Not reached scored incorrect", "resp_nr_incorrect", i),
    ("Weighted, nominal $n$", "resp_weighted_nominal", i),
    ("Weighted, Kish effective $n$", "resp_weighted_ess", i),
    (r"\multicolumn{4}{l}{\emph{Duration: model minus one-class M2}}", None, None),
    ("M3$_2$ vs M2, linear time model", "dur_M3_2_vs_M2_linear", i),
    ("M3$_2$ vs M2, cubic heteroscedastic", "dur_M3_2_vs_M2_cubic", i),
    ("M3$_3$ vs M2, cubic heteroscedastic", "dur_M3_3_vs_M2_cubic", i),
    ("M2 cubic minus M2 linear", "dur_M2_cubic_vs_linear", i),
    (r"\multicolumn{4}{l}{\emph{M1$_3$ deviation profile: $r$ with reference}}", None, None),
    ("Not reached scored incorrect", "dev_corr_nr_incorrect", r2),
    ("Weighted", "dev_corr_weighted", r2),
    ("Smallest weighted M1$_3$ class proportion", "min_class_prop_weighted", r2),
    ("Kish effective sample size", "kish_ess", i),
]
EOL = "\\" * 2
L = [r"\begin{table}[t]\centering\small",
     r"\caption{Sensitivity analyses on the training splits (exploratory; deviation log). $\Delta$BIC rows: negative values favour the class model (last duration row: cubic over linear one-class M2). "
     r"Weighted fits maximise a pseudo-likelihood with final student weights rescaled to mean 1; the Kish row rescales them to the effective sample size.}\label{tab:robustness}",
     r"\begin{tabular}{lrrr}\toprule", "Quantity & Form A & Form B & Form C" + EOL + r"\midrule"]
for lab, key, fmt in spec:
    if key is None:
        L.append(lab + EOL)
    else:
        L.append(lab + " & " + " & ".join(fmt(rows[f][key]) for f in FORMS) + EOL)
L += [r"\bottomrule\end{tabular}", r"\end{table}"]
(ROOT / "manuscript" / "tables" / "tab_robustness.tex").write_text("\n".join(L) + "\n", encoding="utf8")
print(summ.round(3).to_string())

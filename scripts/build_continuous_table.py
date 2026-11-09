"""Summarise stage_continuous_map for the three forms.

Writes results/tables/continuous_map_summary.csv and manuscript/tables/tab_continuous.tex."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M, T = ROOT / "results" / "metrics", ROOT / "results" / "tables"
FORMS = ["math_formA", "math_formB", "math_formC"]
rows = {}
for f in FORMS:
    m = json.loads((M / f"{f}_continuous_map.json").read_text(encoding="utf8"))
    la, ef = m["label_agreement"], m["effort_cv"]
    rows[f] = dict(n=m["n"], agree_mean=la["mean"], share_below_0_8=la["share_below_0_8"], kappa=la["kappa_mean"],
                   eap_reliability=m["eap_reliability"], effort_n=ef["n"], r2_profiles=ef["r2"]["profiles"],
                   r2_continuous=ef["r2"]["continuous"], r2_both=ef["r2"]["profiles_plus_continuous"],
                   mse_gain=ef["mse_gain_continuous_vs_profiles"][0], mse_gain_lo=ef["mse_gain_continuous_vs_profiles"][1],
                   mse_gain_hi=ef["mse_gain_continuous_vs_profiles"][2])
summ = pd.DataFrame(rows)
summ.index.name = "quantity"
summ.to_csv(T / "continuous_map_summary.csv", float_format="%.6f")

EOL = "\\" * 2
def n0(x): return f"{x:,.0f}".replace(",", "{,}")
def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"
def pct(x): return f"{100 * x:.0f}\\%"
def ci(f):
    r = rows[f]
    return f"{r['mse_gain']:.3f} [{r['mse_gain_lo']:.3f}, {r['mse_gain_hi']:.3f}]".replace("-", "$-$")
spec = [
    (r"\multicolumn{4}{l}{\emph{Reliability of the individual description}}", None, None),
    ("Training students", "n", n0),
    ("Mean P(same profile in a parallel test)", "agree_mean", f2),
    ("Students with P $<0.8$", "share_below_0_8", pct),
    (r"Cohen's $\kappa$, observed vs replicated profile", "kappa", f2),
    (r"Reliability of continuous ability (EAP)", "eap_reliability", f2),
    (r"\multicolumn{4}{l}{\emph{Self-reported effort, 5-fold cross-validated $R^2$}}", None, None),
    ("Students with effort", "effort_n", n0),
    ("Four profiles", "r2_profiles", f3),
    ("Continuous ability and time", "r2_continuous", f3),
    ("Both", "r2_both", f3),
]
L = [r"\begin{table}[t]\centering\small",
     r"\caption{Four profiles against the continuous quantities they summarise (training splits; exploratory, deviation log). "
     r"Parallel-test scores were drawn 200 times per student from the 2PL posterior with time held fixed. "
     r"Last row: mean gain in squared error from using continuous ability and time instead of the profiles, with a 95\% bootstrap interval.}\label{tab:continuous}",
     r"\begin{tabular}{lrrr}\toprule", "Quantity & Form A & Form B & Form C" + EOL + r"\midrule"]
for lab, key, fmt in spec:
    L.append(lab + EOL if key is None else lab + " & " + " & ".join(fmt(rows[f][key]) for f in FORMS) + EOL)
L.append("Squared-error gain, continuous vs profiles & " + " & ".join(ci(f) for f in FORMS) + EOL)
L += [r"\bottomrule\end{tabular}", r"\end{table}"]
(ROOT / "manuscript" / "tables" / "tab_continuous.tex").write_text("\n".join(L) + "\n", encoding="utf8")
print(summ.round(3).to_string())

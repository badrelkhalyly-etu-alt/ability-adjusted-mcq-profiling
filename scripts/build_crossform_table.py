"""Cross-form BIC comparison of registered and exploratory models (training splits of Forms A, B, C).

Writes results/tables/crossform_bic.csv, results/metrics/crossform_bic.json and manuscript/tables/tab_crossform.tex.
"""
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CK = ROOT / "results/checkpoints"
MODELS = ["M0", "A2", "A3", "M1_2", "M1_3", "MIRT2", "G3PL", "MIRT2G"]
rows = {}
for f in "ABC":
    form = f"math_form{f}"
    ck = pickle.load(open(CK / f"{form}_train_models.pkl", "rb"))
    comps = pickle.load(open(CK / f"{form}_competitors.pkl", "rb"))
    n = int(ck["tr"].sum())
    allm = {**ck["models"], **comps}
    rows[f"Form {f}"] = {m: float(allm[m].bic(n)) for m in MODELS if m in allm}
    sel = pd.read_csv(ROOT / f"results/tables/{form}_model_selection_train.csv")
    lca = sel[sel.model.str.startswith("LCA")].sort_values("BIC").iloc[0]
    rows[f"Form {f}"][f"LCA (best K)"] = float(lca.BIC)
    rows[f"Form {f}"]["n_train"] = n
t = pd.DataFrame(rows)
t.to_csv(ROOT / "results/tables/crossform_bic.csv")
(ROOT / "results/metrics/crossform_bic.json").write_text(json.dumps(rows, indent=2), encoding="utf8")

n_row = t.loc["n_train"]
b = t.drop(index="n_train")
d = b - b.min()
kind = {"M0": "continuous", "A2": "continuous", "A3": "continuous", "M1_2": "class (type B)", "M1_3": "class (type B)",
        "MIRT2": "continuous (expl.)", "G3PL": "continuous (expl.)", "MIRT2G": "continuous (expl.)", "LCA (best K)": "class (LCA)"}
L = [r"\begin{table}[t]\centering\small",
     r"\caption{BIC relative to the best model on each form (training splits; 0 = best). "
     r"Exploratory continuous competitors were added after the protocol was frozen (deviation log).}\label{tab:crossform}",
     r"\begin{tabular}{llrrr}\toprule",
     r"Model & Type & Form A & Form B & Form C\\",
     r" & & $n={:,}$ & $n={:,}$ & $n={:,}$\\\midrule".format(*[int(x) for x in n_row]).replace(",", "{,}")]
for m in d.index:
    vals = " & ".join(("\\textbf{0}" if v == 0 else f"{v:.0f}") for v in d.loc[m])
    L.append(f"{m.replace('_', chr(92) + '_')} & {kind.get(m, '')} & {vals}\\\\")
L += [r"\bottomrule\end{tabular}", r"\end{table}"]
(ROOT / "manuscript/tables/tab_crossform.tex").parent.mkdir(parents=True, exist_ok=True)   # manuscript/ is not in the repository
(ROOT / "manuscript/tables/tab_crossform.tex").write_text("\n".join(L) + "\n", encoding="utf8")
print(d.round(0))

"""Build publication figures, LaTeX tables and the reproducibility report from saved outputs.

Every number drawn here is read from results/metrics or results/tables; nothing is typed in.
Usage: python scripts/build_results.py [--form math_formA] [--demo]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
RES = ROOT / "results"
FIG = RES / "figures"
TAB = RES / "tables"
MAN = ROOT / "manuscript"

# validated reference categorical palette (dataviz skill, light mode); slots used in fixed order
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "lines.linewidth": 2, "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
                     "font.family": "DejaVu Sans"})


def J(name):
    p = RES / "metrics" / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name, caption, captions):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    captions[name] = caption


def fig_e1(form, cap):
    e1 = J(f"{form}_E1_baseline.json")
    if not e1:
        return
    from run_experiments import load_form
    df, _ = load_form(form, form == "demo")
    lab = np.load(RES / "checkpoints" / f"{form}_kmeans_labels.npy")
    ok = lab >= 0
    x, y = df.total_duration_minutes.to_numpy()[ok], df.final_score.to_numpy()[ok]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
    hb = ax[0].hexbin(x, y, gridsize=35, cmap="Greys", mincnt=1, linewidths=0)
    for p in e1["k4_profiles"]:
        ax[0].scatter(p["minutes"], p["score"], s=60, color=C1, edgecolor="white", linewidth=1.5, zorder=3)
        ax[0].annotate(f"n={p['n']}", (p["minutes"], p["score"]), xytext=(5, 5), textcoords="offset points", fontsize=7, color=INK)
    ax[0].set(xlabel="Total time (min)", ylabel="Score (items correct)", title="(a) Score–time density, K-means centroids (k=4)")
    k = pd.DataFrame(e1["k_curve"])
    ax[1].plot(k.k, k.silhouette, marker="o", color=C1, label="observed")
    ax[1].axhline(e1["null_silhouette"]["q95"], color=C2, ls="--", lw=1.5, label="no-cluster null, 95th pct (k=4)")
    ax[1].axhspan(0.25, 0.5, color=GRID, alpha=0.5, lw=0)
    ax[1].set(xlabel="k", ylabel="Silhouette", title="(b) Silhouette vs k; band = 'weak structure'")
    ax[1].legend(fontsize=7)
    v = np.array(e1["null_silhouette"]["values"])
    ax[2].hist(v, bins=20, color=INK2, alpha=0.6)
    ax[2].axvline(e1["k4"]["silhouette"], color=C1, lw=2)
    ax[2].annotate(f"observed {e1['k4']['silhouette']:.3f}", (e1["k4"]["silhouette"], ax[2].get_ylim()[1] * 0.9), xytext=(4, 0),
                   textcoords="offset points", fontsize=7)
    ax[2].set(xlabel="Silhouette (k=4)", ylabel="Null replications", title="(c) Silhouette under no clusters")
    save(fig, "fig01_score_time_clusters",
         f"Score–time K-means pipeline on PISA 2025 Form A (n = {e1['n_used']}). (a) Score–time density with the four "
         f"K-means centroids. (b) Silhouette by k; the shaded band is the conventional 'weak structure' range. "
         f"(c) Silhouette of K-means (k = 4) on {e1['null_silhouette']['R']} datasets with the same marginals and "
         f"rank correlation but no clusters (Gaussian copula), against the observed value.", cap)


def fig_items(form, cap, ck):
    if ck is None:
        return
    m0 = ck["models"]["M0"]
    a, b = m0.a[0], m0.b[0]
    from run_experiments import load_form, arrays
    df, _ = load_form(form, form == "demo")
    Y = arrays(df)[0]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
    o = np.argsort(b)
    ax[0].scatter(b, a, color=C1, s=25, edgecolor="white", linewidth=1)
    ax[0].set(xlabel="Difficulty b", ylabel="Discrimination a", title="(a) 2PL item parameters (train)")
    R = pd.DataFrame(Y).corr(min_periods=100).to_numpy(copy=True)
    np.fill_diagonal(R, np.nan)
    im = ax[1].imshow(R, cmap="Blues", vmin=0, vmax=np.nanmax(R))
    fig.colorbar(im, ax=ax[1], fraction=0.046, label="φ correlation")
    ax[1].set(title="(b) Inter-item correlations", xlabel="Item", ylabel="Item")
    ax[1].grid(False)
    th = np.linspace(-3, 3, 200)
    from mcq_profiles.irt import icc
    P = icc(a, b, th)
    for j in range(P.shape[1]):
        ax[2].plot(th, P[:, j], color=INK2, lw=0.8, alpha=0.6)
    ax[2].plot(th, P.mean(1), color=C1, lw=2, label="test characteristic / J")
    ax[2].set(xlabel="θ", ylabel="P(correct)", title="(c) Item characteristic curves")
    ax[2].legend(fontsize=7)
    save(fig, "fig02_04_items_icc", "Homogeneous 2PL (M0) fitted on the training split: (a) item parameters; (b) phi "
         "correlations between items (pairwise complete); (c) item characteristic curves.", cap)


def fig_model_comparison(form, cap):
    p = TAB / f"{form}_model_selection_train.csv"
    if not p.exists():
        return
    t = pd.read_csv(p)
    r = t[t.family == "responses"].copy()
    r["dBIC"] = r.BIC - r.loc[r.model == "M0", "BIC"].iloc[0]
    d = t[t.family == "duration"].copy()
    d["dBIC"] = d.BIC - d.loc[d.model == "M2", "BIC"].iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for a_, tt, ref in [(ax[0], r, "M0"), (ax[1], d, "M2")]:
        tt = tt.sort_values("dBIC")
        cols = [C2 if m.startswith(("M1", "M3", "M4")) else (INK2 if m.startswith("LCA") else C1) for m in tt.model]
        a_.barh(tt.model, tt.dBIC, color=cols, height=0.6)
        a_.axvline(0, color=INK, lw=0.8)
        a_.set(xlabel=f"ΔBIC vs {ref} (negative = preferred)")
        a_.invert_yaxis()
    ax[0].set_title("(a) Response models (train)")
    ax[1].set_title("(b) Joint response + duration models")
    save(fig, "fig05_model_comparison", "Training-split BIC differences. Blue: continuous-ability models; orange: models "
         "with class-specific item difficulties; grey: unrestricted LCA.", cap)


def fig_classes(form, cap):
    e7 = J(f"{form}_E7_stability.json")
    if not e7:
        return
    dev0 = np.array(e7["dev_ref"])
    devs = np.array(e7["devs"])
    lo, hi = np.quantile(devs, [0.025, 0.975], axis=0)
    o = np.argsort(dev0)
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
    xs = np.arange(len(dev0))
    ax[0].vlines(xs, lo[o], hi[o], color=INK2, lw=1.5)
    ax[0].scatter(xs, dev0[o], color=C2, s=20, zorder=3)
    ax[0].axhline(0, color=INK, lw=0.8)
    ax[0].set(xlabel="Item (sorted)", ylabel="b(class 2) − b(class 1)", title="(a) Ability-adjusted difficulty differences")
    ax[1].hist(e7["aris"], bins=10, color=C1, alpha=0.8)
    ax[1].set(xlabel="ARI vs full-train solution", ylabel="Bootstrap refits", title=f"(b) Assignment stability (B={e7['B']})")
    ax[2].hist(e7["cors"], bins=10, color=C3, alpha=0.9)
    ax[2].set(xlabel="|r| of difficulty-difference profiles", title="(c) Profile reproducibility")
    save(fig, "fig07_10_class_profiles_stability",
         f"Two-class mixture 2PL with class-specific difficulties (M1_2), training split. (a) Class difference in item "
         f"difficulty after ability adjustment, with 95% bootstrap ranges ({e7['B']} refits). (b) Adjusted Rand index of "
         f"bootstrap assignments against the full-sample solution. (c) Correlation of the difference profiles across refits.", cap)


def fig_posteriors(form, cap):
    a = RES / "checkpoints" / f"{form}_E8_arrays.npy"
    t = TAB / f"{form}_E8_within_score.csv"
    if not a.exists():
        return
    arr = np.load(a)
    score, pc, lz = arr[:, 0], arr[:, 1], arr[:, 2]
    w = pd.read_csv(t) if t.stat().st_size > 2 else pd.DataFrame()
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
    ax[0].hist(np.maximum(pc, 1 - pc), bins=25, color=C1)
    ax[0].set(xlabel="max posterior class probability", ylabel="Students", title="(a) Class assignment certainty (M1_2)")
    if len(w):
        ax[1].plot(w.score, w.sd_classpost, marker="o", color=C1, label="PISA data")
        ax[1].plot(w.score, w.sd_classpost_null, marker="s", color=C2, label="continuous null")
        ax[1].set(xlabel="Total score", ylabel="SD of P(class 2) at fixed score", title="(b) Within-score spread")
        ax[1].legend(fontsize=7)
        ax[2].plot(w.score, w.frac_lz_lt_m2, marker="o", color=C1, label="PISA data")
        ax[2].plot(w.score, w.frac_lz_lt_m2_null, marker="s", color=C2, label="continuous null")
        ax[2].set(xlabel="Total score", ylabel="share l_z < −1.96", title="(c) Person misfit at fixed score")
        ax[2].legend(fontsize=7)
    save(fig, "fig06_09_within_score", "Within-score analysis (training split). (a) Certainty of class assignment. "
         "(b) Spread of the class posterior among students with the same score, observed vs a dataset simulated from the "
         "continuous 2PL with the same items and missingness. (c) Share of aberrant response patterns (l_z < −1.96).", cap)


def fig_null(form, cap):
    t = TAB / f"{form}_E5_null_reps.csv"
    e5 = J(f"{form}_E5_null.json")
    sel = TAB / f"{form}_model_selection_train.csv"
    if not t.exists() or not sel.exists():
        return
    t = pd.read_csv(t)
    s = pd.read_csv(sel).set_index("model").BIC
    obs = min(s["M0"], s["A2"], s.get("A3", np.inf)) - min(s["M1_2"], s["M1_3"])
    obs_d = s["M2"] - s["M3_2"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
    for cfg, col in [("fitted", C1), ("skewed", C2)]:
        tt = t[t.cfg == cfg]
        ax[0].hist(tt.dBIC_resp, bins=15, alpha=0.6, color=col, label=f"null: {cfg} θ")
        ax[1].hist(tt.dBIC_dur, bins=15, alpha=0.6, color=col, label=f"null: {cfg} θ")
        ax[2].hist(tt.kmeans4_silhouette, bins=15, alpha=0.6, color=col, label=f"null: {cfg} θ")
    ax[0].axvline(obs, color=INK, lw=2)
    ax[0].annotate(f"observed {obs:.0f}", (obs, ax[0].get_ylim()[1] * 0.9), xytext=(-4, 0), textcoords="offset points", ha="right", fontsize=7)
    ax[1].axvline(obs_d, color=INK, lw=2)
    ax[1].annotate(f"observed {obs_d:.0f}", (obs_d, ax[1].get_ylim()[1] * 0.9), xytext=(-4, 0), textcoords="offset points", ha="right", fontsize=7)
    e1 = J(f"{form}_E1_baseline.json")
    if e1:
        ax[2].axvline(e1["k4"]["silhouette"], color=INK, lw=2)
    ax[0].set(xlabel="ΔBIC (continuous − class model)", ylabel="Replications", title="(a) Responses: class vs continuous")
    ax[1].set(xlabel="ΔBIC (M2 − M3_2)", title="(b) Responses + duration")
    ax[2].set(xlabel="K-means (k=4) silhouette", title="(c) Source pipeline")
    ax[0].legend(fontsize=7)
    save(fig, "fig11_null_simulation", "Null simulations with no latent classes. Data are generated from the continuous "
         "model fitted to Form A, using either the fitted ability distribution or a skewed one, and the same N, items, "
         "missingness and duration model. Positive ΔBIC means the class model is preferred. Vertical lines mark the "
         "observed PISA values.", cap)


def fig_ablation(form, cap):
    p = TAB / f"{form}_E10_paired.csv"
    if not p.exists():
        return
    t = pd.read_csv(p)
    s = pd.read_csv(TAB / f"{form}_E10_test_scores.csv")
    meta = J(f"{form}_E10_meta.json")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw=dict(width_ratios=[1.2, 1]))
    lab = [f"{a} vs {b}" for a, b in zip(t.better, t.than)]
    y = np.arange(len(t))
    ax[0].errorbar(t.gain_nats_per_item * 1000, y, xerr=[(t.gain_nats_per_item - t.ci_lo) * 1000, (t.ci_hi - t.gain_nats_per_item) * 1000],
                   fmt="o", color=C1, ecolor=INK2, capsize=2, ms=5)
    ax[0].axvline(0, color=INK, lw=0.8)
    ax[0].axvline(1, color=C2, lw=1, ls="--")
    ax[0].set_yticks(y, lab, fontsize=7)
    ax[0].invert_yaxis()
    ax[0].set(xlabel="Log-loss gain (milli-nats/item; >0 = first better)", title="(a) Paired held-out comparisons (95% CI)")
    for m, col in [("M0", C1), ("M1_2", C2), ("KMeans4_score_time", C3)]:
        if meta and m in meta["calibration"]:
            c = np.array(meta["calibration"][m])
            ax[1].plot(c[:, 2], c[:, 3], marker="o", color=col, label=m, ms=4)
    ax[1].plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
    ax[1].set(xlabel="Predicted P(correct)", ylabel="Observed proportion", title="(b) Calibration on masked items")
    ax[1].legend(fontsize=7)
    save(fig, "fig12_14_ablation_calibration", "Held-out evaluation on the test split (30% of students). Masked items "
         "(30% of positions, 5 mask sets) are predicted from the unmasked responses. Models with '_time' also use the "
         "time spent on unmasked items. (a) Paired differences in log loss with student-bootstrap 95% CIs; the dashed "
         "line marks the pre-registered practical threshold (0.001 nats/item). (b) Reliability diagram for the first mask set.", cap)


def fig_dif(form, cap):
    p = TAB / f"{form}_E9_dif_oecd.csv"
    if not p.exists():
        return
    d1 = pd.read_csv(p)
    g = TAB / f"{form}_E9_dif_gender.csv"
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(d1.item - 0.2, d1.dR2, width=0.4, color=C1, label="OECD vs non-OECD")
    if g.exists():
        d2 = pd.read_csv(g)
        ax.bar(d2.item + 0.2, d2.dR2, width=0.4, color=C2, label="gender")
    ax.axhline(0.035, color=INK2, ls="--", lw=1)
    ax.annotate("moderate DIF threshold (0.035)", (0, 0.036), fontsize=7, color=INK2)
    ax.set(xlabel="Item", ylabel="ΔNagelkerke R²", title="Logistic-regression DIF (rest-score matched)")
    ax.legend(fontsize=7)
    save(fig, "fig13_dif", "Differential item functioning on Form A (all students) by group membership. The dashed line "
         "is the Jodoin–Gierl threshold for moderate DIF.", cap)


def fig_continuous_map(form, cap):
    """Exploratory: reliability of K-means labels versus the continuous coordinates they summarise (training split)."""
    m = J(f"{form}_continuous_map.json")
    p = RES / "checkpoints" / f"{form}_continuous_map.npz"
    if not m or not p.exists():
        return
    z = np.load(p)
    order = np.argsort(-z["agree"])                  # draw unstable students last, on top
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("stab", plt.get_cmap("Blues_r")(np.linspace(0, 0.75, 256)))  # stable stays visible
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.2), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.38))
    jit = np.random.default_rng(0).uniform(-0.25, 0.25, len(order))
    sc = ax[0].scatter(z["minutes"][order], z["score"][order] + jit, c=z["agree"][order], cmap=cmap, vmin=0, vmax=1,
                       s=6, linewidths=0)
    for c in z["centers"]:
        ax[0].scatter(c[1], c[0], s=70, color=C2, edgecolor="white", linewidth=1.5, zorder=3)
    cb = fig.colorbar(sc, ax=ax[0], pad=0.01)
    cb.set_label("P(same profile in a parallel test)")
    ax[0].set(xlabel="Total time (min)", ylabel="Score (items correct, jittered)",
              title="(a) Profile label stability, K-means centroids")
    la = m["label_agreement"]
    ax[1].hist(z["agree"], bins=np.linspace(0, 1, 21), color=C1, edgecolor="white", linewidth=1)
    ax[1].axvline(0.8, color=INK2, ls="--", lw=1.2)
    ax[1].annotate(f"{la['share_below_0_8']:.0%} of students\nbelow 0.8", (0.8, ax[1].get_ylim()[1] * 0.85), xytext=(-6, 0),
                   textcoords="offset points", ha="right", fontsize=7.5, color=INK)
    ax[1].set(xlabel="P(same profile in a parallel test)", ylabel="Students",
              title=f"(b) Label agreement; κ = {la['kappa_mean']:.2f}, EAP reliability = {m['eap_reliability']:.2f}")
    save(fig, "fig15_continuous_map",
         f"How stable is a student's score–time profile? Training students of Form A (n = {m['n']}). For each student, "
         f"{m['R']} parallel-test scores were drawn from the 2PL posterior with the student's own time held fixed, and "
         f"each replicated (score, time) point was assigned to the nearest K-means centroid. (a) Students coloured by "
         f"the share of replications that keep their observed profile; orange dots are the centroids. (b) Distribution "
         f"of that share; κ is the mean Cohen's kappa between observed and replicated profiles. Because time is held "
         f"fixed, the instability shown is a lower bound.", cap)


def fig_summary(form, cap, verdict):
    if not verdict:
        return
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.axis("off")
    rows = [[h, v["status"], v["evidence"][:90]] for h, v in verdict["hypotheses"].items()]
    tb = ax.table(cellText=rows, colLabels=["Hypothesis", "Status", "Key evidence"], loc="center", cellLoc="left",
                  colWidths=[0.12, 0.16, 0.72])
    tb.auto_set_font_size(False)
    tb.set_fontsize(7)
    tb.scale(1, 1.4)
    ax.set_title(f"Overall verdict: {verdict['verdict']}", fontsize=9)
    save(fig, "fig15_contribution_summary", "Summary of the pre-registered hypotheses and the overall verdict.", cap)


# ------------------------------------------------------------------ LaTeX
def latex_tables(form):
    out = {}
    p = TAB / f"{form}_model_selection_train.csv"
    if p.exists():
        t = pd.read_csv(p)[["model", "loglik", "n_params", "AIC", "BIC", "converged"]].sort_values("BIC")
        out["tab_model_selection"] = t.to_latex(index=False, float_format="%.1f", caption="Training-split model comparison (Form A).", label="tab:modelsel")
    p = TAB / f"{form}_E10_test_scores.csv"
    if p.exists():
        t = pd.read_csv(p)
        out["tab_e10"] = t.to_latex(index=False, float_format="%.4f", caption="Held-out masked-item prediction (test split, 5 masks).", label="tab:e10")
        t = pd.read_csv(TAB / f"{form}_E10_paired.csv")
        out["tab_e10_paired"] = t.to_latex(index=False, float_format="%.5f", caption="Paired log-loss gains (nats/item) with 95\\% bootstrap CIs.", label="tab:paired")
    p = TAB / f"{form}_E6_positive_summary.csv"
    if p.exists():
        out["tab_e6"] = pd.read_csv(p).drop(columns=["secs", "mean_maxpost"], errors="ignore").to_latex(index=False, float_format="%.2f", caption="Positive-control simulation: power and recovery.", label="tab:e6")
    for k, v in out.items():
        head, sep, body = v.partition(r"\begin{tabular}")
        v = head + sep + body.replace("_", r"\_").replace("NaN", "--") if sep else v  # escape cell underscores only
        (MAN / "tables" / (f"{k}.tex" if form != "demo" else f"demo_SYNTHETIC_{k}.tex")).write_text(v, encoding="utf8")
    return list(out)


def repro_report(form):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    except Exception:
        commit, dirty = "unavailable", None
    import numpy, scipy, sklearn, statsmodels, matplotlib
    vers = {m.__name__: m.__version__ for m in [numpy, scipy, pd, sklearn, statsmodels, matplotlib]}
    metas = {}
    for f in (ROOT / "data/processed").glob("math_form*.json"):
        m = json.loads(f.read_text())
        metas[f.stem] = dict(n=m["n"], items=len(m["items"]), fingerprint=m.get("fingerprint"))
    cfg_hash = hashlib.sha256((ROOT / "configs/experiments.yaml").read_bytes()).hexdigest()[:16]
    lines = ["# Reproducibility report (auto-generated by scripts/build_results.py)", "",
             f"- Git commit: `{commit}`" + (" (working tree had uncommitted changes)" if dirty else ""),
             f"- Python {platform.python_version()} on {platform.platform()}",
             f"- Library versions: " + ", ".join(f"{k} {v}" for k, v in vers.items()),
             f"- Config `configs/experiments.yaml` SHA-256 (16): `{cfg_hash}`",
             "- Seeds: see `configs/seeds.yaml` and `configs/experiments.yaml`", "", "## Datasets", "",
             "| Form | n | items | fingerprint |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {v['items']} | `{v['fingerprint']}` |" for k, v in sorted(metas.items())]
    lines += ["", "## Saved outputs", ""]
    lines += [f"- `results/metrics/{p.name}`" for p in sorted((RES / "metrics").glob(f"{form}*"))]
    lines += [f"- `results/tables/{p.name}`" for p in sorted(TAB.glob(f"{form}*"))]
    lines += [f"- `results/figures/{p.name}`" for p in sorted(FIG.glob("*.pdf"))]
    d = J(f"{form}_fit_diagnostics.json")
    sel = TAB / f"{form}_model_selection_train.csv"
    if sel.exists():
        t = pd.read_csv(sel)
        bad = t[~t.converged.astype(bool)].model.tolist()
        lines += ["", "## Convergence", "", f"Non-converged fits (flagged, not silently used): {bad if bad else 'none'}"]
    lines += ["", "## Numerical tolerance", "",
              "EM stops when the relative change in the penalised log-likelihood falls below 1e-8. Re-runs with the same",
              "seeds on the same platform reproduce the log-likelihoods exactly (test_reproducibility.py). Across",
              "platforms or BLAS builds, differences of up to about 1e-6 in the log-likelihood are expected."]
    (ROOT / "reports" / "reproducibility_report.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", default="math_formA")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    form = "demo" if a.demo else a.form
    global FIG
    if a.demo:  # synthetic outputs are kept apart from empirical figures
        FIG = RES / "figures" / "demo_SYNTHETIC"
    FIG.mkdir(parents=True, exist_ok=True)
    (MAN / "tables").mkdir(parents=True, exist_ok=True)
    cap = {}
    ckp = RES / "checkpoints" / f"{form}_train_models.pkl"
    ck = pickle.load(open(ckp, "rb")) if ckp.exists() else None
    fig_e1(form, cap)
    fig_items(form, cap, ck)
    fig_model_comparison(form, cap)
    fig_classes(form, cap)
    fig_posteriors(form, cap)
    fig_null(form, cap)
    fig_ablation(form, cap)
    fig_dif(form, cap)
    fig_continuous_map(form, cap)
    fig_summary(form, cap, J(f"{form}_verdict.json"))
    (FIG / "captions.json").write_text(json.dumps(cap, indent=2, ensure_ascii=False), encoding="utf8")
    tabs = latex_tables(form)
    if not a.demo:
        repro_report(form)
    print("figures:", list(cap), "\ntables:", tabs)


if __name__ == "__main__":
    main()

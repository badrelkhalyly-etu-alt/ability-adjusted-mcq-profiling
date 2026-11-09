"""Tables and figures for the continuous representation (B) and temporal-regime (C) extensions.

Reads results/continuous/* and results/temporal/* (written by run_continuous_eval.py and run_temporal.py) and
writes results/tables/{continuous_eval,temporal}_summary.csv, manuscript/tables/tab_{continuous_eval,temporal}.tex,
results/figures/fig16_continuous_eval.* and fig17_temporal_regimes.*, results/metrics/extension_decisions.json.
Every number is read from those files."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_results as br  # noqa: E402  (shared style, palette and save())

plt = br.plt
RC, RT_, T, MAN = ROOT / "results" / "continuous", ROOT / "results" / "temporal", ROOT / "results" / "tables", ROOT / "manuscript" / "tables"
MAN.mkdir(parents=True, exist_ok=True)   # manuscript/ is not in the repository
FORMS = [("confirm", "math_formB", "B"), ("confirm", "math_formC", "C")]
EOL = "\\" * 2


def J(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def num(x, d=3):
    return f"{x:.{d}f}".replace("-", "$-$")


def ci(m, lo, hi, d=4):
    return f"{num(m, d)} [{num(lo, d)}, {num(hi, d)}]"


# ------------------------------------------------------------------ B: continuous representation
def continuous():
    rows, dec = [], {}
    for part, form, lab in FORMS + [("dev", "math_formA", "A (dev)")]:
        pf = RC / f"{part}_{form}_paired.csv"
        if not pf.exists():
            continue
        p = pd.read_csv(pf)
        s = J(RC / f"{part}_{form}_summary.json")
        sc = pd.read_csv(RC / f"{part}_{form}_scores.csv").set_index("model")
        for _, r in p.iterrows():
            rows.append(dict(form=lab, kind="logloss_gain", better=r.better, than=r.than, gain=r.gain, lo=r.ci_lo, hi=r.ci_hi,
                             gain_w=r.gain_weighted, lo_w=r.ci_lo_weighted, hi_w=r.ci_hi_weighted))
        iv = s["intervals"]
        for k, v in iv.items():
            rows.append(dict(form=lab, kind="interval", better=k, coverage=v["coverage90"], expected=v.get("expected_coverage", np.nan),
                             width=v["width_mean"], logscore=v["logscore_mean"]))
        rows.append(dict(form=lab, kind="split_half", **{k: v for k, v in s["split_half"].items()}))
        rows.append(dict(form=lab, kind="effort", r2_profiles=s["effort"]["r2_test"]["profiles"],
                         r2_continuous=s["effort"]["r2_test"]["continuous"], r2_both=s["effort"]["r2_test"]["both"],
                         gain=s["effort"]["mse_gain_continuous_vs_profiles"][0], lo=s["effort"]["mse_gain_continuous_vs_profiles"][1],
                         hi=s["effort"]["mse_gain_continuous_vs_profiles"][2]))
        rows.append(dict(form=lab, kind="dashboard", **s["dashboard"]))
        rows.append(dict(form=lab, kind="scores", **{f"logloss_{m}": sc.loc[m, "logloss"] for m in sc.index},
                         **{f"ece_{m}": sc.loc[m, "ece"] for m in sc.index}))
        if part == "confirm":
            g = p.set_index(["better", "than"])
            def sig(a, b):
                r = g.loc[(a, b)]
                return bool(r.ci_lo > 0)
            ctr = iv["CTR"]
            dec[lab] = dict(H6=sig("CTR", "KMeans4_score_time"), H7=sig("CTR", "CTR_kmeans4") and sig("CTR", "CTR_grid2x2"),
                            H8=bool(abs(ctr["coverage90"] - ctr["expected_coverage"]) <= 0.02),
                            H9=bool(s["split_half"]["theta_sb"] > s["split_half"]["kmeans_kappa"]))
    t = pd.DataFrame(rows)
    t.to_csv(T / "continuous_eval_summary.csv", index=False, float_format="%.6f")
    return t, dec


def tab_continuous(t):
    g = t[t.kind == "logloss_gain"]
    lab = {"KMeans4_score_time": "K-means score--time profiles", "CTR_kmeans4": "CTR coordinates, K-means ($k=4$)",
           "CTR_grid2x2": "CTR coordinates, $2\\times2$ grid", "G3PL": "G3PL (no time)", "CTR_no_time": "CTR without time",
           "M3_3_time": "M3$_3$ with total time (classes)", "MIRT2G": "MIRT2G (2D, guessing)", "M1_3": "M1$_3$ (classes)"}
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Independent validation of the continuous representation (CTR) on the untouched test splits of Forms B and C. "
         r"Masked-item log-loss gain of CTR over each comparator in nats per response (positive = CTR better), with paired bootstrap 95\% CIs; "
         r"weighted: final student weights. Protocol frozen before evaluation (\texttt{reports/protocol\_continuous\_temporal.md}).}\label{tab:continuous_eval}",
         r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{lllll}\toprule",
         "CTR vs & Form B & Form B, weighted & Form C & Form C, weighted" + EOL + r"\midrule"]
    for k, name in lab.items():
        cells = []
        for f in ["B", "C"]:
            r = g[(g.form == f) & (g.better == "CTR") & (g.than == k)]
            if len(r):
                r = r.iloc[0]
                cells += [ci(r.gain, r.lo, r.hi), ci(r.gain_w, r.lo_w, r.hi_w)]
            else:
                cells += ["--", "--"]
        L.append(name + " & " + " & ".join(cells) + EOL)
    iv = t[t.kind == "interval"]
    L.append(r"\midrule \multicolumn{5}{l}{\emph{90\% interval for the masked score: coverage / model-expected coverage / mean width}}" + EOL)
    for k, name in [("CTR", "CTR"), ("G3PL", "G3PL"), ("KMeans4_score_time", "K-means profiles"), ("CTR_kmeans4", "CTR, K-means ($k=4$)")]:
        cells = []
        for f in ["B", "C"]:
            r = iv[(iv.form == f) & (iv.better == k)].iloc[0]
            cells += [f"{r.coverage:.3f} / {r.expected:.3f} / {r.width:.2f}", ""]
        L.append(name + " & " + " & ".join(cells) + EOL)
    sh = t[t.kind == "split_half"].set_index("form")
    L.append(r"\midrule \multicolumn{5}{l}{\emph{Split-half stability (odd vs even items)}}" + EOL)
    for k, name in [("theta_sb", "Ability, Spearman--Brown reliability"), ("rel_time_sb", "Relative time, Spearman--Brown reliability"),
                    ("kmeans_kappa", r"K-means profiles, Cohen's $\kappa$"), ("theta_quartile_kappa", r"Ability quartiles, Cohen's $\kappa$")]:
        L.append(name + " & " + " & ".join([f"{sh.loc['B', k]:.2f}", "", f"{sh.loc['C', k]:.2f}", ""]) + EOL)
    ef = t[t.kind == "effort"].set_index("form")
    L.append(r"\midrule \multicolumn{5}{l}{\emph{Self-reported effort, out-of-sample $R^2$}}" + EOL)
    for k, name in [("r2_profiles", "Four profiles"), ("r2_continuous", "Continuous coordinates")]:
        L.append(name + " & " + " & ".join([f"{ef.loc['B', k]:.3f}", "", f"{ef.loc['C', k]:.3f}", ""]) + EOL)
    L += [r"\bottomrule\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    (MAN / "tab_continuous_eval.tex").write_text("\n".join(L) + "\n", encoding="utf8")


def fig_continuous(t, cap):
    g = t[(t.kind == "logloss_gain") & (t.better == "CTR")]
    order = ["KMeans4_score_time", "CTR_kmeans4", "CTR_grid2x2", "M1_3", "G3PL", "CTR_no_time", "MIRT2G", "M3_3_time"]
    names = {"KMeans4_score_time": "K-means score–time profiles", "CTR_kmeans4": "CTR discretised: K-means k=4",
             "CTR_grid2x2": "CTR discretised: 2×2 grid", "M1_3": "M1_3 (classes, no time)", "G3PL": "G3PL (no time)",
             "CTR_no_time": "CTR without time", "MIRT2G": "MIRT2G (2D + guessing)", "M3_3_time": "M3_3 with time (classes)"}
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, (f, col, dy) in enumerate([("B", br.C1, -0.15), ("C", br.C2, 0.15)]):
        for j, k in enumerate(order):
            r = g[(g.form == f) & (g.than == k)]
            if not len(r):
                continue
            r = r.iloc[0]
            ax.errorbar(r.gain, j + dy, xerr=[[r.gain - r.lo], [r.hi - r.gain]], fmt="o", color=col, ms=5, capsize=0, lw=1.5,
                        label=f"Form {f} test" if j == 0 else None)
    ax.axvline(0, color=br.INK2, lw=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[k] for k in order])
    ax.invert_yaxis()
    ax.set(xlabel="Log-loss gain of CTR (nats per masked response; > 0 = CTR better)",
           title="CTR against each comparator, untouched test splits")
    ax.legend(loc="lower right", fontsize=7.5)
    br.save(fig, "fig16_continuous_eval",
            "Independent validation of the continuous ability × relative-time representation (CTR). Points are mean "
            "per-student log-loss gains of CTR over each comparator on masked items of the Form B and Form C test splits "
            "(5 mask sets), with paired bootstrap 95 % intervals. Positive values favour CTR. The protocol was frozen "
            "before these test splits were used.", cap)


# ------------------------------------------------------------------ C: temporal regimes
def temporal():
    rows = []
    sets = [("dev", "math_formA", "A (dev)")] + [("confirm", f, l) for _, f, l in FORMS]
    for part, form, lab in sets:
        sp = RT_ / f"{part}_{form}_summary.json"
        if not sp.exists():
            continue
        s = J(sp)
        for var in ["v_relative", "log_total_time"]:
            tab = pd.read_csv(RT_ / f"{part}_{form}_{var}.csv").set_index("model")
            c = s[var]
            gw = c.get("heldout_gain_weighted", [np.nan] * 3)
            reg = c["regime_summary"][c["best_regime"]]
            rows.append(dict(form=lab, variable=var, best_continuous=c["best_continuous"], best_regime=c["best_regime"],
                             dBIC=c["dBIC_cont_minus_regime"], gain=c["heldout_gain_regime_vs_cont"][0],
                             lo=c["heldout_gain_regime_vs_cont"][1], hi=c["heldout_gain_regime_vs_cont"][2],
                             gain_w=gw[0], lo_w=gw[1], hi_w=gw[2],
                             dBIC_normalcubic_minus_bestcont=tab.loc["normal|cubic|linear", "train_bic"] - tab.loc[c["best_continuous"], "train_bic"],
                             pit_cover_cont=tab.loc[c["best_continuous"], "pit_cover90"], pit_cover_reg=tab.loc[c["best_regime"], "pit_cover90"],
                             pit_ks_cont=tab.loc[c["best_continuous"], "pit_ks"], pit_ks_reg=tab.loc[c["best_regime"], "pit_ks"],
                             smallest_regime_prev=min(reg["prevalence_marginal"])))
    t = pd.DataFrame(rows)
    t.to_csv(T / "temporal_summary.csv", index=False, float_format="%.6f")
    ns = pd.read_csv(RT_ / "null_summary.csv") if (RT_ / "null_summary.csv").exists() else None
    bs = J(RT_ / "bootstrap_stability.json") if (RT_ / "bootstrap_stability.json").exists() else None
    dec = {}
    if ns is not None and bs is not None:
        nulls = ns[~ns.generator.str.startswith("positive")]
        q95 = float(nulls.dBIC_q95.max())
        stable = bool(bs["ari_vs_full"]["median"] >= 0.7 and all(np.array(bs["intercept_gaps"]["q025"]) > 0))
        for var in ["v_relative", "log_total_time"]:
            c = t[(t.variable == var) & t.form.isin(["B", "C"])]
            if len(c) < 2:
                continue
            i_ = (c.dBIC > q95).all()
            ii = (c.lo > 0).all()
            supported = bool(i_ and ii and stable)
            rejected = bool(((c.lo <= 0)).all() and not (c.dBIC > q95).any())
            dec[var] = dict(null_q95_max=q95, criterion_i_bic=bool(i_), criterion_ii_heldout=bool(ii), criterion_iii_stable=stable,
                            decision="supported" if supported else "rejected" if rejected else "indeterminate")
    return t, ns, bs, dec


def tab_temporal(t, ns, bs):
    nm = {"normal|linear|const": "normal, linear", "normal|cubic|linear": "normal, cubic, het.", "normal|spline|spline": "normal, spline",
          "skewnormal|spline|spline": "skew-normal, spline", "t|spline|spline": "$t$, spline", "shash|spline|spline": "SHASH, spline",
          "regimes|K=2": "2 regimes", "regimes|K=3": "3 regimes"}
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Latent time regimes versus flexible continuous time distributions, conditional on the same IRT response model. "
         r"$\Delta$BIC: best continuous minus best regime model on the training data (positive favours regimes). Held-out gain: "
         r"mean per-student $\log p(v\mid Y)$ of the best regime model minus the best continuous model on the evaluation data, with a paired bootstrap 95\% CI. "
         r"$v$: item-adjusted relative log time (primary); $T$: total log time. A: Form A development split; B, C: untouched test splits.}\label{tab:temporal}",
         r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{lllrrll}\toprule",
         "Data & Var. & Best continuous / regime & $\\Delta$BIC & Gain vs normal-cubic & Held-out gain & Weighted held-out gain" + EOL + r"\midrule"]
    for _, r in t.iterrows():
        L.append(f"{r.form} & {'$v$' if r.variable == 'v_relative' else '$T$'} & {nm[r.best_continuous]} / {nm[r.best_regime]} & "
                 f"{num(r.dBIC, 1)} & {num(r.dBIC_normalcubic_minus_bestcont, 1)} & {ci(r.gain, r.lo, r.hi)} & "
                 + (ci(r.gain_w, r.lo_w, r.hi_w) if np.isfinite(r.gain_w) else "--") + EOL)
    if ns is not None:
        gn = {"shash_spline": "SHASH", "t_spline": "$t$", "skewnormal_spline": "skew-normal", "normal_cubic_het": "normal cubic",
              "positive_regimes2": "2 regimes (positive control)"}
        L.append(r"\midrule \multicolumn{7}{l}{\emph{Simulations from fits on the Form A development split ($v$), 20 replications each}}" + EOL)
        L.append(r"Generator & & Regimes win on BIC & $\Delta$BIC q95 & & Held-out gain, median & " + EOL)
        for _, r in ns.iterrows():
            L.append(f"{gn.get(r.generator, r.generator)} & & {r.regime_wins_bic:.2f} & {num(r.dBIC_q95, 1)} & & {num(r.heldout_gain_median, 4)} & " + EOL)
    if bs is not None:
        L.append(r"\midrule \multicolumn{7}{l}{\emph{Bootstrap stability of the " + bs["model"].replace("regimes|K=", "") +
                 "-regime model (Form A development split, $B=" + str(bs["B"]) + "$): median ARI vs full fit " +
                 f"{bs['ari_vs_full']['median']:.2f}" + "}}" + EOL)
    L += [r"\bottomrule\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    (MAN / "tab_temporal.tex").write_text("\n".join(L) + "\n", encoding="utf8")


def fig_temporal(t, ns, cap):
    nsim = pd.read_csv(RT_ / "null_simulations.csv") if (RT_ / "null_simulations.csv").exists() else None
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw=dict(wspace=0.75))
    if nsim is not None:
        gens = list(dict.fromkeys(nsim.generator))
        for i, gname in enumerate(gens):
            d = nsim[nsim.generator == gname].dBIC_cont_minus_regime
            col = br.C2 if gname.startswith("positive") else br.INK2
            ax[0].scatter(d, np.full(len(d), i) + np.random.default_rng(i).uniform(-.15, .15, len(d)), s=10, color=col, alpha=.7)
        for _, r in t[t.variable == "v_relative"].iterrows():
            ax[0].axvline(r.dBIC, color=br.C1, lw=1.5, ls="--")
            ax[0].annotate(r.form.split()[0], (r.dBIC, -0.75), color=br.C1, fontsize=8, ha="center", va="center",
                           bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
        ax[0].axvline(0, color=br.INK, lw=0.8)
        ax[0].set_yticks(range(len(gens)))
        ax[0].set_yticklabels([{"positive_regimes2": "2 regimes (positive control)", "normal_cubic_het": "normal cubic, het.", "skewnormal_spline": "skew-normal, spline", "t_spline": "t, spline", "shash_spline": "SHASH, spline"}.get(g, g) for g in gens])
        ax[0].set(xlabel="ΔBIC, best continuous − best regime (> 0 favours regimes)", title="(a) Simulated data vs observed (dashed)")
        ax[0].set_ylim(len(gens) - 0.5, -1.2)
    lab = []
    for j, (_, r) in enumerate(t.iterrows()):
        col = br.C1 if r.variable == "v_relative" else br.C3
        ax[1].errorbar(r.gain, j, xerr=[[r.gain - r.lo], [r.hi - r.gain]], fmt="o", color=col, ms=5, lw=1.5)
        lab.append(f"{r.form}, {'relative' if r.variable == 'v_relative' else 'total'} time")
    ax[1].axvline(0, color=br.INK2, lw=1)
    ax[1].set_yticks(range(len(lab)))
    ax[1].set_yticklabels(lab)
    ax[1].invert_yaxis()
    ax[1].set(xlabel="Held-out log p(v | Y), regimes − continuous", title="(b) Held-out comparison")
    br.save(fig, "fig17_temporal_regimes",
            "Latent time regimes versus flexible continuous time distributions. (a) ΔBIC (best continuous minus best regime "
            "model) in data simulated from continuous fits and from a two-regime fit on the Form A development split, against "
            "the observed values (dashed; A = development split, B and C = training splits of the validation forms). "
            "(b) Held-out conditional log-density gain of the best regime model over the best continuous model, with paired "
            "bootstrap 95 % intervals, for relative time (blue) and total time (green).", cap)


def main():
    cap = json.loads((br.FIG / "captions.json").read_text(encoding="utf8"))
    out = {}
    t, dec = continuous()
    if len(t):
        if {"B", "C"} <= set(t.form):
            tab_continuous(t)
            fig_continuous(t, cap)
        out["continuous_hypotheses"] = dec
    tt, ns, bs, dect = temporal()
    if len(tt):
        tab_temporal(tt, ns, bs)
        fig_temporal(tt, ns, cap)
        out["temporal_decision"] = dect
    (br.FIG / "captions.json").write_text(json.dumps(cap, indent=2, ensure_ascii=False), encoding="utf8")
    (ROOT / "results" / "metrics" / "extension_decisions.json").write_text(json.dumps(out, indent=1), encoding="utf8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

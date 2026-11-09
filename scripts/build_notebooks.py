"""Generate the Colab notebooks (thin, reproducible wrappers around src/ and scripts/).

Usage: python scripts/build_notebooks.py [REPO_URL]
Each notebook (1) can re-run its stage (DEMO = synthetic, or real PISA data if available) and
(2) displays the saved PISA 2025 aggregate results committed in results/.
"""
import sys
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = r'''# --- Environment (Colab or local) -------------------------------------------
REPO_URL = "__REPO_URL__"   # GitHub repository; a private repository needs a token in the URL or an uploaded copy
DEMO = None            # None = real PISA 2025 data if data/processed/ is present (it is in the repository), else synthetic; or force True/False
import os, sys, subprocess, platform
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not os.path.exists("ability-adjusted-mcq-profiling"):
        if not REPO_URL:
            raise SystemExit("Set REPO_URL or upload the repository folder to /content.")
        subprocess.run(["git", "clone", REPO_URL], check=True)
    os.chdir("ability-adjusted-mcq-profiling")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
else:
    while not os.path.exists("configs") and os.getcwd() != os.path.dirname(os.getcwd()):
        os.chdir("..")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
if DEMO is None:
    DEMO = not os.path.exists("data/processed/math_formA.parquet")
import numpy, scipy, pandas, sklearn
print(platform.python_version(), "numpy", numpy.__version__, "scipy", scipy.__version__,
      "pandas", pandas.__version__, "sklearn", sklearn.__version__)
FORM = "demo" if DEMO else "math_formA"
print("Mode:", "SYNTHETIC DEMONSTRATION - not empirical results" if DEMO else "REAL PISA 2025 data")'''

RUN = r'''import subprocess, sys
def run(stage, form=None):
    cmd = [sys.executable, "scripts/run_experiments.py", "--stage", stage] + (["--demo"] if DEMO else []) \
          + (["--form", form] if form else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-6000:])
    if r.returncode:
        print(r.stderr[-4000:]); raise RuntimeError(f"stage {stage} failed")'''

SHOW = r'''# Saved PISA 2025 results (aggregate outputs committed to the repository; readable without the licensed data)
import json, pandas as pd
from pathlib import Path
from IPython.display import display, Image, Markdown
def show_json(name, keys=None):
    p = Path("results/metrics") / name
    if not p.exists():
        print("missing", p); return
    d = json.loads(p.read_text(encoding="utf8"))
    print(json.dumps({k: d[k] for k in keys if k in d} if keys else d, indent=1, default=str)[:5000])
def show_csv(name):
    p = Path("results/tables") / name
    display(pd.read_csv(p)) if p.exists() else print("missing", p)
def show_fig(name):
    p = Path("results/figures") / f"{name}.png"
    display(Image(str(p))) if p.exists() else print("missing", p)'''

DATA_NOTE = ("**Data.** The analysis-ready PISA 2025 public-use extracts are in `data/processed/` (see `data/README.md`), "
             "so cloning the repository in Colab is enough to run the real analyses; `DEMO` switches to real data automatically. "
             "Set `DEMO = True` for a fast synthetic demonstration. **Demo outputs are not findings.** "
             "The *saved results* cells display the PISA 2025 aggregate outputs committed in `results/`.")

CT = ["profile_x_M1_3", "M1_3_x_scoreband", "profile_x_M1_2", "profile_x_oecd", "profile_x_booklet", "M1_3_x_oecd"]

BOOKS = [
    ("00_environment_colab", "Environment check", "Verify the installation, versions and test suite.",
     [RUN, "r = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests'], capture_output=True, text=True); print(r.stdout[-3000:])"]),
    ("01_data_audit", "Data audit",
     "Validate the analytic form. Checks: duplicates, missing values, item codes, score consistency, timing anomalies and the extremes of total time.",
     [r'''import pandas as pd, numpy as np, json
from run_experiments import load_form
df, meta = load_form(FORM, DEMO)
items = [c for c in df.columns if c.startswith("I") and c[1:].isdigit()]
print("rows", len(df), "items", len(items), "fingerprint", meta.get("fingerprint"))
print("duplicated ids:", df.duplicated(["CNT", "CNTSTUID"]).sum())
print("item codes:", pd.unique(df[items].values.ravel()))
print("score consistency:", np.allclose(df[items].sum(1, skipna=True), df.final_score))
print(df.total_duration_minutes.describe(percentiles=[.01,.05,.5,.95,.99]))
print("missing total time:", df.total_duration_minutes.isna().sum(), " >60 min:", (df.total_duration_minutes > 60).sum())''',
      SHOW, "show_csv('data_descriptives.csv')  # Forms A, B, C"]),
    ("02_baseline_clustering", "E1: The score-time pipeline (K-means on score and time)",
     "K-means (k = 2..8), internal indices, 5-fold held-out silhouette, bootstrap ARI/Jaccard, log-time variant, GMM/BIC, "
     "agglomerative, DBSCAN/HDBSCAN, Hopkins statistic, and a no-cluster null distribution of the silhouette.",
     [RUN, "run('baseline')", SHOW,
      "show_json('math_formA_E1_baseline.json')  # every E1 number cited in the article (k = 2..8, CV silhouette, ARI, GMM, HDBSCAN, null)\n"
      "d = json.loads(Path('results/metrics/math_formA_E1_baseline.json').read_text())['null_silhouette']\n"
      "print('no-cluster null silhouette:', {k: d[k] for k in ['mean', 'q95', 'p_value']})\n"
      "c = json.loads(Path('results/metrics/math_formA_contingency.json').read_text())\n"
      "print('K-means profile names used in the article:', c['profile_names'])",
      "show_fig('fig01_score_time_clusters')"]),
    ("03_irt_models", "E2/E3: Model ladder and continuous competitors (training split)",
     "Fit M0, A_K, M1_K, M2, M3_K, M4_K and LCA_K on the 70% training split, then the exploratory continuous competitors "
     "(2D 2PL, 2PL with guessing, 2D with guessing) and their nulls; replication on Forms B and C.",
     [RUN, "run('fit')",
      "if not DEMO:\n    run('e3b'); run('e3c'); run('replication')   # replication fits Forms B and C\n"
      "    for f in ['math_formB', 'math_formC']:\n        run('competitors', f)",
      SHOW, "show_csv('math_formA_model_selection_train.csv')",
      "show_csv('crossform_bic.csv')  # BIC of every model on Forms A, B, C",
      "show_json('math_formA_E3b_competitors.json', ['bic_train', 'class_contrast_vs_item_properties', 'local_dependence_pairs', 'mdim_null'])\n"
      "show_json('math_formA_E3c_guess_null.json')",
      "show_json('math_formA_fit_diagnostics.json')  # class means, SDs, proportions and duration slopes cited in the article",
      "show_fig('fig05_model_comparison'); show_fig('fig02_04_items_icc')"]),
    ("04_lca_mixirt", "E7 and contingency tables: ability-adjusted classes, stability, cross-classification",
     "Class-specific difficulty deviations after ability adjustment, bootstrap stability, and contingency tables of "
     "K-means profiles x classes x score quartile x OECD x booklet.",
     [RUN, "run('stability')", "if not DEMO:\n    run('contingency')", SHOW,
      "show_json('math_formA_E7_stability.json', ['B', 'ari_median', 'ari_q', 'dev_corr_median', 'aris'])",
      "for t in " + repr(CT) + ":\n"
      "    display(Markdown(f'**{t}**: counts, then row %'))\n"
      "    show_csv(f'math_formA_ct_{t}.csv'); show_csv(f'math_formA_ct_{t}_rowpct.csv')\n"
      "c = json.loads(Path('results/metrics/math_formA_contingency.json').read_text())\n"
      "display(pd.DataFrame({k: v['stats'] for k, v in c.items() if isinstance(v, dict) and 'stats' in v}).T)\n"
      "show_csv('math_formA_M1_3_class_profile.csv')",
      "show_fig('fig07_10_class_profiles_stability')"]),
    ("05_joint_duration_model", "E4/E10: Incremental value of time and masked-item prediction",
     "Held-out masked-item log loss, Brier score and calibration. Duration is measured on unmasked items only, so the timing "
     "of masked items cannot leak into the prediction. The real test set was evaluated once (protocol); the stage is re-run here only in DEMO mode.",
     [RUN, "if DEMO:\n    run('e10')", SHOW,
      "show_csv('math_formA_E10_test_scores.csv'); show_csv('math_formA_E10_paired.csv')\n"
      "m = json.loads(Path('results/metrics/math_formA_E10_meta.json').read_text())\n"
      "print('evaluable test students n =', m['n_test'], '| masks:', len(m['masks']))",
      "show_fig('fig12_14_ablation_calibration')"]),
    ("06_null_simulation", "E5/E6: Null (no classes) and positive-control simulations",
     "False-positive rate of the class-selection rule under a continuous-ability generator fitted to the data; power and "
     "recovery under known classes. Replications are checkpointed, so an interrupted run resumes.",
     [RUN, "run('null')", "run('positive')", SHOW,
      "show_json('math_formA_E5_null.json')",
      "r5 = pd.read_csv('results/tables/math_formA_E5_null_reps.csv')\n"
      "display(r5.groupby('cfg').median(numeric_only=True).T)  # medians per null configuration\n"
      "show_csv('math_formA_E6_positive_summary.csv')\n"
      "r6 = pd.read_csv('results/tables/math_formA_E6_positive_reps.csv')\n"
      "print('E6 datasets:', len(r6))\n"
      "display(r6[r6.dur].groupby(['prop', 'n']).dBIC_dur.agg(['median', 'count', lambda s: int((s > 0).sum())]))",
      "show_fig('fig11_null_simulation')"]),
    ("07_robustness_ablation", "E8/E9/H5: Within-score analysis, DIF and external criterion",
     "Within-score heterogeneity against a null, logistic DIF (OECD and gender), per-subgroup class enumeration, and "
     "self-reported effort (EFFORT1) as an external criterion.",
     [RUN, "run('e8')", "if not DEMO:\n    run('dif'); run('external')", SHOW,
      "show_json('math_formA_E8_personfit.json'); show_csv('math_formA_E8_within_score.csv')",
      "show_json('math_formA_E9_dif.json')\n"
      "show_json('math_formA_H5_external.json', ['base', 'kmeans', 'class2', 'class3', 'kmeans_effort_means'])",
      "show_fig('fig06_09_within_score'); show_fig('fig13_dif')",
      "# Robustness (exploratory, deviation log): not-reached scored incorrect, sampling-weighted fits,\n"
      "# continuous duration model with a cubic speed-ability curve. Long on real data (about 1 h per form).\n"
      "RUN_ROBUSTNESS = DEMO\n"
      "if RUN_ROBUSTNESS:\n    run('robustness', FORM)",
      "for f in ['math_formA', 'math_formB', 'math_formC']:\n"
      "    display(Markdown(f'**{f}**')); show_json(f'{f}_robustness.json'); show_csv(f'{f}_robustness.csv')",
      "# Summary used by Table tab:robustness, and sampling-weight diagnostics (Kish effective n, ESS-rescaled BIC)\n"
      "show_csv('robustness_summary.csv'); show_json('weight_diagnostics.json')",
      "# Continuous alternative (exploratory, training split, deviation log): profile-label reliability under parallel\n"
      "# tests, EAP reliability, cross-validated prediction of self-reported effort. Needs the real PISA extracts.\n"
      "if not DEMO:\n    for f in ['math_formA', 'math_formB', 'math_formC']:\n        run('continuous_map', f)",
      "show_csv('continuous_map_summary.csv'); show_fig('fig15_continuous_map')"]),
    ("08_figures_results", "Figures, tables, verdict and reproducibility report",
     "Rebuild every figure and table from the saved outputs, then display the scientific verdict and the Q1-readiness assessment.",
     ["import subprocess, sys\n"
      "steps = [['build_results.py'] + (['--demo'] if DEMO else []), ['build_captions.py']]\n"
      "if not DEMO:\n    steps += [['build_data_table.py'], ['build_crossform_table.py'], ['build_robustness_table.py'], ['build_continuous_table.py']]\n"
      "for s in steps:\n    print(subprocess.run([sys.executable, 'scripts/' + s[0]] + s[1:], capture_output=True, text=True).stdout[-2000:])",
      "# Every cell of the manuscript tables must equal its source CSV (prints mismatches, then the total)\n"
      "if os.path.exists('manuscript'):   # the manuscript is kept outside the repository\n    print(subprocess.run([sys.executable, 'scripts/check_tables.py'], capture_output=True, text=True).stdout[-3000:])",
      SHOW, "show_json('verdict.json')",
      "display(Markdown(Path('reports/scientific_verdict.md').read_text(encoding='utf8')))",
      "display(Markdown(Path('reports/q1_readiness.md').read_text(encoding='utf8')))"]),
    ("09_continuous_representation", "Continuous ability x relative-time representation with uncertainty (C2, C3)",
     "Build the uncertainty-aware continuous representation (ability from an IRT model with guessing; relative time "
     "adjusted for ability, booklet, items reached and item time intensities) and compare it with discrete score-time "
     "profiles: masked-item prediction, calibration, predictive intervals, split-half stability, survey weights, "
     "self-reported effort. Development used the Form A training split; confirmation used the untouched Form B and C test splits "
     "(protocol: reports/protocol_continuous_temporal.md).",
     ["# 1. Synthetic demonstration (runs anywhere, no PISA data): fit the representation and look at it\n"
      "from mcq_profiles.dashboard import simulate_demo, compute_map, LIMITATIONS\n"
      "demo = simulate_demo(n=1500, J=20, seed=0)\n"
      "co, summ = compute_map(demo, level=0.9)\n"
      "print(summ); co.head()",
      "import matplotlib.pyplot as plt\n"
      "s = co.dropna(subset=['rel_time']).sample(300, random_state=0)\n"
      "fig, ax = plt.subplots(figsize=(6, 4))\n"
      "ax.errorbar(s.theta, s.rel_time, xerr=[s.theta - s.theta_lo, s.theta_hi - s.theta], yerr=[s.rel_time - s.rel_time_lo, s.rel_time_hi - s.rel_time],\n"
      "            fmt='o', ms=3, color='#2a78d6', ecolor='#c8d9f2', elinewidth=0.8)\n"
      "ax.axhline(0, color='#52514e', ls='--', lw=1); ax.set(xlabel='Ability (logits)', ylabel='Relative time (SD)', title='SYNTHETIC demo')\n"
      "plt.show()\n"
      "print(chr(10).join(LIMITATIONS))",
      "# 2. Real analysis (needs the PISA extracts; about 10-40 min per form)\n"
      "import subprocess, sys\n"
      "if not DEMO:\n"
      "    for args in [['--part', 'dev'], ['--part', 'confirm', '--form', 'math_formB'], ['--part', 'confirm', '--form', 'math_formC']]:\n"
      "        r = subprocess.run([sys.executable, 'scripts/run_continuous_eval.py'] + args, capture_output=True, text=True)\n"
      "        print(r.stdout[-3000:]); r.check_returncode()",
      "# 3. Saved PISA 2025 results (aggregate outputs committed to the repository)\n"
      "import json, pandas as pd\nfrom pathlib import Path\nfrom IPython.display import display, Markdown, Image\n"
      "R = Path('results/continuous')\n"
      "for part, form in [('dev', 'math_formA'), ('confirm', 'math_formB'), ('confirm', 'math_formC')]:\n"
      "    f = R / f'{part}_{form}_scores.csv'\n"
      "    if not f.exists():\n        print('missing', f); continue\n"
      "    display(Markdown(f'**{part}, {form}**'))\n"
      "    display(pd.read_csv(f)); display(pd.read_csv(R / f'{part}_{form}_paired.csv'))\n"
      "    print(json.dumps(json.loads((R / f'{part}_{form}_summary.json').read_text()), indent=1)[:3000])",
      "for f in ['fig16_continuous_eval.png']:\n"
      "    p = Path('results/figures') / f\n    display(Image(str(p))) if p.exists() else print('missing', p)",
      "# 4. Streamlit prototype: run locally with  streamlit run app/streamlit_app.py\n"
      "# (in Colab, Streamlit needs a tunnel; the back-end used above is the same code: mcq_profiles.dashboard)"]),
    ("10_temporal_regimes", "Latent time regimes vs flexible continuous time distributions (option C)",
     "Test, falsifiably, whether the residual time heterogeneity reflects discrete latent time regimes or a continuous "
     "non-normal distribution: spline (GAM-type) location and variance, skew-normal, Student t and sinh-arcsinh residuals "
     "versus K = 2, 3 regimes with ordered intercepts and ability-dependent prevalence, all conditional on the same IRT "
     "response model. Null simulations, positive control, bootstrap stability, and an independent test on Forms B and C.",
     ["# 1. Synthetic demonstration: regimes are found when present and not when the distribution is merely skewed\n"
      "import numpy as np\nfrom scipy.special import expit\nfrom scipy.stats import skewnorm\n"
      "from mcq_profiles.latent_classes import ContinuousIRT\n"
      "from mcq_profiles.temporal import ContinuousTimeModel, RegimeTimeModel, item_loglik_grid\n"
      "rng = np.random.default_rng(0); n, J = 3000, 20\n"
      "m = ContinuousIRT(dims=1, guessing=True); m.A = rng.uniform(0.8, 1.8, (J, 1)); m.d = rng.normal(0, 1, J); m.c = rng.uniform(.05, .2, J)\n"
      "th = rng.normal(size=n); Y = (rng.random((n, J)) < m.c + (1 - m.c) * expit(th[:, None] * m.A[:, 0] + m.d)).astype(float)\n"
      "LY = item_loglik_grid(m, Y); g = np.zeros(n, int); x = m.X[:, 0]\n"
      "for name, v in [('true regimes', -0.3 * th + np.where(rng.random(n) < .3, 1.2, 0) + .3 * rng.normal(size=n)),\n"
      "                ('skewed continuum', -0.3 * th + .5 * skewnorm.rvs(4, size=n, random_state=1))]:\n"
      "    c = ContinuousTimeModel('shash', 'spline', 'spline').fit(LY, v, g, x)\n"
      "    r = RegimeTimeModel(2, n_starts=3).fit(LY, v, g, x)\n"
      "    print(f'{name:18s} BIC continuous {c.bic():8.1f}  BIC 2 regimes {r.bic():8.1f}  ->', 'regimes' if r.bic() < c.bic() else 'continuous')",
      "# 2. Real analysis (needs the PISA extracts): development, nulls, bootstrap, confirmation\n"
      "import subprocess, sys\n"
      "if not DEMO:\n"
      "    for args in [['--part', 'dev'], ['--part', 'null'], ['--part', 'boot'], ['--part', 'confirm', '--form', 'math_formB'],\n"
      "                 ['--part', 'confirm', '--form', 'math_formC']]:\n"
      "        r = subprocess.run([sys.executable, 'scripts/run_temporal.py'] + args, capture_output=True, text=True)\n"
      "        print(r.stdout[-3000:]); r.check_returncode()",
      "# 3. Saved PISA 2025 results\n"
      "import json, pandas as pd\nfrom pathlib import Path\nfrom IPython.display import display, Markdown, Image\n"
      "R = Path('results/temporal')\n"
      "for f in sorted(R.glob('*.csv')):\n    display(Markdown(f'**{f.name}**')); display(pd.read_csv(f))\n"
      "for f in sorted(R.glob('*.json')):\n    display(Markdown(f'**{f.name}**')); print(json.dumps(json.loads(f.read_text()), indent=1)[:3000])",
      "p = Path('results/figures/fig17_temporal_regimes.png')\ndisplay(Image(str(p))) if p.exists() else print('missing', p)",
      "p = Path('reports/temporal_regimes_report.md')\ndisplay(Markdown(p.read_text(encoding='utf8'))) if p.exists() else print('missing', p)"]),
]


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, title, goal, cells in BOOKS:
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n**Objective.** {goal}\n\n{DATA_NOTE}\n\n"
                                             "Errors are raised, never swallowed. Every output is written to `results/` with the form name; "
                                             "outputs from synthetic data are prefixed `demo_`."),
                    nbf.v4.new_code_cell(SETUP.replace("__REPO_URL__", url))] + [nbf.v4.new_code_cell(c) for c in cells]
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
        nbf.write(nb, NB / f"{name}.ipynb")
    print("written", len(BOOKS))


if __name__ == "__main__":
    main()

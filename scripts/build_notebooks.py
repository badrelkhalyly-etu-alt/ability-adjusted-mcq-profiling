"""Generate the Colab notebooks (thin, reproducible wrappers around src/ and scripts/)."""
import nbformat as nbf
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = r'''# --- Environment (Colab or local) -------------------------------------------
# Set REPO_URL to the GitHub repository once it exists (it is NOT public yet; no URL is assumed here).
REPO_URL = ""          # e.g. "https://github.com/<owner>/ability-adjusted-mcq-profiling.git"
DEMO = True            # True = synthetic demonstration (no PISA data needed); False = real PISA analysis
import os, sys, subprocess, platform
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not REPO_URL:
        raise SystemExit("Set REPO_URL (repository not yet published) or upload the repository folder to /content.")
    if not os.path.exists("ability-adjusted-mcq-profiling"):
        subprocess.run(["git", "clone", REPO_URL], check=True)
    os.chdir("ability-adjusted-mcq-profiling")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
else:
    while not os.path.exists("configs") and os.getcwd() != os.path.dirname(os.getcwd()):
        os.chdir("..")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy, scipy, pandas, sklearn
print(platform.python_version(), "numpy", numpy.__version__, "scipy", scipy.__version__,
      "pandas", pandas.__version__, "sklearn", sklearn.__version__)
FORM = "demo" if DEMO else "math_formA"
print("Mode:", "SYNTHETIC DEMONSTRATION - not empirical results" if DEMO else "REAL PISA 2025 data")'''

RUN = r'''import subprocess, sys
def run(stage):
    cmd = [sys.executable, "scripts/run_experiments.py", "--stage", stage] + (["--demo"] if DEMO else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-6000:]);
    if r.returncode: print(r.stderr[-4000:]); raise RuntimeError(f"stage {stage} failed")'''

DATA_NOTE = ("**Data.** Real mode needs the PISA 2025 public-use files processed as described in `data/README.md` "
             "(licensed by the OECD and never committed). For Colab, upload `data/processed/math_form*.parquet/.json` and "
             "`student_external.parquet` to the same relative paths, or mount Google Drive. "
             "Demo mode generates synthetic data. **Demo outputs are not findings.**")

BOOKS = [
    ("00_environment_colab", "Environment check", "Verify the installation, versions and test suite.",
     [RUN, "r = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests'], capture_output=True, text=True); print(r.stdout[-3000:])"]),
    ("01_data_audit", "Data audit", "Validate the analytic form. Checks: duplicates, missing values, item codes, score consistency, timing anomalies and the extremes of total time.",
     [r'''import pandas as pd, numpy as np, json
from run_experiments import load_form
df, meta = load_form(FORM, DEMO)
items = [c for c in df.columns if c.startswith("I") and c[1:].isdigit()]
print("rows", len(df), "items", len(items), "fingerprint", meta.get("fingerprint"))
print("duplicated ids:", df.duplicated(["CNT", "CNTSTUID"]).sum())
print("item codes:", pd.unique(df[items].values.ravel()))
print("score consistency:", np.allclose(df[items].sum(1, skipna=True), df.final_score))
print(df.total_duration_minutes.describe(percentiles=[.01,.05,.5,.95,.99]))
print("missing total time:", df.total_duration_minutes.isna().sum(), " >60 min:", (df.total_duration_minutes > 60).sum())''']),
    ("02_baseline_clustering", "E1: Reproduction of the source pipeline (K-means on score and time)",
     "K-means (k = 2..8), internal indices, 5-fold held-out silhouette, bootstrap ARI/Jaccard, log-time variant, GMM/BIC, agglomerative, DBSCAN/HDBSCAN, Hopkins statistic, and a no-cluster null distribution of the silhouette.",
     [RUN, "run('baseline')", "import json; print(json.dumps(json.load(open(f'results/metrics/{FORM}_E1_baseline.json'))['null_silhouette'], indent=1)[:800])"]),
    ("03_irt_models", "E2: Homogeneous IRT and the model ladder (training split)", "Fit M0, A_K, M1_K, M2, M3_K, M4_K and LCA_K on the 70% training split.",
     [RUN, "run('fit')", "import pandas as pd; pd.read_csv(f'results/tables/{FORM}_model_selection_train.csv').sort_values('BIC')"]),
    ("04_lca_mixirt", "E3/E7: Ability-adjusted classes and their stability", "Class-specific difficulty deviations after ability adjustment, plus bootstrap stability.",
     [RUN, "run('stability')", "import json; print(json.load(open(f'results/metrics/{FORM}_E7_stability.json'))['ari_median'])"]),
    ("05_joint_duration_model", "E4/E10: Incremental value of time and masked-item prediction",
     "Held-out masked-item log loss, Brier score and calibration. Duration is measured on unmasked items only, so the timing of masked items cannot leak into the prediction.",
     [RUN, "run('e10')", "import pandas as pd; display(pd.read_csv(f'results/tables/{FORM}_E10_test_scores.csv')); pd.read_csv(f'results/tables/{FORM}_E10_paired.csv')"]),
    ("06_null_simulation", "E5/E6: Null (no classes) and positive-control simulations",
     "False-positive rate of the class-selection rule under a continuous-ability generator fitted to the data; power and recovery under known classes.",
     [RUN, "run('null')", "run('positive')"]),
    ("07_robustness_ablation", "E8/E9/H5: Within-score analysis, DIF and external criterion",
     "Within-score heterogeneity against a null, logistic DIF (OECD and gender), per-subgroup class enumeration, and self-reported effort (EFFORT1) as an external criterion.",
     [RUN, "run('e8')", "if not DEMO:\n    run('dif'); run('external')"]),
    ("08_figures_results", "Figures, LaTeX tables and the reproducibility report", "Rebuild every figure and table from the saved outputs.",
     ["import subprocess, sys; print(subprocess.run([sys.executable, 'scripts/build_results.py'] + (['--demo'] if DEMO else []), capture_output=True, text=True).stdout[-3000:])"]),
]


def main():
    for name, title, goal, cells in BOOKS:
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n**Objective.** {goal}\n\n{DATA_NOTE}\n\n"
                                             "Errors are raised, never swallowed. Every output is written to `results/` with the form name; outputs from synthetic data are prefixed `demo_`."),
                    nbf.v4.new_code_cell(SETUP)] + [nbf.v4.new_code_cell(c) for c in cells]
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
        nbf.write(nb, NB / f"{name}.ipynb")
    print("written", len(BOOKS))


if __name__ == "__main__":
    main()

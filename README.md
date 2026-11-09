# ability-adjusted-mcq-profiling

**Are score–time "behavioural profiles" real classes of examinees, and what should replace them?**
An empirical study on the PISA 2025 mathematics test (OECD public-use files).

Online exams record each student's score and completion time, and learning-analytics tools often cluster these two numbers into "profiles" (fast and accurate, slow and weak, …). This repository contains everything needed to test that practice and to reproduce our results:

1. **Validation of profiles against continuous models.** Every profile-detection rule is calibrated on data simulated from continuous models with no classes, fitted to the same test.
2. **A continuous representation with uncertainty.** Each student gets an ability (item response model with guessing) and a relative time (adjusted for ability, booklet and the items reached), each with an interval.
3. **A test of latent time regimes** against flexible, skewed and heavy-tailed continuous time distributions.
4. **Notebooks** (Google Colab), a test suite and a **dashboard prototype**.

All empirical results come from the **PISA 2025 public-use files**. Anything named `demo_*` comes from synthetic data and is a software check, not a finding.

---

## Main results

| Question | Answer on PISA 2025 |
|---|---|
| Do score–time profiles show distinct classes? | **No.** K-means and latent class analysis find four profiles even in simulated data that has no classes. |
| Do ability-adjusted mixture classes show distinct classes? | **No.** They pass a simple null, but are also "found" in every simulation with guessing or a second dimension, and they do not predict new data better than continuous models. |
| Are there latent time regimes? | **Indeterminate.** A skewed, heavy-tailed continuous distribution explains most of the signal; regimes do not replicate out of sample. |
| Does the continuous representation do better than profiles? | **Yes**, on two untouched test forms evaluated once: better prediction of masked answers (+0.019 to +0.025 nats per response), calibrated intervals (the profiles' intervals are overconfident) and far greater stability (reliability 0.87–0.89 versus profile agreement κ 0.52–0.58). |

What is *not* claimed:
- **No new algorithm.** The models are existing mixture and item-response models.
- **Relative time is not a measure of speed, effort or engagement.** It adds little predictive value, and this small gain disappears with survey weights.
- **Not for high-stakes decisions about individual students.** The study validates no pedagogical outcome.

Hypotheses and decision rules were fixed before analysis: the main protocol at commit `0f30da3`, the extension protocol at commit `77cafcd`. The full numbers are in `results/`.

---

## Quick start in Google Colab

1. Open any notebook from `notebooks/` in Colab (*File → Open notebook → GitHub*, then paste the repository URL).
2. In the first cell, set `REPO_URL` to the repository URL. A private repository needs a personal access token in the URL.
3. Run all cells. The first cell clones the repository, installs the requirements and **uses the PISA 2025 data in `data/processed/` automatically**. Set `DEMO = True` for a fast run on synthetic data.

| Notebook | Content |
|---|---|
| `00_environment_colab` | Installation, versions, test suite |
| `01_data_audit` | Checks on the analytic forms: duplicates, missing values, item codes, timing anomalies |
| `02_baseline_clustering` | Score–time K-means and other clustering methods, stability |
| `03_irt_models` | Item response models (2PL, guessing, two dimensions) |
| `04_lca_mixirt` | Latent class analysis and mixture IRT |
| `05_joint_duration_model` | Joint model of responses and completion time |
| `06_null_simulation` | Simulations from continuous models without classes (false-positive rates) and positive controls |
| `07_robustness_ablation` | Not-reached scoring, survey weights, alternative specifications |
| `08_figures_results` | Figures, tables and final verdict |
| `09_continuous_representation` | The continuous ability × relative-time representation and its evaluation |
| `10_temporal_regimes` | Latent time regimes versus continuous time distributions |

Long stages (model fits, simulations) can take hours on real data. Each notebook also displays the saved results from `results/`, so it can be read without re-running anything.

---

## Local installation

```bash
pip install -r requirements.txt      # or: conda env create -f environment.yml
pytest -q                            # 45 tests
```
Tested with Python 3.14 on Windows 11. Colab's Python 3.11/3.12 works with the nearest package versions.

---

## Data

`data/processed/` contains the analysis-ready extracts of the PISA 2025 public-use files:

| File | Content |
|---|---|
| `math_formA.parquet`, `math_formB.parquet`, `math_formC.parquet` (+ `.json`) | The three mathematics forms used in the study: item responses, item times, totals. The `.json` file records the item map, coding rules and a SHA-256 fingerprint |
| `student_external.parquet` | Effort, perseverance, socio-economic index (ESCS), gender, final student weight `W_FSTUWT` |
| `cog_linear_items.parquet`, `process_linear_items.parquet` | Linear-path item responses and item timings |
| `cog_design.parquet`, `cog_presence.parquet`, `cog_scored_items.csv` | Booklet design and item presence |
| `item_labels.csv`, `timing_label_examples.csv` | Variable labels |

**Columns of a form.**
- Identifiers: `student_id_anonymous` (a hash), `CNT`, `CNTSCHID`, `OECD`, `BOOKID`.
- Responses: `I01…`, coded 0/1, with NaN for not reached.
- Item times in seconds: `RT01…`.
- Totals: `total_duration_minutes`, `final_score`.
- Counts: `n_not_reached`, `n_technical`, `n_time_missing`.

**Source.** OECD (2026), *PISA 2025 Database*, public-use files, release `20260806`. The files contain no names; students are identified only by anonymous OECD codes. Use of the data is governed by the OECD terms of use, and the MIT licence below covers the code only.

**Rebuilding from the OECD files (optional).** The raw SPSS files are 1–13 GB each and are not in the repository.
1. Download `CY09_MS_COG_PUF.zip`, `CY09_MS_COG_PROCESS_PUF.zip` and `CY09_MS_STU_PUF.zip` from the OECD PISA database page.
2. Unzip them into `data/raw/`. Use 7-Zip, because some archives use Deflate64.
3. Run the extraction:
   ```bash
   python scripts/extract_items.py
   python scripts/extract_student.py
   python -c "import sys; sys.path.insert(0,'src'); from mcq_profiles.data import *; c,p=load_extracts(); [save_form(*build_form(c,p,rank=r), n) for r,n in [(0,'math_formA'),(1,'math_formB'),(2,'math_formC')]]"
   ```
Item timing is withheld for the new science items, which is why the study uses mathematics.

---

## Reproducing the analyses

**Main study (Form A, replication on Forms B and C):**
```bash
python scripts/run_experiments.py --stage fit
for s in baseline stability e8 dif external e3b e3c null positive replication e10; do
  python scripts/run_experiments.py --stage $s
done
python scripts/build_data_table.py
python scripts/build_results.py
```

**Continuous representation:**
```bash
python scripts/run_continuous_eval.py --part dev
python scripts/run_continuous_eval.py --part confirm --form math_formB   # then math_formC
```

**Time regimes:**
```bash
python scripts/run_temporal.py --part dev
python scripts/run_temporal.py --part null
python scripts/run_temporal.py --part boot
python scripts/run_temporal.py --part confirm --form math_formB          # then math_formC
```

**Summary tables and figures for both extensions:**
```bash
python scripts/build_extension_outputs.py
```

**Synthetic smoke test (a few minutes):**
```bash
python scripts/run_experiments.py --stage all --demo
```

**Running options.**
- `AALBP_N_JOBS` sets the number of parallel workers (default 8).
- Simulation stages save a checkpoint after each replication in `results/checkpoints/`, so an interrupted run resumes where it stopped.

**Test splits.**
- Each form has a training split and a test split.
- The Form A test split was used once by stage `e10`.
- The Form B and C test splits were used once by the `confirm` parts.
- Re-running these stages reproduces the published numbers. A new hypothesis tested on the same splits would no longer be an independent validation.

---

## Dashboard prototype

```bash
streamlit run app/streamlit_app.py
```
- **What it shows:** each student's ability and relative time, each with an interval. It can compare two students: if their intervals overlap, the data cannot tell them apart.
- **What it does not do:** it assigns **no profiles, no psychological labels and no recommendations**, and it always shows the limits of interpretation.
- **Data:**
  - it opens on synthetic demonstration data;
  - you can load your own CSV, with columns `I1…IJ` (0/1, empty = not reached), `RT1…RTJ` (seconds), and optionally `student_id` and `booklet`;
  - the computation runs locally.

---

## Repository layout

```
configs/            experiment settings and seeds
data/processed/     PISA 2025 analysis-ready extracts (see Data)
src/mcq_profiles/   library
  data.py             build the linear forms from the PISA extracts
  irt.py              item response models
  latent_classes.py   mixture IRT, latent class analysis, continuous IRT (1D/2D, guessing)
  duration.py         completion-time models
  baseline.py         score–time clustering pipeline and stability
  simulation.py       null and positive-control generators
  statistics.py       person fit, DIF, multiple testing, bootstrap
  evaluation.py       masked-item log loss, Brier score, calibration
  continuous_map.py   continuous ability × relative-time representation with intervals
  temporal.py         continuous time distributions and latent time-regime models
  dashboard.py        back-end of the prototype
scripts/            extraction, experiment stages, figures and tables
notebooks/          Colab notebooks 00–10
results/            metrics (JSON), tables (CSV), figures (PNG/SVG/PDF)
app/                Streamlit prototype
tests/              pytest suite
```

---

## Licence and citation

- **Licence.** The code is under the MIT licence (`LICENSE`). The PISA data remain under the OECD terms of use.
- **Citation.** See `CITATION.cff`.
- **Origin of the question.** The score–time profiling question was raised in an unpublished manuscript by N. Safeh.

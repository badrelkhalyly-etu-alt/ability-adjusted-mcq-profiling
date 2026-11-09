# ability-adjusted-mcq-profiling

This repository tests whether score–time "behavioural profiles" of online multiple-choice examinations are classes beyond ability. It does so with null-calibrated mixture IRT.

Using PISA 2025, it asks whether the profiles found by K-means on (score, total time) survive four tests:
- adjustment for continuous ability;
- calibration against a no-class null;
- held-out prediction;
- replication.

> **Data scope.** All empirical results here come from the **PISA 2025 public-use files** (linear-path mathematics forms). Any output named `demo_*` comes from synthetic data and is a software test only.

## Extensions (second frozen protocol, confirmed on untouched Form B and C test splits)
- **Continuous, uncertainty-aware representation (CTR).** Each student gets an IRT ability with guessing and a relative time adjusted for ability, booklet, the items reached and their time intensities, both with intervals. Code: `src/mcq_profiles/continuous_map.py`; evaluation: `scripts/run_continuous_eval.py`; notebook 09.
- **Latent time regimes versus flexible continuous time distributions.** Six continuous families (spline/GAM-type, skew-normal, Student t, sinh-arcsinh) are compared with regime models, under the same response model. Code: `src/mcq_profiles/temporal.py`; analysis: `scripts/run_temporal.py`; notebook 10; report: `reports/temporal_regimes_report.md`.
- **Prototype dashboard.** `streamlit run app/streamlit_app.py`; see `app/README.md`. It runs on synthetic demonstration data or on a local file, and shows no labels.
- Protocol: `reports/protocol_continuous_temporal.md`. Critical audit: `reports/critical_audit.md`. Submission readiness: `reports/q1_readiness.md`.

## What is (and is not) claimed
- **No new algorithm.** The models are existing mixture-IRT and log-normal duration models. They are organised as a ladder (M0, A_K, M1_K, M2, M3_K, M4_K) and calibrated against simulations from a continuous model fitted to the same assessment.
- **Hypotheses were registered first.** H1–H5 and the decision rules were frozen before any model was fitted to item responses. See `reports/experimental_protocol.md`, commit `e54a29f`. Later deviations are logged in the same file.
- **Negative results are reported as such.** See `reports/scientific_verdict.md` once the run is complete.

## Layout
```
configs/        experiment settings and seeds (frozen with the protocol)
src/mcq_profiles/
  data.py           build fixed linear forms from the PISA extracts
  latent_classes.py MixIRT (2PL/Rasch mixtures, class-specific b, censored log-normal duration),
                    LCA, ContinuousIRT (1D/2D, optional guessing)
  baseline.py       the score–time pipeline (K-means, GMM, Ward, DBSCAN/HDBSCAN, stability)
  simulation.py     null and positive-control generators
  statistics.py     Q3, l_z, logistic DIF, BH, Clopper–Pearson, Cramér's V, bootstrap
  evaluation.py     masked-item log loss / Brier / calibration
scripts/        extraction, experiment stages (E1–E10), figures/tables, notebooks
notebooks/      00–10 (Colab-ready; use the PISA 2025 extracts in data/processed/, or DEMO = True for synthetic data)
results/        metrics (JSON), tables (CSV), figures (PDF/SVG/PNG); checkpoints are git-ignored
reports/        manuscript audit, protocol + deviation log, reproducibility report, verdict
literature/     review 2024–2026, Scopus verification, related-work and novelty matrices
tests/          pytest suite (estimator recovery, identification, data coding, reproducibility)
```

## Reproduce
```bash
pip install -r requirements.txt      # or: conda env create -f environment.yml
pytest -q                            # unit tests; no licensed data needed
python scripts/run_experiments.py --stage all --demo   # synthetic smoke test (outputs demo_*)
```
To run the full PISA 2025 analysis:
1. The analysis-ready PISA 2025 extracts are already in `data/processed/` (see `data/README.md` to rebuild them from the OECD files).
2. Run the stages in order:
   ```bash
   python scripts/run_experiments.py --stage fit          # model ladder on the Form A training set
   for s in baseline stability e8 dif external e3b e3c null positive replication e10; do
     python scripts/run_experiments.py --stage $s
   done
   python scripts/build_data_table.py
   python scripts/build_results.py                       # figures, LaTeX tables, reproducibility report
   ```
`AALBP_N_JOBS` sets the number of parallel workers (default 8). BLAS threads are pinned to 1 per worker. The simulation stages (null, positive) checkpoint each replication under `results/checkpoints/`, so an interrupted run resumes where it stopped.

The test set is used only by stage `e10`. The protocol requires that stage to be run once, after all configuration is committed.

## Data and ethics
- `data/processed/` holds anonymised extracts of the PISA 2025 public-use files (OECD), so the notebooks run in Colab from a clone. The raw `.sav` files (1–13 GB each) and model checkpoints are not committed.
- The manuscript is kept outside this repository.
- PISA public-use files are anonymised. Redistributing them is governed by the OECD terms of use.
- Profiles must not be used for high-stakes decisions about individual students. The study validates no pedagogical outcome.

## Licence and citation
Code: MIT (see `LICENSE`). The licence does not cover the PISA data. The score–time profiling question was raised in an unpublished manuscript by N. Safeh (cited in the paper).

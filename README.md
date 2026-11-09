# ability-adjusted-mcq-profiling

This repository tests whether score–time "behavioural profiles" of online multiple-choice examinations are classes beyond ability. It does so with null-calibrated mixture IRT.

Using PISA 2025, it asks whether the profiles found by K-means on (score, total time) survive four tests:
- adjustment for continuous ability;
- calibration against a no-class null;
- held-out prediction;
- replication.

> **Data scope.** All empirical results here come from the **PISA 2025 public-use files** (linear-path mathematics forms). Any output named `demo_*` comes from synthetic data and is a software test only.

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
notebooks/      00–08 (Colab-ready; run in DEMO mode without licensed data)
results/        metrics (JSON), tables (CSV), figures (PDF/SVG/PNG); checkpoints are git-ignored
manuscript/     LaTeX article (article.tex + sections/, tables/)
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
1. Obtain the public-use files and extract them as described in `data/README.md`.
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
- No student-level data are committed. `data/raw/`, `data/processed/`, `*.sav`, `*.parquet` and checkpoints are git-ignored.
- PISA public-use files are anonymised. Redistributing them is governed by the OECD terms of use.
- Profiles must not be used for high-stakes decisions about individual students. The study validates no pedagogical outcome.

## Licence and citation
Code: MIT (see `LICENSE`). The licence does not cover the PISA data. The score–time profiling question was raised in an unpublished manuscript by N. Safeh (cited in the paper).

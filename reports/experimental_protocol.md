# Experimental protocol (pre-registered)

**Frozen:** 2026-11-08, before any model was fitted to real PISA item responses.
**Already inspected at freeze time (descriptives only):** item p-values, total-time quantiles, time–score correlation, and the booklet-level medians of Forms A–C (see `reports/data_audit.md`).
**Change rule:** any deviation is logged in §9 with its date and reason. Confirmatory results are reported under this protocol even if later analyses are added.

## 1. Data

| Role | Data | n | Items | Timing |
|---|---|---|---|---|
| **Primary (confirmatory)** | PISA 2025, mathematics linear path, most frequent fixed form (**Form A**) | 5,680 students, 82 countries | 22 (5 polytomous items dichotomised to full credit) | item-level, total = sum |
| Replication 1 | Form B (2nd most frequent) | 5,375 | 25 | item-level |
| Replication 2 | Form C (3rd most frequent) | 4,798 | 27 | item-level |
| External validators (never used in fitting) | STU questionnaire: EFFORT1 (self-reported effort on the test, 1–10), PERSEV, ESCS, gender, grade | — | — | — |

Why the linear math path: (i) every student on a form saw the same items, as in the source MCQ exam; (ii) in the 2025 PUF, item timing is withheld for the new science items (CACERS*); (iii) adaptive paths make item exposure depend on earlier performance, which confounds pattern-based classes.

Coding rules (primary):
- 0/1 credit.
- Polytomous items: 1 = maximum code.
- Not reached → missing (PISA convention).
- Technical issue → missing.
- Total time = sum of item times in minutes; not-reached items contribute 0.
- Students with no timing at all keep their responses but are excluded from the duration likelihood (logT unobserved).

Sensitivity rules:
- not reached = incorrect;
- polytomous: any credit = 1;
- trimming at the 0.5 % and 99.5 % time quantiles versus no trimming;
- nominal censoring at 60 min versus none.

No observation is deleted silently. Every exclusion is counted in `results/tables/`.

**Unweighted analysis.** The estimand is the structure of *this* form's response data, not a population parameter. Survey weights (W_FSTUWT) are used only in a sensitivity re-fit of the selected model.

## 2. Split

Form A is split by student into **train 70 % / test 30 %**, stratified by country, with seed `20261108`.
- All model fitting, selection and tuning use the train set only.
- The test set is evaluated **once**, after `configs/experiments.yaml` is committed.

**Masked-item prediction task (E10, H1/H2).**
- One fixed random set of 30 % of the item positions is masked: 7 of 22 items for Form A (seed 20261108).
- This is repeated for 5 mask sets (seeds +1…+4), and results are averaged.
- Test students' latent posteriors use only unmasked responses.
- The duration variable for E10 is **time on unmasked items only**, so the time spent on a masked item cannot leak its correctness.
- Item parameters for masked items come from train students only.

## 3. Models

| Id | Responses | Latent | Duration |
|---|---|---|---|
| M0 | 2PL | θ ~ N(0,1) | — |
| A_K | 2PL, items shared | θ ~ Σ_c π_c N(μ_c, σ_c²): flexible *continuous* ability, "type A" classes | — |
| M1_K | 2PL, class-specific b_cj (ridge 1.0; equal mean difficulty across classes), "type B" | θ \| c ~ N(μ_c, σ_c²) | — |
| M2 | M0 | — | log T \| θ ~ N(α + γ_book + βθ, s²) |
| M3_K | M1_K | — | log T \| θ, c ~ N(α_c + γ_book + β_c θ, s_c²) |
| M4_K | M3_K | + booklet-specific class prevalence | as M3 |
| LCA_K | unrestricted latent classes | — | — |

- Baselines on (score, time), reproducing the source study: K-means (k = 2…8, z-scores), GMM (full covariance, BIC), agglomerative (Ward), HDBSCAN.
- K ∈ {2, 3} for mixtures, extended to 4 only if K = 3 is selected. LCA K = 1…6.
- 5 random starts. A fit counts as converged only if the relative change is < 1e-8 within 1,500 iterations. Non-converged fits are flagged and never selected silently.

## 4. Hypotheses, estimands and decision rules

**H1 (residual heterogeneity).** H1 is supported only if **all** of the following hold:
- (a) Train BIC prefers some M1_K over both M0 and the best A_K.
- (b) The calibrated test holds: the observed train ΔBIC (best A or M0 minus selected M1_K) exceeds the 95th percentile of ΔBIC across ≥ 40 parametric-bootstrap null datasets. These are generated from the fitted **best continuous model** (M0 or A_K, chosen by BIC) with the same N and missingness pattern.
- (c) Test masked-item log loss of the selected M1_K is lower than that of the best continuous model, and the 95 % paired bootstrap CI (2,000 resamples of students) excludes 0.
- (d) The smallest class is ≥ 5 % and the relative entropy is ≥ 0.6.

**H2 (incremental value of total time).** Supported if:
- the test masked-item log loss of M3 (or M2 when H1 fails) is lower than that of M1 (or M0), with the CI excluding 0; **and**
- the improvement is ≥ 0.001 nats/item (practical threshold, fixed now).

Secondary: the held-out log-density of log-time under M2/M3 versus an intercept-only log-normal.

**H3 (robustness).** Evaluated only if H1 holds:
- the median matched ARI over 20 bootstrap refits is ≥ 0.5;
- the correlation of class-specific difficulty-deviation profiles across refits is ≥ 0.7;
- the same K is selected on Form B or C.

**H4 (non-reducibility).** Evaluated if H1 holds. Classes must not be reproduced by:
- score tertiles, θ quintiles, booklet or OECD status: Cramér's V < 0.5 for each;
- the K-means score–time profiles: ARI < 0.5.

**H5 (external informativeness).** Regress EFFORT1 on EAP θ, score, total time and booklet:
- (i) with K-means profile dummies added;
- (ii) with posterior class probabilities added.

Report ΔR² and the cluster-robust (school) F-test. This tests construct relevance only; pedagogical usefulness stays **unverified** because no pedagogical outcome is available.

**Verdict mapping.**
- A: H1–H4 hold and H5 shows a non-trivial association.
- B: H1 holds but H3 or H4 fails, or H1 holds only in part (a)+(c).
- C: H1(a–c) fails and continuous models suffice.
- D: computational or identification failure prevents a decision.

## 5. Simulations

**E5 null (false positives).**
- Generator: the fitted best continuous model on Form A train, with its θ distribution and the duration model of M2 (booklet effects included), and the observed missingness pattern copied row-wise.
- R = 40 replications for config N1 (the fitted model).
- R = 40 replications for config N2: the same, but θ drawn from a skewed (sinh-arcsinh) distribution, to mimic non-normal ability.
- For each replication, record:
  - the selected model;
  - ΔBIC;
  - K-means silhouette at k = 4;
  - the LCA K chosen by BIC.
- Reported: false-positive rate = P(selection rule picks a class model) with its Clopper–Pearson CI.

**E6 positive control.** Grid of:
- class difficulty-shift magnitude {0.5, 1.0, 1.5} logits on half the items (centred);
- mixing proportion {0.5, 0.2};
- N {2,000, 5,000};
- J = 22, using Form A item parameters;
- duration informativeness {none, class intercept shift 0.3 log-min}, only at magnitude 1.0;
- R = 10 replications per cell.

Reported: power, recovered K, ARI, RMSE of b-deviations, and runtime.

## 6. Other experiments

- **E1:** reproduction of the source pipeline on Form A, reporting silhouette, Davies–Bouldin, Calinski–Harabasz, bootstrap ARI, a log-time variant and a GMM/BIC variant.
- **E7:** stability (see H3).
- **E8:** within-score analysis. Among students with equal total score (restricted to scores with n ≥ 100):
  - compare item-pattern log-likelihood under M0 versus M1 (person fit l_z);
  - check whether class posteriors vary at a fixed score beyond what θ uncertainty implies, by comparing their variance with that from M1-simulated data at the same score.
- **E9:** DIF and invariance.
  - Logistic-regression DIF (rest-score matching; OECD vs non-OECD, gender, booklet position), with Benjamini–Hochberg FDR at 0.05 and the ΔNagelkerke R² effect size.
  - Class enumeration re-run separately in OECD and non-OECD subsets.
  - M4 versus M3.

## 7. Software

Python 3.14 (local) / 3.11+ (Colab), numpy, scipy, pandas, scikit-learn, pyreadstat, statsmodels, matplotlib, joblib. The mixture-IRT estimator is our own EM implementation (`src/mcq_profiles/latent_classes.py`), validated by the unit tests and by E6.

## 8. Claims we will not make

Psychological or affective labels (anxious, bored, disengaged) for classes. EFFORT1 is self-reported effort, so it supports, at most, "associated with self-reported effort".

## 9. Deviation log

| Date | Deviation | Reason |
|---|---|---|
| — | — | — |

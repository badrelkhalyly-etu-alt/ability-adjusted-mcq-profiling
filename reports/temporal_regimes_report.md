# Latent time regimes in PISA 2025: final report (option C)

**Decision under the frozen rule (`reports/protocol_continuous_temporal.md`, commit `fe7dab6`): INDETERMINATE for both time variables.** Regimes are neither supported nor rejected. The evidence leans towards a non-normal continuum: a well-fitting continuous distribution explains most of the signal, and what remains does not predict new students reproducibly.

*Data: PISA 2025 public-use files, linear mathematics forms. Development: Form A training split (internal 70/30 split). Validation: the untouched Form B and C test splits. Machine-readable outputs: `results/temporal/`, `results/tables/temporal_summary.csv`, `results/metrics/extension_decisions.json`. Figure: `results/figures/fig17_temporal_regimes.*`.*

## 1. Question
The paper reported that class duration models beat a cubic heteroscedastic *normal* time model by 497–958 BIC points. Is that evidence of discrete time regimes, or of a continuous, non-normal time distribution that the normal model missed?

## 2. What explains the residual advantage
Comparisons are made with the same IRT response model (G3PL), so likelihoods are strictly comparable. Replacing the normal cubic model by a flexible continuous distribution improves training BIC as follows (column "best continuous vs normal cubic" in `temporal_summary.csv`):

| Data | Relative time | Total time |
|---|---|---|
| Form A (development) | 220 | 158 |
| Form B | 193 | 225 |
| Form C | 185 | 221 |

The best continuous model is the SHASH distribution (sinh-arcsinh, with skewness and tail weight), with spline location and spline log-scale in θ, on every form and for both variables. **The bulk of the "residual heterogeneity" is the shape of a continuous distribution** (skew and tails), not discreteness.

## 3. Regimes against the best continuous model

| Data | Variable | ΔBIC (continuous − regimes; > 0 favours regimes) | Held-out gain of regimes (nats/student, 95 % CI) | Weighted held-out gain |
|---|---|---|---|---|
| A, development | relative | −12.5 | −0.004 [−0.012, 0.004] | 0.008 [−0.011, 0.030] |
| A, development | total | −25.3 | −0.011 [−0.028, 0.001] | −0.007 [−0.036, 0.017] |
| **B, test** | relative | 21.9 | −0.005 [−0.013, 0.003] | **−0.023 [−0.038, −0.008]** |
| **B, test** | total | 37.3 | −0.009 [−0.018, 0.001] | −0.005 [−0.023, 0.014] |
| **C, test** | relative | 56.3 | 0.002 [−0.007, 0.010] | 0.003 [−0.015, 0.020] |
| **C, test** | total | −6.3 | **0.006 [0.001, 0.012]** | **0.017 [0.007, 0.028]** |

ΔBIC is computed on each form's training split. The held-out gain uses its untouched test split, or the internal evaluation part for Form A.

**Calibration.** The probability integral transform (PIT) is well calibrated for both model types. The 90 % PIT coverage is 0.89–0.92, and the KS distance is 0.02–0.05.

## 4. Calibration of the decision (Form A development split)

| Generator (20 replications each) | Regimes win on BIC | ΔBIC, 95th percentile |
|---|---|---|
| SHASH, spline | 0.00 | −31.4 |
| t, spline | 0.00 | −9.2 |
| skew-normal, spline | 0.00 | −35.5 |
| normal cubic, heteroscedastic | 0.05 | −20.9 |
| **2 regimes (positive control)** | **1.00** | 47.0 (median 31.9) |

- **Discrimination.** The procedure separates a true two-regime structure from the four continuous non-normal alternatives on BIC almost perfectly.
- **Positive control, held-out.** The held-out gain had a median of +0.005 nats per student.
- **Bootstrap** (Form A development split, B = 20). The median ARI of regime assignments against the full fit is 0.83. The intercept gap between the two regimes lies in [0.33, 0.48].

## 5. Decision rule, applied as frozen

| Criterion (must hold on both B and C) | Relative time | Total time |
|---|---|---|
| (i) ΔBIC > largest null 95th percentile (−9.2) | met (21.9; 56.3) | met (37.3; −6.3)* |
| (ii) held-out gain > 0, CI excluding 0 | **not met** (B: CI includes 0) | **not met** (B: CI includes 0) |
| (iii) bootstrap stability (ARI ≥ 0.7, gap CI excluding 0) | met | met |
| **Decision** | **indeterminate** | **indeterminate** |

\* The largest null 95th percentile is negative. Criterion (i) is therefore weaker than ΔBIC > 0: Form C total time passes it although the continuous model has the lower BIC. This was noticed after freezing and is reported, not corrected. It does not change any decision, because criterion (ii) fails.

"Rejected" would require criterion (i) to fail on both forms. It does not, because regimes win on BIC in B and C for relative time.

## 6. Interpretation (beyond the rule)
- **Does the BIC signal exceed what continuous nulls produce?** On B and C, yes: 22–56 points against null 95th percentiles of about −9 to −35. However, the nulls were calibrated at the Form A development sample size (n ≈ 2,800), while the B and C training splits are larger (n ≈ 3,400–3,800), and BIC differences grow with n. The comparison is approximate.
- **What the regimes look like.** In every fit, the two regimes split students roughly 45/55 at every ability level. They differ mainly in **spread**: a wide and a narrow regime with log-scales of about −0.9 and −1.6, and intercepts about 0.5 apart. That is a scale mixture, which is the classical way a heavy-tailed continuous distribution is approximated by components. On Form C, the three-regime solution adds a degenerate regime holding 0.4 % of students, with almost zero variance.
- **Prediction.** The regime models do not predict new students' times better than SHASH:
  - on Form B they are slightly worse, significantly so with survey weights;
  - on Form C, total time, they are slightly better, by 0.006 nats per student;
  - no direction replicates.
- **Ability versus composition versus time.**
  - Regimes are defined conditionally on ability, with ability-dependent prevalence, so they are not ability differences.
  - Item composition (which items were reached, and their time intensities) is removed in the relative-time variable. The pattern of results is the same for relative and total time, so composition does not create or remove the regime signal.

**Bottom line.** Discrete time regimes are **not established** in PISA 2025 mathematics. The large residual advantage of class duration models over normal time models is mostly the skewed, heavy-tailed shape of the continuous time distribution. The small remainder could be either a further shape feature or a regime structure; it is in-sample only and does not predict new students reproducibly.

## 7. Limits
- Regimes here are person-level. Item-level states, such as rapid guessing on single items, were not tested and are not ruled out.
- Null calibration used 20 replications per generator, all at the Form A development sample size.
- The response model was fixed at its training estimate. Uncertainty in the item parameters is not propagated.
- Weighted analyses use a pseudo-likelihood. With multi-country weights the effective sample size is small, so these intervals are approximate.

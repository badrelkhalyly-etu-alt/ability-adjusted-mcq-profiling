# Protocol: continuous representation (B) and latent time regimes (C)

**Status:** frozen before any Form B or Form C test-split data were used by these analyses (git commit recorded in the deviation log of `experimental_protocol.md`). Development used only the Form A training split (internal 70/30 split, `run_temporal.dev_split`, seed 11). The Form A test split, already examined in E10, is not used for any new claim.

**Independent validation:** Forms B and C test splits (1,608 and 1,435 students; items and students disjoint from Form A). Models are fitted on each form's training split and evaluated once on its test split. Before this protocol, the B and C training splits were used for model-ladder BIC fits, robustness fits and the label-reliability analysis; their test splits were never evaluated.

---

## Part C: latent time regimes versus flexible continuous time distributions

### Question
Is the residual time heterogeneity (class duration models beat a cubic heteroscedastic normal model by 497–958 BIC points) evidence of discrete latent time regimes, or of a continuous non-normal distribution that the normal models did not capture?

### Variables
- **Primary:** `v_relative`, the item-intensity-adjusted mean log item time over the items each student reached (`continuous_map.relative_time_index`). It removes the effect of which items were reached and of their time intensities (composition).
- **Secondary:** `log_total_time`, total log time on the form (the variable of models M2/M3).

### Common response model
G3PL (2PL with lower asymptotes), fitted once on the training responses. Every time model is conditional on θ on the same 41-node grid with the same response likelihood. Log-likelihoods, BIC values and held-out conditional log-densities log p(v | Y) are therefore strictly comparable across time models.

### Models (`mcq_profiles.temporal`)
| Model | Location in θ | Log-scale in θ | Residual |
|---|---|---|---|
| normal, linear, const | linear | constant | normal (M2 analogue) |
| normal, cubic, linear | cubic | linear | normal (the competitor already reported) |
| normal, spline, spline | B-spline (df 5) | B-spline (df 3) | normal (GAM-type, conditional variance) |
| skew-normal, spline, spline | spline | spline | skew-normal |
| t, spline, spline | spline | spline | Student t (heavy tails) |
| SHASH, spline, spline | spline | spline | sinh-arcsinh (skew and tail weight) |
| regimes K = 2, 3 | regime intercept + slope, shared θ², θ³ | regime log-scale, shared θ slope | normal within regime; prevalence logistic in θ |

All models include booklet intercepts. Regimes are ordered by intercept. Their prevalence depends on θ, so regimes cannot be ability groups by construction. Each regime model is fitted from 5 random starts.

### Selection and comparison
On each training split, the best continuous model and the best regime model are each selected by BIC. The comparison is the held-out per-student mean of log p(v | Y), best regime minus best continuous, with a paired bootstrap 95 % CI (2,000 resamples). The same comparison is repeated with final student weights (`W_FSTUWT`) as a sensitivity analysis.

### Null and positive-control calibration (Form A development split)
Twenty replications each from five generators fitted on the development split:
- four continuous nulls: SHASH-spline, t-spline, skew-normal-spline and normal cubic heteroscedastic;
- one positive control: the fitted 2-regime model.

Each replication is fitted with all eight models. Quantities:
- the rate at which the regime model wins on BIC under each null, which is the false-discovery rate;
- the 95th percentile of ΔBIC (best continuous minus best regime) under the nulls;
- power under the positive control.

### Decision rule (primary variable; the secondary variable is reported with the same rule)
- **Regimes supported:** on **both** Forms B and C, (i) the training ΔBIC (best continuous minus best regime) exceeds the largest null 95th percentile, **and** (ii) the held-out gain of the best regime model is positive with a 95 % CI excluding 0, **and** (iii) the development bootstrap shows stable regimes: median ARI of regime assignments vs the full fit ≥ 0.7, and a 95 % interval for the intercept gap excluding 0.
- **Regimes rejected:** on **both** forms the held-out gain CI includes 0 or lies below 0, **and** criterion (i) fails, so that a continuous non-normal model fits at least as well.
- **Indeterminate:** any other pattern.

---

## Part B: uncertainty-aware continuous representation (CTR)

### Representation
`ContinuousIRTDur(guessing=True, deg=3, het=True)` fitted jointly on responses and `v_relative`, computed from unmasked items. It gives two coordinates, each with a posterior SD and a 90 % interval:
- ability, from responses only;
- relative time, the standardised residual of v given ability and booklet.

Relative time is not a psychological speed: it mixes pace, checking, pauses and rapid guessing.

### Evaluation design
The masked-item design of E10 is reused: 30 % of item positions, the same 5 mask seeds, and models refitted on the fitting set with time from unmasked items only.

| Comparison (paired bootstrap of per-student log loss) | Contribution |
|---|---|
| CTR vs K-means score–time profiles | C2 |
| CTR vs K-means (k = 4) on the CTR coordinates; CTR vs a 2 × 2 median grid on the CTR coordinates (same information, discretised) | C2: information loss from discretisation |
| CTR vs G3PL and vs CTR without time | value of time |
| CTR vs M3_3 with time, vs MIRT2G, vs M1_3 | class models; two-sided, no direction predicted |

Additional measures:
- **Calibration:** ECE of item predictions. The 90 % predictive interval for the number correct on the masked items is evaluated for coverage, mean width and log score, for CTR, G3PL, the K-means profiles and the discretised CTR.
- **Weights:** every paired comparison is repeated with `W_FSTUWT`-weighted per-student losses.
- **Stability:** a split-half analysis (odd vs even items). It compares the correlation and Spearman–Brown reliability of the ability and relative-time coordinates with Cohen's κ of K-means profiles across halves (labels matched by centroid) and with κ of ability quartiles.
- **External criterion:** out-of-sample R² for self-reported effort (EFFORT1), regression fitted on the fitting set and evaluated on the evaluation set: profiles vs continuous coordinates vs both.
- **Dashboard indicators (descriptive):**
  - mean width of the ability interval;
  - share of students whose ability interval excludes the median;
  - share whose relative-time interval excludes 0;
  - EAP reliability and separation strata.

### Confirmatory hypotheses (Forms B and C test splits; each must hold on both forms)
- **H6 (C2):** CTR has lower masked-item log loss than the K-means score–time profiles, with a CI excluding 0.
- **H7 (C2):** CTR has lower log loss than both discretised versions of its own coordinates, with CIs excluding 0.
- **H8 (C3):** the CTR 90 % masked-score interval is calibrated: observed coverage minus the coverage the model itself expects (its probability mass inside each student's interval) lies within ±0.02. Because masked scores are discrete (about 7 items), an equal-tailed 90 % interval holds more than 90 % of the mass, so raw coverage above 0.90 is expected. *Development note:* the first draft used raw coverage within [0.85, 0.95]; on the Form A development split, raw coverage was 0.964 because of this discreteness. The criterion was changed before freezing.
- **H9 (C3, stability):** the split-half Spearman–Brown reliability of CTR ability exceeds the split-half κ of the K-means profiles.

All comparisons with class models and with MIRT2G, the weighted analyses and the effort and dashboard indicators are descriptive or exploratory and are reported whatever their direction.

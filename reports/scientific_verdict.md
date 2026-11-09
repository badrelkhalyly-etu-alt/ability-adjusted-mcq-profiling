# Scientific verdict: **C, no evidence of residual latent classes**

*Data: PISA 2025 public-use files, linear mathematics Forms A (confirmatory), B and C (replications). None of these results concern the 1,979 FSBM students of the source study.*
*Protocol frozen at commit `e54a29f`. Deviations are logged in `reports/experimental_protocol.md`. Machine-readable version: `results/metrics/verdict.json`.*

## Verdict scale (from the research brief)
| Verdict | Meaning |
|---|---|
| A | Meaningful, reproducible residual latent heterogeneity with independent predictive or pedagogical value |
| B | Some improvement, but important limitations remain |
| **C** | **Existing homogeneous or continuous models explain the data sufficiently well** |
| D | Available data are insufficient to establish the claim |

## Hypotheses
| Hypothesis | Registered criterion | Result | Status |
|---|---|---|---|
| H1 residual heterogeneity | BIC prefers type-B classes, ΔBIC above the null q95, held-out gain with CI excluding 0, all against M0 and A_K | ΔBIC = 562 vs A2. Null q95 = −130 (fitted θ) and −126 (skewed θ), 0/80 false positives. Held-out M1_3 vs A2: +0.0070 nats [0.0053, 0.0085] | Met against the registered competitors |
| H1, exploratory robustness | Same comparisons against continuous models with guessing or a 2nd dimension | G3PL has the best BIC on all 3 forms (M1 is 40 / 25 / 86 points behind). 2D and guessing nulls trigger the class rule 20/20 times each. Held-out M1_3 vs G3PL: +0.0009 [−0.0004, 0.0023]; vs MIRT2G: −0.0004 [−0.0016, 0.0008] | **Fails** |
| H2 value of time | Gain ≥ 0.001 nats per item | M2 vs M0: +0.0015 [0.0009, 0.0021]. M3_3 vs M1_3: +0.0025 [0.0011, 0.0039]. Item-level speed vs total time: +0.0002 (n.s.) | Met at the point estimate; small effect |
| H3 robustness | Stable under bootstrap; replicates on Forms B and C | Profile of difficulty deviations r = 0.96. Person ARI bimodal (14/20 refits ≈ 0.88, 6/20 ≈ 0). Class model ranking replicates on B and C | Partial: item contrast yes, person classes no |
| H4 non-reducibility | Not reproduced by score, ability, booklet or region | Class × score quartile V = 0.64. 94.6 % of class 3 lies in the top quartile. Class × OECD V = 0.20 | **Not supported** |
| H5 external value | Predicts EFFORT1 beyond ability, score and time | ΔR² = 0.002 (K-means), 0.001 (M1_2), 0.004 (M1_3) | Not supported (negligible) |

## Source pipeline
- K-means k = 4 reproduces the four profiles, with silhouette 0.353 (source: 0.355).
- The no-cluster copula null has mean 0.363 (p = 0.94).
- The 80 IRT null replications give a silhouette of 0.39 to 0.40, and LCA selects K = 4 in 80/80.
- Used as a predictor of masked items, the profiles are 0.021 nats per item worse than a 2PL.

## Positive controls (power)
- A difficulty shift of 0.5 logit is never detected.
- A shift of 1.0 logit is detected 40–100 % of the time.
- A shift of 1.5 logit is always detected.
- When classes are detected, the ARI is only 0.31–0.67.

The protocol could have detected substantial classes. The negative result is therefore not a power failure.

## Why C and not B or D
- **Not B.** The only improvement over the registered competitors vanishes once guessing or a second dimension is modelled, both on held-out data and by BIC on three forms. The "classes" track item properties: their contrast correlates with the guessing asymptote (r = 0.55) and with the second-dimension loading (r = 0.77). They largely reproduce score levels.
- **Not D.** Power is adequate for shifts of 1 logit or more (3,363–3,980 training students per form). The test set and two replication forms give consistent answers.
- The small, reliable gain from total time (H2) concerns a continuous covariate, not classes.

## Alternative publishable question (verdict C)
The limits of behavioural profiling from minimal examination data. The paper shows, with a reusable pre-registered protocol, three things:
1. Silhouette, bootstrap stability and LCA enumeration cannot distinguish score–time profiles from a continuum.
2. Even ability-adjusted mixture IRT classes that pass null calibration against the 2PL are explained by guessing and multidimensionality.
3. Discrete profiles lose predictive information relative to continuous ability.

## Caveats
- The exploratory competitors (G3PL, MIRT2, MIRT2G) and the τ̂ ablation were added after the protocol was frozen and are logged as deviations.
- The duration null is linear in θ. The large duration ΔBIC (2,090) is consistent with a curvilinear speed–ability relation, which was not modelled as a continuous competitor.
- PISA is low-stakes and multi-format. The source examination is a graded university MCQ exam. The conclusions concern the method, not the FSBM students.

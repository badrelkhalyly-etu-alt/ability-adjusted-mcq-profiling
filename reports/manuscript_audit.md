# Critical audit of the source manuscript

**Audited document:** *Beyond the Score: Behavioural Profiles from Minimal Process Data in Online Multiple-Choice Examinations*, Nisrine Safeh, Faculté des Sciences Ben M'Sick, Université Hassan II de Casablanca (file `Behavioural_Profiles_MCQ_.docx`, read in full on 2026-11-08: 125 paragraphs, 6 tables, 0 embedded result figures except Figure 3/4 captions).

Authorship of the source study remains with N. Safeh. Nothing in this repository changes that attribution. Any extension built on it requires the agreement of the author and of the data custodians (FSBM).

**Data status.** The Casablanca raw data (1,997 records) are **not** available to this project. Every arithmetic check below uses only numbers printed in the manuscript. The empirical extension in this repository uses the OECD PISA 2025 public-use files instead (see `data/README.md`). PISA results must never be reported as results about the 1,979 FSBM students.

---

## 1. Scientific contributions claimed

| # | Claimed contribution | Where | Status in the manuscript |
|---|---|---|---|
| C1 | Partition structure from standardized total time and final score, three algorithms compared, stability reported (RQ1) | §1, §4.2–4.3, §5.2–5.3 | Partly delivered: indices in Table 3, 5-fold held-out silhouette. Assignment stability, which was promised, is not reported. |
| C2 | Characterisation of four profiles and theory-based readings (CVT, cognitive load, self-efficacy, self-worth) (RQ2) | §4.1, §5.4, §7.1, Tables 2, 4, 5 | Delivered as description. The readings are explicitly hypotheses, and rival explanations are listed (good practice). |
| C3 | Design and ethical safeguards for teacher-facing profile delivery; Émotionnella prototype (RQ3) | §6, §7.3 | Described only. Not evaluated (stated). |
| C4 | Observation that time alone does not separate performance (A vs C, B vs D) | §5.4 | Delivered, but partly an artefact of the partition (the author acknowledges this in §7.1 and §7.4). |

## 2. Reported experimental evidence

| Evidence | Value reported | Arithmetic check (from printed numbers) |
|---|---|---|
| Sample by programme | PC 892, BCG 690, MIP 281, IA 134; total 1,997 | Sums to 1,997 ✓ |
| Clustering sample | 1,979 (sum of cluster sizes) | 643+690+402+244 = 1,979 ✓ |
| Mean score | 14.64/20 | Size-weighted mean of cluster means = 14.644 ✓ |
| Mean duration | 17.83 min (SD 10.57, median 17.6, Q3 22.6) | Size-weighted mean of cluster mean times = **17.57 min ≠ 17.83** ✗ (0.26 min gap: descriptives appear to use a different sample or pre-cap durations) |
| Seconds per item | 68 / 38 / 67 / 29 | = mean time × 60 / 20 ✓ |
| % of limit | 76 / 42 / 75 / 32 | = mean time / 30 ✓ |
| K-means k = 4 | silhouette 0.355, DB 0.871, CH 1,592.0 | Not checkable without data |
| DBSCAN | 2 clusters, silhouette 0.559, CH 8.6 | A silhouette of 0.559 together with CH = 8.6 on n ≈ 1,979 is suspicious. It most likely means one dominant cluster plus a small one, or noise points included or excluded inconsistently. Treatment of noise not stated. |
| Agglomerative | 4 clusters, silhouette 0.332 | Linkage not stated |
| 5-fold held-out silhouette | 0.336–0.380, mean 0.362 (SD 0.022) | Internally consistent |
| Item difficulty | p from ≈0.54 (Q6) to >0.85 (Q3, Q5); max inter-item r ≈ 0.34 | KR-20 not reported |
| Programme means | IA ≈ 18/20 and ≈ 13 min; MIP > 16; PC and BCG ≈ 14, 17–18 min | Cluster × programme table not reported |

## 3. Methodological weaknesses

1. **Weak structure, likely a continuum.** A silhouette of 0.355 falls in the "weak structure" band. A two-dimensional, near-unimodal score–time cloud split by K-means always gives quadrant-like cells. The four profiles (high/low score × long/short time) match the result expected from partitioning a continuous bivariate distribution. No clusterability test is reported: the Hopkins statistic, gap statistic and GMM/BIC are all promised in §4.3 and none appear. **No null reference distribution** (silhouette under a fitted unimodal model) is given. Without one, 0.355 cannot be interpreted.
2. **Two variables only, and one is a sum score.** The item matrix is discarded for clustering. Score is a noisy proxy for ability (KR-20 unknown; inter-item r ≤ 0.34 suggests modest reliability). Students with the same score but different item patterns cannot be separated, by construction.
3. **Total time is censored and contaminated.** The limit is 30 minutes, and recorded durations reach 332 min because of a platform fault. Such values were "removed or capped". The sentence is unfinished, and the two treatments cannot both apply. Capping makes a point mass at 30 that K-means treats as real distance. The share of students near the limit (promised in §5.5) is not reported.
4. **Standardization and Euclidean distance** give score and time equal weight with no substantive reason. A log-time sensitivity check is promised but not reported.
5. **Model selection driven by interpretability.** k = 4 was chosen partly for "interpretability for teachers" (§4.3). This makes the number of profiles a design choice, not a finding.
6. **The stability evidence measures the wrong thing.** The held-out silhouette checks centroid fit, not whether assignments are reproducible (the author acknowledges this). Bootstrap ARI and Jaccard are promised but not reported.
7. **Circular interpretation.** The profiles are described with the same two variables that define them (acknowledged). The affective readings (anxiety, boredom, hopelessness) have no measurement behind them.
8. **Programme confounding.** IA is a selective track (≈18/20, ≈13 min) and probably falls mostly in profile B. If so, the "high-score, short" profile partly re-labels programme membership. No Cramér's V or cross-tabulation is reported.
9. **Item-level measurement invariance.** Four programmes with different curricula can show DIF on MTU items. This is not examined.
10. **No external criterion.** There is no later grade, self-report, re-test or second examination. The profiles cannot be validated.

## 4. Reproducibility audit

| Item | Present? |
|---|---|
| Raw data / data access statement | No (no data availability statement) |
| Code | No (Streamlit app described, no repository) |
| Software versions | scikit-learn cited, version not given |
| Hyperparameters | K-means init/n_init not given; DBSCAN eps/min_samples not given (described as sensitive); linkage for agglomerative not given |
| Seeds | Seed 42 for CV folds only |
| Preprocessing decisions | Partly given; capping vs removal unresolved; exclusion of 18 records not itemised |
| Figures 1–2 | Placeholders ("redraw project Figure 29/30") |
| Section 2.4 Positioning | Heading only, empty |
| Section 5.5 | Lists six analyses, reports **none** |

Verdict: the study **cannot be reproduced** from the manuscript.

## 5. Interpretation risks

- **Reification:** labels such as "anxiety" or "boredom" (Table 5) can be read as diagnoses even with the caveats. The dashboard's per-student view and recommendations multiply this risk (the author recommends disabling them, which is correct).
- **Speed ambiguity:** total duration mixes reading speed, language proficiency, checking, rapid guessing, early exit and technical interruptions. With no item-level latency, none can be separated (acknowledged).
- **Partition artefact:** the A–C contrast (same time, 6.2-point gap) follows from cutting the plane. It is not an empirical finding about time and performance unless the full time–score relation is shown.
- **Selection:** a single course module (MTU), a single sitting (Dec 2024), first-year students. Generalisation is not established.

## 6. Unreported or incomplete results

1. Time–score correlation (Pearson and Spearman) and the shape of the relation.
2. The distribution of durations, including the mass within 2 min of the limit and the number of values above 30 min.
3. Bootstrap ARI and cluster-wise Jaccard.
4. Log-time K-means; GMM with BIC; gap and Hopkins statistics.
5. Cluster × programme cross-tabulation with Cramér's V.
6. KR-20, person-fit index, item patterns by profile (needed to tell low knowledge from random responding in D).
7. Elbow and silhouette curves (Figure 1) and the scatter plot (Figure 2).
8. An itemised account of the 18 excluded records (1,997 → 1,979), and the sample on which the descriptives (17.83 min) were computed.
9. The Section 2.4 positioning text.

## 7. The 1,997 → 1,979 discrepancy

The manuscript gives two exclusion mechanisms: "records without a final score were removed", and durations above the limit "removed or capped". The 18 missing records are compatible with (a) 18 missing scores and capping of timing faults, (b) removal of timing faults only, or (c) a mixture. The descriptive mean time (17.83) does not equal the cluster-weighted mean (17.57). This indicates that the descriptives were computed on a different set of records or on uncapped values. **This cannot be resolved without the raw export.** It should be resolved before any re-analysis is published.

## 8. Incremental versus genuinely novel

| Element | Assessment |
|---|---|
| K-means/DBSCAN/agglomerative on score × time | Incremental. Standard EDM practice since at least Baker & Yacef (2009). |
| Profile labels from CVT | Interpretive framing, not empirically tested. |
| Dashboard | Engineering prototype, not evaluated. |
| Honest framing ("segmentation of a continuum") | A strength. The manuscript already concedes the central weakness. |
| **What would be novel** | Showing, against a calibrated null, whether discrete response-pattern heterogeneity exists **beyond continuous ability**, and whether coarse total time adds information that item-level time captures better. This is the question the extension addresses (see `reports/experimental_protocol.md`). |

## 9. Consequence for the extension

The extension cannot re-analyse the FSBM data. It therefore:

1. answers the methodological question (do score–time "profiles" survive an ability-adjusted, null-calibrated test?) on PISA 2025 public data, which has item-level and total timing;
2. includes a **minimal-data condition** (sum score + total time only, as in Safeh) next to the richer conditions, so the comparison applies directly to the original design;
3. makes no empirical claim about the FSBM cohort. Applying the pipeline to the FSBM data, if the custodians authorise it, is the external replication the original study lacks.

# D10: Q1 readiness assessment

**Scientific verdict:** C, no evidence of residual latent classes (see `scientific_verdict.md`).

**Publication readiness: not yet submittable.** The science is complete and coherent as a methodological and negative-results paper. Authorship, data permissions and several robustness items must be settled first.

## 1. Recommended framing and target
The paper is publishable as **"why score–time profiles from minimal process data are not evidence of student types: a null-calibrated, pre-registered protocol"**. It is not publishable as a new profiling method.

| Rank | Journal | Why it fits | Main risk |
|---|---|---|---|
| 1 | *Educational and Psychological Measurement* (Sage; SJR Q1) | Simulation + empirical design; negative results welcome when the method has demonstrated power (we have E6) | Read as a routine mixture-IRT application. Lead with the null-calibration and continuous-competitor findings |
| 2 | *Large-scale Assessments in Education* (IEA/Springer; SJR Q1, no APC) | PISA data; its scope welcomes evidence that "meaningfully changes confidence in an established finding" | Requires weights, plausible-value and design treatment. PISA must not look like a convenient testbed |
| 3 | *Journal of Learning Analytics* / *Computers & Education* (see `journal_strategy.md`) | Practitioner implication for profile dashboards | Expects pedagogical outcomes, which we do not have |

All quartiles are SJR (Scopus). **No JCR quartile was verified** (see `journal_strategy.md`).

## 2. Strengths reviewers will credit
- A frozen protocol (commit `e54a29f`) with an explicit deviation log, and a test set evaluated once.
- Null simulations fitted to the operational data (80 replications), positive controls (160 replications) and held-out masked-item prediction with paired bootstrap CIs.
- Replication on two independent forms with different students and items.
- A negative result that is informative. The K-means pipeline and LCA "find" four profiles in data without classes, and discrete profiles predict worse than continuous ability.
- Open code, unit tests, CI and Colab notebooks.

## 3. Remaining publication risks, by severity

### Blocking
1. **Authorship and consent.** The extension builds on N. Safeh's unpublished manuscript. Before submission, co-authorship, ordering and her approval must be agreed. `CITATION.cff` and the manuscript author block are placeholders.
2. **Pre-registration status.** The protocol was frozen in a local git commit, not on a public registry (OSF/AsPredicted). Reviewers may discount the "pre-registered" label. Either describe it as an "internally frozen protocol with a timestamped commit" or publish the repository history.
3. **Post-hoc competitors.** The decisive comparisons (G3PL, MIRT2, MIRT2G) were added after the Form A results were seen. Forms B and C mitigate this: when the competitors were added, their outcome on those forms was unknown. It must still be stated prominently, and ideally confirmed on a further form or cycle.

### Major
4. **Sampling design.** Model fitting ignores weights and clustering, and plausible values are not used. This is defensible for structural questions, but LSA-type journals will ask for it. A weighted sensitivity fit is needed.
5. **Duration model.** The duration null is linear in θ. The large duration ΔBIC (2,090) has not been tested against a continuous model with a non-linear time–ability regression. Add that competitor: a spline in θ within M2.
6. **Model space.** No residual speed–accuracy model (Bolsinova et al.) and no item-position model was fitted. The class models allow only difficulty shifts.
7. **Simulation scope.** The null and positive-control designs are tied to Form A (22 items, n ≈ 4,000). EPM reviewers will want factors such as test length, N and the RT–ability correlation.
8. **Not-reached sensitivity.** The protocol lists scoring not-reached as incorrect as a sensitivity analysis, but it is not yet reported in the manuscript.

### Minor
9. Starts per fit: 5 on real data and 3 in simulations. 8 of 480 null fits did not converge (flagged).
10. BIC uses nominal parameter counts despite ridge-penalised deviations.
11. The paired comparison M3_3_time vs MIRT2G was not pre-specified, so no CI is reported for it.
12. **Literature.** The 2026 references come from Crossref/Scopus metadata. Some abstracts were not read (see `literature/scopus_verification.md`). Every citation must be re-checked before submission.
13. **Link to the source study.** No FSBM data were analysed. Reviewers of a paper citing an unpublished manuscript may ask for it. Running the released pipeline on FSBM data, with permission, is the strongest possible addition.

## 4. Path to submission (estimated work)
1. Settle authorship and consent with N. Safeh. Decide the repository visibility.
2. Add a weighted fit, a non-linear duration competitor and the not-reached sensitivity analysis. Each reuses existing code (≈ 1–2 days of compute).
3. Optionally extend the simulation factors for EPM.
4. Re-verify the references, then prepare the cover letter around verdict C and its alternative question.

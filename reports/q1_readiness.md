# Submission-readiness report (critical), IEEE Access

**Scientific status.**
- **Main study:** verdict C. Score–time profiles and ability-adjusted mixture classes are not distinct from continuous models.
- **Extension B:** a continuous, uncertainty-aware representation, confirmed on independent data. All four frozen hypotheses held on both validation forms.
- **Extension C:** latent time regimes. Indeterminate under the frozen rule, leaning towards a non-normal continuum.

**Readiness: close to submittable.** The science and the PDF are complete. Editorial items and a few judgement calls remain; they are listed below.

The submission file is `manuscript/access.tex` → `access.pdf`: 24 pages, official IEEE Access template, no undefined references, no overfull lines. The Overleaf bundle is `manuscript/access_overleaf.zip`. The single-column preprint is `article.tex` (36 pages).

## 1. Contributions and the evidence behind each

| Contribution | Experiment and comparison | Data | Evidence |
|---|---|---|---|
| C1. Null-calibrated validation of profiles | Profile rules calibrated on data simulated from fitted continuous models (2PL, guessing, 2D, four non-normal time models); positive controls | Form A (main study); Form A development split (time) | K-means/LCA find 4 profiles without classes. Mixture-class rule: 0/80 false positives under a 2PL null, 20/20 under guessing or 2D nulls. Time-regime rule: 0–5 % false positives under non-normal continua, 100 % power |
| C2. Information lost by discrete profiling | Masked-item log loss: CTR vs K-means profiles and vs two discretisations of CTR's own coordinates | Form B and C test splits, once | Profiles lose 0.019–0.025 nats per response; discretising CTR loses 0.026–0.036. Profiles keep about 70 % of ability variance and 28–30 % of relative-time variance |
| C3. Uncertainty-aware continuous representation | Predictive-interval calibration; split-half stability; comparison with class models and MIRT2G; survey weights | Form B and C test splits, once | Calibrated (coverage − expected within 0.005); profile intervals overconfident by 4–5 points. Reliability 0.87–0.89 vs profile κ 0.52–0.58. Matches M3_3 with time, MIRT2G and M1_3 |
| C4. Reproducible evaluation and visualisation framework | Code, 45 tests, notebooks 00–10, Streamlit prototype | — | Every table cell checked against its CSV (`check_tables.py`). The notebooks run in demonstration mode. 45/45 tests pass. An earlier intermittent failure of the prototype test was traced to two Python installations on the build machine |

## 2. IEEE Access requirements

| Requirement | Status |
|---|---|
| Abstract ≤ 250 words, one paragraph, no citations | Done (237 words) |
| Index terms | Done |
| Official template | Done: the zip supplied by the authors, class files unmodified. Caption wrapping fixed in our figure environment |
| Authors | As supplied by the authors. Corresponding authors: B. El Khalyly and N. Safeh |
| Affiliations | **To be confirmed** for B. El Khalyly (faculty or lab) and A. Yalid (inferred from the e-mail domain). **To be completed** for A. Belkhadim and Z. Ibn Batouta. None was invented |
| Biographies and photographs | **Missing** (placeholders for 5 authors) |
| Funding statement | **Missing** (placeholder) |
| Data and code availability | Repository named; **URL to add** once the authors create the GitHub repository |
| APC and journal quartile | **Not verified**. Check on the IEEE Access site and in JCR/Scopus before claiming "Q1" |

## 3. Risks a reviewer will raise, by severity

### Major
1. **Length and focus.** At 24 pages, the paper reports three studies: the negative result, representation B and time regimes C. Option: move E8, E9, the contingency tables and part of C to supplementary material.
2. **Protocols frozen in local git commits, not on a public registry.** The text says "frozen before analysis". Publish the repository history (commits `e54a29f` and `fe7dab6`) with the paper.
3. **Post-hoc competitors in the main study** (G3PL, MIRT2, MIRT2G). This is mitigated, because extensions B and C were confirmatory on untouched forms.
4. **The value of time is small and not robust to survey weights**: CTR vs G3PL +0.0008–0.0013 unweighted, about 0 weighted. The paper states this. Do not oversell relative time as predictive. Its role is descriptive.
5. **Survey design.** Weighted analyses use a pseudo-likelihood. Effective sample sizes are small with multi-country weights, and clustering is not modelled.

### Moderate
6. **Rule for time regimes.** The frozen BIC threshold is negative (−9.2), which weakens criterion (i). This is reported and does not change the decision. The nulls were calibrated at the development sample size.
7. **H9 compares different metrics** (reliability vs κ). The like-for-like comparison, ability quartiles κ = 0.38, is reported and shows that cut points cause the instability.
8. **Weighted M1_3 better than CTR on Form C** (−0.0034 [−0.0060, −0.0009]), not on Form B. It is reported as exploratory.
9. **Scope.** Low-stakes PISA 2025 mathematics, 15-year-olds. There is no high-stakes examination, and item-level engagement states were not tested.

### Minor
10. Five starts on real data, three in simulations; 20 replications per null generator in C.
11. BIC uses nominal parameter counts with ridge-penalised deviations.
12. The 2026 references come from Crossref or Scopus metadata. Re-check each citation before submission.

## 4. Remaining steps before submission
1. Authors: affiliations, biographies with photographs, funding, CRediT statement.
2. Create the public GitHub repository, then run `python scripts/build_notebooks.py <URL>`, add the URL to the paper and rebuild the Overleaf zip.
3. Decide what to move to supplementary material if a shorter paper is preferred.
4. Re-verify the references and check the APC.
5. Cover letter around C1–C4, stating the negative result plainly.

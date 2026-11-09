# D10: Q1 readiness assessment

**Scientific verdict:** C, no evidence of residual latent classes (see `scientific_verdict.md`).

**Publication readiness: not yet submittable.** The science is complete and coherent as a methodological and negative-results paper. Authorship, data permissions and several robustness items must be settled first.

## 1. Target journal: IEEE Access (authors' choice, 9 November 2026)
The paper is publishable as **"why score–time profiles from minimal process data are not evidence of student types: a null-calibrated protocol"**. It is not publishable as a new profiling method.

The submission version is `manuscript/access.tex` → `manuscript/access.pdf`: IEEE Access two-column class, 17 pages. The single-column preprint `article.tex` shares the same sections, tables and figures.

| IEEE Access requirement | Status |
|---|---|
| Single-paragraph abstract, ≤ 250 words, no citations or equations | Done (`sections/abstract_ieee.tex`, 239 words) |
| Index terms (keywords) | Done (9 terms) |
| IEEE numeric reference style | Done (`IEEEtranN`; author–year commands map to numbers) |
| Template class | Done: official IEEE Access LaTeX template (Overleaf zip supplied by the authors, 9 November 2026), class files unmodified. `manuscript/access_overleaf.zip` (built by `scripts/make_overleaf_bundle.py`) imports directly into Overleaf |
| Author names, affiliations, corresponding author, funding footnote | **Placeholders**: authorship not yet agreed |
| Biography (and photograph) for every author | **Placeholder**: to be written by each author |
| Article processing charge (open access) | Not verified here; check the current amount on the IEEE Access website |
| Journal quartile / impact factor | **Not verified**. A secondary source reports a 2025 JIF of 4.2 and a CiteScore of 9.3, but not the quartile. Check JCR and Scopus before relying on a "Q1" claim; the category matters |

**Fit and risk.** IEEE Access is a broad, multidisciplinary engineering journal with a binary accept/reject decision and fast review. It publishes educational data mining and learning analytics work. Main risks:
1. Reviewers outside psychometrics may read a negative result as "nothing found". The introduction and abstract must lead with the engineering message: a profiling feature built into e-assessment platforms produces spurious types, and here is a protocol to test it.
2. The mixture-IRT notation is dense for this readership. Keep the model ladder table and move derivations to the supplementary material if reviewers ask.
3. Length: 17 two-column pages. Contingency tables and part of E8/E9 can move to supplementary material if needed.

## 2. Strengths reviewers will credit
- A frozen protocol (commit `e54a29f`) with an explicit deviation log, and a test set evaluated once.
- Null simulations fitted to the operational data (80 replications), positive controls (160 replications) and held-out masked-item prediction with paired bootstrap CIs.
- Replication on two independent forms with different students and items.
- A negative result that is informative. The K-means pipeline and LCA "find" four profiles in data without classes, and discrete profiles predict worse than continuous ability.
- Open code, unit tests, CI and Colab notebooks.

## 3. Remaining publication risks, by severity

### Blocking
1. **Authorship.** The paper is now framed on PISA 2025 alone (authors' decision, 9 November 2026); N. Safeh's unpublished manuscript is cited once, as the origin of the question. Author list set on 9 November 2026: N. Safeh, B. El Khalyly, A. Yalid, A. Belkhadim, Z. Ibn Batouta; corresponding authors B. El Khalyly and N. Safeh. Affiliations of three authors, biographies and funding remain to be completed. `CITATION.cff` and the manuscript author block are placeholders.
2. **Pre-registration status.** The protocol was frozen in a local git commit, not on a public registry (OSF/AsPredicted). Reviewers may discount the "pre-registered" label. Either describe it as an "internally frozen protocol with a timestamped commit" or publish the repository history.
3. **Post-hoc competitors.** The decisive comparisons (G3PL, MIRT2, MIRT2G) were added after the Form A results were seen. Forms B and C mitigate this: when the competitors were added, their outcome on those forms was unknown. It must still be stated prominently, and ideally confirmed on a further form or cycle.

### Major
4. **Sampling design.** Done in part (Section "Sensitivity analyses"). A weighted pseudo-likelihood favours M1_3 with the nominal n but not with the Kish effective n (518–614), and the weighted classes change across forms. Reviewers may still ask for a design-based treatment of clustering, which has no standard for mixture IRT. The ESS rescaling is post hoc and must be presented as such.
5. **Duration model.** Done: a cubic, heteroscedastic M2 explains roughly 70–75 % of the duration ΔBIC, but class duration models still lead by 497–958. The paper now says the remainder is unresolved (discrete classes or a non-normal continuous speed distribution). A reviewer may ask for the second model.
6. **Model space.** No residual speed–accuracy model (Bolsinova et al.) and no item-position model was fitted. The class models allow only difficulty shifts.
7. **Simulation scope.** The null and positive-control designs are tied to Form A (22 items, n ≈ 4,000). EPM reviewers will want factors such as test length, N and the RT–ability correlation.
8. **Not-reached sensitivity.** Done: the conclusion is unchanged on all three forms.

### Minor
9. Starts per fit: 5 on real data and 3 in simulations. 8 of 480 null fits did not converge (flagged).
10. BIC uses nominal parameter counts despite ridge-penalised deviations.
11. The paired comparison M3_3_time vs MIRT2G was not pre-specified, so no CI is reported for it.
12. **Literature.** The 2026 references come from Crossref/Scopus metadata. Some abstracts were not read (see `literature/scopus_verification.md`). Every citation must be re-checked before submission.
13. **Scope.** Results concern PISA 2025 (low-stakes). Transfer to high-stakes examinations is untested; another PISA cycle or domain would be the cheapest external replication.

## 4. Path to submission (estimated work)
1. Complete the author list. Decide the repository visibility.
2. Done: weighted fit, non-linear duration competitor and not-reached sensitivity analysis (training data only). Optional: a non-normal continuous speed model for the duration remainder.
3. Optionally extend the simulation factors for EPM.
4. Re-verify the references, then prepare the cover letter around verdict C and its alternative question.

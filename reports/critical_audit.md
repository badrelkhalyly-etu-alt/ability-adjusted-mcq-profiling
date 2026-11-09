# Critical audit of the manuscript claims (phase 1)

Scope: abstract, results and discussion of `manuscript/access.tex`, checked against `results/`. No result was changed to favour a conclusion. Each problem found is listed with what was done about it.

## 1. Numerical consistency
- Every cell of the manuscript tables is checked automatically against its source CSV by `scripts/check_tables.py`: 586 cells, 0 mismatches at the last run, before the extension tables. The extension tables are rebuilt by `scripts/build_extension_outputs.py` directly from the result files.
- Numbers in the text were audited earlier (about 240 values). The values added for the extensions are read from `results/continuous/*` and `results/temporal/*`, and were re-checked when the new sections were written.

## 2. Simple nulls versus richer continuous nulls
**Problem.** The abstract and contribution 1 said "no false positives in 80 null replications". That is true only for the null generated from the fitted **2PL** (with fitted or skewed ability). Nulls generated from richer continuous models, a 2D 2PL or a 2PL with guessing, triggered the class rule in **20/20** replications each (E3b, E3c).

**Consequence.** Null calibration protects against spurious classes only if the null is as rich as the true continuous structure. That is a central lesson, not a footnote.

**Done.** The abstract and contribution C1 now state both results: 0/80 under the 2PL null, and 20/20 under the guessing and 2D nulls.

## 3. Apparent contradiction: the best predictive model is a class model with time
**Problem.** On the Form A test split, the model with the lowest masked-item log loss was M3_3 with total time (0.4917), ahead of MIRT2G (0.4936) and G3PL (0.4949). The paper argued for continuous models, but no continuous model **with both guessing and time** had been compared with M3_3_time, and no interval was reported for M3_3_time against MIRT2G. M3_3_time therefore combined two advantages, time and flexible response structure, while its continuous competitors had at most one.

**Done.** The continuous representation CTR (guessing plus a flexible time model) was compared with M3_3_time on the untouched Form B and C test splits, two-sided and with paired bootstrap intervals. See `results/continuous/confirm_*_paired.csv` and the manuscript section on independent validation. The Form A result is still reported as it was.

## 4. Survey weights
- The weighted pseudo-likelihood with the nominal n favoured M1_3. With the Kish effective n (518–614), the continuous model won again, and the weighted classes changed from form to form. The Kish rescaling is post hoc and is labelled as such.
- In the extensions, every held-out comparison is also reported with losses weighted by `W_FSTUWT`. In option C, the weighted result reverses direction on Form B for relative time, against regimes. This is reported.

## 5. Missing or weak comparisons, now addressed

| Gap | Status |
|---|---|
| No continuous competitor with guessing and time | Added: CTR, confirmed on B and C |
| Duration "classes" only tested against normal time models | Added: six continuous families, including skewed and heavy-tailed ones, plus regime models (option C) |
| Information loss from discretisation only shown with K-means on score and time | Added: discretisations of the *same* continuous coordinates (K-means k = 4 and a 2 × 2 grid) |
| Profile uncertainty never quantified | Added: predictive intervals for masked scores, with coverage against model-expected coverage, and split-half stability |
| New analyses evaluated on the already examined Form A test split | Avoided: Form A test split not reused; Forms B and C test splits used once, after the protocol was frozen (`fe7dab6`) |

## 6. Remaining weaknesses that reviewers may raise
1. **Pre-registration.** The protocols were frozen in local git commits, not on a public registry. The text now says "frozen before analysis", not "pre-registered".
2. **Post-hoc competitors in the main study.** G3PL, MIRT2 and MIRT2G were added after the Form A results were seen. Forms B and C mitigate this, and the extensions are confirmatory on B and C only.
3. **Label-reliability analysis (section "A continuous alternative to profiles").** It uses the 2PL posterior, although G3PL fits better. The CTR validation uses G3PL.
4. **Null calibration of option C.** It was done at the Form A development sample size. The frozen BIC threshold is negative (−9.2), which weakens criterion (i). This was noticed after freezing, is reported, and does not change the decision.
5. **Split-half comparison (H9).** It contrasts a reliability coefficient for ability with Cohen's κ for profiles: different metrics. The paper also reports κ for ability quartiles, which is the like-for-like comparison. It turned out lower than the profile κ, because discretising a continuum creates boundary instability, whereas the profiles also use time, which is highly reliable.
6. **Effort criterion.** EFFORT1 is a single post-test self-report. All R² values are below 0.04.
7. **Scope.** Low-stakes PISA 2025 mathematics only. There is no high-stakes examination and no item-level engagement states.

## 7. Editorial metadata
- **Authors.** As provided by the authors. Affiliations for three authors are marked "[to be confirmed]" or "[to be completed]". None was invented.
- **Template.** Official IEEE Access template; figure captions now wrap.
- **References.** Every reference added during the extensions (IEEE Access articles, Jones & Pewsey 2009, Rigby & Stasinopoulos 2005) was checked against Crossref.

# Novelty and gap matrix: AALBP vs the closest work (2026-11-08)

Status: **FT** means full text read, **ABS** means abstract only. Our study is called "AALBP" below. It uses the PISA 2025 public-use files (to be verified; PISA 2022 is the fallback) and has these components:

- an LC-IRT / mixture-IRT model compared with a homogeneous 2PL model;
- a hierarchical speed–accuracy model;
- null simulations from a fitted continuous model;
- positive controls;
- DIF and invariance checks;
- held-out masked-item prediction;
- a duration model;
- an ablation comparing total time with item-level time.

## 1. Paper-by-paper comparison

| # | Paper | What it solves | Data required | Key assumptions | How AALBP differs (or does not) | Evidence AALBP needs to claim novelty over it |
|---|---|---|---|---|---|---|
| 1 | **R4** Yıldırım Hoş & Uysal Saraç 2025 (FT) | Shows that item-level RT covariates of class membership improve fit of a 2-class (ability / non-ability) mixture 2PL in PISA 2022 math | **Item-level RTs**, item responses (N=1,180; 11 MC items; Türkiye) | Jeon–De Boeck mixture. Non-ability class is intercept-only. Double-centred log-RT enters class membership. Model comparison only by AIC, BIC and LRT | **Same data family (PISA) and same kind of question.** It does not compare against a one-class *joint* RT model, has no null simulation, and checks no invariance across countries. Its non-ability class differs mainly in item thresholds, which fits a lower-θ slice | Show that, against a one-class van der Linden model and a null calibrated to the same data, the 2-class gain is (or is not) larger than the null distribution. Show replication across countries |
| 2 | **R3** Huang, Luo & Jeon 2025 (ABS) | Strategy switching within examinees, at the item level, driven by RT; strategies are not predefined | **Item-level RTs**; PISA 2018 Science | Item-by-person latent strategy; RT informs strategy | AALBP targets **person-level** profiles (Safeh's construct). R3 implies those may be the wrong unit. AALBP should **include an item-level class model as a competitor**, not ignore it | Out-of-sample comparison of person-level classes, item-level switching, and continuous (θ, τ) on the same PISA data |
| 3 | **R2** Bacci, Fabbricatore & Iannario 2026 (ABS) | Joint model of accuracy and RT with item-position effects and within-person change in speed and ability | **Item-level RTs plus item positions**; computer-based learning assessment | Continuous latent traits with random effects; no classes | A **continuous competitor**. Drift from position or fatigue can create apparent "slow-low" or "fast-low" profiles. AALBP uses it, or a simplified position-effect term, as an alternative explanation | Show that classes keep predictive value after position effects are modelled |
| 4 | **R1** Bakk & Fabbricatore 2026 (FT) | Two-step estimation of LC-IRT with DIF and multidimensionality; avoids interpretational confounding | Item responses and covariates; **no RTs** | Class-specific normal θ. Measurement model is fixed in step 2. Number of classes is chosen before covariates | AALBP **uses** this estimation logic for covariates and DIF. It adds timing and a continuous null. R1's own empirical classes separated high from low competence, which is the ambiguity AALBP tests | None against R1's method; AALBP is an application and validation of it. Cite it as the estimator, not as a competitor |
| 5 | **R5** AlHakmani & Sheng 2024 (FT) | Bayesian MixIRT (NUTS) on TIMSS 2019; reports 2–3 classes | Item responses only | Ordinal constraint on class mean ability, which builds ability ordering into the classes; WAIC, LOO, DIC, WBIC | AALBP **adds** the null calibration that R5 lacks. R5's WBIC preferred one class, which shows conclusions depend on the index | False-positive rates of each index (WAIC, LOO, BIC, BLRT) under a null fitted to the data |
| 6 | **Zheng et al. 2023** (FT, eTIMSS 2019) | Descriptive mixture on item log-RTs plus item scores; 4–5 behaviour classes per country with labels | **Item-level RTs**, item scores (4 countries, n 242–1,595) | Normal mixtures for log-RT, no IRT ability, BLRT; different class counts by country | **The closest analogue to Safeh-style profiling in an ILSA.** AALBP adds ability adjustment, a continuous null, formal invariance across countries, and a total-time vs item-time ablation | Show whether Zheng-type classes survive conditioning on (θ, τ) and replicate across countries under invariance |
| 7 | **Sen, Cohen & Kim 2016** (ABS) and **Alexeev, Templin & Cohen 2011** (ABS) | Show that mixture IRT extracts spurious classes from single-class data that are non-normal or misspecified | Simulated responses | Generic θ distributions (bimodal, uniform, skewed) or a 2PL fitted as Rasch | **The null-simulation idea is not new.** AALBP's null differs because (a) it is a **joint (θ, τ)** model, (b) it is **fitted to the operational data** (item parameters, booklet design, time intensities, missingness), and (c) it is applied to the **whole pipeline**, including K-means on (score, time) | Report false-positive class detection under data-calibrated joint nulls; show it differs from generic-normal nulls |
| 8 | **Ma et al. 2026, Psychological Methods** (ABS) | Compares class-based (CDM) with continuous (2PL) models by **held-out predictive performance**; class models overfit | Item responses (11 CDM datasets) | Predictive criterion; CDM classes | **Predictive comparison of class and continuous models is not new.** AALBP extends it to mixture-IRT and behavioural classes **with timing data** and to predicting held-out times as well as responses | Show predictive gain, or no gain, of classes for masked items **and** for held-out item or total times |
| 9 | **Nagy & Ulitzsch 2022** (ABS); **Huang 2026 Mix-HM** (ABS) | Item-level engagement or pacing classes inside joint models (Mplus multilevel mixture IRT; Bayesian Mix-HM) | **Item-level RTs** | Engagement is an item-level latent class (Nagy). Pacing switches and local dependence (Huang) | Mature competitor models for "behaviour beyond (θ, τ)". AALBP should use one as a **positive-control generator** and as a candidate model | Include in the model set; show detection power under positive controls |

## 2. What the minimal data allow, and what PISA allows

Safeh's original data are not available to us. The middle column describes what a Safeh-type dataset (20 MC items, total score and total exam time) would allow in principle.

| Analysis | Safeh-type data (total score + total time; item responses only if recorded) | PISA (item responses + item-level timing + questionnaire) |
|---|---|---|
| K-means / LPA on (total score, total time), the Safeh replication | Possible | Possible (by computing totals) |
| Mixture 2PL / LC-IRT on item responses | Possible if the item response matrix exists | Possible |
| Joint model with total time as one lognormal or censored duration | Possible | Possible |
| R4 / Jeon–De Boeck mixture (item RT predicts class) | **Not possible** | Possible |
| R3 item-level strategy switching; Molenaar–De Boeck response mixtures | **Not possible** | Possible |
| R2 joint RT and item position | **Not possible** | Possible (positions are known from the booklet or MST design) |
| van der Linden hierarchical model with item time intensities | **Not possible** (only one time per person) | Possible |
| Rapid-guessing flags (Wise & Kong) | **Not possible** | Possible |
| DIF / invariance across countries or subgroups | Only gender or local subgroups, if recorded | Possible |

So **R2, R3, R4, Zheng 2023, Nagy & Ulitzsch 2022 and Huang 2026 all require item-level times that Safeh's data do not contain**. PISA supplies item-level timing in recent computer-based cycles; R4 used it for PISA 2022. Whether the PISA 2025 public-use files include it, and when, has **not** been verified here.

## 3. Has the proposed combination already been done?

We checked whether the combination "ability-adjusted latent classes + null-simulation false-positive calibration + total-time vs item-level-time ablation" exists in the literature.

- **Ability-adjusted latent classes** (mixture IRT / LC-IRT, with or without RT): **done many times** (Rost 1990; Jeon & De Boeck 2019; R1, R3, R4, R5; Huang 2026).
- **Null simulations of spurious classes**: **done**, for generic mixture IRT (Alexeev 2011; Sen 2016) and for mixtures in general (Bauer & Curran 2003; Cole 2026). It has not been done with data-calibrated joint (θ, τ) nulls in the papers we found.
- **Predictive comparison of class and continuous models**: **done** for CDM versus IRT (Ma 2026), without timing data.
- **Total-time vs item-level-time ablation for class detection**: **not found** in the papers searched.
- **The whole package applied to the question of whether minimal score–time profiles are classes or a continuum**: **not found.**

Caveat: we searched Crossref, OpenAlex, Europe PMC and the open web, not Scopus or Web of Science. Conference papers (for example NCME and IMPS proceedings) and dissertations were not systematically searched. Sen's 2014 University of Georgia dissertation on spurious classes appeared in results and should be checked.

## 4. Honest classification of the contribution

- **Not a new algorithm or model.** Every model component exists: mixture 2PL, LC-IRT with two-step covariates, the hierarchical speed–accuracy model, lognormal or censored duration, BLRT, and the information criteria.
- **Primary contribution: a validation protocol (AALBP).** It is a pre-specified pipeline that tests a claimed behavioural profile against the following:
  1. a **data-calibrated continuous joint null**, giving the false-positive rate of the exact pipeline, including silhouette and BLRT null distributions;
  2. **positive controls**, giving power;
  3. **held-out prediction** of responses and times, in the spirit of Ma 2026;
  4. **invariance across countries**;
  5. **incremental external validity** beyond (θ, τ);
  6. a **total-time vs item-time ablation**.

  Novelty lies in the combination and calibration, not in any single component.
- **Secondary contribution: a new empirical finding.** Whether PISA (2025 or 2022) response-and-time data contain person-level classes beyond a continuous (θ, τ) structure, and whether these replicate across countries. This would be new for PISA 2025 if the data are released. For PISA 2022, R4 is a direct precedent and our result would partly be a re-examination of it.
- **Tertiary contribution: a direct reassessment of minimal-process-data profiling.** We will quantify what Safeh-type (score, total time) clustering can and cannot identify, using the ablation.

## 5. Revised research question (recommended)

Replace "Do item responses plus time reveal reproducible latent classes beyond IRT ability?" (which R4 and Zheng 2023 already partly answer, positively but without calibration) with:

> **RQ1.** When the analysis is calibrated against a continuous joint ability–speed model fitted to the same PISA data, how often do common profiling pipelines detect "behavioural profiles" by chance? The pipelines are K-means or LPA on total score and total time, mixture IRT, and RT-covariate mixtures in the style of R4. Do the profiles found in the real data exceed these null rates?
>
> **RQ2.** How much of the evidence for classes depends on item-level timing, compared with total time only? This is the ablation, and it bears directly on minimal-process-data studies such as Safeh's.
>
> **RQ3.** Do any surviving classes (a) improve held-out prediction of masked item responses and of item or total times over continuous (θ, τ), (b) replicate under measurement invariance across countries, and (c) predict external questionnaire criteria (effort, anxiety, ESCS) beyond (θ, τ)?

Expected defensible outcomes:

- **"Classes survive calibration."** This is a new, strong empirical claim.
- **"Score–time profiles are indistinguishable from continuum slices; only item-level rapid-guessing states survive."** This is equally publishable as a validation or negative result, and it directly qualifies Safeh's emotion-labelled profiles.

## 6. Risks to novelty to resolve before writing

1. Read the full texts of R3, Huang 2026, Ma 2026 and Sen 2016 (all currently abstract only) to confirm none of them already runs data-calibrated joint nulls or time-granularity ablations.
2. Check Sen (2014, UGA dissertation) and recent NCME/IMPS proceedings for "spurious classes response times".
3. Confirm that the PISA 2025 public-use files exist and contain item-level timing; otherwise reframe on PISA 2022 and cite R4 as the direct precedent being re-examined.

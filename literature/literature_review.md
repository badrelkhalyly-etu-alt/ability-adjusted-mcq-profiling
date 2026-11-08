# Literature review: are score–time "profiles" latent classes or slices of a continuum?

Prepared 2026-11-08 for the AALBP extension of Safeh, *Beyond the Score*. The review covers 69 papers. Full metadata and verification status for each paper are in `related_work_matrix.csv`, and the BibTeX keys are in `references.bib`. Status labels are used strictly:

- **FULL TEXT** means the methods and results were read.
- **ABSTRACT** means only the abstract was read.
- **BIB** means only the DOI and metadata were checked. For these papers, the description of the content comes from established knowledge of the paper, not from reading it again now. Do not cite them for specific numbers until they have been read.

---

## 1. The problem this review is organised around

Safeh's manuscript clusters 1,979 students on two variables, total score and total time. K-means gives four groups with a silhouette of 0.355, and these groups receive emotion labels. The extension asks whether such groups show **qualitative** heterogeneity, meaning latent classes with different response processes. The alternative is that they are partitions of a **continuous** joint distribution of ability (θ) and speed (τ). The literature bears on this question through five largely separate strands:

- mixture and latent-class IRT;
- joint speed–accuracy models, and mixtures within them;
- engagement and rapid-guessing models;
- methodological work on spurious classes and class enumeration;
- invariance and external validation.

Few papers bring these strands together, and that combination is the space the extension can occupy.

## 2. Mixture IRT and LC-IRT: what "a class" means

The founding models are Rost's mixed Rasch model (`rost1990rasch`, BIB) and Mislevy & Verhelst's strategy mixtures (`mislevy1990modeling`, BIB). In both, item parameters differ across classes, so a class means a different response process and not merely a different ability level. Two later traditions blur this meaning:

1. **Discrete-ability LC-IRT.** The Bartolucci tradition (`bartolucci2007class`, BIB; `bacci2014class`) uses latent classes as support points of a discretised ability distribution. Here a class is a slice of the continuum by construction.
2. **LC-IRT with class-specific θ distributions.** **R1** (`bakk2026twostep`, FULL TEXT) uses a mixture of classes, each with its own normal θ distribution. It then estimates covariate and DIF effects in a second step, after the measurement model, which protects the definition of the classes from interpretational confounding. This builds on `bakk2018twostep` and `vermunt2010latent`. R1 is methodologically strong but contains an informative detail. In its empirical example (n = 300), BIC selected two classes, and the authors report that this solution "mainly distinguished students with higher versus lower statistical competence." That is a class solution that sorts mostly on ability, which is exactly the ambiguity our study addresses. R1 has no timing data and no generating condition with a single continuous population.

The ILSA applications show the same pattern on a larger scale. **R5** (`alhakmani2024detecting`, FULL TEXT) fitted Bayesian MixRasch, Mix2PL and Mix3PL models to TIMSS 2019 data from four Gulf states (N = 1,618; 26 items):

- most indices preferred two or three classes;
- WBIC preferred **one** class for every model family;
- label switching was resolved by ordering the classes on mean ability, so the classes are partly defined by ability level;
- there was no null simulation, even though the paper cites Sen et al. (2016).

`effatpanah2025fitting` (ABSTRACT) labels three mixed-Rasch classes in IELTS listening as low, medium and high "stimulus processors". These are process labels attached to classes ordered by proficiency, the same interpretive move as Safeh's emotion labels.

**Takeaway:** in applied mixture IRT, a "class" is often not a qualitative difference. Without explicit tests, it can be a discretised θ, a non-normal θ, or latent DIF (see `wallin2026regularised`, ABSTRACT; `choi2015differential`; `cho2016ncme`).

## 3. Joint time–accuracy models: the continuous baseline

van der Linden's lognormal model (`vanderlinden2006lognormal`, BIB) and hierarchical framework (`vanderlinden2007hierarchical`, BIB) are the standard continuous model. The framework has three parts: an IRT model for accuracy, a lognormal model for item time (person speed τ, item time intensity β), and a bivariate normal (θ, τ) distribution at the population level. **In this model, every score–time configuration Safeh observed can arise without any classes.** Low-score/fast and high-score/slow students are simply regions of a correlated (θ, τ) distribution.

Several papers relax the conditional-independence assumption of this model without introducing person classes:

- `bolsinova2017modelling` (BIB) models residual dependence between time and accuracy directly;
- `bolsinova2017conditional` (BIB) reviews where such dependence comes from, including strategy mixtures, fluctuations in speed or ability, and item-level effects;
- `molenaar2016hidden` (BIB) uses state switching;
- **R2** (`bacci2026unified`, ABSTRACT) adds item position and within-person change in speed and ability to a bivariate GLIRT.

All of these are continuous or dynamic alternatives that a class model must outperform before classes are claimed. Goldhammer et al. (`goldhammer2014time`, BIB) show that the time-on-task effect changes sign depending on task difficulty and skill. A fixed reading of total time, such as "slow = anxious", is therefore not identifiable from time alone. `sinharay2020assessing` (BIB) warns that the lognormal RT model can misfit. This matters for our null design, because a skewed or misfitting time distribution is a known route to spurious classes (Section 5).

## 4. Mixtures in time and accuracy: the direct competitors

Response-time mixtures began with Schnipke & Scrams (`schnipke1997modeling`, BIB), who modelled solution behaviour versus rapid guessing. They have since become a dense literature:

- **Within the hierarchical model.** `wang2015mixture` (BIB) mixes solution and guessing behaviour inside the hierarchical framework. `huang2026unveiling` (ABSTRACT) proposes Mix-HM, in which respondents switch pacing speed and item-level local dependencies are estimated. It reports that the conventional hierarchical model distorts the ability–speed relationship when such dependencies are present.
- **Item-level, within-person classes.** `molenaar2018response` (ABSTRACT) classifies individual responses rather than persons. **R3** (`huang2025response`, ABSTRACT) lets examinees switch between ability-based and non-ability strategies item by item, illustrated on PISA 2018 Science. `nagy2022multilevel` (ABSTRACT) unifies RT-threshold and model-based engagement classes in an Mplus multilevel mixture IRT and finds that the adjustments to proficiency are small.
- **Person-level classes from RT covariates.** `jeon2019analysis` (ABSTRACT) introduced a mixture with an ability class and a non-ability class whose membership depends on RT. **R4** (`yildirimhos2025how`, FULL TEXT) applies it to PISA 2022 Türkiye mathematics (N = 1,180; 11 MC items). The two-class model with RT covariates fits much better than a one-class 2PL (BIC 13,806.9 vs 14,395.7; entropy 0.814; 48%/52% split). Two problems remain. First, the one-class comparator carries **no time information at all**, so the improvement in fit cannot be separated from ordinary dependence between time and accuracy under a continuous (θ, τ) model. Second, the non-ability class has higher thresholds on almost every item, which is also what a lower-ability slice would show.
- **Engagement models.** Engagement can be modelled with RT, responses and omissions (`ulitzsch2020hierarchical`, `ulitzsch2020using`, BIB; `ulitzsch2024model`, ABSTRACT), as carelessness (`ulitzsch2024using`, BIB), or as person-level effort classes (`liu2020identifying`, BIB; `pokropek2016grade`, BIB).

**Descriptive mixtures in ILSAs.** `zheng2023identification` (FULL TEXT) is the closest analogue to Safeh's design. It fits finite mixtures to 28 item log-RTs plus item scores in eTIMSS 2019 for four countries:

- it finds 4–5 classes per country (Steady, Disengaged, Speeded, Efficient);
- the classes are chosen by BLRT with entropy above 0.9;
- prevalence differs by country, with Disengaged classes above 20% in England and the UAE and an "Efficient" class only in Singapore;
- there is no IRT ability adjustment and no continuous null;
- class solutions differ across countries, so measurement invariance is not established.

`erdemkara2025exploring` (ABSTRACT, PISA 2022) and `han2023behavioral` (ABSTRACT, PISA 2015 ColPS) use latent profile analysis on time, action counts and effort, again without adjusting for ability.

The review by Anghel, Khorramdel & von Davier (**R6**, `anghel2024use`, FULL TEXT; 221 studies) confirms that RT models and clustering of test-takers are major themes in the LSA process-data literature. It does not treat spurious classes or validation standards as a topic.

**Takeaway:** the field already has many models that detect behavioural classes from **item-level** times. The open question is not whether a class model can be fitted, but whether its classes **beat a properly specified continuous joint model** and replicate across samples and countries.

## 5. Spurious classes: the null problem is known, but rarely applied

The methodological literature has been explicit for two decades:

- `bauer2003distributional` (BIB): homogeneous but non-normal data produce extra mixture classes.
- `lubke2006distinguishing` (BIB) and `deboeck2005conceptual` (BIB) frame the choice between categories and dimensions.
- For mixture IRT specifically:
  - `alexeev2011spurious` (ABSTRACT): data generated from a one-class 2PL model required more than one class when fitted with a mixed Rasch model;
  - `sen2016impact` (ABSTRACT): MixRasch extracted spurious classes when θ was bimodal or uniform, while Mix2PL and Mix3PL were more robust.
- `cole2026exploring` (ABSTRACT): data from a single class with response styles were not recovered as one class by any scoring method. Nuisance person processes are a recent, confirmed source of spurious classes, and speed is a candidate nuisance process of exactly that kind.
- Model-based alternatives to classes for non-normal θ: Gaussian-mixture θ (`cheng2025estimating`, ABSTRACT) and a test for latent non-normality (`guastadisegni2025generalized`, ABSTRACT). Both show that a mixture for θ can be a density approximation rather than a set of substantive classes.
- Enumeration practice:
  - `nylund2007deciding` (BIB): BIC and BLRT perform best among the criteria compared;
  - `whittaker2024meta` (ABSTRACT): meta-analysis showing BIC accuracy rises with N;
  - `sen2026covariate` (ABSTRACT): BIC and CAIC are the most stable; entropy and likelihood-ratio tests are unreliable.

The important gap: the BLRT and most enumeration simulations use a **(K−1)-class mixture** as the null, not a **continuous latent-trait model**. Applied ILSA papers (R4, R5, Zheng 2023) report BLRT or IC wins over fewer classes. None of them estimates how often its own pipeline would find classes in data generated from a fitted continuous θ–τ model with that dataset's design, item parameters and time distributions. For clustering (Safeh), the same applies to silhouette values (`rousseeuw1987silhouettes`) and stability (`hennig2007clusterwise`). A silhouette of 0.355 has no meaning until it is compared with the null distribution of silhouettes under a continuous model.

Identifiability and label switching (`stephens2000dealing`, `jasra2005markov`, BIB) matter in practice. Any comparison across bootstrap replicates or countries needs class matching, and the ordinal constraints used in R5 build ability ordering into the class definition.

## 6. Prediction as a criterion: classes versus continuum

`ma2026comparison` (ABSTRACT, *Psychological Methods*, 2026) is the most direct precedent for our held-out evaluation. It compares cognitive diagnosis models (class-based) with the 2PL on predictive performance across 11 empirical datasets:

- the CDMs overfit;
- with MAP scoring, the CDM was never better than the 2PL;
- with marginal mastery probabilities, the CDM fitted better in only 6 of 11 datasets.

The paper changes the burden of proof. Class models should earn their place through **out-of-sample predictive gain** over a continuous model, not through in-sample fit. Its classes are CDM attribute profiles, not mixture-IRT or behavioural classes, and it does not use timing data. Our masked-item prediction and held-out time prediction therefore extend its logic rather than invent it.

## 7. Invariance, DIF and engagement across countries

Cross-country use of PISA brings in confounds that can look like classes:

- latent DIF (`choi2015differential`, `masyn2017measurement`, `wallin2026regularised`, R1);
- differential response-time functioning: in PISA 2018, 7–10% of items showed time DIF between multilingual and other students, with little overlap with accuracy DIF (`park2025measurement`, ABSTRACT);
- differential rapid guessing across language groups (`deng2025linking`, ABSTRACT, PISA 2018).

`rios2025effort` (ABSTRACT) shows that rapid guessing correlated with ability has little effect on aggregate θ. That reassures for scoring purposes, but the same correlation is what makes "disengaged" profiles hard to tell apart from "low-ability" profiles. `deribo2026devising` (ABSTRACT) proposes a four-step reporting workflow for rapid guessing, which our pipeline should follow. `chen2023who` (ABSTRACT) shows a meta-analytic way to summarise how class prevalence varies across education systems.

## 8. External validation and explainability

Class validation in this literature usually means looking at covariate profiles: gender and typing difficulty in Zheng 2023, SES and language in Chen 2023, institutional climate in R7 (`wang2025profiling`, BIB only, abstract not accessible). Bias-adjusted stepwise methods are available for this (`vermunt2010latent`, `bakk2018twostep`, R1). External correlates are necessary but **not sufficient**, because a slice of the continuum also correlates with covariates of ability. The diagnostic test is incremental validity: does class membership predict an external criterion (questionnaire effort, anxiety, ESCS) beyond (θ, τ)? `khosravi2022explainable` (ABSTRACT) provides the explainability framing. Labels presented to stakeholders, such as Safeh's emotion labels, must be backed by the evidence the model can actually provide.

## 9. What remains unsolved

1. **Continuous-null calibration of the pipeline in operational data.** Spurious-class simulations exist for generic non-normal θ (Sen 2016, Alexeev 2011, Bauer & Curran 2003). We found no study that generates data from a **fitted** hierarchical θ–τ model (a van der Linden-type model with the empirical item parameters, booklet design and time distributions) and estimates the false-positive rate of the **exact** profiling pipeline applied, whether K-means on score and time, latent profile analysis, or mixture IRT with RT covariates.
2. **Granularity of time.** RT-based class models (R3, R4, Zheng 2023, Huang 2026, Nagy & Ulitzsch 2022) all need item-level times. No study we found compares, on the same data, what can be identified from **total time only** (Safeh's minimal-data setting) with what can be identified from **item-level times**. That comparison is a direct test of whether minimal process data can support behavioural profiling.
3. **Classes versus continuum, judged by prediction.** Ma et al. (2026) did this for CDMs versus IRT, without timing data. It has not been done for behavioural or mixture-IRT classes defined jointly by time and accuracy.
4. **Invariance of class solutions across countries.** Zheng 2023 finds different numbers of classes per country. Formal tests of whether the same class structure holds, including invariance of time intensities (Park 2025), are missing.
5. **Incremental external validity.** Whether class membership predicts external criteria beyond (θ, τ) is rarely tested. Correlations with covariates are usually reported as though they were validation.
6. **Data caveat for our own study.** We could not verify that the PISA 2025 public-use files, or their item-level timing variables, are available. Secondary sources give 8 September 2026 for the main PISA 2025 release; we did not confirm the database release. The adaptive or multistage design used in recent PISA cycles also creates structured missingness, which must be modelled (missing at random given routing) and built into the null simulations. If PISA 2025 timing data are not available, PISA 2022 can substitute (R4 shows it has item-level timing).

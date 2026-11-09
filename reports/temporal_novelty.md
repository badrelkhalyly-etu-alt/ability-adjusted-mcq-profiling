# Novelty of the temporal-regime analysis (option C) relative to recent literature

*All sources below are in `manuscript/references.bib` and were checked against Crossref. The 2026 entries come from metadata and abstracts; see `literature/scopus_verification.md`.*

## What already exists

| Line of work | What it does | Gap relative to this study |
|---|---|---|
| Mixtures of item-level response times and accuracy (Wang & Xu 2015; Molenaar & De Boeck 2018; Liu et al. 2020; Ulitzsch et al. 2020, 2024) | Separates solution behaviour from rapid guessing or disengagement, or lets item parameters depend on fast vs slow responses, using **item-level** times | These models assume the latent states exist and estimate them. They do not test the states against a flexible **continuous** time distribution with the same response model, and they do not use person-level summary time |
| Hidden Markov IRT for responses and times (Molenaar et al. 2016) | Lets persons switch between latent states during a test | Same: the comparison is with a homogeneous log-normal model, not with skewed or heavy-tailed continuous alternatives |
| Hierarchical speed–accuracy model (van der Linden 2007) and its residual-dependence extensions (Bolsinova et al. 2017) | Treats speed as a continuous latent trait, with log-normal times | Normality of log time given speed is assumed. Non-normal continuous alternatives are not compared with class models |
| Spurious classes from non-normality (Bauer & Curran 2003; Alexeev et al. 2011; Sen et al. 2016) | Shows that mixtures over-extract classes when a continuous distribution is skewed or misspecified | The work is general (growth mixtures, mixture IRT). We found no application to response-time regimes in large-scale assessment that **calibrates the regime decision against several fitted non-normal continuous nulls** |
| Classes versus dimensions (De Boeck et al. 2005; Lubke & Neale 2006) | Gives a conceptual and likelihood framework for deciding between categories and continua | It is not applied to timing data with a fixed IRT response model and held-out evaluation |
| Distribution families: GAMLSS (Rigby & Stasinopoulos 2005), sinh-arcsinh (Jones & Pewsey 2009) | Model location, scale and shape (skewness, tail weight) as smooth functions of covariates | Not used as competitors to latent time regimes in psychometric timing data |

## What this study adds

All points below are methodological. None is a new algorithm.

1. **A falsifiable test of time regimes.** It compares a regime model, whose prevalence depends on ability so that regimes cannot be ability groups, with six continuous families: GAM-type spline location and variance, skew-normal, Student t and SHASH residuals. All are conditional on **the same** IRT response model, so the likelihoods are strictly comparable.
2. **Three sources of evidence combined in one decision rule:**
   - a null-calibrated BIC threshold from several fitted non-normal continuous generators;
   - a held-out conditional log-density on independent forms;
   - the bootstrap stability of the regimes.
3. **A test of item composition.** The test is run on two time variables:
   - an item-intensity-adjusted relative time, which removes which items were reached and how long they take;
   - total time, which keeps them.
   Comparing the two shows how much apparent regime structure is due to item composition.
4. **Independent validation.** The test sets of two PISA 2025 forms, with disjoint items and students, were never used before the protocol was frozen.

## Limits of the claim of novelty

- A mixture of two normals is itself a flexible continuous distribution, so data on one time variable can only partly separate "discrete regimes" from "a non-normal continuum". The decision rule therefore asks whether regimes predict better than the best continuous competitors *and* behave like regimes (stable, beyond what continuous nulls produce), not whether they "exist".
- Item-level regime models, which use within-person switching across items, are richer than our person-level regimes. A negative result here does not rule out item-level engagement states such as rapid guessing on individual items.

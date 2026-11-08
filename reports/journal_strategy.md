# Journal strategy: ability-adjusted latent response classes (mixture IRT + response time, PISA 2025)

Prepared 2026-11-08. Every quartile, APC and URL below was checked on that date. Where something could not be checked, it says so.

## 0. How each item was verified (read first)

| Item | Source used | Status |
|---|---|---|
| SJR value / SJR quartile | SCImago journal pages (scimagojr.com). **Direct fetches returned HTTP 403.** The values came from the search engine's index of those same pages, so they are SCImago's figures but were not read off a page loaded live. | Partly verified. Open each SCImago link in a browser before citing a figure. |
| SJR category-level quartile | Same as above. The per-category breakdown came through for only some journals. | Marked per journal |
| Web of Science JCR quartile | Clarivate JCR needs a subscription. **No JCR quartile in this file was verified.** SJR (Scopus-based) quartiles are **not** JCR quartiles. They use different databases, metrics and category schemes, and the two often differ. | **Unverified for all journals** |
| APC / OA model | Publisher price lists downloaded directly: Elsevier APC xlsx (prices as of 05-Oct-2026), Wiley hybrid OA xlsx (updated 12-Oct-2026), Sage Choice 2026 xlsx, Springer journal pages, Cambridge fees page, JLA policies page | Verified, except Taylor & Francis (MBR) |
| Recent relevant papers | Crossref API metadata (DOI, year, journal ISSN filter) | Verified to exist in the named journal. Abstracts were not read. Relevance is judged from titles only. |

The Elsevier and Taylor & Francis journal websites also returned 403. Their scope wording comes from the publishers' price lists and from secondary listings, as marked.

Data timing: OECD launched PISA 2025 results on 8 Sep 2026 ([OECD launch speech](https://www.oecd.org/en/about/news/speech-statements/2026/09/international-launch-of-pisa-2025.html)). I did not check whether the public-use cognitive item and timing files are out yet. This must be confirmed before any submission timeline is set.

---

## 1. Educational and Psychological Measurement (EPM), Sage

- **Scope fit: high.** The journal publishes "theoretical and applied papers in the measurement field": new methods, or novel uses of existing techniques ([Sage journal page](https://us.sagepub.com/en-us/nam/educational-and-psychological-measurement/journal200914)). It regularly publishes mixture-IRT/response-time work and simulation-plus-empirical designs.
- **SJR:** 2025 SJR 2.630. Q1 in *Developmental and Educational Psychology*, ranked 11th in that category; Q1 there every year 2015–2025 ([SCImago](https://www.scimagojr.com/journalsearch.php?q=13563&tip=sid)). Other Scopus categories were not seen.
- **JCR:** quartile unverified. The Sage page shows an impact factor of 4.4 (5-year 4.5). That is a Clarivate number, but the page gives no year or category quartile.
- **OA / APC:** hybrid (Sage Choice). 2026 APC **USD 3,910 / GBP 2,867** ([Sage Choice 2026 price list, xlsx](https://www.sagepub.com/docs/default-source/rp-pages/info-for-authors/hybrid-oa-sage-choice/sage-choice-price-list-2026---external.xlsx), linked from [Sage publishing options](https://www.sagepub.com/journals/information-for-authors/publishing-options)). Subscription publication is free.
- **Methodological expectations:** a simulation study with clear factors and outcomes (Type I error, power, bias, classification accuracy), plus an empirical illustration. Reproducible code is increasingly expected.
- **Recent relevant papers (Crossref-verified):**
  - Nagy et al. (2021), multilevel mixture IRT with response times as predictors or indicators of engagement. doi:10.1177/00131644211045351
  - Bakk et al. (2026), two-step estimation of latent class-IRT models in the presence of DIF and multidimensionality. doi:10.1177/00131644261467757
  - Man et al. (2022), mixture hierarchical model of responses and RTs for preknowledge. doi:10.1177/00131644221136142
  - Rios et al. (2023), RT-threshold scoring under rapid guessing. doi:10.1177/00131644231168398
  - Holopainen et al. (2026), misclassification by rapid-guessing identification methods. doi:10.1177/00131644261419426
- **Desk-rejection risks:**
  - Being read as "routine application of mixture IRT to a new dataset" with no methodological contribution.
  - The K-means/online-exam backstory reading as learning analytics, which is out of scope here.
- **Extra evidence needed:**
  - Frame the null-calibrated class-enumeration procedure as the contribution: the false-positive rate of BIC/aBIC/bootstrap LRT for spurious classes under continuous-ability + lognormal-RT generating models.
  - A factorial simulation covering N, test length, RT–ability correlation, misspecified RT distribution and missing-by-design.
  - Positive controls that show power.
  - Explicitly position the work against Bakk et al. (2026) and Nagy et al. (2021).
- **Suitability:** positive result high, negative result **high**, provided the null is the calibrated output of a method that has been shown to have power.

## 2. Large-scale Assessments in Education (LSA), IEA/IERI, Springer Nature

- **Scope fit: high on the data side.** Covers "design, measurement, analysis … of large-scale assessments", and names PISA explicitly. The scope says the journal values studies that test competing explanations or provide "evidence that meaningfully changes confidence in an established finding" ([aims and scope](https://link.springer.com/journal/40536/aims-and-scope)). That wording is a good fit for a rigorous negative result.
- **SJR:** 2025 SJR 1.097, Q1 in *Education*, Q1 every year 2019–2025 ([SCImago](https://www.scimagojr.com/journalsearch.php?q=21100901162&tip=sid)).
- **JCR:** unverified, including whether the journal is in SSCI at all.
- **OA / APC:** fully OA. **No fee to authors.** "the article processing charge (APC) is covered by the International Association for the Evaluation of Educational Achievement" ([fees and funding](https://link.springer.com/journal/40536/how-to-publish-with-us)).
- **Methodological expectations:** the scope page treats "correct treatment of … sampling weights, plausible values, replication methods, clustering, and cross-group comparability" as a baseline requirement.
- **Recent relevant papers (Crossref-verified):**
  - Han et al. (2023), latent profile analysis of response times and actions in PISA 2015 CPS. doi:10.1186/s40536-023-00185-5
  - Kuang et al. (2023), disengagement and its effect on item parameters, PISA 2015 vs 2018 (US). doi:10.1186/s40536-023-00152-0
  - Papanastasiou et al. (2024), process-data time-management indicators. doi:10.1186/s40536-024-00193-z
  - Deng (2025), linking error from rapid guessing in multigroup concurrent IRT. doi:10.1186/s40536-025-00265-8
- **Desk-rejection risks:**
  - The scope says that "the large-scale assessment serves primarily as a convenient source of data" will be out of scope. If PISA is only a convenient testbed, this applies.
  - Ignoring weights, the multistage adaptive design, or cross-country comparability.
  - Analysing a single small country subset.
- **Extra evidence needed:**
  - Use weights, or justify unweighted model comparison.
  - Handle PISA's adaptive design: routing-induced missingness of item-level times.
  - Check class-structure invariance across several countries or languages; this is where the DIF checks belong.
  - Say what the result means for PISA reporting, for example whether process-data "profiles" add anything beyond the proficiency scale.
  - Drop the K-means exam story to a short motivation.
- **Suitability:** positive result high, negative result **high**. Zero APC.

## 3. Behavior Research Methods (BRM), Psychonomic Society, Springer

- **Scope fit: medium–high.** The scope is "methods, techniques, and instrumentation of research in experimental psychology … particularly … computer technology" ([aims and scope](https://link.springer.com/journal/13428/aims-and-scope)). In practice the journal publishes much psychometric and latent-class methods work, often with tutorials and software.
- **SJR:** 2025 SJR 2.382. Q1 in *Experimental and Cognitive Psychology* (4th), *Developmental and Educational Psychology*, *Psychology (misc.)* and *Arts and Humanities (misc.)* ([SCImago](https://www.scimagojr.com/journalsearch.php?tip=sid&q=146223)).
- **JCR:** unverified.
- **OA / APC:** hybrid. APC **GBP 2,690 / USD 4,090 / EUR 3,090** plus VAT ([how to publish](https://link.springer.com/journal/13428/how-to-publish-with-us)).
- **Methodological expectations:**
  - Open code, ideally a reusable package or pipeline.
  - Simulation evidence.
  - A worked example.
  - Sorgente et al. (2025) set out practice guidelines that reviewers will likely apply to latent class work.
- **Recent relevant papers (Crossref-verified):**
  - Huang et al. (2025), RT-based mixture IRT for dynamic response strategies. doi:10.3758/s13428-024-02555-5
  - Uglanova et al. (2026), when mixture models identify careless responding: simulation and validity evidence. doi:10.3758/s13428-026-03165-z
  - Sorgente et al. (2025), systematic review of LCA practice vs guidelines. doi:10.3758/s13428-025-02812-1
  - Lu et al. (2025), joint GIRT + RT model. doi:10.3758/s13428-025-02855-4
- **Desk-rejection risks:**
  - Substantive educational finding with no reusable method or tool.
  - Educational-assessment framing seen as outside experimental psychology.
- **Extra evidence needed:**
  - Package the null-calibration and positive-control workflow as an open, documented tool, for example an R/Python function.
  - Present it as a general "is my mixture real?" check, with PISA as the case study.
- **Suitability:** positive result medium–high, negative result **high**. A demonstrated false-positive-control tool is valuable whatever the empirical answer.

## 4. Journal of Educational Measurement (JEM), NCME, Wiley

- **Scope fit: high, but the bar is high.** It is NCME's flagship methods journal and publishes RT, PISA-design and latent-class measurement papers. The official scope page could not be fetched (Wiley returned 403).
- **SJR:** 2025 SJR 1.027, best quartile Q1. *Applied Psychology* is Q2 for 2023–2025 ([SCImago](https://www.scimagojr.com/journalsearch.php?q=66374&tip=sid)). The category that carries the Q1 was not visible.
- **JCR:** unverified.
- **OA / APC:** hybrid. APC **USD 3,350 / GBP 2,660 / EUR 3,050** ([Wiley hybrid OA price list, xlsx, updated 12-Oct-2026](https://authors.wiley.com/asset/Wiley-Journal-APCs-OnlineOpen.xlsx), linked from [Wiley hybrid OA page](https://authors.wiley.com/author-resources/Journal-Authors/open-access/hybrid-open-access/index.html)).
- **Methodological expectations:** a new or substantially improved measurement method, or a rigorous evaluation of an existing one, with operational relevance.
- **Recent relevant papers (Crossref-verified):**
  - Shin et al. (2024), robustness of IRT models under PISA MSAT designs. doi:10.1111/jedm.12409
  - Mutak et al. (2024), intraindividual relation of ability and speed. doi:10.1111/jedm.12391
  - Gergely et al. (2025), "Toward Psychometric Learning Analytics" (Urnings + RTs). doi:10.1111/jedm.70026
  - Binici et al. (2022), latent class analysis to validate performance standards. doi:10.1111/jedm.12325
- **Desk-rejection risks:** the contribution seen as incremental, i.e. an application rather than a method.
- **Extra evidence needed:** the same as for EPM, plus an explicit operational recommendation for PISA-type MSAT data. Shin et al. (2024) is a template for how to handle MSAT.
- **Suitability:** positive result medium–high, negative result medium. Possible if the calibration procedure is novel, harder than EPM.

## 5. Applied Psychological Measurement (APM), Sage

- **Scope fit: medium–high.** Applied psychometric methods. Short-form contributions such as class-enumeration notes appear here.
- **SJR:** 2025 SJR 1.287, best quartile Q1 ([SCImago](https://www.scimagojr.com/journalsearch.php?q=12028&tip=sid&clean=0)). The category breakdown for 2015–2025 was not visible; *Psychology (misc.)* was Q1 in 1999–2014.
- **JCR:** unverified.
- **OA / APC:** hybrid (Sage Choice). 2026 APC **USD 3,910 / GBP 2,867** ([Sage Choice 2026 xlsx](https://www.sagepub.com/docs/default-source/rp-pages/info-for-authors/hybrid-oa-sage-choice/sage-choice-price-list-2026---external.xlsx)).
- **Recent relevant papers (Crossref-verified):**
  - Sen et al. (2025), the elbow-plot method for class enumeration in factor mixture models. doi:10.1177/01466216251344288
  - Jin et al. (2022), modelling rapid guessing in computer-based testlets. doi:10.1177/01466216221125177
  - Kim et al. (2022), glca R package for multiple-group LCA. doi:10.1177/01466216221084197
- **Desk-rejection risks:** the empirical PISA part being too large relative to the methodological core.
- **Extra evidence needed:** a tight simulation on the enumeration false-positive rate. A shorter manuscript would suit.
- **Suitability:** positive result medium, negative result medium–high, especially as a focused note on spurious classes.

## 6. Multivariate Behavioral Research (MBR), SMEP, Taylor & Francis (Routledge)

- **Scope fit: medium.** "dissemination, evaluation, and application of quantitative methods to the behavioral sciences". Regular articles are judged on "accessibility to a wide audience, potential to change the field, and innovativeness". The journal also takes tutorials and encourages sharing code and data ([aims and scope](https://www.tandfonline.com/journals/hmbr20/about-this-journal), read via search index because the direct fetch was blocked).
- **SJR:** 2025 SJR 2.017, Q1 in *Statistics and Probability* (18 of 304) ([SCImago](https://www.scimagojr.com/journalsearch.php?q=12711&tip=sid)). Other categories were not seen.
- **JCR:** unverified.
- **OA / APC:** hybrid (T&F Open Select). **APC unverified.** T&F prices are only available through the interactive cost finder ([OA options finder](https://authorservices.taylorandfrancis.com/publishing-open-access/oa-options-finder/)), which I did not query.
- **Recent relevant papers (Crossref-verified):**
  - Hauenstein et al. (2022), mixture explanatory IRT for dynamic processes. doi:10.1080/00273171.2021.2011699
  - Kang (2024), latent structure of responses and RTs. doi:10.1080/00273171.2024.2436406
  - Cheng et al. (2025), IRT under Gaussian-mixture latent traits. doi:10.1080/00273171.2025.2512345. This one is directly relevant: a non-normal continuous trait can produce spurious classes.
- **Desk-rejection risks:** not general or innovative enough for the broad behavioural-methods audience.
- **Extra evidence needed:**
  - Generalise the finding to "mixture IRT + RT class enumeration in large samples".
  - Show the distinction between latent-trait non-normality and true latent classes, building on Cheng et al. (2025).
- **Suitability:** positive result medium, negative result medium. Better as a tutorial-style paper.

## 7. British Journal of Mathematical and Statistical Psychology (BJMSP), BPS, Wiley

- **Scope fit: low–medium.** The journal favours new models, estimation algorithms and critiques of models (per secondary listings; the Wiley page was blocked). An evaluation of existing models on PISA is unlikely to clear its novelty bar.
- **SJR:** Q1 in *Psychology (misc.)* for 2025 ([SCImago](https://www.scimagojr.com/journalsearch.php?q=12773&tip=sid)). The numeric SJR value is **unverified**.
- **JCR:** unverified.
- **OA / APC:** hybrid. APC **USD 3,570 / GBP 2,380 / EUR 2,990** ([Wiley hybrid OA xlsx](https://authors.wiley.com/asset/Wiley-Journal-APCs-OnlineOpen.xlsx)).
- **Recent relevant papers (Crossref-verified):**
  - He et al. (2024), sparse latent class model incorporating response times. doi:10.1111/bmsp.12380
  - Gorney et al. (2023), person-fit using scores and RTs. doi:10.1111/bmsp.12320
  - Wallin et al. (2026), regularised latent-class IRT for non-invariance. doi:10.1111/bmsp.70076
- **Desk-rejection risks:** no new statistical model or theory.
- **Suitability:** positive result low–medium, negative result low, unless the calibration method has formal, new statistical properties.

## 8. Journal of Educational and Behavioral Statistics (JEBS), AERA/ASA, Sage

- **Scope fit: low–medium.** A statistics-methods venue. It publishes RT mixture models, but expects statistical novelty.
- **SJR:** 2025 SJR 1.445, best quartile Q1 ([SCImago](https://www.scimagojr.com/journalsearch.php?q=16444&tip=sid)). The category breakdown was **not** seen.
- **JCR:** unverified.
- **OA / APC:** hybrid (Sage Choice). 2026 APC **USD 3,640 / GBP 2,669** ([Sage Choice 2026 xlsx](https://www.sagepub.com/docs/default-source/rp-pages/info-for-authors/hybrid-oa-sage-choice/sage-choice-price-list-2026---external.xlsx)).
- **Recent relevant papers (Crossref-verified):**
  - Kanopka et al. (2024), position-sensitive mixture item response model. doi:10.3102/10769986241289399
  - Ulitzsch et al. (2023), RTs for careless responding and response styles. doi:10.3102/10769986231173607
  - Xi et al. (2024), mixture response model for item preknowledge. doi:10.3102/10769986241268460
- **Suitability:** positive result low–medium, negative result low.

## 9. Psychometrika, Psychometric Society, Cambridge University Press

- **Scope fit: low.** The flagship theory journal. It publishes RT latent-response mixture models (Ulitzsch et al. 2022), but an applied evaluation is unlikely to fit.
- **SJR:** 2025 SJR 1.928, Q1 (seen in Mathematics / Applied Mathematics listings) ([SCImago](https://www.scimagojr.com/journalsearch.php?q=14061&tip=sid)). The full category table was **not** seen.
- **JCR:** unverified.
- **OA / APC:** **fully gold OA**. APC **USD 3,655 / GBP 2,610**; Society member rate USD 2,740 / GBP 1,960. Many institutional agreements cover the charge ([Cambridge fees and pricing](https://www.cambridge.org/core/journals/psychometrika/information/author-instructions/fees-and-pricing)). A search snippet showed older figures (USD 3,550); the live page was used here.
- **Recent relevant papers (Crossref-verified):**
  - Ulitzsch et al. (2022), RT-based latent response mixture model. doi:10.1007/s11336-021-09817-7
  - Kwon et al. (2026), copula-based joint model for speed and accuracy. doi:10.1017/psy.2026.10143
- **Suitability:** positive result low, negative result very low. Not recommended.

## 10. Journal of Learning Analytics (JLA), SoLAR

- **Scope fit: medium, for the "behavioural profiles" lineage.** Learning-analytics research on "collecting, analyzing, and reporting data to improve learning" ([about](https://learning-analytics.info/index.php/JLA/about)). The paper would fit as a measurement-grounded critique of clustering-based learner profiling from score and time data.
- **SJR:** 2025 SJR 0.911, best quartile Q1 ([SCImago](https://www.scimagojr.com/journalsearch.php?q=21100932472&tip=sid)). The category was **not** seen. The SJR fell from 1.496 (2023).
- **JCR:** unverified.
- **OA / APC:** diamond OA. "does not charge authors submission or processing charges … costs are covered by … SoLAR" ([JLA policies](https://learning-analytics.info/index.php/JLA/about/editorialPolicies)).
- **Recent relevant papers (Crossref-verified):**
  - Papageorgiou et al. (2025), nonlinear effort–time dynamics of engagement. doi:10.18608/jla.2025.8663
  - Albano et al. (2023), NLP management of MCQ item repositories. doi:10.18608/jla.2023.7897
  - Closest psychometric-LA bridge, published in JEM rather than JLA: Gergely et al. (2025).
- **Desk-rejection risks:**
  - Too psychometric, with no implication for learning or practice.
  - PISA is not a learning environment.
- **Extra evidence needed:**
  - Put the K-means online-exam study in the foreground.
  - Show directly that K-means score/time profiles collapse once ability is adjusted for, on both the exam and PISA.
  - State the implication for LA dashboards and profiling.
- **Suitability:** positive result medium, negative result medium–high, as a corrective for the field.

## 11. Computers & Education (C&E), Elsevier

- **Scope fit: low.** The journal is about the pedagogical uses of digital technology. Per secondary listings, narrow single-institution evaluations, and work where the computer is only a delivery platform, are not accepted (the Elsevier guide could not be fetched; [secondary summary](https://peeref.com/journals/2003)). A psychometric model comparison on PISA has no pedagogy-technology question.
- **SJR:** 2025 SJR 4.434, Q1 in *Education* every year 2001–2025 ([SCImago](https://www.scimagojr.com/journalsearch.php?q=17645&tip=sid)). Other categories were not seen.
- **JCR:** unverified.
- **OA / APC:** hybrid. APC **USD 4,630 / EUR 4,220 / GBP 3,710**, excluding taxes ([Elsevier APC price list xlsx, as of 05-Oct-2026](https://legacyfileshare.elsevier.com/els_com_pricing/article-publishing-charge.xlsx), linked from [Elsevier pricing](https://www.elsevier.com/about/policies-and-standards/pricing)).
- **Recent relevant papers (Crossref-verified):**
  - Barthakur et al. (2022), integrating learning analytics and measurement theory in online courses. doi:10.1016/j.compedu.2022.104603
  - Jiang et al. (2025), timer and proctoring settings for online exams. doi:10.1016/j.compedu.2025.105436
- **Desk-rejection risk:** very high, on scope.
- **Suitability:** positive result low, negative result very low.

## 12. Computers & Education: Artificial Intelligence (CAEAI), Elsevier

- **Scope fit: very low.** The journal covers AI in education and AI education. Mixture IRT is not an AI contribution.
- **SJR:** 2025 SJR 7.083. Q1 in *Artificial Intelligence* (1st of 549) and *Education*, 2022–2025 ([SCImago](https://www.scimagojr.com/journalsearch.php?q=21101074919&tip=sid)).
- **JCR:** unverified.
- **OA / APC:** fully OA. APC **USD 3,620 / EUR 3,300 / GBP 2,900** ([Elsevier APC xlsx](https://legacyfileshare.elsevier.com/els_com_pricing/article-publishing-charge.xlsx)).
- **Recent paper:** Schmucker et al. (2026), item-writing flaws and IRT difficulty/discrimination. doi:10.1016/j.caeai.2026.100632. This shows IRT appears there only alongside an AI or LLM angle.
- **Suitability:** not recommended.

## 13. Journal of Computer Assisted Learning (JCAL), Wiley

- **Scope fit: low–medium.** Covers ICT to support learning. Test-taking engagement papers do appear there.
- **SJR:** 2025 SJR 2.155, Q1 in *E-learning* (8th of 69) ([SCImago](https://www.scimagojr.com/journalsearch.php?q=26183&tip=sid)).
- **JCR:** unverified.
- **OA / APC:** hybrid. APC **USD 4,360 / GBP 2,920 / EUR 3,650** ([Wiley hybrid OA xlsx](https://authors.wiley.com/asset/Wiley-Journal-APCs-OnlineOpen.xlsx)).
- **Recent relevant paper:** Nagy et al. (2022), rapid guessing and test-taking persistence in modelling engagement. doi:10.1111/jcal.12719
- **Desk-rejection risk:** the reviewers will expect a learning or ICT contribution, and the psychometric framing will look out of scope.
- **Suitability:** positive result low–medium, negative result low.

---

## 14. Cross-cutting evidence any top venue will expect

1. **Null calibration:** simulate from the fitted continuous model (2PL/3PL + lognormal RT, van der Linden hierarchical). Use realistic non-normal ability and RT distributions and PISA-like missing-by-design. Report the false-positive rate of each enumeration criterion. Without this, "K > 1 classes" is uninterpretable.
2. **Positive controls:** state the power to recover planted classes, such as rapid-guessing or strategy classes, at plausible effect sizes. A negative result is only credible if power was shown.
3. **Distinguish non-normal trait from class structure,** following Cheng et al. (2025, MBR). Run a semi-parametric or Gaussian-mixture ability model as the competitor.
4. **Masked-item held-out prediction:** the predictive gain of mixture over continuous models, with uncertainty, on held-out items and persons.
5. **PISA design:** weights, adaptive routing, cross-country invariance or DIF of class solutions, and multiple countries.
6. **Pre-registration or a frozen analysis plan,** plus open code. This strengthens a negative result at EPM, BRM and LSA.

## 15. Ranked shortlist

| Rank | Journal | SJR 2025 (SCImago, Q) | APC (OA route) | If positive | If negative | Why |
|---|---|---|---|---|---|---|
| 1 | **Educational and Psychological Measurement** | 2.630, Q1 Dev. & Educ. Psych. | USD 3,910 optional (hybrid) | High | High | Best methods fit; publishes RT mixture and rapid-guessing simulation work |
| 2 | **Large-scale Assessments in Education** | 1.097, Q1 Education | **None** (IEA-funded OA) | High | High | PISA-native; scope explicitly values evidence that "changes confidence in an established finding"; must meet design requirements |
| 3 | **Behavior Research Methods** | 2.382, Q1 Exp. & Cog. Psych. | USD 4,090 optional (hybrid) | Med–high | High | Best if framed as a reusable false-positive-control tool |
| 4 | Journal of Educational Measurement | 1.027, best Q1 | USD 3,350 optional (hybrid) | Med–high | Medium | High bar; needs a methodological novelty claim |
| 5 | Applied Psychological Measurement | 1.287, best Q1 | USD 3,910 optional (hybrid) | Medium | Med–high | Fallback as a focused enumeration note |
| 6 | Journal of Learning Analytics | 0.911, best Q1 | None (diamond) | Medium | Med–high | Fallback if framed as a critique of K-means profiling |
| 7 | Multivariate Behavioral Research | 2.017, Q1 Stat. & Prob. | Unverified (hybrid) | Medium | Medium | Needs a broader methods framing |
| — | BJMSP, JEBS, Psychometrika | Q1 | see above | Low | Low | Need new statistical methodology |
| — | C&E, CAEAI, JCAL | Q1 | see above | Low | Very low | Out of scope (pedagogy, AI or ICT) |

**Suggested sequence:** EPM, then LSA, then BRM. Choose LSA first if the PISA design treatment (weights, MSAT, multi-country) is complete and a zero APC matters. Choose BRM first if the calibration workflow is released as a package.

**Gaps still open before submission:**
- Confirm every SJR figure by loading the SCImago pages in a browser; direct fetch was blocked.
- Get JCR quartiles from Clarivate through institutional access; none were verified.
- Get the MBR APC from the T&F cost finder.
- Find the numeric SJR for BJMSP.
- Find the category-level SJR quartiles for JEM, APM, JEBS, JLA and Psychometrika.
- Read the official Elsevier and Wiley author guides, which were blocked.
- Confirm that PISA 2025 public-use item-level timing data are released.

# Scopus / SJR verification log (2026-11-08)

## Method and honest limits

- **Scopus** (scopus.com) was not accessed: no subscription session was available. **No article-level Scopus record was verified for any paper.** Every row in `related_work_matrix.csv` says so in `scopus_article_verified`.
- **Journal level.** SCImago Journal Rank (SJR) is computed from Scopus data, so a SCImago journal profile with current-year values is used as evidence that the journal is indexed in Scopus.
- **Access problem.** Direct fetches of scimagojr.com (pages and the CSV export) returned **HTTP 403 / Cloudflare challenge**. The quartiles below therefore come from scimagojr.com journal-profile pages as shown in web-search results restricted to `scimagojr.com`, which were summarised by the search tool. This is secondary confirmation. Re-check each value in a browser before putting it in the manuscript.
- **Article metadata.** Article existence and metadata (authors, title, journal, volume, issue, pages) were verified for **all 70 DOIs** through the Crossref REST API. Abstracts were checked through Crossref, OpenAlex or Europe PMC where available.

## Journal-level table

| Journal | Papers in matrix | How indexing was checked | Quartile (year; category) | Source of quartile |
|---|---|---|---|---|
| Educational and Psychological Measurement | R1, A23, A26, A34, A47 | SCImago profile id 13563 found | 2025 Q1 (Applied Mathematics; Developmental & Educational Psychology); SJR 2.630 | scimagojr snippet. One fragment showed 6.305, likely a parsing error |
| Multivariate Behavioral Research | R2, A07, A20, A22, A24, A36 | SCImago id 12711 | 2024 Q1 and 2025 Q1 (Statistics & Probability; Experimental & Cognitive Psychology); SJR 2024 2.711 | scimagojr snippet |
| Behavior Research Methods | R3, A17, A30 | SCImago id 146223 | 2024 Q1 and 2025 Q1 (Experimental & Cognitive Psychology; Developmental & Educational Psychology); SJR 2024 2.462 | scimagojr snippet |
| Education and Science (Eğitim ve Bilim) | R4 | SCImago id 21100283777 | 2025 **Q3** (Education); SJR 0.250 | scimagojr snippet |
| Large-scale Assessments in Education | R5, R6, A27, A28, A46, A49, A51, A54 | SCImago id 21100901162 | 2025 Q1 (Education); SJR 1.097; Q1 2019–2025 | scimagojr snippet |
| Computers & Education | R7 | SCImago id 17645 | 2025 Q1 (Education; E-learning); SJR 4.434 | scimagojr snippet |
| Computers and Education: Artificial Intelligence | A60 | SCImago id 21101074919 | 2025 Q1; SJR 7.083 (category not captured) | scimagojr snippet |
| Psychological Methods | A05, A32, A33 | SCImago id 13461 | 2025 Q1 (Psychology misc.); SJR 4.769 | scimagojr snippet |
| Journal of Educational and Behavioral Statistics | A04, A25, A31, A53, A62 | SCImago id 16444 | 2025 Q1 (Education); SJR 1.445 | scimagojr snippet |
| Psychometrika | A02, A03, A16, A18, A44, A55, A61 | SCImago id 14061 | 2025 Q1 (Psychology misc.; Applied Mathematics); SJR 1.928 | scimagojr snippet |
| Educational Measurement: Issues and Practice | A13, A41, A45, A48 | SCImago id 100147352 | 2025 Q1 (Education); SJR 0.836 | scimagojr snippet |
| International Electronic Journal of Elementary Education | A29 | SCImago id 19700176214 | 2025 **Q3** (Education); SJR 0.284 | scimagojr snippet |
| British Journal of Mathematical and Statistical Psychology | A15, A21, A37, A38 | SCImago id 12773 | 2025 Q1 (Psychology misc.); SJR value not captured | scimagojr snippet |
| Journal of Educational Measurement | A09, A12, A14 | SCImago id 66374 seen in search index | NOT VERIFIED | — |
| Applied Psychological Measurement | A01, A10 | SCImago id 12028 seen in search index | NOT VERIFIED | — |
| Applied Measurement in Education | A11 | SCImago id 29450 seen in search index | NOT VERIFIED | — |
| Structural Equation Modeling | A06, A35, A42 | not checked | NOT VERIFIED | — |
| International Journal of Testing | A39, A40, A52 | not checked | NOT VERIFIED | — |
| Journal of Educational Psychology; Psychological Review; Frontiers in Psychology; Political Analysis; JRSS-B; Statistical Science; CSDA; J. Computational and Applied Mathematics; Communications in Statistics – Theory and Methods | seminal / methodological refs | not checked | NOT VERIFIED | — |

## Article-level status (mandatory papers)

| Paper | DOI resolves (Crossref) | Metadata matches user-supplied citation? | Article-level Scopus | Content access |
|---|---|---|---|---|
| R1 Bakk & Fabbricatore 2026 | Yes | Yes. EPM online first 2026-08-21; **no volume or pages yet** (article number 00131644261467757) | Not verified | Full text (Europe PMC PMC13498696, CC-BY) |
| R2 Bacci, Fabbricatore & Iannario 2026 | Yes | Yes. MBR online first 2026-06-27, pp. 1–26; **no volume yet**. Exact title: "A Unified Framework for Jointly modelling Response Times and Item Position Effects in Computer-Based Learning Assessments" | Not verified | Abstract only (Europe PMC / PubMed 42364132; tandfonline 403) |
| R3 Huang, Luo & Jeon 2025 | Yes | Yes. BRM 57(1), article 54, published online 2025-01-09 | Not verified | Abstract only (PubMed 39789213) |
| R4 Yıldırım Hoş & Uysal Saraç 2025 | Yes | Yes. 50, 129–146. Crossref lists no issue; the PDF header confirms "2025 Supplement 1" | Not verified | Full text (open PDF, methods and results read) |
| R5 AlHakmani & Sheng 2024 | Yes | Yes. LSAE 12(1):37 | Not verified | Full text (open access) |
| R6 Anghel, Khorramdel & von Davier 2024 | Yes | Yes. LSAE 12(1):13 | Not verified | Full text (open access; key sections) |
| R7 Wang, Yin & King 2025 | Yes | Yes. C&E 227:105209 (April 2025) | Not verified | **Bibliography only.** The abstract could not be retrieved (ScienceDirect 403; absent from Crossref, OpenAlex and Semantic Scholar) |

**No user-supplied DOI failed.** All seven resolve, and the metadata match the user's descriptions. The only differences are the missing volume and issue for the two 2026 online-first articles (R1, R2) and the missing issue in Crossref for R4.

## DOIs I tried that failed (not used)

- `10.1111/j.1745-3984.2005.00007.x`, my guess for Cohen & Bolt (2005, JEM): Crossref 404. Not included.
- `10.1177/0146621610377080`, my guess for Meyer (2010, APM): it resolves to a **different** paper (Green 2011). Not included.

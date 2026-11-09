# Data

## What is in the repository
`data/processed/` contains the analysis-ready extracts of the **PISA 2025 public-use files** (OECD). These are the files the scripts and notebooks read, so the analyses can be re-run from a clone of the repository, including in Google Colab.

| File | Content | Size |
|---|---|---|
| `math_formA.parquet`, `math_formB.parquet`, `math_formC.parquet` (+ `.json` sidecars) | The three analytic mathematics forms: responses, item times and totals | about 1 MB each |
| `student_external.parquet` | Effort, perseverance, ESCS, gender and the final student weight `W_FSTUWT` for the analytic students | 0.4 MB |
| `cog_linear_items.parquet`, `process_linear_items.parquet` | Linear-path item responses and item timings, extracted from the cognitive and process files | 12 MB and 65 MB |
| `cog_design.parquet`, `cog_presence.parquet`, `cog_scored_items.csv` | Booklet design and item presence | 12 MB and 38 MB |
| `item_labels.csv`, `timing_label_examples.csv` | Item and timing-variable labels | small |

**Source and attribution.** OECD (2026), *PISA 2025 Database*, public-use files with the release date stamp `20260806`. The files contain no names. Students are identified only by the OECD's anonymous country, school and student codes. Users must follow the OECD's terms of use for the PISA data. The MIT licence of this repository covers the code only, not the data.

**Not in the repository.** The raw SPSS files (`data/raw/*.sav`) are 1–13 GB each, which is above GitHub's file-size limit. They are only needed to rebuild `data/processed/` from scratch.

## Rebuilding `data/processed/` from the OECD files (optional)
1. Download the PISA 2025 public-use SPSS files from the OECD PISA database page. You need these files:
   - `CY09_MS_COG_PUF.zip`
   - `CY09_MS_COG_PROCESS_PUF.zip`
   - `CY09_MS_STU_PUF.zip`
2. Unzip them into `data/raw/`. Some of the archives use Deflate64, so use 7-Zip or `zipfile-deflate64`.
3. Run the extraction scripts:
   ```
   python scripts/extract_items.py      # linear-path science/math items + item timing  -> data/processed/
   python scripts/extract_student.py    # EFFORT1, PERSEV, ESCS, gender, weight for analytic students
   python -c "import sys; sys.path.insert(0,'src'); from mcq_profiles.data import *; c,p=load_extracts(); [save_form(*build_form(c,p,rank=r), n) for r,n in [(0,'math_formA'),(1,'math_formB'),(2,'math_formC')]]"
   ```
   Each analytic file has a `.json` sidecar. The sidecar records the item map, the coding rules and a SHA-256 fingerprint.

Item-level timing is **withheld for the new science items (CACERS\*)**. Mathematics linear-path forms have timing for every item, and that is why they are the primary data.

## Schema of an analytic form
- Identifiers: `student_id_anonymous` (a hash of CNT and CNTSTUID), `CNT`, `CNTSCHID`, `OECD`, `BOOKID`.
- Responses: `I01..IJJ`, coded 0/1 or NaN (NaN = not reached or technical issue).
- Item times: `RT01..RTJJ`, in seconds.
- Totals: `total_duration_minutes`, `final_score`.
- Counts: `n_not_reached`, `n_technical`, `n_time_missing`.

## Synthetic data
`--demo` generates small synthetic datasets in memory (see `src/mcq_profiles/simulation.py`). Their outputs are always named `demo_*`. They are never empirical findings.

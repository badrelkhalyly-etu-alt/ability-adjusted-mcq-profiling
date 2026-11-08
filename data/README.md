# Data

## Nothing in this folder is committed
`data/raw/` and `data/processed/` are git-ignored. They contain licensed, student-level records.

## PISA 2025 (empirical data of this study)
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

The 2025 release files used here carry the date stamp `20260806`. Item-level timing is **withheld for the new science items (CACERS\*)**. Mathematics linear-path forms have timing for every item, and that is why they are the primary data.

### Schema of an analytic form
- Identifiers: `student_id_anonymous` (a hash of CNT and CNTSTUID), `CNT`, `CNTSCHID`, `OECD`, `BOOKID`.
- Responses: `I01..IJJ`, coded 0/1 or NaN (NaN = not reached or technical issue).
- Item times: `RT01..RTJJ`, in seconds.
- Totals: `total_duration_minutes`, `final_score`.
- Counts: `n_not_reached`, `n_technical`, `n_time_missing`.

## Source-study data (FSBM, Casablanca)
These data are **not available** to this project. The pipeline accepts the same schema: `student_id_anonymous`, `programme`, `total_duration_minutes`, `final_score`, `Q1..Q20`. To use it, rename `Q*` to `I*` and `programme` to `BOOKID`. It may only be run after the author and the data custodians have authorised it.

## Synthetic data
`--demo` generates small synthetic datasets in memory (see `src/mcq_profiles/simulation.py`). Their outputs are always named `demo_*`. They are never empirical findings.

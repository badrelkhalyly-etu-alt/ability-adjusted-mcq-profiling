# Ability and relative-time map (Streamlit prototype)

A descriptive dashboard that shows, for each student of one test sitting, two measured quantities with their uncertainty:

| Quantity | What it is | What it is not |
|---|---|---|
| **Ability** (logits) | Posterior mean under a unidimensional IRT model with item-specific guessing, from the item responses only, with a 90 % interval (80 % and 95 % selectable) | A grade, a diagnosis, a potential |
| **Relative time** (SD units) | The student's mean log item time over the items they reached, adjusted for item time intensities, compared with the time expected for a student of the same ability on the same booklet. 0 = as expected, +1 = one residual SD longer | Speed, effort, anxiety or engagement. It mixes working pace, checking, pauses, distraction and rapid guessing |

The app assigns **no profiles, no psychological labels and no recommendations**. The reason comes from the companion study: on PISA 2025, score–time profiles were not distinct from a continuum, lost predictive information, and about 30 % of students would change profile on a parallel test.

## Run

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app opens with **synthetic demonstration data** generated in the app. It contains no real student data.

## Your own data

Upload a CSV with one row per student:

| Column | Content |
|---|---|
| `student_id` (optional) | Any identifier |
| `I1` … `IJ` | Item responses, 0 or 1; empty = not reached or not administered |
| `RT1` … `RTJ` | Time on each item in seconds (same numbering as `I`) |
| `booklet` (optional) | Booklet or form code. The time model has one intercept per booklet |

The computation runs locally, and nothing is sent anywhere. Use pseudonymised identifiers, and follow your institution's rules for student data.

## What the screen shows

- **Summary:**
  - ability reliability;
  - the number of statistically distinct ability strata, (4G+1)/3 with G = √(rel/(1−rel));
  - the mean ability interval width.
- **The map:** each student is a point with interval bars on both axes.
- **Compare two students:** says whether their intervals overlap. If they do, the data cannot tell the two students apart on that quantity.
- **Table:** all coordinates with their intervals, downloadable as CSV.
- **Limits of interpretation:** always displayed.

## Method

The back-end is `src/mcq_profiles/dashboard.py`, which calls `continuous_map.ContinuousIRTDur` (`deg=3`, `het=True`), the representation evaluated in the paper:
1. The model is fitted jointly on the responses and the relative log-time index.
2. Ability is then read from the responses alone.
3. Relative time is the standardised residual of the index given ability and booklet, averaged over the ability posterior.

Its interval combines:
- the ability uncertainty;
- the sampling error of the index, which is the within-student SD of item residuals divided by √(number of timed items).

Students with fewer than 3 timed items have no relative time.

## Tests

`tests/test_dashboard.py`:
- the back-end (shapes, intervals, input validation, absence of labels);
- a headless run of the app with `streamlit.testing.v1.AppTest`.

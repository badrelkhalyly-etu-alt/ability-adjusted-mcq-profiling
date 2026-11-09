"""Descriptive ability x relative-time map with uncertainty (prototype).

Run:  streamlit run app/streamlit_app.py
Data: synthetic demo data, or your own CSV with columns I1..IJ (0/1, empty = not reached), RT1..RTJ (seconds),
optional ``booklet`` and ``student_id``. Nothing is uploaded anywhere: computation runs locally.
The app shows two measured quantities and their intervals. It assigns no profiles, no psychological labels and
no recommendations.
"""
import io
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcq_profiles.dashboard import LIMITATIONS, compute_map, distinguishable, simulate_demo  # noqa: E402

st.set_page_config(page_title="Ability and relative time map", layout="wide")
st.title("Ability and relative time, with uncertainty")
st.caption("Descriptive prototype. Not for grading, placement, diagnosis or other high-stakes decisions.")

with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Synthetic demo data", "Upload a CSV"], index=0)
    level = st.select_slider("Interval level", options=[0.8, 0.9, 0.95], value=0.9)
    seed = st.number_input("Demo seed", 0, 9999, 0) if source.startswith("Synthetic") else 0
    up = st.file_uploader("CSV file", type="csv") if source == "Upload a CSV" else None


@st.cache_data(show_spinner="Fitting the model...")
def run(df_json: str, level: float):
    df = pd.read_json(io.StringIO(df_json), orient="split")
    return compute_map(df, level=level)


if source.startswith("Synthetic"):
    data = simulate_demo(seed=int(seed))
    st.info("Synthetic data generated in the app. No real student data are included in this prototype.")
elif up is not None:
    data = pd.read_csv(up)
else:
    st.warning("Upload a CSV to continue.")
    st.stop()

try:
    co, summ = run(data.to_json(orient="split"), level)
except ValueError as e:
    st.error(f"Input not valid: {e}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Students", f"{summ['n_students']:,}")
c2.metric("Ability reliability", f"{summ['ability_reliability']:.2f}")
c3.metric("Distinct ability strata", f"{summ['separation_strata']:.1f}")
c4.metric("Mean ability interval width", f"{summ['mean_ability_interval_width']:.2f}")
if not summ["converged"]:
    st.warning("The model did not fully converge; treat the map with caution.")

st.subheader("Map")
st.write(f"Each point is a student; bars are {int(level * 100)} % intervals. Relative time is standardised: "
         "0 = the time expected for a student of the same ability on the same items; +1 = one SD longer.")
show = co.dropna(subset=["rel_time"])
max_n = st.slider("Students shown", 50, int(min(len(show), 2000)), int(min(len(show), 400)), step=50)
sub = show.sample(max_n, random_state=1) if len(show) > max_n else show
base = alt.Chart(sub)
pts = base.mark_circle(size=28, color="#2a78d6", opacity=0.8).encode(
    x=alt.X("theta:Q", title="Ability (logits)"), y=alt.Y("rel_time:Q", title="Relative time (SD)"),
    tooltip=["student_id", alt.Tooltip("theta:Q", format=".2f"), alt.Tooltip("theta_lo:Q", format=".2f"),
             alt.Tooltip("theta_hi:Q", format=".2f"), alt.Tooltip("rel_time:Q", format=".2f"),
             alt.Tooltip("rel_time_lo:Q", format=".2f"), alt.Tooltip("rel_time_hi:Q", format=".2f"), "items_timed"])
xbar = base.mark_rule(color="#9fbfe8", opacity=0.6).encode(x="theta_lo:Q", x2="theta_hi:Q", y="rel_time:Q")
ybar = base.mark_rule(color="#9fbfe8", opacity=0.6).encode(x="theta:Q", y="rel_time_lo:Q", y2="rel_time_hi:Q")
zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#52514e", strokeDash=[4, 4]).encode(y="y:Q")
st.altair_chart((xbar + ybar + pts + zero).properties(height=460), use_container_width=True)

st.subheader("Compare two students")
a, b = st.columns(2)
i = a.selectbox("Student A", range(len(co)), format_func=lambda k: co.student_id.iloc[k])
j = b.selectbox("Student B", range(len(co)), index=min(1, len(co) - 1), format_func=lambda k: co.student_id.iloc[k])
for q, name in [("theta", "ability"), ("rel_time", "relative time")]:
    if np.isnan(co[q].iloc[i]) or np.isnan(co[q].iloc[j]):
        st.write(f"- {name}: not available for one of the students (too few timed items).")
    else:
        verdict = "distinguishable" if distinguishable(co, i, j, q) else "not distinguishable (intervals overlap)"
        st.write(f"- {name}: {verdict}.")

st.subheader("Table")
st.dataframe(co.round(3), use_container_width=True, hide_index=True)
st.download_button("Download the table (CSV)", co.to_csv(index=False), "ability_relative_time.csv", "text/csv")

st.subheader("Limits of interpretation")
for t in LIMITATIONS:
    st.markdown(f"- {t}")

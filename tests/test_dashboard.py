from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mcq_profiles.dashboard import LIMITATIONS, compute_map, distinguishable, parse_table, simulate_demo

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_demo_data_has_no_real_identifiers_and_parses():
    df = simulate_demo(n=200, J=10)
    ids, Y, RT, g = parse_table(df)
    assert Y.shape == (200, 10) and RT.shape == (200, 10)
    assert ids[0].startswith("S")
    assert np.isnan(Y).any()                         # not-reached items present


def test_compute_map_outputs_intervals_and_summary():
    co, s = compute_map(simulate_demo(n=800, J=15, seed=3), level=0.9, max_iter=200)
    assert len(co) == 800
    assert (co.theta_lo <= co.theta).all() and (co.theta <= co.theta_hi).all()
    ok = co.rel_time.notna()
    assert (co.rel_time_lo[ok] <= co.rel_time_hi[ok]).all()
    assert 0.5 < s["ability_reliability"] < 1 and s["separation_strata"] > 1
    assert not {"profile", "label", "cluster"} & set(co.columns)   # no discrete labels are produced


def test_invalid_inputs_are_rejected():
    df = simulate_demo(n=50, J=5)
    with pytest.raises(ValueError):
        parse_table(df.drop(columns=["RT3"]))
    bad = df.copy()
    bad["I1"] = 2
    with pytest.raises(ValueError):
        parse_table(bad)


def test_distinguishable():
    co = pd.DataFrame(dict(theta=[0, 3], theta_lo=[-0.5, 2.5], theta_hi=[0.5, 3.5]))
    assert distinguishable(co, 0, 1)
    co.loc[1, "theta_lo"] = 0.4
    assert not distinguishable(co, 0, 1)


def test_limitations_forbid_labels_and_high_stakes_use():
    text = " ".join(LIMITATIONS).lower()
    assert "high-stakes" in text and "not a measure of speed" in text


def test_streamlit_app_runs_with_demo_data():
    import streamlit.testing.v1 as st_testing
    at = st_testing.AppTest.from_file(str(APP), default_timeout=300).run()
    assert not at.exception, at.exception
    assert any("Ability reliability" in m.label for m in at.metric)
    assert any("Limits of interpretation" in h.value for h in at.subheader)

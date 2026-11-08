import numpy as np
import pandas as pd
from mcq_profiles.preprocessing import train_test_split_stratified, item_mask, log_duration, unmasked_duration
from mcq_profiles.statistics import cramers_v, clopper_pearson, kr20, bh_fdr, match_labels
from mcq_profiles.evaluation import logloss_matrix


def test_split_is_stratified_and_disjoint():
    df = pd.DataFrame({"CNT": np.repeat(["A", "B", "C"], [100, 50, 10])})
    tr, te = train_test_split_stratified(df, 1, 0.3)
    assert not (tr & te).any() and (tr | te).all()
    assert te[df.CNT == "A"].sum() == 30 and te[df.CNT == "C"].sum() == 3


def test_mask_size():
    m = item_mask(22, 0.3, 0)
    assert m.sum() == 7


def test_log_duration_never_drops_silently():
    t = np.array([1.0, 10.0, 0.0, 50.0, np.nan])
    lt, c = log_duration(t, censor=30)
    assert np.isnan(lt[2]) and np.isnan(lt[4]) and c[3] and lt[3] == np.log(30)


def test_unmasked_duration():
    rt = np.array([[60.0, 60.0, np.nan], [np.nan, np.nan, np.nan]])
    miss = np.array([[False, False, True], [True, True, True]])
    out = unmasked_duration(rt, miss, np.array([True, False, False]))
    assert out[0] == 1.0 and out[1] == 0.0


def test_stats_helpers():
    assert cramers_v([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    lo, hi = clopper_pearson(0, 40)
    assert lo == 0 and 0.08 < hi < 0.1
    Y = (np.random.default_rng(0).random((200, 10)) < 0.5).astype(float)
    assert -0.5 < kr20(Y) < 0.5
    assert bh_fdr(np.array([0.001, 0.5, 0.04]), 0.05).tolist() == [True, False, False]
    lab, _ = match_labels(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]))
    assert lab.tolist() == [0, 0, 1, 1]


def test_logloss_ignores_missing():
    Y = np.array([[1.0, np.nan]])
    ll, n = logloss_matrix(Y, np.array([[0.5, 0.9]]))
    assert n[0] == 1 and np.isclose(ll[0], np.log(2))

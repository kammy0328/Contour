"""Tracking benchmark scoring (instruction 2026-11-19 §1): a joint left out never raises pck5. Two answers, one
predicted right: a wrong second prediction and no second prediction both give pck5 0.50 (missing 0 % and 50 %);
no answer at all is not measured. Before the fix the omission read pck5 1.00 over n = 1."""
import numpy as np
import pytest

from contour.learn import benchmark as BM


def _case(second):
    gt = np.full((1, 1, 2, 3), np.nan)
    gt[0, 0, 0, :2] = [100, 100]
    gt[0, 0, 1, :2] = [200, 200]
    trk = np.full((1, 1, 2, 3), np.nan)
    trk[0, 0, 0, :2] = [100, 100]
    if second is not None:
        trk[0, 0, 1, :2] = second
    present = np.ones((1, 1), bool)
    scale = np.full((1, 1), 100.0)
    return trk, gt, present, scale


def test_wrong_and_missing_score_the_same_pck():
    wrong = BM.score(*_case([260, 200]))
    miss = BM.score(*_case(None))
    assert wrong["pck5"] == pytest.approx(0.5) and wrong["missing"] == pytest.approx(0.0) and wrong["n_eval"] == 2
    assert miss["pck5"] == pytest.approx(0.5) and miss["missing"] == pytest.approx(0.5) and miss["n_eval"] == 2
    assert wrong["metric_version"] == 2
    assert miss["err_pct"] == pytest.approx(0.0)                  # the predicted one is exact
    assert miss["score_pct"] == pytest.approx(BM.CAP * 100 / 2)    # the missed one costs the cap
    assert wrong["score_pct"] == pytest.approx(BM.CAP * 100 / 2)   # 0.6 off, capped at 0.2


def test_no_answer_is_not_measured():
    trk, gt, present, scale = _case([200, 200])
    gt[:] = np.nan
    r = BM.score(trk, gt, present, scale)
    assert r["n_eval"] == 0 and r["pck5"] is None and r["missing"] is None and r["score_pct"] is None


def test_hidden_or_unscaled_answers_are_not_scored():
    trk, gt, present, scale = _case(None)
    vis = np.ones((1, 1, 2), np.uint8)
    vis[0, 0, 1] = 0                                              # the missed joint was not visible
    r = BM.score(trk, gt, present, scale, vis)
    assert r["n_eval"] == 1 and r["pck5"] == 1.0 and r["missing"] == 0.0
    r2 = BM.score(trk, gt, present, np.full((1, 1), np.nan))
    assert r2["n_eval"] == 0 and r2["pck5"] is None
    only = np.zeros((1, 1, 2), bool)
    only[0, 0, 0] = True                                          # only the joint a person marked
    assert BM.score(trk, gt, present, scale, None, only)["n_eval"] == 1


def test_summary_prefers_the_capped_score_and_keeps_old_results_apart():
    new = {"a": BM.score(*_case(None)), "b": BM.score(*_case([200, 200]))}
    assert BM.metric_of(new) == 2
    assert BM.summary(new) == pytest.approx((BM.CAP * 100 / 2 + 0.0) / 2)
    old = {"a": dict(err_pct=3.0, pck5=0.9, n=10)}
    assert BM.metric_of(old) == 1 and BM.summary(old) == 3.0

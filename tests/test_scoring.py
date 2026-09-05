"""
test_scoring.py -- the extraction metric library. Pure, no LLM.

Run:  pytest -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from scoring import (  # noqa: E402
    aggregate,
    score_row,
    values_equal,
)


# --- values_equal ------------------------------------------------------
def test_bool_exact():
    assert values_equal("chest_pain_present_now", True, True) is True
    assert values_equal("chest_pain_present_now", True, False) is False


def test_enum_exact():
    assert values_equal("duration", "over_5_min", "over_5_min") is True
    assert values_equal("duration", "over_5_min", "under_5_min") is False


def test_enum_set_order_independent():
    assert values_equal("radiation_sites", ["arm", "jaw"], ["jaw", "arm"]) is True
    assert values_equal("radiation_sites", ["arm", "jaw"], ["arm"]) is False


def test_int_exact_for_age():
    assert values_equal("age", 58, 58) is True
    assert values_equal("age", 58, 59) is False


def test_severity_has_a_one_point_tolerance():
    # categorical language ("excruciating") maps to one gold number, but
    # 8/9/10 are clinically identical under the protocol's own severe band
    # and drive the same disposition -- exact match would fail a
    # clinically-correct answer that landed on a different point in-band.
    assert values_equal("severity_1_10", 9, 8) is True
    assert values_equal("severity_1_10", 9, 10) is True
    assert values_equal("severity_1_10", 9, 7) is False   # crosses out of tolerance
    assert values_equal("severity_1_10", 7, 8) is True    # adjacent still counts


def test_onset_hours_tolerant():
    # "this morning" ~ 6h vs model's 5h -> within tolerance
    assert values_equal("onset_hours_ago", 6, 5) is True
    # 48h vs 40h -> within 25%
    assert values_equal("onset_hours_ago", 48, 40) is True
    # 6h vs 30h -> not close
    assert values_equal("onset_hours_ago", 6, 30) is False


def test_heart_rate_small_tolerance():
    assert values_equal("heart_rate_bpm", 140, 143) is True
    assert values_equal("heart_rate_bpm", 140, 160) is False


def test_free_text_is_unscored():
    assert values_equal("suspected_cause", "reflux", "heart attack") is None
    assert values_equal("location", "center", "left") is None


# --- score_row -------------------------------------------------------
def test_perfect_row():
    g = {"duration": "over_5_min", "age": 58, "chest_pain_present_now": True}
    rs = score_row(g, dict(g))
    assert rs.wrong == [] and rs.missing == [] and rs.spurious == []
    assert rs.f1 == 1.0
    assert rs.disp_match is True


def test_missing_and_spurious():
    g = {"duration": "over_5_min", "severity_1_10": 9}
    p = {"duration": "over_5_min", "radiation_sites": ["arm"]}
    rs = score_row(g, p)
    assert "severity_1_10" in rs.missing
    assert "radiation_sites" in rs.spurious
    assert rs.recall < 1.0 and rs.precision < 1.0


def test_slot_error_that_does_not_change_disposition():
    # severity 10 vs 8: outside the +-1 tolerance so still flagged "wrong" at
    # the slot level, but both are >= 8 ("severe") -> same disposition either way
    g = {"chest_pain_present_now": True, "duration": "under_5_min",
         "severity_1_10": 10}
    p = {"chest_pain_present_now": True, "duration": "under_5_min",
         "severity_1_10": 8}
    rs = score_row(g, p)
    assert rs.wrong == ["severity_1_10"]
    assert rs.disp_match is True          # 8 and 10 are both "severe" -> ED


def test_slot_error_that_flips_disposition_is_caught():
    g = {"chest_pain_present_now": True, "duration": "over_5_min", "age": 58}
    p = {"chest_pain_present_now": True, "duration": "under_5_min", "age": 58}
    rs = score_row(g, p)
    assert rs.disp_gold == "CALL_EMS_911_NOW"
    assert rs.disp_match is False


def test_critical_slot_tracking():
    g = {"duration": "over_5_min", "radiation_sites": ["jaw"]}
    p = {"duration": "few_seconds"}
    rs = score_row(g, p)
    assert rs.critical["duration"] == "wrong"
    assert rs.critical["radiation_sites"] == "missing"


# --- aggregate -----------------------------------------------------
def test_aggregate_basic():
    rows = [
        score_row({"duration": "over_5_min"}, {"duration": "over_5_min"}),
        score_row({"duration": "over_5_min"}, {"duration": "few_seconds"}),
        score_row({"age": 40}, {"age": 40, "severity_1_10": 3}),  # 1 spurious
    ]
    agg = aggregate(rows, ttft=[0.4, 0.5, 0.6], total=[1.0, 1.2, 1.4])
    assert agg["rows"] == 3
    assert 0.0 <= agg["slot_f1"] <= 1.0
    assert agg["hallucination_rate"] > 0.0        # the spurious severity_1_10
    assert agg["latency_total_s"]["p50"] == 1.2
    assert "duration" in agg["per_slot"]

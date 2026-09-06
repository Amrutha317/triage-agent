"""test_eval_ci.py -- the pure metric + percentile helpers in eval_ci.py."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import eval_ci as C  # noqa: E402


def test_pct_matches_known_quantiles():
    xs = list(range(101))  # 0..100
    assert C._pct(xs, 0.0) == 0
    assert C._pct(xs, 1.0) == 100
    assert C._pct(xs, 0.5) == 50
    assert abs(C._pct(xs, 0.025) - 2.5) < 1e-9


def test_harness_metrics_and_detect():
    rows = [
        {"gold_disposition": "CALL_EMS_911_NOW", "pred_disposition": "CALL_EMS_911_NOW",
         "gold_rule_id": "r1", "pred_rule_id": "r1"},
        {"gold_disposition": "CALL_EMS_911_NOW", "pred_disposition": "HOME_CARE",
         "gold_rule_id": "r2", "pred_rule_id": "home"},
        {"gold_disposition": "HOME_CARE", "pred_disposition": "GO_TO_ED_NOW",
         "gold_rule_id": "r3", "pred_rule_id": "ed"},
    ]
    kind, fn = C.detect(rows)
    assert kind == "harness"
    m = fn(rows)
    assert abs(m["triage_accuracy"] - 1 / 3) < 1e-9
    assert m["red_flag_recall_911"] == 0.5              # 1 of 2 gold-911 caught
    # denominator is all 3 rows (both dispositions rank-known):
    assert abs(m["under_triage_rate"] - 1 / 3) < 1e-9   # row 2: 911 -> HOME_CARE
    assert abs(m["over_triage_rate"] - 1 / 3) < 1e-9    # row 3: HOME_CARE -> ED


def test_extraction_metrics():
    rows = [
        {"exact": True, "tp_keys": ["a", "b"], "fp_keys": [], "fn_keys": [],
         "val_ok": ["a", "b"], "val_bad": []},
        {"exact": False, "tp_keys": ["a"], "fp_keys": ["x"], "fn_keys": ["c"],
         "val_ok": ["a"], "val_bad": []},
    ]
    kind, fn = C.detect(rows)
    assert kind == "extraction"
    m = fn(rows)
    assert m["row_exact_match"] == 0.5
    assert abs(m["key_precision"] - 3 / 4) < 1e-9   # tp 3, fp 1
    assert abs(m["key_recall"] - 3 / 4) < 1e-9      # tp 3, fn 1
    assert m["hallucination_row_rate"] == 0.5


def test_distress_metrics():
    g1 = {k: False for k in C._DISTRESS_FLAGS}
    g1["life_threatening"] = True
    p1 = dict(g1)                                   # perfect
    p2 = {k: False for k in C._DISTRESS_FLAGS}      # missed it
    rows = [{"gold": g1, "pred": p1}, {"gold": g1, "pred": p2}]
    kind, fn = C.detect(rows)
    assert kind == "distress"
    m = fn(rows)
    assert m["micro_recall"] == 0.5                 # 1 of 2 true flags caught
    assert m["micro_precision"] == 1.0             # no false positives
    assert m["flag_exact_match"] == 0.5


def test_bootstrap_one_brackets_the_point_estimate():
    rows = [{"exact": i % 3 == 0, "tp_keys": ["a"], "fp_keys": [], "fn_keys": [],
             "val_ok": ["a"], "val_bad": []} for i in range(60)]
    _, fn = C.detect(rows)
    res = C.bootstrap_one(rows, fn, seed=1)
    pt, lo, hi = res["row_exact_match"]
    assert lo <= pt <= hi
    assert 0.0 <= lo < hi <= 1.0

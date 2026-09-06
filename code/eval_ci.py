"""
eval_ci.py -- bootstrap confidence intervals for the eval result JSONs.

The eval sets are small (59 scenarios, 87-112 extraction rows), so a raw
"0.915 -> 0.907" needs a CI before it can be called a change. This reads the
`rows` array that eval_harness.py / eval_extraction.py / eval_distress.py write
and reports percentile bootstrap CIs on the headline metrics.

    python code/eval_ci.py outputs/eval_baseline_all.json
    python code/eval_ci.py outputs/extract_base.json outputs/extract_lora_v3.json

With two files it also reports a PAIRED bootstrap CI on the difference (both
runs cover the same rows in the same order), and flags whether that CI excludes
zero -- i.e. whether the adapter moved the metric beyond noise.

Pure stdlib. Auto-detects harness / extraction / distress JSON from the row shape.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

_B = 5000
_ALPHA = 0.05

_DISPO_ORDER = [
    "CALL_EMS_911_NOW", "SEE_MORE_APPROPRIATE_GUIDELINE", "GO_TO_ED_NOW",
    "GO_TO_ED_UCC_NOW_OR_PCP_TRIAGE", "SEE_PCP_WITHIN_24_HOURS",
    "CALL_PCP_WITHIN_24_HOURS", "SEE_PCP_WITHIN_3_DAYS", "HOME_CARE",
]
_RANK = {d: i for i, d in enumerate(_DISPO_ORDER)}
_DISTRESS_FLAGS = ("severe_difficulty_breathing", "confused_or_hard_to_awaken",
                   "shock_signs", "visible_facial_diaphoresis",
                   "life_threatening", "very_sick_or_weak")


# --- metric families: each maps a list[row] -> {metric_name: value|None} ------
def _harness_metrics(rows: list[dict]) -> dict:
    n = len(rows) or 1
    triage = sum(r["pred_disposition"] == r["gold_disposition"] for r in rows) / n
    workflow = sum(
        r["pred_disposition"] == r["gold_disposition"]
        and r["pred_rule_id"] == r["gold_rule_id"]
        for r in rows
    ) / n
    rf = [r for r in rows if r["gold_disposition"] == "CALL_EMS_911_NOW"]
    rf_recall = (sum(r["pred_disposition"] == "CALL_EMS_911_NOW" for r in rf) / len(rf)
                 if rf else None)
    scored = [(r["gold_disposition"], r["pred_disposition"]) for r in rows
              if r["pred_disposition"] in _RANK and r["gold_disposition"] in _RANK]
    under = (sum(_RANK[p] > _RANK[g] for g, p in scored) / len(scored)
             if scored else None)
    over = (sum(_RANK[p] < _RANK[g] for g, p in scored) / len(scored)
            if scored else None)
    return {"triage_accuracy": triage, "workflow_accuracy": workflow,
            "red_flag_recall_911": rf_recall,
            "under_triage_rate": under, "over_triage_rate": over}


def _extraction_metrics(rows: list[dict]) -> dict:
    n = len(rows) or 1
    tp = sum(len(r["tp_keys"]) for r in rows)
    fp = sum(len(r["fp_keys"]) for r in rows)
    fn = sum(len(r["fn_keys"]) for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    vo = sum(len(r["val_ok"]) for r in rows)
    vb = sum(len(r["val_bad"]) for r in rows)
    return {
        "row_exact_match": sum(r["exact"] for r in rows) / n,
        "key_precision": p, "key_recall": rc, "key_f1": f1,
        "value_accuracy": vo / (vo + vb) if vo + vb else None,
        "hallucination_row_rate": sum(bool(r["fp_keys"]) for r in rows) / n,
    }


def _distress_metrics(rows: list[dict]) -> dict:
    tp = fp = fn = 0
    for r in rows:
        for k in _DISTRESS_FLAGS:
            g, p = bool(r["gold"].get(k)), bool(r["pred"].get(k))
            tp += g and p
            fp += p and not g
            fn += g and not p
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    exact = sum(r["gold"] == r["pred"] for r in rows) / (len(rows) or 1)
    return {"micro_precision": prec, "micro_recall": rec, "micro_f1": f1,
            "flag_exact_match": exact}


def detect(rows: list[dict]):
    r = rows[0]
    if "pred_disposition" in r:
        return "harness", _harness_metrics
    if "tp_keys" in r:
        return "extraction", _extraction_metrics
    if "gold" in r and isinstance(r["gold"], dict):
        return "distress", _distress_metrics
    sys.exit(f"unrecognized row shape: keys={sorted(r)[:8]}")


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    i = q * (len(xs) - 1)
    lo = int(i)
    return xs[lo] if lo + 1 >= len(xs) else xs[lo] + (i - lo) * (xs[lo + 1] - xs[lo])


def bootstrap_one(rows: list[dict], fn, seed: int = 0) -> dict:
    rnd = random.Random(seed)
    n = len(rows)
    point = fn(rows)
    keys = list(point)
    samples: dict = {k: [] for k in keys}
    for _ in range(_B):
        idx = [rnd.randrange(n) for _ in range(n)]
        res = fn([rows[i] for i in idx])
        for k in keys:
            if res[k] is not None:
                samples[k].append(res[k])
    out = {}
    for k in keys:
        s = samples[k]
        out[k] = (point[k],
                  _pct(s, _ALPHA / 2) if s else None,
                  _pct(s, 1 - _ALPHA / 2) if s else None)
    return out


def bootstrap_pair(a_rows: list[dict], b_rows: list[dict], fn, seed: int = 0) -> dict:
    rnd = random.Random(seed)
    n = len(a_rows)
    pa, pb = fn(a_rows), fn(b_rows)
    keys = list(pa)
    diffs: dict = {k: [] for k in keys}
    for _ in range(_B):
        idx = [rnd.randrange(n) for _ in range(n)]
        ra = fn([a_rows[i] for i in idx])
        rb = fn([b_rows[i] for i in idx])
        for k in keys:
            if ra[k] is not None and rb[k] is not None:
                diffs[k].append(rb[k] - ra[k])
    out = {}
    for k in keys:
        d = diffs[k]
        lo = _pct(d, _ALPHA / 2) if d else None
        hi = _pct(d, 1 - _ALPHA / 2) if d else None
        sig = (lo is not None and (lo > 0 or hi < 0))
        out[k] = (pa[k], pb[k],
                  (pb[k] - pa[k]) if (pa[k] is not None and pb[k] is not None) else None,
                  lo, hi, sig)
    return out


def _f(x) -> str:
    return "  n/a " if x is None else f"{x:+.3f}" if x < 0 else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", metavar="RESULT.json")
    args = ap.parse_args()

    loaded = [(f, json.load(open(f))) for f in args.files]
    kinds = set()
    for _, d in loaded:
        k, _fn = detect(d["rows"])
        kinds.add(k)
    if len(kinds) != 1:
        sys.exit(f"files are different eval kinds: {kinds}")
    kind = kinds.pop()
    _, fn = detect(loaded[0][1]["rows"])
    print(f"# {kind} eval, bootstrap B={_B}, {int((1 - _ALPHA) * 100)}% percentile CI\n")

    if len(loaded) == 1:
        f, d = loaded[0]
        res = bootstrap_one(d["rows"], fn)
        print(f"{f}   (n={len(d['rows'])})\n")
        print(f"  {'metric':24} {'estimate':>9}   95% CI")
        for k, (pt, lo, hi) in res.items():
            ci = f"[{_f(lo)}, {_f(hi)}]" if lo is not None else "n/a"
            print(f"  {k:24} {_f(pt):>9}   {ci}")
        return

    (fa, da), (fb, db) = loaded[0], loaded[1]
    if len(da["rows"]) != len(db["rows"]):
        sys.exit(f"row counts differ ({len(da['rows'])} vs {len(db['rows'])}) -- "
                 "paired bootstrap needs the same rows in the same order")
    res = bootstrap_pair(da["rows"], db["rows"], fn)
    print(f"A = {fa}")
    print(f"B = {fb}   (n={len(da['rows'])})\n")
    print(f"  {'metric':24} {'A':>8} {'B':>8} {'Δ(B-A)':>8}   95% CI on Δ        beyond noise?")
    for k, (a, b, dd, lo, hi, sig) in res.items():
        ci = f"[{_f(lo)}, {_f(hi)}]" if lo is not None else "n/a"
        print(f"  {k:24} {_f(a):>8} {_f(b):>8} {_f(dd):>8}   {ci:18} {'YES' if sig else 'no'}")


if __name__ == "__main__":
    main()

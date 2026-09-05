"""
scoring.py -- metric library for the extraction bake-off and the eval harness.

Pure functions, no LLM, no network. Given a gold slot dict and a predicted
slot dict, score them; given many rows, aggregate.

Two views of "how good was the extraction":

  * slot-level  -- precision / recall / F1 over (slot, value) pairs, plus a
                   per-slot breakdown and a hallucination rate.
  * outcome-level -- run gold vs predicted slots through the decision engine
                   and check whether the DISPOSITION still comes out the same.
                   This is the metric that actually matters: a slot error that
                   doesn't change the disposition is cosmetic; one that does is
                   a safety issue.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

import slots as S
from decision_engine import evaluate

# Slots whose value directly decides a disposition -- reported separately.
CRITICAL_SLOTS: set[str] = {
    "chest_pain_present_now", "duration", "onset_hours_ago", "severity_1_10",
    "pain_qualities", "radiation_sites", "pattern_comes_and_goes",
    "pattern_worsening", "nitroglycerin_status", "history_of_heart_disease",
    "age",
}

# Free-text slots are not scored for correctness.
_UNSCORED = {sid for sid, s in S.SLOTS_BY_ID.items() if s.type == S.TEXT}

# Numeric tolerance per slot: equal if |g-p| <= abs_tol OR <= rel_tol*max(|g|,|p|)
_NUM_TOL: dict[str, tuple[float, float]] = {
    "age": (0, 0.0),
    # +-1 not 0: categorical language ("unbearable", "excruciating") maps to a
    # single gold number, but 8/9/10 are clinically identical under the
    # protocol's own severe band (8-10) and drive the identical disposition
    # (ed_severe_pain: severity_1_10 >= 8). Exact-match would fail a
    # clinically-correct answer that just picked a different point in the
    # same band as the gold label.
    "severity_1_10": (1, 0.0),
    "heart_rate_bpm": (5, 0.0),
    "temperature_f": (0.3, 0.0),
    "onset_hours_ago": (0.5, 0.25),
}
_NUM_TOL_DEFAULT_INT = (0, 0.0)
_NUM_TOL_DEFAULT_FLOAT = (0.0, 0.20)


def values_equal(slot_id: str, gold: Any, pred: Any) -> bool | None:
    """True/False, or None if this slot is not scored for correctness."""
    slot = S.SLOTS_BY_ID.get(slot_id)
    if slot is None or slot_id in _UNSCORED:
        return None
    t = slot.type
    if t == S.BOOL:
        return bool(gold) == bool(pred)
    if t == S.ENUM:
        return gold == pred
    if t == S.ENUM_SET:
        return set(gold or []) == set(pred or [])
    if t in (S.INT, S.FLOAT):
        try:
            g, p = float(gold), float(pred)
        except (TypeError, ValueError):
            return False
        abs_tol, rel_tol = _NUM_TOL.get(
            slot_id,
            _NUM_TOL_DEFAULT_INT if t == S.INT else _NUM_TOL_DEFAULT_FLOAT,
        )
        return abs(g - p) <= abs_tol or abs(g - p) <= rel_tol * max(abs(g), abs(p), 1e-9)
    return None


@dataclass
class RowScore:
    matched: list[str] = field(default_factory=list)    # key in both, value right
    wrong: list[str] = field(default_factory=list)      # key in both, value wrong
    missing: list[str] = field(default_factory=list)    # in gold, not predicted
    spurious: list[str] = field(default_factory=list)   # predicted, not in gold
    unscored: list[str] = field(default_factory=list)   # free-text etc.
    json_valid: bool = True
    critical: dict[str, str] = field(default_factory=dict)  # slot -> hit|wrong|missing|spurious
    disp_gold: str | None = None
    disp_pred: str | None = None
    disp_match: bool = False

    @property
    def precision(self) -> float:
        denom = len(self.matched) + len(self.wrong) + len(self.spurious)
        return len(self.matched) / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = len(self.matched) + len(self.wrong) + len(self.missing)
        return len(self.matched) / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def score_row(gold: dict, pred: dict, *, json_valid: bool = True) -> RowScore:
    gold = gold or {}
    pred = pred or {}
    rs = RowScore(json_valid=json_valid)

    for key in set(gold) | set(pred):
        eq = None
        if key in gold and key in pred:
            eq = values_equal(key, gold[key], pred[key])
            if eq is None:
                rs.unscored.append(key)
            elif eq:
                rs.matched.append(key)
            else:
                rs.wrong.append(key)
        elif key in gold:
            if values_equal(key, gold[key], gold[key]) is None:
                rs.unscored.append(key)
            else:
                rs.missing.append(key)
        else:  # key in pred only
            if values_equal(key, pred[key], pred[key]) is None:
                rs.unscored.append(key)
            else:
                rs.spurious.append(key)

        if key in CRITICAL_SLOTS:
            if key in rs.matched:
                rs.critical[key] = "hit"
            elif key in rs.wrong:
                rs.critical[key] = "wrong"
            elif key in rs.missing:
                rs.critical[key] = "missing"
            elif key in rs.spurious:
                rs.critical[key] = "spurious"

    rs.disp_gold = evaluate(gold).disposition
    rs.disp_pred = evaluate(pred).disposition
    rs.disp_match = rs.disp_gold == rs.disp_pred
    return rs


def _pct(xs: list[float]) -> dict:
    if not xs:
        return {"p50": None, "p90": None, "mean": None}
    xs = sorted(xs)
    return {
        "mean": statistics.fmean(xs),
        "p50": statistics.median(xs),
        "p90": xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))],
    }


def aggregate(rows: list[RowScore], *, ttft: list[float] | None = None,
              total: list[float] | None = None) -> dict:
    n = len(rows) or 1
    tp = sum(len(r.matched) for r in rows)
    fp = sum(len(r.wrong) + len(r.spurious) for r in rows)
    fn = sum(len(r.wrong) + len(r.missing) for r in rows)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # per-slot
    per_slot: dict[str, dict[str, int]] = {}
    for r in rows:
        for k in r.matched:
            per_slot.setdefault(k, _z())["tp"] += 1
        for k in r.wrong:
            per_slot.setdefault(k, _z())["wrong"] += 1
        for k in r.missing:
            per_slot.setdefault(k, _z())["missing"] += 1
        for k in r.spurious:
            per_slot.setdefault(k, _z())["spurious"] += 1
    per_slot_f1 = {}
    for k, c in per_slot.items():
        p = c["tp"] / (c["tp"] + c["wrong"] + c["spurious"]) if (c["tp"] + c["wrong"] + c["spurious"]) else 1.0
        rr = c["tp"] / (c["tp"] + c["wrong"] + c["missing"]) if (c["tp"] + c["wrong"] + c["missing"]) else 1.0
        per_slot_f1[k] = {"f1": (2 * p * rr / (p + rr) if (p + rr) else 0.0),
                          "precision": p, "recall": rr, **c}

    crit_hits = sum(1 for r in rows for v in r.critical.values() if v == "hit")
    crit_total = sum(len(r.critical) for r in rows)

    return {
        "rows": len(rows),
        "json_valid_rate": sum(r.json_valid for r in rows) / n,
        "slot_precision": prec,
        "slot_recall": rec,
        "slot_f1": f1,
        "hallucination_rate": sum(len(r.spurious) for r in rows) / max(tp + fp, 1),
        "critical_slot_accuracy": crit_hits / crit_total if crit_total else None,
        "disposition_agreement": sum(r.disp_match for r in rows) / n,
        "row_exact_match": sum(
            1 for r in rows if not r.wrong and not r.missing and not r.spurious
        ) / n,
        "per_slot": dict(sorted(per_slot_f1.items(), key=lambda kv: kv[1]["f1"])),
        "latency_ttft_s": _pct(ttft or []),
        "latency_total_s": _pct(total or []),
    }


def _z() -> dict[str, int]:
    return {"tp": 0, "wrong": 0, "missing": 0, "spurious": 0}


def format_summary(agg: dict, name: str = "") -> str:
    L = [f"=== {name} ===" if name else "==="]
    for k in ("rows", "json_valid_rate", "slot_precision", "slot_recall",
              "slot_f1", "hallucination_rate", "critical_slot_accuracy",
              "disposition_agreement", "row_exact_match"):
        v = agg.get(k)
        L.append(f"  {k:24} {v:.3f}" if isinstance(v, float) else f"  {k:24} {v}")
    lt = agg.get("latency_total_s", {})
    if lt.get("p50") is not None:
        L.append(f"  latency_total_s          p50={lt['p50']:.2f} p90={lt['p90']:.2f}")
    worst = list(agg.get("per_slot", {}).items())[:6]
    if worst:
        L.append("  weakest slots (f1):")
        for k, c in worst:
            L.append(f"    {k:26} f1={c['f1']:.2f}  tp={c['tp']} wrong={c['wrong']} "
                     f"miss={c['missing']} spur={c['spurious']}")
    return "\n".join(L)


if __name__ == "__main__":
    gold = {"duration": "over_5_min", "radiation_sites": ["arm", "jaw"],
            "severity_1_10": 7, "chest_pain_present_now": True}
    pred = {"duration": "over_5_min", "radiation_sites": ["arm"],
            "severity_1_10": 8, "chest_pain_present_now": True,
            "suspected_cause": "heartburn"}
    rs = score_row(gold, pred)
    print(rs)
    print(format_summary(aggregate([rs]), "demo"))

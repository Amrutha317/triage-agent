"""
generate_scenarios.py -- build data/eval/scenarios.jsonl from cases that
already exist and are already trusted: the 55 cases in
tests/test_decision_engine.py (which pin rules.yaml to the protocol) plus the
11 cases in tests/audit/golden_cases.yaml (an independent adversarial pass
over rules.yaml, built after the deterministic core -- see tests/audit/).

No new authoring. gold_disposition/gold_rule_id are RECOMPUTED here from the
live engine (not copied from the test files) so this script also acts as a
staleness check -- if rules.yaml changes and a case's outcome shifts, this
will surface it rather than silently baking in an old answer.

Only terminal cases (a real disposition, not an "insufficient info" probe)
become scenarios -- those need a COMPLETE fact set so a simulated patient can
answer every question the agent asks. Partial-info cases are already covered
by test_state_machine.py's loop-safety / fallback tests, which is the right
place for them, not here.

Run:  python code/generate_scenarios.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from decision_engine import RulesEngine  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "eval", "scenarios.jsonl")
ENGINE = RulesEngine()


def from_test_decision_engine() -> list[dict]:
    from test_decision_engine import CASES  # the 55 (name, slots, exp_disp, exp_rule)

    out = []
    for name, slots, exp_disp, exp_rule in CASES:
        if exp_disp is None:
            continue  # partial-info probe, not a full scenario
        d = ENGINE.evaluate(slots)
        assert d.disposition == exp_disp and d.rule_id == exp_rule, (
            f"staleness: '{name}' now evaluates to {d.disposition}/{d.rule_id}, "
            f"test file says {exp_disp}/{exp_rule} -- rules.yaml changed underneath"
        )
        out.append({
            "name": name,
            "source": "test_decision_engine",
            "facts": slots,
            "gold_disposition": d.disposition,
            "gold_rule_id": d.rule_id,
        })
    return out


def from_golden_cases() -> list[dict]:
    import yaml

    path = os.path.join(os.path.dirname(__file__), "..", "tests", "audit", "golden_cases.yaml")
    cases = yaml.safe_load(open(path, encoding="utf-8"))["cases"]
    out = []
    for c in cases:
        d = ENGINE.evaluate(c["slots"])
        if d.disposition is None:
            continue  # e.g. the two deliberate reachability-hole probes
        out.append({
            "name": c["name"],
            "source": "golden_cases",
            "why": c.get("why", ""),
            "facts": c["slots"],
            "gold_disposition": d.disposition,
            "gold_rule_id": d.rule_id,
        })
    return out


def main() -> None:
    scenarios = from_test_decision_engine() + from_golden_cases()

    # de-dupe by (gold_disposition, gold_rule_id, sorted facts) -- a few
    # golden_cases overlap in intent with test_decision_engine cases
    def _hashable(v):
        if isinstance(v, list):
            return tuple(_hashable(x) for x in v)
        return v

    seen = set()
    deduped = []
    for s in scenarios:
        key = (s["gold_disposition"], s["gold_rule_id"],
               tuple(sorted(
                   ((k, _hashable(v)) for k, v in s["facts"].items()),
                   key=lambda kv: kv[0],
               )))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for s in deduped:
            fh.write(json.dumps(s) + "\n")

    rule_ids = {s["gold_rule_id"] for s in deduped}
    all_rule_ids = {r["id"] for _, r in ENGINE.iter_rules()}
    tiers = {}
    for s in deduped:
        tiers[s["gold_disposition"]] = tiers.get(s["gold_disposition"], 0) + 1

    print(f"wrote {len(deduped)} scenarios -> {OUT}")
    print(f"rule ids covered: {len(rule_ids)} / {len(all_rule_ids)}")
    missing = all_rule_ids - rule_ids
    if missing:
        print(f"NOT covered (need at least one more scenario each): {sorted(missing)}")
    print("tier distribution:")
    for name in ENGINE.order:
        print(f"  {name:34} {tiers.get(name, 0)}")


if __name__ == "__main__":
    main()

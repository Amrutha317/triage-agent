#!/usr/bin/env python3
"""Run golden_cases.yaml against rules.yaml. Reuses check_rules.py's evaluator."""
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_rules import decide, _DEFAULT_RULES  # noqa: E402

rules_doc = yaml.safe_load(open(_DEFAULT_RULES))
_HERE = os.path.dirname(os.path.abspath(__file__))
cases = yaml.safe_load(open(os.path.join(_HERE, "golden_cases.yaml")))["cases"]

rule_ca = {r["id"]: r.get("care_advice")
           for d in rules_doc["dispositions"] for r in d.get("rules", [])}

fails = 0
for c in cases:
    disp, rid = decide(rules_doc, c["slots"])
    ok = (disp == c["expect"]) and (rid == c.get("expect_rule"))
    ca_note = ""
    if ok and "expect_care_advice" in c:
        got = rule_ca.get(rid)
        if got != c["expect_care_advice"]:
            ok = False
            ca_note = f"  care_advice: got {got}, want {c['expect_care_advice']}"

    drift = ""
    ps = c.get("protocol_says")
    if isinstance(ps, str) and ps != (disp or "") and ps.isupper():
        drift = f"   [DEVIATION from protocol: {ps}]"

    print(f"{'PASS' if ok else 'FAIL'}  {c['name']}")
    print(f"        -> {disp} / {rid}{drift}{ca_note}")
    if not ok:
        fails += 1
        print(f"        expected {c['expect']} / {c.get('expect_rule')}")

print(f"\n{len(cases) - fails}/{len(cases)} passed")
sys.exit(1 if fails else 0)

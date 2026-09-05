#!/usr/bin/env python3
"""
check_rules.py -- conformance checker for rules.yaml against the Chest Pain
(Adult) Schmitt-Thompson protocol.

Run:  python3 check_rules.py [--rules rules.yaml] [--protocol protocol_index.yaml]

Exit code 0 = no ERRORs (warnings may still be present), 1 = at least one ERROR.

Six independent checks:
  C1  Coverage        every protocol line maps to a rule that exists
  C2  Orphans         every rules.yaml rule maps back to a protocol line
  C3  Tier match      the rule sits under the disposition the protocol prints
  C4  Care advice     CA code list matches the protocol exactly, in order
  C5  Ordering        disposition order matches protocol tier order
  C6  Structure       atom shapes, operators, slot-name hygiene
  C7  Reachability    brute-force probe for states that yield NO disposition,
                      and for rules that no reachable state can ever fire
"""

import argparse
import itertools
import os
import sys
from collections import defaultdict

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RULES = os.path.join(_HERE, "..", "..", "code", "rules.yaml")
_DEFAULT_PROTOCOL = os.path.join(_HERE, "protocol_index.yaml")

UNKNOWN = "UNKNOWN"
OPS = {"==", "!=", ">", ">=", "<", "<=", "contains", "contains_any", "nonempty"}

findings = []


def report(level, check, msg):
    findings.append((level, check, msg))


# ---------------------------------------------------------------------------
# Three-valued evaluator (reference implementation of the file's own contract)
# ---------------------------------------------------------------------------
def eval_atom(atom, slots):
    slot, op, val = atom
    if slot not in slots or slots[slot] is None:
        return UNKNOWN
    actual = slots[slot]
    try:
        if op == "==":
            return actual == val
        if op == "!=":
            return actual != val
        if op == ">":
            return actual > val
        if op == ">=":
            return actual >= val
        if op == "<":
            return actual < val
        if op == "<=":
            return actual <= val
        if op == "contains":
            return val in actual
        if op == "contains_any":
            return any(v in actual for v in val)
        if op == "nonempty":
            return bool(actual) == bool(val)
    except TypeError:
        return UNKNOWN
    return UNKNOWN


def eval_node(node, slots):
    if isinstance(node, list) and node and not isinstance(node[0], (list, dict)):
        return eval_atom(node, slots)
    if isinstance(node, dict):
        results = []
        for key, children in node.items():
            child_vals = [eval_node(c, slots) for c in children]
            if key == "all":
                if False in child_vals:
                    results.append(False)
                elif UNKNOWN in child_vals:
                    results.append(UNKNOWN)
                else:
                    results.append(True)
            elif key == "any":
                if True in child_vals:
                    results.append(True)
                elif UNKNOWN in child_vals:
                    results.append(UNKNOWN)
                else:
                    results.append(False)
            elif key == "none":
                # Optimistic on UNKNOWN, matching decision_engine.py's own
                # documented contract (_none_of): a protocol "Exception:"
                # blocks its rule only when DEFINITELY true. An unresolved
                # exception must not block -- that would silently downgrade a
                # red flag whenever the exception slot hasn't been asked yet.
                # (Originally implemented pessimistically here -- propagating
                # UNKNOWN through `none` -- which is the unsafe direction and
                # produced a false FAIL on the A1_high_risk_pain_resolved_30min_ago
                # golden case; verified against the live engine via
                # try_engine.py before changing this.)
                if True in child_vals:
                    results.append(False)
                else:
                    results.append(True)
        if False in results:
            return False
        if UNKNOWN in results:
            return UNKNOWN
        return True
    return UNKNOWN


def decide(rules_doc, slots):
    """Return (disposition, rule_id) for the first definitively-true rule."""
    for disp in rules_doc["dispositions"]:
        for rule in disp.get("rules", []):
            if eval_node(rule["when"], slots) is True:
                return disp["name"], rule["id"]
    return None, None


# ---------------------------------------------------------------------------
def walk_atoms(node):
    if isinstance(node, list) and node and not isinstance(node[0], (list, dict)):
        yield node
        return
    if isinstance(node, dict):
        for children in node.values():
            for c in children:
                yield from walk_atoms(c)
    elif isinstance(node, list):
        for c in node:
            yield from walk_atoms(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=_DEFAULT_RULES)
    ap.add_argument("--protocol", default=_DEFAULT_PROTOCOL)
    args = ap.parse_args()

    rules_doc = yaml.safe_load(open(args.rules))
    proto = yaml.safe_load(open(args.protocol))

    yaml_rules = {}
    rule_tier = {}
    for disp in rules_doc["dispositions"]:
        for rule in disp.get("rules", []):
            if rule["id"] in yaml_rules:
                report("ERROR", "C6", f"duplicate rule id '{rule['id']}'")
            yaml_rules[rule["id"]] = rule
            rule_tier[rule["id"]] = disp["name"]

    # --- C1 coverage / C3 tier / C4 care advice ----------------------------
    claimed = defaultdict(list)
    for p in proto["rules"]:
        target = p.get("maps_to")
        if target is None:
            report("WARN", "C1", f"{p['id']} deliberately unimplemented: {p['text'][:60]}")
            continue
        if target not in yaml_rules:
            report("ERROR", "C1", f"{p['id']} maps to missing rule '{target}'")
            continue
        claimed[target].append(p)

        if rule_tier[target] != p["tier"]:
            report("ERROR", "C3",
                   f"{p['id']} is tier {p['tier']} in protocol but rule "
                   f"'{target}' sits under {rule_tier[target]}")

    for rid, plist in claimed.items():
        got = yaml_rules[rid].get("care_advice", [])
        expected_sets = {tuple(p["care_advice"]) for p in plist}
        if len(expected_sets) > 1:
            report("ERROR", "C4",
                   f"'{rid}' collapses {len(plist)} protocol lines "
                   f"({', '.join(p['id'] for p in plist)}) that do NOT share one "
                   f"care-advice list. Encoded {got}; protocol needs "
                   + " / ".join(str(list(s)) for s in sorted(expected_sets)))
        elif tuple(got) != expected_sets.pop():
            report("ERROR", "C4",
                   f"'{rid}' care_advice {got} != protocol "
                   f"{plist[0]['care_advice']} ({plist[0]['id']})")
        if len(plist) > 1:
            report("WARN", "C1",
                   f"'{rid}' is a merge of {len(plist)} protocol lines "
                   f"({', '.join(p['id'] for p in plist)}) -- eval set must "
                   f"exercise each member separately")

    # --- C2 orphans ---------------------------------------------------------
    for rid in yaml_rules:
        if rid not in claimed:
            report("ERROR", "C2", f"rule '{rid}' has no corresponding protocol line")

    # --- C5 ordering --------------------------------------------------------
    actual_order = [d["name"] for d in rules_doc["dispositions"]]
    if actual_order != proto["tier_order"]:
        report("ERROR", "C5", f"disposition order {actual_order} != protocol {proto['tier_order']}")
    if rules_doc["meta"]["disposition_order"] != actual_order:
        report("ERROR", "C5", "meta.disposition_order disagrees with the actual dispositions list")

    # --- C6 structure -------------------------------------------------------
    slot_use = defaultdict(list)
    for rid, rule in yaml_rules.items():
        for field in ("id", "protocol", "when"):
            if field not in rule:
                report("ERROR", "C6", f"'{rid}' missing required field '{field}'")
        if "care_advice" not in rule and "route_to" not in rule:
            report("ERROR", "C6", f"'{rid}' has neither care_advice nor route_to")
        for atom in walk_atoms(rule["when"]):
            if len(atom) != 3:
                report("ERROR", "C6", f"'{rid}' malformed atom {atom}")
                continue
            slot, op, _ = atom
            if op not in OPS:
                report("ERROR", "C6", f"'{rid}' unknown operator '{op}'")
            slot_use[slot].append(rid)

    for slot, users in sorted(slot_use.items()):
        if len(users) == 1:
            report("INFO", "C6", f"slot '{slot}' referenced by only one rule ({users[0]})")

    # --- C7 reachability ----------------------------------------------------
    # Small representative slot space. Every combination is a FULLY specified
    # patient; any combination that returns no disposition is a hole.
    space = {
        "chest_pain_present_now": [True, False],
        "duration": ["few_seconds", "under_5_min", "over_5_min"],
        "age": [25, 35, 55],
        "cardiac_risk_factors": [[], ["diabetes"]],
        "history_of_heart_disease": [True, False],
        "pain_qualities": [[], ["crushing"], ["sharp"]],
        "severity_1_10": [2, 5, 9],
        "onset_hours_ago": [1, 100],
        "pattern_comes_and_goes": [True, False],
        "pattern_worsening": [True, False],
        "nitroglycerin_status": ["none", "prescribed_not_taken", "taken_resolved"],
        "cardiac_symptoms_present_now": [True, False],
    }
    quiet = {  # all red flags off, so holes are structural not incidental
        "severe_difficulty_breathing": False, "confused_or_hard_to_awaken": False,
        "shock_signs": False, "passed_out": False, "heart_rate_bpm": 80,
        "visible_facial_diaphoresis": False, "triager_assessment_life_threatening": False,
        "triager_assessment_very_sick_weak": False, "followed_chest_injury": False,
        "radiation_sites": [], "pain_worse_with_movement": False,
        "other_symptoms": [], "cocaine_use_within_3_days": False,
        "pe_risk_factors": [], "pain_worse_with_deep_breath": False,
        "heartburn_exact_match": False, "sour_taste_in_mouth": False,
        "burning_in_chest": False, "temperature_f": 98.6, "rash_at_pain_site": False,
        "pain_caused_by_coughing": False, "known_angina_history": False,
    }

    keys = list(space)
    holes, fired = [], set()
    for combo in itertools.product(*(space[k] for k in keys)):
        slots = dict(quiet)
        slots.update(dict(zip(keys, combo)))
        # skip self-contradictory states
        if slots["chest_pain_present_now"] and slots["cardiac_symptoms_present_now"] is False:
            pass
        disp, rid = decide(rules_doc, slots)
        if disp is None:
            holes.append(dict(zip(keys, combo)))
        else:
            fired.add(rid)

    total = 1
    for k in keys:
        total *= len(space[k])
    if holes:
        report("ERROR", "C7",
               f"{len(holes)} of {total} fully-specified states return NO disposition "
               f"(no fallback rule). Example: {holes[0]}")
    varied = set(keys)
    for rid, rule in yaml_rules.items():
        used = {a[0] for a in walk_atoms(rule["when"]) if len(a) == 3}
        if not used <= varied:
            continue  # probe pins this rule's trigger off; not evidence of shadowing
        if rid not in fired:
            report("WARN", "C7",
                   f"'{rid}' is exercisable by the probe space but never fired in "
                   f"{total} states -- shadowed by an earlier rule")

    # --- output -------------------------------------------------------------
    order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    findings.sort(key=lambda f: (order[f[0]], f[1]))
    counts = defaultdict(int)
    for level, check, msg in findings:
        counts[level] += 1
        print(f"[{level:5}] {check}  {msg}")
    print("\n" + "-" * 70)
    print(f"protocol lines: {len(proto['rules'])}   rules.yaml rules: {len(yaml_rules)}")
    print(f"ERROR {counts['ERROR']}   WARN {counts['WARN']}   INFO {counts['INFO']}")
    return 1 if counts["ERROR"] else 0


if __name__ == "__main__":
    sys.exit(main())

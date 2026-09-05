"""
try_engine.py -- poke the decision engine by hand.

Usage:
  python code/try_engine.py                 # interactive REPL
  python code/try_engine.py age=58 chest_pain_present_now=true duration=over_5_min

In the REPL, type `slot=value` pairs (one or many per line) to set slots and
see the disposition recompute. Values are parsed as:
  true/false -> bool,  12 / 3.5 -> number,  [a,b] -> list,  else -> string
Commands:  show | reset | slots | quit
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from decision_engine import RulesEngine          # noqa: E402
from slots import SLOTS_BY_ID, ASK_ORDER         # noqa: E402

ENGINE = RulesEngine()


def parse_value(raw: str):
    raw = raw.strip()
    low = raw.lower()
    if low in ("true", "yes", "t"):
        return True
    if low in ("false", "no", "f"):
        return False
    if low in ("none", "null", ""):
        return None
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        return [p.strip() for p in inner.split(",") if p.strip()] if inner else []
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def apply_pairs(state: dict, tokens: list[str]) -> None:
    for tok in tokens:
        if "=" not in tok:
            print(f"  ?? ignored '{tok}' (need slot=value)")
            continue
        k, v = tok.split("=", 1)
        k = k.strip()
        if k not in SLOTS_BY_ID:
            print(f"  ?? unknown slot '{k}'")
            continue
        state[k] = parse_value(v)


def render(state: dict) -> None:
    d = ENGINE.evaluate(state)
    print("\n  slots set:")
    for k, v in state.items():
        print(f"    {k} = {v!r}")
    print()
    if d.disposition is None:
        # what would the state machine ask next?
        nxt = next((s for s in ASK_ORDER if s not in state), None)
        print("  DISPOSITION: (none yet - not enough information)")
        if nxt:
            print(f"  next question -> {nxt}: {SLOTS_BY_ID[nxt].question}")
    else:
        print(f"  DISPOSITION : {d.disposition}")
        print(f"  rule fired  : {d.rule_id}")
        print(f"  protocol    : {d.protocol_text}")
        if d.rule_out:
            print(f"  rule_out    : {', '.join(d.rule_out)}  (for clinician, not shown to patient)")
        if d.first_aid:
            print(f"  first aid   : CA {d.first_aid}")
        if d.care_advice:
            print(f"  care advice : CA {d.care_advice}")
        if d.route_to:
            print(f"  route to    : {d.route_to}")
        if d.triager_judgment:
            print("  NOTE: fired on a subjective 'triager judgment' slot (logged)")
        if d.pregnant_flag:
            print("  NOTE: pregnant=true -> NLG adds L&D-capable ED note for ED+ tiers")
        if len(d.all_matches) > 1:
            print(f"  (all matching rules, priority order: {d.all_matches})")
    print()


def main() -> None:
    state: dict = {}
    args = [a for a in sys.argv[1:] if "=" in a]
    if args:
        apply_pairs(state, args)
        render(state)
        return

    print(__doc__)
    print("Enter slot=value pairs. `show`, `reset`, `slots`, `quit`.\n")
    while True:
        try:
            line = input("triage> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("quit", "exit", "q"):
            break
        if line == "reset":
            state = {}
            print("  (cleared)")
            continue
        if line == "show":
            render(state)
            continue
        if line == "slots":
            for sid in ASK_ORDER:
                s = SLOTS_BY_ID[sid]
                extra = f" {list(s.values)}" if s.values else ""
                print(f"    {sid} ({s.type}){extra}")
            continue
        apply_pairs(state, line.split())
        render(state)


if __name__ == "__main__":
    main()

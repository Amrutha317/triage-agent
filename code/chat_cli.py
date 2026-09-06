"""
chat_cli.py -- a terminal chat over agent.TriageSession. No UI dependencies.

A fallback / headless alternative to app.py for demoing the full triage flow.
Each line you type is a patient message; the agent's question or final
disposition is printed, followed by a compact view of the internal state that
produced it (distress flags, slots extracted this turn, the rule that fired,
per-call latency) -- the same panel app.py shows, inline.

Needs a live LLM server (same env as everything else):
    export TRIAGE_BASE_URL=http://localhost:8000/v1
    export TRIAGE_MODEL=<the served model name>

Run:
    python code/chat_cli.py
    python code/chat_cli.py --no-state      # hide the internal-state lines
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from agent import TriageSession  # noqa: E402
from llm_client import LLMClient  # noqa: E402

_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _state_lines(rec) -> str:
    out = [f"{_DIM}  turn={rec.turn_kind}"]
    if rec.distress:
        out.append(f"  distress={rec.distress}")
    out.append(f"  extracted={rec.extracted or {}}")
    if rec.asked_slots:
        out.append(f"  asking={rec.asked_slots}")
    if rec.guardrail_triggered:
        out.append(f"  GUARDRAIL fired -> {rec.guardrail_violations} (safe template used)")
    if rec.turn_kind == "final":
        out.append(f"  disposition={rec.disposition}  rule={rec.rule_id}")
    t = []
    if rec.extract_total is not None:
        t.append(f"extract {rec.extract_total:.2f}s")
    if rec.distress_total is not None:
        t.append(f"distress {rec.distress_total:.2f}s")
    if t:
        out.append("  " + " / ".join(t))
    return "\n".join(out) + _RESET


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-state", action="store_true",
                    help="hide the internal-state lines")
    args = ap.parse_args()

    if not os.environ.get("TRIAGE_BASE_URL"):
        print("WARNING: TRIAGE_BASE_URL not set -- needs a running LLM server "
              "(bootstrap.sh).\n")

    session = TriageSession(llm_client=LLMClient())
    print(f"{_BOLD}ASSISTANT{_RESET}  {session.greeting()}\n")

    while not session.is_done:
        try:
            msg = input(f"{_BOLD}you >{_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n(quit)")
            return
        if not msg:
            continue
        if msg in {"/quit", "/exit"}:
            return

        rec = session.step(msg)
        print(f"\n{_BOLD}ASSISTANT{_RESET}  {rec.message}")
        if not args.no_state:
            print(_state_lines(rec))
        print()

    d = session.final_decision
    print(f"{_BOLD}=== FINAL: {d.disposition}  (rule {d.rule_id}) ==={_RESET}")
    print(f"asked {len(session.fsm._asked)} question-slots over "
          f"{session.summary().n_turns} turns")


if __name__ == "__main__":
    main()

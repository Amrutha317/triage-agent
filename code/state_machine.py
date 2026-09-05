"""
state_machine.py -- the deterministic conversation controller.

Owns the dialogue: what to ask next, when to stop asking, when to escalate.
It never phrases anything for the patient (that is the NLG layer) and never
decides a disposition itself (that is decision_engine). It only sequences
questions and hands back either the next slot to ask or a final Decision.

Question selection is NOT a hardcoded script. On every turn:

  1. Run the rules engine on everything known so far.
  2. If a rule fires -> escalate immediately (red-flag short-circuit), after
     one optional exception-confirmation question, plus a pregnancy/L&D
     question for ED / ED-UCC finals (never for 911 -- that call stays short).
  3. Otherwise ask whatever the MOST SEVERE still-reachable rule needs next.
     Severity order = the order rules appear in rules.yaml. So the dialogue
     naturally walks the protocol from "call 911" down to "home care",
     stopping the instant something fires.
  4. When no rule can fire any more (every rule is definitely FALSE) or a
     safety cap is hit, fall through to a conservative default
     (GO_TO_ED_UCC_NOW_OR_PCP_TRIAGE) per the protocol's "when in doubt,
     refer" guidance.

The public surface is small:

    fsm = TriageStateMachine()
    turn = fsm.start()                      # -> Turn(kind="ask", ...)
    turn = fsm.ingest({turn.slot: value})   # feed one or more slots, get next Turn
    ...
    turn.kind == "final"  ->  turn.decision is the Decision to act on
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(__file__))

from decision_engine import (  # noqa: E402
    Decision,
    RulesEngine,
    Tri,
    _eval_clause,
    _or,
    exception_slots,
    unresolved_slots,
)
from slots import GROUP_OF, QUESTION_GROUPS, SLOTS_BY_ID  # noqa: E402

FALLBACK_DISPOSITION = "GO_TO_ED_UCC_NOW_OR_PCP_TRIAGE"
_ED_TIER_FINALS = {"GO_TO_ED_NOW", "GO_TO_ED_UCC_NOW_OR_PCP_TRIAGE"}
# Cap on question TURNS (each turn asks one question-group). There are 13
# groups total, so 16 never forces a false fallback while still bounding a
# pathological loop.
DEFAULT_MAX_QUESTION_TURNS = 16


@dataclass
class Turn:
    kind: str                         # "ask" | "final"
    slots: list[str] = field(default_factory=list)      # kind == "ask": the group
    questions: list[str] = field(default_factory=list)  # parallel raw prompts for NLG
    decision: Decision | None = None  # kind == "final"
    asked: list[str] = field(default_factory=list)
    reason: str = ""                  # why this group / why final (transcript + eval)
    group: str | None = None          # question-group name (kind == "ask")

    @property
    def slot(self) -> str | None:
        """First slot of the group -- convenience for single-slot call sites."""
        return self.slots[0] if self.slots else None

    @property
    def question(self) -> str | None:
        return self.questions[0] if self.questions else None


class TriageStateMachine:
    def __init__(
        self,
        engine: RulesEngine | None = None,
        max_question_turns: int = DEFAULT_MAX_QUESTION_TURNS,
    ):
        self.engine = engine or RulesEngine()
        self.max_question_turns = max_question_turns
        self.slots: dict = {}
        self._asked: list[str] = []
        self._turns = 0                       # question-groups asked so far
        self._exception_checked: set[str] = set()
        self._asked_pregnancy = False
        self._final: Turn | None = None

    # -- public ---------------------------------------------------------------
    def start(self) -> Turn:
        return self._next_turn()

    def ingest(self, new_slots: dict | None) -> Turn:
        if self._final is not None:
            return self._final
        for k, v in (new_slots or {}).items():
            if k in SLOTS_BY_ID and v is not None:
                self.slots[k] = v
        return self._next_turn()

    @property
    def is_done(self) -> bool:
        return self._final is not None

    def run_to_completion(self, oracle: dict, max_turns: int = 80) -> Turn:
        """Drive the whole conversation from an answer key.

        `observed_only` slots present in the oracle are fed up front -- they
        model what the red-flag / distress classifier produces before any
        question is asked. Every asked slot is then answered from `oracle`
        (left unanswered if the key is missing -- simulates a patient who
        dodges). Returns the final Turn."""
        seed = {
            k: v for k, v in oracle.items()
            if k in SLOTS_BY_ID and SLOTS_BY_ID[k].observed_only
        }
        turn = self.ingest(seed) if seed else self.start()
        guard = 0
        while turn.kind == "ask":
            guard += 1
            if guard > max_turns:
                raise RuntimeError(f"did not converge in {max_turns} turns")
            turn = self.ingest({s: oracle.get(s) for s in turn.slots})
        return turn

    # -- internals ----------------------------------------------------------
    def _next_turn(self) -> Turn:
        states = self.engine.rule_states(self.slots)
        ordered = list(self.engine.iter_rules())

        # Highest-priority rule that currently fires (if any). We must not
        # finalize on it while some rule ABOVE it is still reachable and
        # answerable -- that higher rule could outrank it.
        fired_idx = next(
            (i for i, (_, r) in enumerate(ordered)
             if states[r["id"]] is Tri.TRUE),
            None,
        )
        ceiling = fired_idx if fired_idx is not None else len(ordered)

        for disp_name, rule in ordered[:ceiling]:
            if states[rule["id"]] is not Tri.UNKNOWN:
                continue
            slot = self._first_askable(unresolved_slots(rule["when"], self.slots))
            if slot is None:
                continue
            if self._turns >= self.max_question_turns:
                break
            return self._ask(
                slot,
                reason=f"needed to assess {disp_name} / {rule['id']}",
            )

        if fired_idx is not None:
            decision = self.engine.evaluate(self.slots)
            pre = self._pre_final_question(decision)
            if pre is not None:
                return pre
            return self._finalize(decision, reason=f"rule {decision.rule_id} fired")

        why = (
            "question cap reached"
            if self._turns >= self.max_question_turns
            else "no protocol rule can still match"
        )
        return self._finalize(self._fallback(why), reason=why)

    def _ask(self, slot: str, reason: str | None = None) -> Turn:
        """Ask the whole question-group `slot` belongs to (one natural question
        for the patient), collecting every still-askable slot in that group."""
        group_name = GROUP_OF[slot]
        group_slots = [
            s for s in QUESTION_GROUPS[group_name]
            if s == slot or self._is_askable(s)
        ]
        self._asked.extend(group_slots)
        self._turns += 1
        return Turn(
            kind="ask",
            slots=group_slots,
            questions=[SLOTS_BY_ID[s].question for s in group_slots],
            asked=list(self._asked),
            reason=reason or self._reason_for(slot),
            group=group_name,
        )

    def _first_askable(self, names: list[str]) -> str | None:
        for n in names:
            if self._is_askable(n):
                return n
        return None

    def _finalize(self, decision: Decision, reason: str) -> Turn:
        self._final = Turn(
            kind="final", decision=decision, asked=list(self._asked), reason=reason
        )
        return self._final

    def _pre_final_question(self, decision: Decision) -> Turn | None:
        """At most: the firing rule's unanswered exception question(s), one per
        turn, then a single pregnancy question for ED-tier finals."""
        # 1. exception confirmation -- ask the protocol "Exception:" question(s)
        #    for the firing rule, one per turn, while the exception is still
        #    LIVE (could still turn out to apply). A dead exception (already
        #    definitely false) is skipped.
        if decision.rule_id not in self._exception_checked:
            rule = self.engine.rule_by_id(decision.rule_id)
            none_children = rule["when"].get("none", []) if rule else []
            exception_live = (
                bool(none_children)
                and _or([_eval_clause(c, self.slots) for c in none_children])
                is not Tri.FALSE
            )
            pending = (
                [s for s in exception_slots(rule["when"]) if self._is_askable(s)]
                if exception_live
                else []
            )
            if pending:
                slot = pending[0]
                if len(pending) == 1:
                    self._exception_checked.add(decision.rule_id)
                return self._ask(
                    slot,
                    reason=f"confirming exception to {decision.rule_id} "
                           f"before finalizing {decision.disposition}",
                )
            self._exception_checked.add(decision.rule_id)

        # 2. pregnancy / L&D completeness -- ED and ED/UCC only, never 911
        #    (an EMS-911 call must stay short: "one piece of information").
        if (
            decision.disposition in _ED_TIER_FINALS
            and not self._asked_pregnancy
            and self.slots.get("pregnant") is None
        ):
            self._asked_pregnancy = True
            return self._ask(
                "pregnant",
                reason=f"L&D routing check before finalizing {decision.disposition}",
            )
        return None

    def _is_askable(self, name: str) -> bool:
        s = SLOTS_BY_ID.get(name)
        return (
            s is not None
            and not s.observed_only
            and name not in self._asked
            and self.slots.get(name) is None
        )

    def _reason_for(self, slot: str) -> str:
        states = self.engine.rule_states(self.slots)
        for disp_name, rule in self.engine.iter_rules():
            if states[rule["id"]] is Tri.UNKNOWN and slot in unresolved_slots(
                rule["when"], self.slots
            ):
                return f"needed to assess {disp_name} / {rule['id']}"
        return "protocol assessment question"

    def _fallback(self, why: str) -> Decision:
        return Decision(
            disposition=FALLBACK_DISPOSITION,
            rule_id="fallback_no_rule_matched",
            protocol_text=(
                "No specific protocol rule matched. Per the protocol's "
                "Background guidance -- 'when in doubt, it is best to refer the "
                "patient to the emergency department or talk to the provider' -- "
                f"routing to provider/ED triage ({why})."
            ),
            care_advice=[42, 17, 1],
            pregnant_flag=bool(self.slots.get("pregnant")),
        )


if __name__ == "__main__":
    fsm = TriageStateMachine()
    oracle = {
        "chest_pain_present_now": True,
        "duration": "over_5_min",
        "age": 58,
    }
    t = fsm.start()
    while t.kind == "ask":
        ans = oracle.get(t.slot)
        print(f"ASK  {t.slot:<28} -> {ans!r}    ({t.reason})")
        t = fsm.ingest({t.slot: ans})
    print(f"\nFINAL: {t.decision.disposition}  via {t.decision.rule_id}")
    print(f"       {t.decision.protocol_text}")
    print(f"       asked {len(t.asked)} question(s): {t.asked}")

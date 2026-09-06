"""
agent.py -- one full conversation, as a class. Wires everything built so far
into the runtime flow:

    patient text
        |
        +--> llm_client.classify_distress()  --> red-flag / triager slots
        +--> llm_client.extract_slots()       --> {slot: value}
        |
        v
    state_machine.ingest(merged slots)  --> Turn (ask | final)
        |
        v
    llm_client.render(turn)  --> raw text
        |
        v
    guardrails.guard_question / guard_final  --> text actually shown

No new logic lives here -- TriageSession is pure wiring over modules that are
each independently tested. That's deliberate: if this file has a bug, it's a
wiring bug, not a triage-logic bug, and the blast radius is easy to reason
about.

`llm_client` is injected (constructor param), not hardcoded, so this class
runs identically against the real LLMClient (needs a live server) or
oracle_client.OracleLLMClient (offline, perfect extraction -- see
eval_harness.py --offline and tests/test_agent.py).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(__file__))

from decision_engine import Decision, RulesEngine  # noqa: E402
from guardrails import guard_final, guard_question  # noqa: E402
from slots import ALL_SLOTS  # noqa: E402
from state_machine import TriageStateMachine  # noqa: E402

# Observed red-flag / triager-judgment slots. These are set from HOW the patient
# presents, not from a fact they can retract -- so they are monotonic across
# turns: once any turn's distress read raises one, a later calmer turn must not
# clear it. (keyword_redflags already guarantees this within a turn; this
# extends it across turns.)
_STICKY_OBSERVED = frozenset(s.id for s in ALL_SLOTS if s.observed_only)

GREETING = (
    "Hi, I'm a chest-pain triage assistant. I can't diagnose you, but I'll "
    "ask some questions and tell you the right next step. If this is a "
    "medical emergency, hang up and call 911 now. Otherwise -- what's going "
    "on?"
)

_HISTORY_CHAR_LIMIT = 2000


@dataclass
class TurnRecord:
    patient_text: str
    turn_kind: str                       # "ask" | "final"
    asked_slots: list[str] = field(default_factory=list)
    message: str = ""
    guardrail_triggered: bool = False
    guardrail_violations: list[str] = field(default_factory=list)
    extracted: dict = field(default_factory=dict)
    distress: dict = field(default_factory=dict)
    extract_ttft: float | None = None
    extract_total: float | None = None
    distress_ttft: float | None = None
    distress_total: float | None = None
    wall_seconds: float = 0.0
    disposition: str | None = None
    rule_id: str | None = None


class TriageSession:
    def __init__(self, llm_client=None, engine: RulesEngine | None = None):
        if llm_client is None:
            from llm_client import LLMClient
            llm_client = LLMClient()
        self.client = llm_client
        self.fsm = TriageStateMachine(engine=engine)
        self.turns: list[TurnRecord] = []
        self._history: list[str] = []
        self._pending_asked: list[str] = []
        self._done = False
        self.final_decision: Decision | None = None

    def greeting(self) -> str:
        return GREETING

    @property
    def is_done(self) -> bool:
        return self._done

    def _history_text(self) -> str:
        text = "\n".join(self._history[-6:])   # last few patient turns is enough context
        return text[-_HISTORY_CHAR_LIMIT:]

    def step(self, patient_text: str) -> TurnRecord:
        """One turn: patient free text in, a TurnRecord (message + everything
        needed for logging/eval) out. Once the conversation is done, repeated
        calls just return the same final record."""
        if self._done:
            return self.turns[-1]

        t0 = time.perf_counter()
        self._history.append(patient_text)

        distress = self.client.classify_distress(patient_text, history=self._history_text())
        known = dict(self.fsm.slots)
        extract = self.client.extract_slots(
            patient_text, known_slots=known, asked_slots=list(self._pending_asked)
        )

        merged: dict[str, Any] = {}
        merged.update(distress.data or {})
        merged.update(extract.data or {})    # facts win over inferred distress on overlap

        # Never let a calmer later turn clear an observed red flag an earlier
        # turn already raised (see _STICKY_OBSERVED). Without this, the distress
        # classifier -- which re-runs every turn and always emits
        # life_threatening / very_sick_or_weak as booleans -- silently
        # downgrades a turn-1 escalation to False on turn 3.
        for k in _STICKY_OBSERVED:
            if merged.get(k) is False and self.fsm.slots.get(k) is True:
                del merged[k]

        turn = self.fsm.ingest(merged)
        self._pending_asked = list(turn.slots) if turn.kind == "ask" else []

        rendered = self.client.render(turn)
        if turn.kind == "final":
            guarded = guard_final(turn.decision, rendered)
            self._done = True
            self.final_decision = turn.decision
        else:
            guarded = guard_question(turn, rendered)

        record = TurnRecord(
            patient_text=patient_text,
            turn_kind=turn.kind,
            asked_slots=list(turn.slots) if turn.kind == "ask" else [],
            message=guarded.text,
            guardrail_triggered=not guarded.ok,
            guardrail_violations=list(guarded.violations),
            extracted=dict(extract.data or {}),
            distress=dict(distress.data or {}),
            extract_ttft=extract.ttft_seconds,
            extract_total=extract.total_seconds,
            distress_ttft=distress.ttft_seconds,
            distress_total=distress.total_seconds,
            wall_seconds=time.perf_counter() - t0,
            disposition=turn.decision.disposition if turn.kind == "final" else None,
            rule_id=turn.decision.rule_id if turn.kind == "final" else None,
        )
        self.turns.append(record)
        return record

    def run_conversation(
        self, respond_fn: Callable[[list[str] | None], str], max_turns: int = 30
    ) -> "ConversationResult":
        """Drive a whole conversation. `respond_fn(asked_slots)` returns the
        patient's next message -- `asked_slots` is None for the open-ended
        greeting turn, else the list of slot ids the agent's last message
        asked about (see patient_sim.make_patient_sim)."""
        asked: list[str] | None = None
        for _ in range(max_turns):
            patient_text = respond_fn(asked)
            record = self.step(patient_text)
            asked = record.asked_slots or None
            if self._done:
                break
        return self.summary()

    def summary(self) -> "ConversationResult":
        return ConversationResult(
            done=self._done,
            disposition=self.final_decision.disposition if self.final_decision else None,
            rule_id=self.final_decision.rule_id if self.final_decision else None,
            n_turns=len(self.turns),
            turns=list(self.turns),
            final_slots=dict(self.fsm.slots),
        )


@dataclass
class ConversationResult:
    done: bool
    disposition: str | None
    rule_id: str | None
    n_turns: int
    turns: list[TurnRecord]
    final_slots: dict


if __name__ == "__main__":
    # Offline smoke test -- no LLM, no pod. Proves the wiring, not model quality.
    from oracle_client import OracleLLMClient
    from patient_sim import make_patient_sim

    facts = {
        "chest_pain_present_now": True, "duration": "over_5_min", "age": 58,
        "pain_qualities": ["crushing"], "severity_1_10": 8,
    }
    session = TriageSession(llm_client=OracleLLMClient(facts))
    result = session.run_conversation(make_patient_sim(facts))
    print(f"disposition: {result.disposition} / {result.rule_id}  ({result.n_turns} turns)")
    for t in result.turns:
        who = "ASK " if t.turn_kind == "ask" else "FINAL"
        print(f"  {who:5} patient: {t.patient_text!r}")
        print(f"        agent  : {t.message!r}")

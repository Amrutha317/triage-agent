"""
oracle_client.py -- a perfect-extraction stand-in for LLMClient, used to test
and demo agent.py / eval_harness.py WITHOUT a live model.

This is the agent-level equivalent of state_machine.run_to_completion's
`oracle` dict: instead of asserting "the FSM reaches the right disposition
given perfect slots," it asserts "the whole conversation loop (classify,
extract, ingest, render, guardrail) reaches the right disposition given a
perfect NLU layer." A real LLMClient can only do worse than this at
extraction -- so a bug that shows up even with the oracle is a wiring bug,
not a model-quality problem, and is cheap to catch before ever touching the
pod.

Deliberately duck-types the same three methods TriageSession calls:
classify_distress, extract_slots, render -- nothing else needs to match.
"""

from __future__ import annotations

from llm_client import LLMResult, build_final_text, build_question_text, keyword_redflags

_DISTRESS_SLOTS = (
    "severe_difficulty_breathing", "confused_or_hard_to_awaken", "shock_signs",
    "visible_facial_diaphoresis", "triager_assessment_life_threatening",
    "triager_assessment_very_sick_weak",
)


class OracleLLMClient:
    def __init__(self, facts: dict, extraction_dropout: set[str] | None = None):
        """`facts` is the scenario's ground truth. `extraction_dropout` is an
        optional set of slot ids to deliberately never extract, e.g. to test
        that the FSM's fallback/loop-safety still converges when the "model"
        misses something -- most callers leave it empty (perfect oracle)."""
        self.facts = facts
        self.extraction_dropout = extraction_dropout or set()
        self.llm_questions = False
        self.llm_final = False

    def classify_distress(self, patient_text: str, history: str = "") -> LLMResult:
        out = {
            k: self.facts[k] for k in _DISTRESS_SLOTS
            if k in self.facts and k not in self.extraction_dropout
        }
        # the always-on keyword net runs for real even in oracle mode -- it's
        # part of the system under test, not part of the LLM being faked
        out.update(keyword_redflags(patient_text))
        return LLMResult(data=out, text="", ttft_seconds=0.005, total_seconds=0.01, ok=True)

    def extract_slots(
        self, patient_text: str, known_slots: dict | None = None,
        asked_slots: list[str] | None = None,
    ) -> LLMResult:
        asked_slots = asked_slots or []
        out = {
            s: self.facts[s] for s in asked_slots
            if s in self.facts and s not in self.extraction_dropout
        }
        return LLMResult(data=out, text="", ttft_seconds=0.02, total_seconds=0.05, ok=True)

    def render(self, turn) -> str:
        if turn.kind == "final":
            return build_final_text(turn.decision)
        return build_question_text(turn)

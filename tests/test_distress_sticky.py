"""
test_distress_sticky.py -- an observed red-flag / triager-judgment slot raised
on one turn must not be cleared by a later, calmer turn's distress read.

Regression test for the bug where classify_distress (which re-runs every turn
and always emits life_threatening / very_sick_or_weak as booleans) downgraded a
turn-1 escalation to False on a subsequent mundane turn, sinking the scenario
to HOME_CARE.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from agent import TriageSession  # noqa: E402
from llm_client import LLMResult, build_final_text, build_question_text  # noqa: E402


class _FlagThenCalmClient:
    """Turn 1: distress flags life_threatening. Every later turn: distress says
    everything is false (a patient who led with panic then answered calmly).
    Extraction always returns the bland facts of a fleeting-pain scenario."""

    llm_questions = False
    llm_final = False

    def __init__(self):
        self._turn = 0

    def classify_distress(self, patient_text, history=""):
        self._turn += 1
        if self._turn == 1:
            data = {"triager_assessment_life_threatening": True,
                    "triager_assessment_very_sick_weak": True}
        else:
            data = {"triager_assessment_life_threatening": False,
                    "triager_assessment_very_sick_weak": False}
        return LLMResult(data=data, ok=True, ttft_seconds=0.0, total_seconds=0.0)

    def extract_slots(self, patient_text, known_slots=None, asked_slots=None):
        facts = {"age": 30, "chest_pain_present_now": False,
                 "duration": "few_seconds", "severity_1_10": 2,
                 "onset_hours_ago": 5.0, "pattern_comes_and_goes": True,
                 "pattern_worsening": False, "radiation_sites": [],
                 "pain_qualities": ["sharp"], "cardiac_risk_factors": [],
                 "pe_risk_factors": [], "other_symptoms": [],
                 "history_of_heart_disease": False, "known_angina_history": False,
                 "pain_caused_by_coughing": False,
                 "cardiac_symptoms_present_now": False}
        out = {s: facts[s] for s in (asked_slots or []) if s in facts}
        return LLMResult(data=out, ok=True, ttft_seconds=0.0, total_seconds=0.0)

    def render(self, turn):
        if turn.kind == "final":
            return build_final_text(turn.decision)
        return build_question_text(turn)


def _respond_factory():
    calls = {"n": 0}

    def respond(asked_slots):
        calls["n"] += 1
        if asked_slots is None:
            return ("This feels like a real emergency, please help. Each time it "
                    "only lasts a few seconds. It's a 2 out of 10.")
        return "Okay. " + " ".join(f"About {s}." for s in asked_slots)

    return respond


def test_turn1_escalation_survives_later_calm_turns():
    session = TriageSession(llm_client=_FlagThenCalmClient())
    result = session.run_conversation(_respond_factory(), max_turns=30)

    assert result.done
    assert result.disposition == "CALL_EMS_911_NOW", (
        f"turn-1 life_threatening flag was cleared by a later calm turn -> "
        f"{result.disposition} / {result.rule_id}"
    )
    assert result.final_slots.get("triager_assessment_life_threatening") is True

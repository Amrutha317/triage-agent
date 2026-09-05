"""
test_state_machine.py -- the deterministic conversation controller.

Covers: red-flag short-circuit, severity-ordered question selection,
exception-confirmation questions, the pregnancy/L&D check, the conservative
fallback, loop-safety when a patient dodges, and a cross-check that driving
the FSM turn-by-turn lands on the same disposition the engine gives for the
full slot dict.

Run:  pytest -q
"""

import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "code"))
sys.path.insert(0, _HERE)

from state_machine import TriageStateMachine, FALLBACK_DISPOSITION  # noqa: E402
from test_decision_engine import CASES, ENGINE                       # noqa: E402


# --- red-flag short-circuit ------------------------------------------------
def test_classifier_redflag_escalates_with_zero_questions():
    # Real flow: the distress classifier + slot extractor run on the patient's
    # first utterance and their output is the FIRST ingest -- no question asked.
    fsm = TriageStateMachine()
    turn = fsm.ingest({"severe_difficulty_breathing": True})
    assert turn.kind == "final"
    assert turn.decision.disposition == "CALL_EMS_911_NOW"
    assert turn.decision.rule_id == "ems_severe_dyspnea"
    assert turn.asked == []          # escalated without asking anything


def test_911_final_does_not_ask_pregnancy():
    fsm = TriageStateMachine()
    turn = fsm.run_to_completion(
        {"chest_pain_present_now": True, "duration": "over_5_min", "age": 58}
    )
    assert turn.decision.disposition == "CALL_EMS_911_NOW"
    assert "pregnant" not in turn.asked


def test_prolonged_pain_not_present_now_is_not_a_911():
    fsm = TriageStateMachine()
    turn = fsm.run_to_completion({
        "chest_pain_present_now": False,
        "duration": "over_5_min",
        "age": 60,
        "onset_hours_ago": 8,
    })
    assert turn.decision.disposition != "CALL_EMS_911_NOW"
    assert turn.decision.disposition == "GO_TO_ED_UCC_NOW_OR_PCP_TRIAGE"


# --- question selection is severity-first --------------------------------
def test_first_questions_serve_the_most_severe_reachable_rule():
    fsm = TriageStateMachine()
    t = fsm.start()
    # the very first question must be in service of a CALL_EMS_911_NOW rule
    assert "CALL_EMS_911_NOW" in t.reason


def test_early_exit_question_count_is_small_for_a_clear_redflag():
    fsm = TriageStateMachine()
    turn = fsm.run_to_completion(
        {"chest_pain_present_now": True, "duration": "over_5_min", "age": 58}
    )
    assert turn.decision.disposition == "CALL_EMS_911_NOW"
    # a clear 911 must land in a handful of question TURNS, not the full sweep
    assert fsm._turns <= 4


# --- exception-confirmation questions -----------------------------------
def test_radiation_asks_movement_exception_then_confirms_ed():
    fsm = TriageStateMachine()
    t = fsm.start()
    seen_movement_question = False
    # answer everything benign; only radiation to arm is positive
    oracle = {
        "chest_pain_present_now": True,
        "duration": "under_5_min",
        "age": 30,
        "radiation_sites": ["arm"],
        "severity_1_10": 4,
        "pattern_comes_and_goes": False,
        "pattern_worsening": False,
        "pain_worse_with_movement": False,
        "pregnant": False,
    }
    guard = 0
    while t.kind == "ask":
        guard += 1
        assert guard < 40
        if "pain_worse_with_movement" in t.slots:
            seen_movement_question = True
            assert "exception" in t.reason
        t = fsm.ingest({s: oracle.get(s) for s in t.slots})
    assert seen_movement_question, "FSM never asked the movement exception"
    assert t.decision.disposition == "GO_TO_ED_NOW"
    assert t.decision.rule_id == "ed_radiation_to_shoulder_arm_jaw"


def test_radiation_exception_true_blocks_ed_radiation():
    fsm = TriageStateMachine()
    turn = fsm.run_to_completion({
        "chest_pain_present_now": True,
        "duration": "under_5_min",
        "age": 30,
        "radiation_sites": ["arm"],
        "severity_1_10": 4,
        "pattern_comes_and_goes": False,
        "pattern_worsening": False,
        "pain_worse_with_movement": True,       # musculoskeletal -> exception
        "nitroglycerin_status": "not_prescribed",
        "known_angina_history": False,
        "cardiac_symptoms_present_now": False,
        "pregnant": False,
    })
    assert turn.decision.rule_id != "ed_radiation_to_shoulder_arm_jaw"


# --- pregnancy / L&D check --------------------------------------------
def test_ed_tier_final_asks_pregnancy_once():
    fsm = TriageStateMachine()
    turn = fsm.run_to_completion({
        "severity_1_10": 9,               # ed_severe_pain fires immediately
        "chest_pain_present_now": True,
        "duration": "under_5_min",
        "pregnant": True,
    })
    assert turn.decision.disposition == "GO_TO_ED_NOW"
    assert turn.asked.count("pregnant") == 1
    assert turn.decision.pregnant_flag is True


# --- conservative fallback -------------------------------------------
def test_unmatched_presentation_falls_back_to_provider_triage():
    fsm = TriageStateMachine()
    turn = fsm.run_to_completion({
        "chest_pain_present_now": True,
        "duration": "under_5_min",
        "age": 30,
        "onset_hours_ago": 2,
        "severity_1_10": 3,
        "radiation_sites": [],
        "pattern_comes_and_goes": False,
        "pattern_worsening": False,
        "pain_qualities": ["aching"],
        "cardiac_risk_factors": [],
        "pe_risk_factors": [],
        "other_symptoms": [],
        "history_of_heart_disease": False,
        "known_angina_history": False,
        "nitroglycerin_status": "not_prescribed",
        "cardiac_symptoms_present_now": True,      # still symptomatic -> lower tiers blocked
        "pain_worse_with_deep_breath": False,
        "pain_caused_by_coughing": False,
        "cocaine_use_within_3_days": False,
        "heartburn_exact_match": False,
        "heart_rate_bpm": 80,
        "temperature_f": 98.6,
        "rash_at_pain_site": False,
        "pregnant": False,
    })
    assert turn.decision.disposition == FALLBACK_DISPOSITION
    assert turn.decision.rule_id == "fallback_no_rule_matched"


# --- loop safety -----------------------------------------------------
def test_patient_dodges_every_question_still_terminates():
    fsm = TriageStateMachine(max_question_turns=20)
    turn = fsm.run_to_completion({})          # answers nothing, ever
    assert turn.kind == "final"
    assert turn.decision.disposition == FALLBACK_DISPOSITION
    assert fsm._turns <= 20


def test_no_slot_is_asked_twice():
    fsm = TriageStateMachine()
    turn = fsm.run_to_completion({})
    assert len(turn.asked) == len(set(turn.asked))


# --- cross-check against the engine --------------------------------
_TERMINAL_CASES = [c for c in CASES if c[2] is not None]


@pytest.mark.parametrize("name,slots,exp_disp,exp_rule",
                         _TERMINAL_CASES, ids=[c[0] for c in _TERMINAL_CASES])
def test_fsm_matches_engine_disposition(name, slots, exp_disp, exp_rule):
    """Driving the FSM turn-by-turn from the case's slots as the answer key
    must reach the same disposition the engine computes for the full dict."""
    fsm = TriageStateMachine()
    turn = fsm.run_to_completion(dict(slots))
    assert turn.kind == "final"
    assert turn.decision.disposition == exp_disp, (
        f"{name}: FSM -> {turn.decision.disposition}/{turn.decision.rule_id}, "
        f"engine -> {exp_disp}/{exp_rule}"
    )

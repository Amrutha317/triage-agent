"""
test_guardrails.py -- the last-mile safety check on rendered text.

Pure, no LLM, no network. Every failing check must fall back to EXACTLY the
deterministic template text (build_question_text / build_final_text), never
a warning bolted onto the bad text.

Run:  pytest -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decision_engine import Decision  # noqa: E402
from guardrails import check_text, guard_final, guard_question  # noqa: E402
from llm_client import build_final_text, build_question_text  # noqa: E402
from state_machine import Turn  # noqa: E402


# --- check_text: pure violation list ------------------------------------
def test_clean_text_has_no_violations():
    assert check_text("Please go to the Emergency Department now.") == []


def test_catches_stated_diagnosis():
    assert check_text("This is a heart attack.") != []
    assert check_text("You have GERD.") != []


def test_catches_false_reassurance():
    assert check_text("Don't worry, it's probably nothing.") != []
    assert check_text("You're probably fine.") != []


def test_catches_unapproved_drug_on_a_final_message():
    assert check_text("Take some ibuprofen for the pain.", is_final=True) != []
    assert check_text("You could try nitroglycerin.", is_final=True) != []


def test_aspirin_is_not_flagged_as_unapproved():
    # aspirin is the one templated first-aid drug (CA 1610)
    assert check_text("You may chew one adult aspirin.", is_final=True) == []


def test_drug_mention_is_fine_in_a_question_not_a_final():
    # "did you take your nitroglycerin?" is a normal protocol question
    # (nitroglycerin_status slot) -- only a FINAL directive is the risk
    assert check_text(
        "Do you have nitroglycerin prescribed, and did you take it for this pain?",
        is_final=False,
    ) == []


def test_911_disposition_requires_the_emergency_cue():
    assert check_text("Please get help soon.", disposition="CALL_EMS_911_NOW") != []
    assert check_text("Call 911 now.", disposition="CALL_EMS_911_NOW") == []


def test_911_disposition_rejects_hedging():
    hedgy = "You might want to consider calling 911 if you feel like it."
    assert check_text(hedgy, disposition="CALL_EMS_911_NOW") != []


def test_hedging_is_fine_on_non_911_dispositions():
    # "consider" language is appropriate at a lower tier
    assert check_text(
        "You might want to see your doctor within a few days.",
        disposition="SEE_PCP_WITHIN_3_DAYS",
    ) == []


# --- guard_question / guard_final: fail-closed substitution -------------
def test_guard_question_passes_clean_text_through():
    turn = Turn(kind="ask", slots=["age"], questions=["How old are you?"])
    r = guard_question(turn, "How old are you?")
    assert r.ok is True
    assert r.text == "How old are you?"


def test_guard_question_falls_back_to_template_on_violation():
    turn = Turn(kind="ask", slots=["age"], questions=["How old are you?"])
    bad = "Don't worry, just tell me your age."
    r = guard_question(turn, bad)
    assert r.ok is False
    assert r.text == build_question_text(turn)   # exact template, not a patched version
    assert r.original == bad


def test_guard_final_passes_clean_text_through():
    d = Decision(disposition="HOME_CARE", rule_id="home_fleeting")
    text = build_final_text(d)
    r = guard_final(d, text)
    assert r.ok is True
    assert r.text == text


def test_guard_final_falls_back_to_template_on_911_hedging():
    d = Decision(disposition="CALL_EMS_911_NOW", rule_id="ems_severe_dyspnea")
    bad = "You might want to consider calling someone if you feel like it."
    r = guard_final(d, bad)
    assert r.ok is False
    assert r.text == build_final_text(d)
    assert "911" in r.text                        # the real template is unambiguous


def test_guard_final_falls_back_on_invented_diagnosis():
    d = Decision(disposition="GO_TO_ED_NOW", rule_id="ed_radiation_to_shoulder_arm_jaw")
    bad = "This is a heart attack, please go to the ED."
    r = guard_final(d, bad)
    assert r.ok is False
    assert r.text == build_final_text(d)
    assert "heart attack" not in r.text.lower()


def test_guard_final_fallback_never_contains_a_banned_pattern():
    """The templates themselves must be guardrail-clean -- if the fallback
    ever tripped its own check, there would be nowhere safe left to go."""
    for disp in ("CALL_EMS_911_NOW", "GO_TO_ED_NOW", "GO_TO_ED_UCC_NOW_OR_PCP_TRIAGE",
                "SEE_PCP_WITHIN_24_HOURS", "CALL_PCP_WITHIN_24_HOURS",
                "SEE_PCP_WITHIN_3_DAYS", "HOME_CARE"):
        d = Decision(disposition=disp, rule_id="x")
        text = build_final_text(d)
        assert check_text(text, disposition=disp) == [], f"{disp}: template itself trips a check"

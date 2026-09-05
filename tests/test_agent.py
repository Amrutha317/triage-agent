"""
test_agent.py -- TriageSession wiring, proven with OracleLLMClient (perfect
extraction, no network). This tests the PLUMBING -- merge order, guardrail
integration, turn tracking, termination -- not model quality. A bug that
shows up here is a wiring bug; it can't be blamed on the LLM.

Run:  pytest -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from agent import TriageSession  # noqa: E402
from oracle_client import OracleLLMClient  # noqa: E402
from patient_sim import make_patient_sim  # noqa: E402


def run(facts, extraction_dropout=None):
    client = OracleLLMClient(facts, extraction_dropout=extraction_dropout)
    session = TriageSession(llm_client=client)
    result = session.run_conversation(make_patient_sim(facts))
    return session, result


def test_classic_911_reaches_the_right_disposition_in_few_turns():
    facts = {"chest_pain_present_now": True, "duration": "over_5_min", "age": 58,
             "pain_qualities": ["crushing"]}
    _, result = run(facts)
    assert result.done
    assert result.disposition == "CALL_EMS_911_NOW"
    assert result.rule_id == "ems_prolonged_pain_age"
    assert result.n_turns <= 6


def test_red_flag_volunteered_on_open_turn_escalates_fast():
    # patient leads with an observed-only red-flag phrase on the FIRST turn
    facts = {"severe_difficulty_breathing": True}
    _, result = run(facts)
    assert result.disposition == "CALL_EMS_911_NOW"
    assert result.rule_id == "ems_severe_dyspnea"
    assert result.n_turns == 1     # classifier catches it on message #1, no FSM Q needed


def test_keyword_net_fires_even_if_the_oracle_didnt_set_the_flag():
    """The always-on keyword check in OracleLLMClient.classify_distress runs
    for real (it's the system under test, not part of the fake) -- so
    literal emergency phrasing escalates even if `facts` never set the
    corresponding slot, same as the real LLM path.

    It fires ems_triager_life_threatening -- the LAST of the 11 EMS rules --
    so the FSM correctly keeps checking the other 10 (age, risk factors,
    heart disease, quality, heart rate...) before committing, per its
    never-finalize-while-something-higher-could-still-fire rule. None of
    those CAN resolve from these sparse facts, so it must still converge to
    the same disposition once nothing higher is left to ask -- just not
    necessarily in one turn."""
    facts = {"duration": "under_5_min", "severity_1_10": 2}   # otherwise benign
    client = OracleLLMClient(facts)
    session = TriageSession(llm_client=client)
    record = session.step("please call 911, I think this is an emergency")
    guard = 0
    while not session.is_done and guard < 30:
        guard += 1
        record = session.step("I'm not sure about that.")
    assert session.is_done
    assert record.disposition == "CALL_EMS_911_NOW"


def test_gerd_needs_all_three_conditions_through_the_full_loop():
    facts = {"heartburn_exact_match": True, "burning_in_chest": True,
             "sour_taste_in_mouth": True}
    _, result = run(facts)
    assert result.disposition == "CALL_PCP_WITHIN_24_HOURS"
    assert result.rule_id == "cpcp24_gerd"


def test_gerd_two_of_three_does_not_conclude_gerd():
    facts = {"heartburn_exact_match": True, "burning_in_chest": True,
             "sour_taste_in_mouth": False, "duration": "few_seconds", "onset_hours_ago": 5}
    _, result = run(facts)
    assert result.rule_id != "cpcp24_gerd"


def test_extraction_dropout_still_converges_via_fallback():
    """Even if the 'model' (oracle here) never manages to extract anything
    useful, the conversation must still terminate -- on the documented
    fallback, not hang or crash."""
    facts = {"duration": "under_5_min", "chest_pain_present_now": True,
             "severity_1_10": 3, "cardiac_symptoms_present_now": True}
    _, result = run(facts, extraction_dropout=set(facts))
    assert result.done
    assert result.disposition == "GO_TO_ED_UCC_NOW_OR_PCP_TRIAGE"
    assert result.rule_id == "fallback_no_rule_matched"


def test_guardrail_intercepts_a_bad_render_and_substitutes_the_template():
    from decision_engine import Decision
    from llm_client import build_final_text

    class BadRenderClient(OracleLLMClient):
        def render(self, turn):
            if turn.kind == "final":
                return "Don't worry, you're probably fine, no need to see anyone."
            return super().render(turn)

    facts = {"severe_difficulty_breathing": True}
    client = BadRenderClient(facts)
    session = TriageSession(llm_client=client)
    record = session.step("I can't breathe")
    assert record.turn_kind == "final"
    assert record.guardrail_triggered is True
    assert "911" in record.message
    assert record.message == build_final_text(session.final_decision)


def test_session_records_a_full_transcript():
    facts = {"chest_pain_present_now": True, "duration": "over_5_min", "age": 58,
             "pain_qualities": ["crushing"]}
    session, result = run(facts)
    assert len(session.turns) == result.n_turns
    assert session.turns[-1].turn_kind == "final"
    assert all(t.message for t in session.turns)


def test_repeated_step_after_done_is_idempotent():
    facts = {"severe_difficulty_breathing": True}
    session, result = run(facts)
    last = session.turns[-1]
    again = session.step("anything")
    assert again is last     # doesn't re-run the pipeline once finished

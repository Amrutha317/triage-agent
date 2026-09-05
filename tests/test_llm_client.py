"""
test_llm_client.py -- the pure, LLM-free parts of llm_client.

These are the safety-relevant bits: whatever the model returns, `validate_slots`
must yield only schema-valid values, must never let an observed_only slot
through (A4), and must drop garbage. No network.

Run:  pytest -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from llm_client import (  # noqa: E402
    DISPOSITION_SCRIPT,
    _parse_json,
    keyword_redflags,
    validate_slots,
)


# --- _parse_json ---------------------------------------------------------
def test_parse_plain_object():
    assert _parse_json('{"age": 58}') == {"age": 58}


def test_parse_object_inside_prose_or_fence():
    assert _parse_json('sure:\n```json\n{"duration": "few_seconds"}\n```') == {
        "duration": "few_seconds"
    }


def test_parse_non_object_returns_empty():
    assert _parse_json("I'm sorry, I can't help") == {}
    assert _parse_json("[1, 2, 3]") == {}
    assert _parse_json("") == {}


# --- validate_slots ----------------------------------------------------
def test_keeps_valid_and_coerces_types():
    out = validate_slots({"duration": "under_5_min", "age": "58",
                          "severity_1_10": "9", "onset_hours_ago": "0.5"})
    assert out == {"duration": "under_5_min", "age": 58,
                   "severity_1_10": 9, "onset_hours_ago": 0.5}


def test_filters_invalid_enum_set_members():
    assert validate_slots({"radiation_sites": ["arm", "moon", "jaw"]}) == {
        "radiation_sites": ["arm", "jaw"]
    }


def test_drops_unknown_enum_value_entirely():
    assert validate_slots({"duration": "ages"}) == {}


def test_empty_list_is_preserved():
    # "asked, patient has none" is a real signal for list slots
    assert validate_slots({"other_symptoms": []}) == {"other_symptoms": []}


def test_observed_only_slots_are_rejected():
    # A4: these can only come from the distress classifier, never extraction
    raw = {
        "severe_difficulty_breathing": True,
        "shock_signs": True,
        "triager_assessment_life_threatening": True,
        "confused_or_hard_to_awaken": True,
        "duration": "over_5_min",
    }
    assert validate_slots(raw) == {"duration": "over_5_min"}


def test_unknown_keys_dropped():
    assert validate_slots({"bogus": 1, "age": 40}) == {"age": 40}


def test_bad_bool_dropped_string_bool_coerced():
    assert validate_slots({"chest_pain_present_now": "yes",
                           "pregnant": "maybe"}) == {"chest_pain_present_now": True}


def test_non_dict_input():
    assert validate_slots(None) == {}
    assert validate_slots([1, 2]) == {}


# --- keyword_redflags: the always-on second check, independent of the LLM --
def test_breathing_phrase_maps_to_the_specific_slot_and_life_threatening():
    hits = keyword_redflags("I can't breathe, help")
    assert hits["severe_difficulty_breathing"] is True
    assert hits["triager_assessment_life_threatening"] is True


def test_passed_out_phrase_maps_to_specific_slot():
    hits = keyword_redflags("my husband just passed out on the floor")
    assert hits["passed_out"] is True
    assert hits["triager_assessment_life_threatening"] is True


def test_general_emergency_phrase_sets_only_the_general_flag():
    hits = keyword_redflags("please call 911 now")
    assert hits["triager_assessment_life_threatening"] is True
    assert "severe_difficulty_breathing" not in hits
    assert "passed_out" not in hits


def test_mundane_text_matches_nothing():
    assert keyword_redflags("It's a dull ache, been there about 20 minutes.") == {}
    assert keyword_redflags("") == {}
    assert keyword_redflags(None) == {}


def test_does_not_overtrigger_on_mere_symptom_mentions():
    # "crushing" alone (a pain quality) must NOT trip the general pattern --
    # that's the extractor's job (pain_qualities), not an emergency keyword.
    assert keyword_redflags("it feels like crushing pressure") == {}


def test_case_insensitive():
    assert keyword_redflags("I CANNOT BREATHE") == {
        "severe_difficulty_breathing": True,
        "triager_assessment_life_threatening": True,
    }


# --- disposition scripts ---------------------------------------------
def test_every_disposition_has_a_script():
    from decision_engine import RulesEngine

    dispositions = set(RulesEngine().order)
    missing = dispositions - set(DISPOSITION_SCRIPT)
    assert not missing, f"no patient script for: {sorted(missing)}"


def test_911_script_is_unambiguous():
    assert "911" in DISPOSITION_SCRIPT["CALL_EMS_911_NOW"]

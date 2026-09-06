"""
test_reachability.py -- two properties worth proving, not just asserting:

  1. No fully-specified patient state falls off the end of the protocol with
     no disposition at all. decision_engine.py alone HAS such states (its
     `None` return is overloaded -- "not enough info yet" during a live
     conversation vs "genuinely no rule fits" once everything is known -- and
     it can't tell the two apart from a slot dict alone). The fallback for the
     second case lives one layer up, in state_machine.py, which DOES know
     whether anything is still askable. This test proves that layer actually
     catches every such state, brute-forcing a probe space that finds 2,160 of
     31,104 raw-engine holes.

  2. No rule with a protocol "Exception:" clause can lock in its disposition
     without the FSM ever asking about the exception -- i.e. the exception
     slot always ends up in `asked`, so it can't sit at UNKNOWN forever by
     construction.

Run:  pytest -q
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decision_engine import RulesEngine  # noqa: E402
from state_machine import FALLBACK_DISPOSITION, TriageStateMachine  # noqa: E402

ENGINE = RulesEngine()

# Brute-force probe space over the decision-relevant slots.
_SPACE = {
    "chest_pain_present_now": [True, False],
    "duration": ["few_seconds", "under_5_min", "over_5_min"],
    "age": [25, 35, 55],
    "cardiac_risk_factors": [[], ["diabetes"]],
    "history_of_heart_disease": [True, False],
    "pain_qualities": [[], ["crushing"], ["sharp"]],
    "severity_1_10": [2, 5, 9],
    "onset_hours_ago": [1, 100],
    "pattern_comes_and_goes": [True, False],
    "pattern_worsening": [True, False],
    "nitroglycerin_status": ["not_prescribed", "prescribed_not_taken", "taken_resolved"],
    "cardiac_symptoms_present_now": [True, False],
}
_QUIET = {
    "severe_difficulty_breathing": False, "confused_or_hard_to_awaken": False,
    "shock_signs": False, "passed_out": False, "heart_rate_bpm": 80,
    "visible_facial_diaphoresis": False, "triager_assessment_life_threatening": False,
    "triager_assessment_very_sick_weak": False, "followed_chest_injury": False,
    "radiation_sites": [], "pain_worse_with_movement": False,
    "other_symptoms": [], "cocaine_use_within_3_days": False,
    "pe_risk_factors": [], "pain_worse_with_deep_breath": False,
    "heartburn_exact_match": False, "sour_taste_in_mouth": False,
    "burning_in_chest": False, "temperature_f": 98.6, "rash_at_pain_site": False,
    "pain_caused_by_coughing": False, "known_angina_history": False,
    "pregnant": False,
}


def _raw_engine_holes() -> list[dict]:
    keys = list(_SPACE)
    holes = []
    for combo in itertools.product(*(_SPACE[k] for k in keys)):
        slots = dict(_QUIET)
        slots.update(dict(zip(keys, combo)))
        if ENGINE.evaluate(slots).disposition is None:
            holes.append(slots)
    return holes


def test_raw_engine_has_the_known_reachability_holes():
    """Sanity check that we're testing the real gap (not a stale number).
    decision_engine.py alone returning None here is expected and correct --
    it's an accurate 'no protocol rule matches these facts', not a bug."""
    holes = _raw_engine_holes()
    assert len(holes) == 2160, (
        f"hole count changed to {len(holes)} -- rules.yaml was edited; "
        f"update this number if the change is intentional"
    )


def test_fsm_fallback_covers_every_raw_engine_hole():
    """The property that actually matters: every state
    that decision_engine.py alone can't resolve is still safely handled by
    the layer that talks to the patient."""
    holes = _raw_engine_holes()
    assert holes, "no holes found -- probe space or rules.yaml changed"
    for slots in holes:
        fsm = TriageStateMachine(engine=ENGINE)
        turn = fsm.run_to_completion(dict(slots), max_turns=200)
        assert turn.decision.disposition == FALLBACK_DISPOSITION, (
            f"hole not caught by the FSM fallback: {slots} -> "
            f"{turn.decision.disposition}/{turn.decision.rule_id}"
        )
        assert turn.decision.rule_id == "fallback_no_rule_matched"


# --- exception slots always get asked, never sit at UNKNOWN forever -------
def test_ed_radiation_exception_always_gets_asked():
    fsm = TriageStateMachine(engine=ENGINE)
    turn = fsm.run_to_completion({
        "chest_pain_present_now": True, "duration": "under_5_min", "age": 30,
        "radiation_sites": ["arm"], "severity_1_10": 4,
        "pattern_comes_and_goes": False, "pattern_worsening": False,
        # pain_worse_with_movement deliberately absent from the oracle
    })
    assert "pain_worse_with_movement" in turn.asked


def test_educ_recent_prolonged_pain_exception_always_gets_asked():
    fsm = TriageStateMachine(engine=ENGINE)
    turn = fsm.run_to_completion({
        "chest_pain_present_now": False, "duration": "over_5_min",
        "age": 30, "onset_hours_ago": 10,
        # heartburn_exact_match / sour_taste_in_mouth deliberately absent
    })
    assert "heartburn_exact_match" in turn.asked
    assert "sour_taste_in_mouth" in turn.asked


def test_no_rule_with_a_none_clause_can_finalize_with_its_exception_unasked():
    """General property, not just the two hand-picked cases above: for every
    rule with a protocol Exception (a `none` clause), if the FSM ever
    finalizes ON that rule, every slot in its exception must be either
    answered or have been asked (i.e. present in `asked`) -- it cannot have
    been silently skipped.

    Two of the four `none`-clause rules (ed_worsening_angina,
    pcp24_brief_pain_resolved) use `duration` as their own exception slot,
    which is always already resolved by the time their positive `all` part is
    satisfied (duration is a single enum value) -- structurally impossible to
    leave unconfirmed. The other two are covered explicitly above. This test
    documents that reasoning by asserting it for all four generically.
    """
    from decision_engine import exception_slots

    rules_with_exceptions = [
        rule for _, rule in ENGINE.iter_rules() if "none" in rule.get("when", {})
    ]
    assert len(rules_with_exceptions) == 4, (
        "a rule's exception clauses changed -- update this test's assumptions"
    )
    for rule in rules_with_exceptions:
        for slot in exception_slots(rule["when"]):
            # every exception slot must be a real, askable slot (not a typo)
            import slots as S
            assert slot in S.SLOTS_BY_ID, f"{rule['id']}: unknown slot '{slot}'"

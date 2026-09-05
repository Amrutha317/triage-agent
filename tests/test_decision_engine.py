"""
test_decision_engine.py -- pins every rule in rules.yaml to the source protocol.

Each CASES entry is (name, slots, expected_disposition, expected_rule_id).
expected_disposition == None means "the engine must NOT reach a disposition
with this much information".

Run:  pytest -q          (from triage-agent/)
This suite passing == rules.yaml is considered finalized.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decision_engine import RulesEngine  # noqa: E402

ENGINE = RulesEngine()


# A minimally-complete "benign" baseline we can perturb one field at a time.
# On its own it should reach HOME_CARE via home_fleeting.
BENIGN = {
    "age": 30,
    "chest_pain_present_now": False,
    "duration": "few_seconds",
    "severity_1_10": 2,
    "onset_hours_ago": 5,
    "pattern_comes_and_goes": True,
    "pattern_worsening": False,
    "radiation_sites": [],
    "pain_qualities": ["sharp"],
    "cardiac_risk_factors": [],
    "pe_risk_factors": [],
    "other_symptoms": [],
    "history_of_heart_disease": False,
    "known_angina_history": False,
    "pain_caused_by_coughing": False,
    "cardiac_symptoms_present_now": False,
}


def merge(**over):
    d = dict(BENIGN)
    d.update(over)
    return d


CASES = [
    # ---- CALL_EMS_911_NOW ---------------------------------------------------
    ("ems: severe dyspnea",
     merge(severe_difficulty_breathing=True),
     "CALL_EMS_911_NOW", "ems_severe_dyspnea"),

    ("ems: altered mental status",
     merge(confused_or_hard_to_awaken=True),
     "CALL_EMS_911_NOW", "ems_altered_mental_status"),

    ("ems: shock signs",
     merge(shock_signs=True),
     "CALL_EMS_911_NOW", "ems_shock"),

    ("ems: syncope",
     merge(passed_out=True),
     "CALL_EMS_911_NOW", "ems_syncope"),

    ("ems: >5min pain + age 45 + present now",
     merge(chest_pain_present_now=True, duration="over_5_min", age=45),
     "CALL_EMS_911_NOW", "ems_prolonged_pain_age"),

    ("ems: >5min pain + age 35 + a risk factor + present now",
     merge(chest_pain_present_now=True, duration="over_5_min", age=35,
           cardiac_risk_factors=["diabetes"]),
     "CALL_EMS_911_NOW", "ems_prolonged_pain_risk_factors"),

    ("ems: >5min pain + known heart disease + present now",
     merge(chest_pain_present_now=True, duration="over_5_min",
           history_of_heart_disease=True),
     "CALL_EMS_911_NOW", "ems_prolonged_pain_known_heart_disease"),

    ("ems: >5min pain + crushing quality + present now",
     merge(chest_pain_present_now=True, duration="over_5_min",
           pain_qualities=["crushing"]),
     "CALL_EMS_911_NOW", "ems_prolonged_pain_crushing_quality"),

    ("ems: bradycardia",
     merge(heart_rate_bpm=44),
     "CALL_EMS_911_NOW", "ems_abnormal_heart_rate"),

    ("ems: tachycardia",
     merge(heart_rate_bpm=165),
     "CALL_EMS_911_NOW", "ems_abnormal_heart_rate"),

    ("ems: visible facial diaphoresis",
     merge(visible_facial_diaphoresis=True),
     "CALL_EMS_911_NOW", "ems_visible_diaphoresis"),

    ("ems: triager judges life-threatening",
     merge(triager_assessment_life_threatening=True),
     "CALL_EMS_911_NOW", "ems_triager_life_threatening"),

    # ---- A1: >5min pain but NOT present now must NOT be a 911 --------------
    ("A1: >5min pain resolved, age 60, recent -> ED/UCC not 911",
     merge(chest_pain_present_now=False, duration="over_5_min", age=60,
           onset_hours_ago=10),
     "GO_TO_ED_UCC_NOW_OR_PCP_TRIAGE", "educ_recent_prolonged_pain"),

    # ---- SEE_MORE_APPROPRIATE_GUIDELINE ----------------------------------
    ("redirect: followed a chest injury",
     merge(followed_chest_injury=True),
     "SEE_MORE_APPROPRIATE_GUIDELINE", "injury_redirect"),

    ("ordering: injury redirect does NOT outrank a real 911",
     merge(followed_chest_injury=True, severe_difficulty_breathing=True),
     "CALL_EMS_911_NOW", "ems_severe_dyspnea"),

    # ---- GO_TO_ED_NOW ----------------------------------------------------
    ("ed: severe pain 9/10",
     merge(severity_1_10=9),
     "GO_TO_ED_NOW", "ed_severe_pain"),

    ("ed: worsening comes-and-goes angina (not fleeting)",
     merge(duration="under_5_min", pattern_comes_and_goes=True,
           pattern_worsening=True),
     "GO_TO_ED_NOW", "ed_worsening_angina"),

    ("exception: worsening angina that lasts only seconds -> NOT ED_worsening",
     merge(duration="few_seconds", pattern_comes_and_goes=True,
           pattern_worsening=True, onset_hours_ago=5),
     "HOME_CARE", "home_fleeting"),

    ("ed: radiation to arm",
     merge(radiation_sites=["arm"]),
     "GO_TO_ED_NOW", "ed_radiation_to_shoulder_arm_jaw"),

    ("exception: radiation but clearly worse with movement -> NOT ED_radiation",
     merge(radiation_sites=["arm"], pain_worse_with_movement=True),
     "HOME_CARE", "home_fleeting"),

    ("ed: difficulty breathing (non-severe)",
     merge(other_symptoms=["difficulty_breathing"]),
     "GO_TO_ED_NOW", "ed_difficulty_breathing"),

    ("ed: hemoptysis",
     merge(other_symptoms=["hemoptysis"]),
     "GO_TO_ED_NOW", "ed_hemoptysis"),

    ("ed: cocaine use in last 3 days",
     merge(cocaine_use_within_3_days=True),
     "GO_TO_ED_NOW", "ed_cocaine_use"),

    ("ed: PE risk factor (recent surgery)",
     merge(pe_risk_factors=["recent_major_surgery_1mo"]),
     "GO_TO_ED_NOW", "ed_pe_risk_surgery_or_clotting_disorder"),

    ("ed: PE risk factor (inherited clotting disorder)",
     merge(pe_risk_factors=["inherited_clotting_disorder"]),
     "GO_TO_ED_NOW", "ed_pe_risk_surgery_or_clotting_disorder"),

    ("ed: PE risk factor (hip/leg fracture or cast)",
     merge(pe_risk_factors=["leg_or_hip_fracture_or_cast_1mo"]),
     "GO_TO_ED_NOW", "ed_pe_risk_immobility_or_clot_history"),

    ("ed: PE risk factor (prolonged bedrest)",
     merge(pe_risk_factors=["prolonged_bedrest_1mo"]),
     "GO_TO_ED_NOW", "ed_pe_risk_immobility_or_clot_history"),

    ("ed: PE risk factor (long-distance travel)",
     merge(pe_risk_factors=["long_travel_6h_1mo"]),
     "GO_TO_ED_NOW", "ed_pe_risk_immobility_or_clot_history"),

    ("ed: PE risk factor (prior DVT/PE)",
     merge(pe_risk_factors=["prior_dvt_or_pe"]),
     "GO_TO_ED_NOW", "ed_pe_risk_immobility_or_clot_history"),

    ("ed: PE risk factor (active/recent cancer) -- distinct CA list",
     merge(pe_risk_factors=["cancer_active_or_treated_6mo"]),
     "GO_TO_ED_NOW", "ed_pe_risk_cancer"),

    # ---- GO_TO_ED_UCC_NOW_OR_PCP_TRIAGE --------------------------------
    ("educ: pleuritic pain (worse on deep breath)",
     merge(pain_worse_with_deep_breath=True),
     "GO_TO_ED_UCC_NOW_OR_PCP_TRIAGE", "educ_pleuritic_pain"),

    ("educ: >5min pain within 72h, low risk, resolved",
     merge(chest_pain_present_now=False, duration="over_5_min", age=25,
           onset_hours_ago=40),
     "GO_TO_ED_UCC_NOW_OR_PCP_TRIAGE", "educ_recent_prolonged_pain"),

    ("exception: recent >5min pain BUT exact heartburn + sour taste -> not educ",
     merge(chest_pain_present_now=False, duration="over_5_min", age=25,
           onset_hours_ago=40, heartburn_exact_match=True,
           sour_taste_in_mouth=True, burning_in_chest=False),
     None, None),

    ("educ: triager judges very sick/weak",
     merge(triager_assessment_very_sick_weak=True),
     "GO_TO_ED_UCC_NOW_OR_PCP_TRIAGE", "educ_triager_very_sick"),

    # ---- SEE_PCP_WITHIN_24_HOURS -------------------------------------
    ("pcp24: >5min pain >3 days ago, nothing now",
     merge(chest_pain_present_now=False, duration="over_5_min",
           onset_hours_ago=120, cardiac_symptoms_present_now=False),
     "SEE_PCP_WITHIN_24_HOURS", "pcp24_old_resolved_prolonged_pain"),

    ("pcp24: brief (<5min) pain fully resolved",
     merge(chest_pain_present_now=False, duration="under_5_min",
           cardiac_symptoms_present_now=False, nitroglycerin_status="not_prescribed"),
     "SEE_PCP_WITHIN_24_HOURS", "pcp24_brief_pain_resolved"),

    ("pcp24: fever over 100.4",
     merge(temperature_f=101.3),
     "SEE_PCP_WITHIN_24_HOURS", "pcp24_fever"),

    ("pcp24: rash at pain site",
     merge(rash_at_pain_site=True),
     "SEE_PCP_WITHIN_24_HOURS", "pcp24_rash_at_pain_site"),

    # ---- CALL_PCP_WITHIN_24_HOURS (GERD -- the only path) ------------
    ("gerd: all three conditions present",
     merge(heartburn_exact_match=True, burning_in_chest=True,
           sour_taste_in_mouth=True),
     "CALL_PCP_WITHIN_24_HOURS", "cpcp24_gerd"),

    ("gerd NOT concluded on only two of three (no sour taste)",
     merge(heartburn_exact_match=True, burning_in_chest=True,
           sour_taste_in_mouth=False),
     "HOME_CARE", "home_fleeting"),

    ("gerd NOT concluded when only 'burning' quality reported",
     merge(pain_qualities=["burning"]),
     "HOME_CARE", "home_fleeting"),

    # ---- SEE_PCP_WITHIN_3_DAYS --------------------------------------
    ("pcp3: <5min pain, nitro prescribed but not taken",
     merge(chest_pain_present_now=True, duration="under_5_min",
           nitroglycerin_status="prescribed_not_taken"),
     "SEE_PCP_WITHIN_3_DAYS", "pcp3_stable_angina_nitro_not_taken"),

    ("shadow: <5min pain fully resolved after nitro -> 24h rule wins (protocol order)",
     merge(chest_pain_present_now=False, duration="under_5_min",
           nitroglycerin_status="taken_resolved",
           cardiac_symptoms_present_now=False),
     "SEE_PCP_WITHIN_24_HOURS", "pcp24_brief_pain_resolved"),

    ("pcp3: nitro-resolved rule only reachable while a cardiac symptom persists",
     merge(chest_pain_present_now=False, duration="under_5_min",
           nitroglycerin_status="taken_resolved",
           cardiac_symptoms_present_now=True),
     "SEE_PCP_WITHIN_3_DAYS", "pcp3_stable_angina_nitro_resolved"),

    ("pcp3: known angina, comes and goes, not worsening",
     merge(duration="under_5_min", pattern_comes_and_goes=True,
           known_angina_history=True, pattern_worsening=False,
           nitroglycerin_status="not_prescribed", chest_pain_present_now=True),
     "SEE_PCP_WITHIN_3_DAYS", "pcp3_known_stable_angina"),

    ("pcp3: fleeting cough pains persisting >3 days",
     merge(duration="few_seconds", pain_caused_by_coughing=True,
           onset_hours_ago=96),
     "SEE_PCP_WITHIN_3_DAYS", "pcp3_cough_fleeting_persistent"),

    ("pcp3: fleeting pains persisting >3 days (no cough)",
     merge(duration="few_seconds", onset_hours_ago=96),
     "SEE_PCP_WITHIN_3_DAYS", "pcp3_fleeting_persistent"),

    # ---- HOME_CARE ------------------------------------------------
    ("home: fleeting cough pains, recent",
     merge(duration="few_seconds", pain_caused_by_coughing=True,
           onset_hours_ago=6),
     "HOME_CARE", "home_cough_fleeting"),

    ("home: fleeting pains, recent",
     merge(duration="few_seconds", onset_hours_ago=6),
     "HOME_CARE", "home_fleeting"),

    # ---- ordering / priority -------------------------------------
    ("ordering: EMS outranks ED when both would match",
     merge(chest_pain_present_now=True, duration="over_5_min", age=70,
           severity_1_10=9),
     "CALL_EMS_911_NOW", "ems_prolonged_pain_age"),

    ("ordering: ED_NOW outranks ED/UCC when both match",
     merge(severity_1_10=9, pain_worse_with_deep_breath=True),
     "GO_TO_ED_NOW", "ed_severe_pain"),

    # ---- three-valued logic: not enough info yet ----------------
    ("insufficient: only age known",
     {"age": 50},
     None, None),

    ("insufficient: >5min pain present now but age/risk/hx/quality unknown",
     {"chest_pain_present_now": True, "duration": "over_5_min"},
     None, None),

    ("insufficient: duration unknown -> fleeting home rule cannot fire",
     {k: v for k, v in BENIGN.items() if k != "duration"},
     None, None),
]


@pytest.mark.parametrize("name,slots,exp_disp,exp_rule",
                         CASES, ids=[c[0] for c in CASES])
def test_case(name, slots, exp_disp, exp_rule):
    d = ENGINE.evaluate(slots)
    assert d.disposition == exp_disp, (
        f"{name}: got {d.disposition} via {d.rule_id}, expected {exp_disp}"
    )
    if exp_rule is not None:
        assert d.rule_id == exp_rule, (
            f"{name}: got rule {d.rule_id}, expected {exp_rule}"
        )


@pytest.mark.parametrize("pe_factor,exp_ca", [
    ("recent_major_surgery_1mo", [41, 80, 81, 17, 1]),
    ("inherited_clotting_disorder", [41, 80, 81, 17, 1]),
    ("leg_or_hip_fracture_or_cast_1mo", [41, 80, 81, 84, 17, 1]),
    ("prolonged_bedrest_1mo", [41, 80, 81, 84, 17, 1]),
    ("long_travel_6h_1mo", [41, 80, 81, 84, 17, 1]),
    ("prior_dvt_or_pe", [41, 80, 81, 84, 17, 1]),
    ("cancer_active_or_treated_6mo", [41, 81, 1]),
])
def test_pe_risk_factor_care_advice_matches_protocol_exactly(pe_factor, exp_ca):
    """Each of the 7 protocol PE-risk lines prints its own CA list -- the
    merged single-rule encoding used to always emit [41,80,81,84,17,1],
    which was wrong for surgery/inherited-clotting (extra CA 17) and badly
    wrong for cancer (extra CA 80, 84, 17). Caught by an independent
    conformance check against the source PDF; fixed by splitting into 3
    rules by CA group instead of resolving it in code."""
    d = ENGINE.evaluate(merge(pe_risk_factors=[pe_factor]))
    assert d.disposition == "GO_TO_ED_NOW"
    assert d.care_advice == exp_ca, (
        f"{pe_factor}: got CA {d.care_advice}, protocol says {exp_ca}"
    )


def test_every_rule_id_is_covered_by_a_positive_case():
    """Guard against a rule existing in yaml that no test ever fires."""
    covered = {r for _, _, _, r in CASES if r is not None}
    all_rule_ids = {
        rule["id"]
        for disp in ENGINE.spec["dispositions"]
        for rule in disp["rules"]
    }
    missing = all_rule_ids - covered
    assert not missing, f"rules with no positive test: {sorted(missing)}"

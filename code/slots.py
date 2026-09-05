"""
slots.py -- the single source of truth for every piece of information the
triage agent can collect, its type, its legal values, and the question the
agent asks to obtain it.

decision_engine.py consumes a {slot_id: value} dict; 
state_machine.py uses ASK_ORDER + this schema to decide what to ask next; 
the LLM slot extractor is prompted with the JSON schema derived from SLOTS.

Design notes
------------
* A slot value of ``None`` / absent means "not yet known" -> the decision
  engine treats any atom referencing it as UNKNOWN (three-valued logic).
* A list slot set to ``[]`` means "asked, patient has none" -> `contains` is
  a definite False, `nonempty` is a definite False. The extractor must emit
  ``[]`` (not omit the key) once the question has been asked and answered.
* GUARD_SLOTS are never asked in sequence. They are evaluated every turn from
  (a) the red-flag / distress classifier and (b) any explicit statement the
  patient volunteers. They exist so a red flag can short-circuit the script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# --- slot types -------------------------------------------------------------
BOOL = "bool"
INT = "int"
FLOAT = "float"
ENUM = "enum"
ENUM_SET = "enum_set"   # list of enum members, may be empty
TEXT = "text"           # free text, never on its own a basis for a disposition


@dataclass(frozen=True)
class Slot:
    id: str
    type: str
    question: str
    values: tuple[str, ...] = ()          # for ENUM / ENUM_SET
    guard: bool = False                   # evaluated every turn, not part of ASK_ORDER
    observed_only: bool = False           # never a question: set only by the
                                          # distress classifier or an explicit
                                          # observation, never asked by the FSM
    decision_relevant: bool = True        # False = collected for context/NLG only
    notes: str = ""


# =============================================================================
# CORE SLOTS  -- the 11 protocol Initial Assessment Questions (+ age, A-item)
# asked roughly in this order unless a guard slot short-circuits first.
# =============================================================================
CORE_SLOTS: list[Slot] = [
    Slot("age", INT,
         "First, how old are you?",
         notes="Not one of the 11 numbered questions but required by the "
               "age>44 / age>30 EMS rules. Collected up front."),

    Slot("chest_pain_present_now", BOOL,
         "Are you having the chest pain right now, as we speak?"),

    Slot("location", TEXT,
         "Where exactly does it hurt?",
         decision_relevant=False,
         notes="Context for NLG and for sanity-checking radiation. No rule "
               "keys off raw location."),

    Slot("radiation_sites", ENUM_SET,
         "Does the pain spread anywhere else - into your neck, jaw, arm, "
         "shoulder, or back?",
         values=("none", "neck", "jaw", "arm", "shoulder", "back", "abdomen")),

    Slot("onset_hours_ago", FLOAT,
         "When did this chest pain first start? Roughly how many hours or "
         "days ago?",
         notes="Stored in hours. The 72 h boundary is the only threshold the "
               "rules use."),

    Slot("pattern_comes_and_goes", BOOL,
         "Does the pain come and go, or has it been constant since it started?",
         notes="True = comes and goes. Constant -> False."),

    Slot("pattern_worsening", BOOL,
         "Compared with when it started, is it happening more often or getting "
         "more severe?",
         notes="True if EITHER frequency or severity is increasing "
               "(protocol OR's them)."),

    Slot("worse_with_exertion", BOOL,
         "Does it get worse when you exert yourself - climbing stairs, walking "
         "fast?",
         decision_relevant=False,
         notes="Asked by protocol Q4 but no triage rule keys off it. Context "
               "only."),

    Slot("duration", ENUM,
         "When you get the pain, how long does a typical episode last - a few "
         "seconds, under 5 minutes, or more than 5 minutes?",
         values=("few_seconds", "under_5_min", "over_5_min"),
         notes="Mutually exclusive (A3). 'few_seconds' is the fleeting bucket "
               "the protocol repeatedly carves out."),

    Slot("severity_1_10", INT,
         "On a scale of 1 to 10, how bad is the pain right now?",
         notes="8-10 = SEVERE -> Go to ED Now (ed_severe_pain)."),

    Slot("pain_qualities", ENUM_SET,
         "How would you describe the pain - for example pressure, crushing, "
         "heavy, tight, sharp, burning, aching?",
         values=("pressure", "crushing", "heavy", "tightness", "squeezing",
                 "sharp", "burning", "aching", "tearing", "stabbing")),

    Slot("cardiac_risk_factors", ENUM_SET,
         "Do you have any of these: diabetes, high blood pressure, high "
         "cholesterol, obesity, peripheral vascular disease, a strong family "
         "history of heart disease, or are you a smoker?",
         values=("diabetes", "hypertension", "high_cholesterol",
                 "obesity_bmi_30_plus", "smoker", "pvd", "strong_family_history"),
         notes="A5: history/modifiable risk factors only. Age scored "
               "separately; known CAD -> history_of_heart_disease."),

    Slot("history_of_heart_disease", BOOL,
         "Have you ever been told you have heart disease - angina, a heart "
         "attack, heart failure, a stent or bypass surgery - or do you take "
         "nitroglycerin?"),

    Slot("known_angina_history", BOOL,
         "Have you specifically been diagnosed with angina before?",
         notes="Distinct from history_of_heart_disease; used only by "
               "pcp3_known_stable_angina."),

    Slot("nitroglycerin_status", ENUM,
         "Do you have nitroglycerin prescribed, and if so did you take it for "
         "this pain and did it help?",
         values=("not_prescribed", "prescribed_not_taken",
                 "taken_resolved", "taken_not_resolved")),

    Slot("pe_risk_factors", ENUM_SET,
         "In the past month have you had major surgery, a leg or hip fracture "
         "or cast, a long illness in bed, or travel of 6 hours or more? Any "
         "history of blood clots, an inherited clotting disorder, or current "
         "or recent cancer treatment?",
         values=("recent_major_surgery_1mo", "leg_or_hip_fracture_or_cast_1mo",
                 "prolonged_bedrest_1mo", "long_travel_6h_1mo",
                 "prior_dvt_or_pe", "inherited_clotting_disorder",
                 "cancer_active_or_treated_6mo")),

    Slot("suspected_cause", TEXT,
         "What do you think is causing the chest pain?",
         decision_relevant=False,
         notes="NEVER terminal. A patient answering 'it's just my reflux' does "
               "not end the assessment - protocol explicitly warns against "
               "this."),

    Slot("other_symptoms", ENUM_SET,
         "Any other symptoms - dizziness, nausea, vomiting, sweating, fever, "
         "difficulty breathing, cough, or coughing up blood?",
         values=("dizziness", "nausea", "vomiting", "sweating", "fever",
                 "difficulty_breathing", "cough", "hemoptysis", "palpitations")),

    Slot("pregnant", BOOL,
         "Is there any chance you are pregnant?",
         notes="A2: does not change the computed disposition in v1; attaches a "
               "flag + L&D note for ED+ tiers."),

    Slot("lmp", TEXT,
         "When was your last menstrual period?",
         decision_relevant=False),
]


# =============================================================================
# GUARD SLOTS  -- checked every turn, never asked in sequence. Set by the
# red-flag / distress classifier or volunteered by the patient.
# =============================================================================
GUARD_SLOTS: list[Slot] = [
    Slot("severe_difficulty_breathing", BOOL,
         "(observed) Struggling for each breath / speaks only in single words",
         guard=True, observed_only=True),
    Slot("confused_or_hard_to_awaken", BOOL,
         "(observed) Disoriented, slurred speech, hard to keep awake",
         guard=True, observed_only=True),
    Slot("shock_signs", BOOL,
         "(observed) Cold/pale/clammy skin, too weak to stand, faint pulse",
         guard=True, observed_only=True),
    Slot("passed_out", BOOL,
         "Did you faint or lose consciousness?",
         guard=True),
    Slot("heart_rate_bpm", FLOAT,
         "If you can measure your pulse, what is it in beats per minute?",
         guard=True),
    Slot("visible_facial_diaphoresis", BOOL,
         "(observed) Visible sweat on the face / sweat dripping",
         guard=True, observed_only=True,
         notes="Only from an explicit observation, not from the patient "
               "reporting 'sweating' (that is other_symptoms)."),
    Slot("followed_chest_injury", BOOL,
         "Did this pain start after a fall, blow, or other injury to the chest?",
         guard=True,
         notes="True -> SEE_MORE_APPROPRIATE_GUIDELINE, stop triage."),
    Slot("cocaine_use_within_3_days", BOOL,
         "Have you used cocaine in the last 3 days?",
         guard=True),
    Slot("pain_worse_with_deep_breath", BOOL,
         "Does taking a deep breath make the pain worse?",
         guard=True),
    Slot("pain_worse_with_movement", BOOL,
         "Is the pain clearly worse when you move, twist, or press on the spot?",
         guard=True,
         notes="Exception condition for ed_radiation_to_shoulder_arm_jaw."),
    Slot("rash_at_pain_site", BOOL,
         "Is there a rash or small blisters in the same area as the pain?",
         guard=True),
    Slot("temperature_f", FLOAT,
         "Have you taken your temperature? What was it?",
         guard=True),
    Slot("pain_caused_by_coughing", BOOL,
         "Do the pains happen when you cough?",
         guard=True),
    Slot("cardiac_symptoms_present_now", BOOL,
         "Right now, are you having any dizziness, shortness of breath, "
         "nausea, sweating, or palpitations?",
         guard=True,
         notes="'NO cardiac symptoms now' gate for the two lower resolved-pain "
               "rules."),
    Slot("heartburn_exact_match", BOOL,
         "Does this feel EXACTLY like heartburn you have been diagnosed with "
         "before - not just similar?",
         guard=True),
    Slot("burning_in_chest", BOOL,
         "Is there a burning feeling in your chest?",
         guard=True),
    Slot("sour_taste_in_mouth", BOOL,
         "Do you have a sour or acid taste in your mouth?",
         guard=True),
    Slot("triager_assessment_life_threatening", BOOL,
         "(distress classifier) The patient sounds like a life-threatening "
         "emergency",
         guard=True, observed_only=True,
         notes="A4: set only by the standalone distress classifier; every "
               "firing logged."),
    Slot("triager_assessment_very_sick_weak", BOOL,
         "(distress classifier) The patient sounds very sick or weak",
         guard=True, observed_only=True,
         notes="A4."),
]


ALL_SLOTS: list[Slot] = CORE_SLOTS + GUARD_SLOTS
SLOTS_BY_ID: dict[str, Slot] = {s.id: s for s in ALL_SLOTS}

# Order the state machine asks core slots in, mirroring the protocol's
# Initial Assessment Questions. Guard slots are interleaved opportunistically
# by the state machine, not from this list.
ASK_ORDER: list[str] = [
    "age",
    "chest_pain_present_now",
    "location",
    "radiation_sites",
    "onset_hours_ago",
    "pattern_comes_and_goes",
    "pattern_worsening",
    "duration",
    "severity_1_10",
    "pain_qualities",
    "worse_with_exertion",
    "cardiac_risk_factors",
    "history_of_heart_disease",
    "known_angina_history",
    "nitroglycerin_status",
    "pe_risk_factors",
    "suspected_cause",
    "other_symptoms",
    "pregnant",
    "lmp",
]


# =============================================================================
# QUESTION GROUPS -- the FSM asks one GROUP per turn, not one slot. The NLG
# layer renders a group as a single natural question ("Any other symptoms --
# nausea, sweating, shortness of breath, fever, cough?"). Groups mirror the
# protocol's 11 Initial Assessment Questions plus intake and the narrow
# screens. Every askable slot belongs to exactly one group; observed_only
# slots belong to none.
# =============================================================================
QUESTION_GROUPS: dict[str, list[str]] = {
    "intake": ["age", "chest_pain_present_now"],
    "episode": ["duration", "onset_hours_ago", "pattern_comes_and_goes",
                "pattern_worsening", "worse_with_exertion"],
    "severity_character": ["severity_1_10", "pain_qualities", "location"],
    "radiation": ["radiation_sites"],
    "cardiac_history": ["history_of_heart_disease", "known_angina_history",
                        "cardiac_risk_factors", "nitroglycerin_status"],
    "pulmonary_history": ["pe_risk_factors"],
    "associated_symptoms": ["other_symptoms", "cardiac_symptoms_present_now"],
    "cause": ["suspected_cause"],
    "acute_events": ["passed_out", "followed_chest_injury"],
    "focal_screen": ["pain_worse_with_deep_breath", "pain_worse_with_movement",
                     "pain_caused_by_coughing", "rash_at_pain_site"],
    "exposure_vitals": ["cocaine_use_within_3_days", "heart_rate_bpm",
                        "temperature_f"],
    "gerd_screen": ["heartburn_exact_match", "burning_in_chest",
                    "sour_taste_in_mouth"],
    "pregnancy": ["pregnant", "lmp"],
}

GROUP_OF: dict[str, str] = {
    sid: gname for gname, members in QUESTION_GROUPS.items() for sid in members
}


def _validate_groups() -> None:
    askable = {s.id for s in ALL_SLOTS if not s.observed_only}
    grouped = set(GROUP_OF)
    missing = askable - grouped
    extra = grouped - {s.id for s in ALL_SLOTS}
    assert not missing, f"askable slots with no question group: {sorted(missing)}"
    assert not extra, f"group references unknown slots: {sorted(extra)}"
    seen: set[str] = set()
    for members in QUESTION_GROUPS.values():
        dupes = seen & set(members)
        assert not dupes, f"slot in more than one group: {sorted(dupes)}"
        seen |= set(members)


_validate_groups()


def json_schema_for_extractor(include_observed: bool = False) -> dict[str, Any]:
    """Machine-readable schema handed to the LLM slot-extraction prompt.

    The LLM returns a subset of these keys -- only the slots it can fill from
    the latest patient turn -- and nothing else. `observed_only` slots are
    excluded by default (A4): those are set by the distress classifier from how
    the patient presents, never by literal extraction.
    """
    props: dict[str, Any] = {}
    for s in ALL_SLOTS:
        if s.observed_only and not include_observed:
            continue
        if s.type == BOOL:
            props[s.id] = {"type": "boolean"}
        elif s.type == INT:
            props[s.id] = {"type": "integer"}
        elif s.type == FLOAT:
            props[s.id] = {"type": "number"}
        elif s.type == ENUM:
            props[s.id] = {"type": "string", "enum": list(s.values)}
        elif s.type == ENUM_SET:
            props[s.id] = {
                "type": "array",
                "items": {"type": "string", "enum": list(s.values)},
            }
        else:  # TEXT
            props[s.id] = {"type": "string"}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": props,
    }


if __name__ == "__main__":
    import json

    print(f"{len(CORE_SLOTS)} core slots, {len(GUARD_SLOTS)} guard slots, "
          f"{len(ALL_SLOTS)} total")
    print(json.dumps(json_schema_for_extractor(), indent=2))

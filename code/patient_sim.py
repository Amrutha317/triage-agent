"""
patient_sim.py -- deterministic simulated patient for eval.

Given a scenario's ground-truth `facts` dict, produces short, patient-style
text answering whatever the agent asks. Deterministic and free (no LLM) --
that's what makes it usable for the eval harness's per-turn latency numbers
(no simulated-patient latency to subtract out) and reproducible across runs.

Two response modes:
  * open-ended (asked_slots=None) -- the very first turn, before the FSM has
    asked anything. Volunteers red flags first (a real distressed patient
    leads with "I can't breathe", not with their age), then a short chief
    complaint from whatever core facts are set.
  * targeted (asked_slots=[...]) -- answers exactly the slots in that
    question-group from `facts`. A fact absent from `facts` is silently
    skipped (the sim "doesn't know" / wasn't asked about it in the scenario
    design) rather than guessed -- this mirrors a patient not volunteering
    something nobody asked about.

Not built for realism of phrasing -- built for controlled, reproducible
coverage of every slot value the schema supports.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(__file__))

# Phrases for the subjective/observed-only signals -- these don't have a
# direct question (the FSM never asks them), so they only ever appear in the
# open-ended chief-complaint turn, for the distress classifier to pick up.
_OBSERVED_ONLY_PHRASES = {
    "severe_difficulty_breathing": "I can barely breathe, I can only get a "
        "few words out at a time.",
    "confused_or_hard_to_awaken": "I feel really out of it, hard to think "
        "straight.",
    "shock_signs": "My skin feels cold and clammy and I feel like I might "
        "collapse.",
    "visible_facial_diaphoresis": "I'm drenched in sweat, it's pouring down "
        "my face.",
    "triager_assessment_life_threatening": "This feels like a real "
        "emergency, please help.",
    "triager_assessment_very_sick_weak": "I feel extremely sick and weak "
        "all over.",
}

# Slots volunteered unprompted on the open-ended turn, in priority order, if
# present and "positive" in `facts`. Everything else only gets mentioned once
# specifically asked.
_CHIEF_COMPLAINT_PRIORITY = [
    "passed_out",
    "severe_difficulty_breathing", "confused_or_hard_to_awaken", "shock_signs",
    "visible_facial_diaphoresis", "triager_assessment_life_threatening",
    "triager_assessment_very_sick_weak",
    "duration", "severity_1_10", "pain_qualities", "location", "radiation_sites",
    "onset_hours_ago",
]


def _phrase(slot_id: str, value: Any) -> str | None:
    if slot_id in _OBSERVED_ONLY_PHRASES and value:
        return _OBSERVED_ONLY_PHRASES[slot_id]

    if value is None:
        return None

    if slot_id == "age":
        return f"I'm {value} years old."
    if slot_id == "chest_pain_present_now":
        return "I'm having the pain right now." if value else "The pain isn't happening right now."
    if slot_id == "location":
        return f"It hurts in my {value}." if value else None
    if slot_id == "radiation_sites":
        sites = [s for s in value if s and s != "none"]
        return f"It spreads into my {', '.join(sites)}." if sites else "It doesn't spread anywhere else."
    if slot_id == "onset_hours_ago":
        if value < 1:
            return f"It started about {max(1, round(value * 60))} minutes ago."
        if value < 48:
            return f"It started about {value:.0f} hours ago."
        return f"It started about {value / 24:.0f} days ago."
    if slot_id == "pattern_comes_and_goes":
        return "It comes and goes." if value else "It's been constant since it started."
    if slot_id == "pattern_worsening":
        return "It's happening more often or getting worse." if value else "It's not getting any worse."
    if slot_id == "worse_with_exertion":
        return "It's worse when I exert myself." if value else "Exertion doesn't seem to change it."
    if slot_id == "duration":
        m = {"few_seconds": "Each time it only lasts a few seconds.",
             "under_5_min": "It lasts less than 5 minutes.",
             "over_5_min": "It's lasted more than 5 minutes."}
        return m.get(value)
    if slot_id == "severity_1_10":
        return f"I'd rate the pain {value} out of 10."
    if slot_id == "pain_qualities":
        return f"It feels {', '.join(value)}." if value else "I can't really describe how it feels."
    if slot_id == "cardiac_risk_factors":
        return f"I have {', '.join(value)}." if value else "I don't have any of those risk factors."
    if slot_id == "history_of_heart_disease":
        return "Yes, I have a history of heart disease." if value else "No history of heart disease."
    if slot_id == "known_angina_history":
        return "I've been diagnosed with angina before." if value else "I've never been diagnosed with angina."
    if slot_id == "nitroglycerin_status":
        m = {"not_prescribed": "I don't have nitroglycerin prescribed.",
             "prescribed_not_taken": "I have nitroglycerin but haven't taken it for this.",
             "taken_resolved": "I took my nitroglycerin and the pain went away completely.",
             "taken_not_resolved": "I took my nitroglycerin but the pain didn't go away."}
        return m.get(value)
    if slot_id == "pe_risk_factors":
        return f"I've had {', '.join(value)}." if value else "None of those apply to me."
    if slot_id == "suspected_cause":
        return f"I think it might be {value}." if value else None
    if slot_id == "other_symptoms":
        return f"I also have {', '.join(value)}." if value else "No other symptoms."
    if slot_id == "pregnant":
        return "There's a chance I could be pregnant." if value else "No chance I'm pregnant."
    if slot_id == "lmp":
        return f"My last period was {value}." if value else None
    if slot_id == "passed_out":
        return "Yes, I fainted." if value else "No, I haven't passed out."
    if slot_id == "heart_rate_bpm":
        return f"My pulse is about {value} beats per minute."
    if slot_id == "followed_chest_injury":
        return "This started after an injury to my chest." if value else "This wasn't caused by any injury."
    if slot_id == "cocaine_use_within_3_days":
        return "Yes, I've used cocaine in the last 3 days." if value else "No cocaine use."
    if slot_id == "pain_worse_with_deep_breath":
        return "Taking a deep breath makes it worse." if value else "Breathing deeply doesn't change it."
    if slot_id == "pain_worse_with_movement":
        return "It's worse when I move or twist." if value else "Moving doesn't make it worse."
    if slot_id == "rash_at_pain_site":
        return "There's a rash where it hurts." if value else "No rash."
    if slot_id == "temperature_f":
        return f"My temperature is {value} degrees."
    if slot_id == "pain_caused_by_coughing":
        return "It happens when I cough." if value else "It's not related to coughing."
    if slot_id == "cardiac_symptoms_present_now":
        return "I do have some of those right now." if value else "None of those right now."
    if slot_id == "heartburn_exact_match":
        return "It feels exactly like my usual heartburn." if value else "It doesn't feel like my usual heartburn."
    if slot_id == "burning_in_chest":
        return "There's a burning feeling in my chest." if value else "No burning feeling."
    if slot_id == "sour_taste_in_mouth":
        return "I have a sour taste in my mouth." if value else "No sour taste."
    return None


def make_patient_sim(facts: dict) -> Callable[[list[str] | None], str]:
    """Returns respond(asked_slots) -> patient_text, closed over `facts`."""

    def respond(asked_slots: list[str] | None) -> str:
        if asked_slots is None:
            parts = []
            for slot in _CHIEF_COMPLAINT_PRIORITY:
                if slot not in facts:
                    continue
                v = facts[slot]
                if v in (None, [], False):
                    continue
                p = _phrase(slot, v)
                if p and p not in parts:
                    parts.append(p)
            if not parts:
                return "I'm having some chest discomfort and wanted to check what to do."
            return " ".join(parts)

        parts = []
        for slot in asked_slots:
            if slot not in facts:
                continue
            p = _phrase(slot, facts[slot])
            if p:
                parts.append(p)
        return " ".join(parts) if parts else "I'm not sure about that."

    return respond


if __name__ == "__main__":
    sim = make_patient_sim({
        "chest_pain_present_now": True, "duration": "over_5_min", "age": 58,
        "pain_qualities": ["crushing"], "severity_1_10": 8,
    })
    print("open:", sim(None))
    print("targeted:", sim(["age", "chest_pain_present_now"]))

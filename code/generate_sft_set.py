"""
generate_sft_set.py -- deterministic SFT builder for the triage LoRA adapter.

Writes data/train/sft_v1.jsonl in the EXACT chat format finetune_lora.py
expects, one object per line:

    {"messages": [{"role": "system", ...}, {"role": "user", ...}],
     "response": "<gold assistant JSON>",
     "category": "<bucket, for slicing eval only -- ignored by the trainer>"}

Two prompt families, byte-for-byte matched to llm_client.py so the adapter
actually transfers at serving time:

  * extract   system = llm_client.EXTRACT_SYS   user = LLMClient.extract_slots(...)
  * distress  system = llm_client.DISTRESS_SYS  user = LLMClient.classify_distress(...)

The canonical slot schema is code/slots.py -- imported, never re-typed here.
Every gold_slots dict below is run through llm_client.validate_slots() at build
time; if validation would change it, the build fails loudly (a bad label can't
reach the trainer).

WHY THIS SET LOOKS THE WAY IT DOES
---------------------------------
It is built to move the 5 failures in outputs/eval_baseline_all.json, all of
which are LLM-layer errors (the rules engine is deterministic and already
correct on all 5 offline):

  1. ems_triager_life_threatening  missed  -- distress classifier under-called
     `life_threatening` on a plain plea for help  -> distress positives below.
  2. ed_difficulty_breathing       -> 911  -- distress classifier promoted a
     mild "shortness of breath" to `severe_difficulty_breathing`
                                            -> distress HARD NEGATIVES below.
  3. educ_pleuritic_pain           -> 911  -- distress classifier read
     "hurts to breathe in" as a breathing emergency
                                            -> distress HARD NEGATIVES below.
  4. educ_triager_very_sick        missed  -- distress classifier under-called
     `very_sick_or_weak`                    -> distress positives below.
  5. pcp3_stable_angina_nitro_resolved -> ED -- extractor hallucinated
     other_symptoms=[difficulty_breathing] from a vague "some of those"
                                            -> extract ANTI-HALLUCINATION below.

v2 (measured, then rolled back): tried distress rows emitting only the true
  optional flags + 5 training epochs. Both over-corrected -- the classifier
  collapsed (micro-F1 0.74 -> 0.29, optional-flag recall -> 0) and the extractor
  over-fit (guard-slot key-F1 1.0 -> 0.67). Do NOT re-introduce either.

v3 CHANGES (current)
  * distress rows: reverted to all-six-flags-every-row (v1 behaviour).
  * KEPT the v2 extraction rows for value-accuracy misses:
    bystander_not_patient ("my dad had a heart attack at 50" != age 50),
    stent_not_angina, central_location_not_extracted, onset_precision.
  * train with --epochs 3 (not 5).

Run:
    python code/generate_sft_set.py
    python code/generate_sft_set.py --out data/train/sft_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from llm_client import DISTRESS_SYS, EXTRACT_SYS, validate_slots  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT = os.path.join(_HERE, "..", "data", "train", "sft_v1.jsonl")


# ---------------------------------------------------------------------------
# user-message builders -- copied verbatim from llm_client.py so training and
# serving see the identical string. If llm_client changes these, change here.
# ---------------------------------------------------------------------------
def _extract_user(patient_text: str, known: dict | None = None,
                  asked: list[str] | None = None) -> str:
    ctx = ""
    if known:
        ctx += ("Already known (do not repeat unless the patient corrects it): "
                + json.dumps(known, default=str) + "\n")
    if asked:
        ctx += "The assistant just asked about: " + ", ".join(asked) + "\n"
    return (ctx + f'Patient message:\n"""{patient_text.strip()}"""\n\n'
            "Return JSON with only the fields the patient gave.")


def _distress_user(patient_text: str, history: str = "") -> str:
    return ((f"Conversation so far:\n{history}\n\n" if history else "")
            + f'Latest patient message:\n"""{patient_text.strip()}"""')


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def ex_row(text: str, gold: dict, category: str,
           known: dict | None = None, asked: list[str] | None = None) -> dict:
    cleaned = validate_slots(gold)
    if cleaned != gold:
        raise SystemExit(
            f"[{category}] gold_slots {gold!r} for {text!r} is not schema-valid "
            f"-- validate_slots would produce {cleaned!r}. Fix the label."
        )
    return {
        "messages": [
            {"role": "system", "content": EXTRACT_SYS},
            {"role": "user", "content": _extract_user(text, known, asked)},
        ],
        "response": _dumps(gold),
        "category": category,
    }


def di_row(text: str, category: str, *, rationale: str, history: str = "",
           sdb=False, conf=False, shock=False, diap=False,
           lt=False, vsw=False) -> dict:
    # v3: back to emitting all six flags explicitly every row (v1 behaviour).
    # v2's "true-only" format collapsed the classifier at 5 epochs -- the model
    # learned that the four optional flags "don't appear" and stopped emitting
    # them at all (severe_difficulty_breathing / confused recall 1.0 -> 0.0,
    # micro-F1 0.74 -> 0.29). Emitting every key every row is clumsy but keeps
    # each flag in the output vocabulary and learnable.
    obj = {
        "severe_difficulty_breathing": sdb,
        "confused_or_hard_to_awaken": conf,
        "shock_signs": shock,
        "visible_facial_diaphoresis": diap,
        "life_threatening": lt,
        "very_sick_or_weak": vsw,
        "rationale": rationale,
    }
    return {
        "messages": [
            {"role": "system", "content": DISTRESS_SYS},
            {"role": "user", "content": _distress_user(text, history)},
        ],
        "response": _dumps(obj),
        "category": category,
    }


# ===========================================================================
# EXTRACTION DATA
# ===========================================================================
def extraction_rows() -> list[dict]:
    R: list[dict] = []
    add = R.append

    # -- duration: few_seconds --------------------------------------------
    for t in ["It's just a few seconds each time, then it's gone.",
              "Quick little jabs, a second or two at most.",
              "A brief stab, over almost before it starts.",
              "Fleeting -- a couple of seconds, tops.",
              "Just a split-second twinge now and then.",
              "Each one is like a one-second zap.",
              "Barely a moment -- blink and it's passed.",
              "Sharp pang for about a second, then nothing."]:
        add(ex_row(t, {"duration": "few_seconds"}, "duration"))
    # -- duration: under_5_min ------------------------------------------
    for t in ["Each episode is maybe three or four minutes.",
              "A couple of minutes, then it eases off.",
              "Less than five minutes, usually two or three.",
              "About four minutes each time it happens.",
              "Roughly a minute or two and it settles.",
              "Short -- never gets past about three minutes.",
              "Two, three minutes and then it lets go."]:
        add(ex_row(t, {"duration": "under_5_min"}, "duration"))
    # -- duration: over_5_min (resolved / not stated present) -----------
    for t in ["More than five minutes for sure, closer to ten.",
              "Each bout runs a good ten or fifteen minutes.",
              "Well over five minutes when it hits.",
              "The last one lasted a solid twenty minutes before it stopped.",
              "Easily ten minutes at a stretch."]:
        add(ex_row(t, {"duration": "over_5_min"}, "duration"))
    # -- duration: over_5_min AND ongoing now (co-extract present) -------
    for t in ["It's been going twenty minutes now and hasn't let up.",
              "Started about half an hour ago and it's still here.",
              "Going on fifteen minutes straight, right now.",
              "Forty minutes and counting -- it's happening as we speak.",
              "It hasn't stopped since it started an hour ago."]:
        add(ex_row(
            t, {"duration": "over_5_min", "chest_pain_present_now": True},
            "duration"))

    # -- pattern (flat bools) --------------------------------------------
    for t in ["It comes and goes.", "It's on and off all day.",
              "Comes in waves, then backs right off.",
              "Not constant -- it keeps coming back.",
              "It'll flare up, ease, then flare again.",
              "Intermittent -- a few minutes on, a while off."]:
        add(ex_row(t, {"pattern_comes_and_goes": True}, "pattern"))
    for t in ["It's been constant since it started.",
              "Steady the whole time, never eased.",
              "Once it started it just stayed put.",
              "No let-up at all -- it's been there non-stop.",
              "Continuous since this morning."]:
        add(ex_row(t, {"pattern_comes_and_goes": False}, "pattern"))
    for t in ["It's happening more often than yesterday.",
              "Each episode is worse than the last.",
              "It's definitely getting more intense.",
              "More frequent and stronger now than at the start.",
              "The gaps between them are getting shorter."]:
        add(ex_row(t, {"pattern_worsening": True}, "pattern"))
    for t in ["It's about the same as when it started.",
              "Not getting any worse.",
              "Same intensity, no real change.",
              "It's held steady -- no better, no worse."]:
        add(ex_row(t, {"pattern_worsening": False}, "pattern"))
    add(ex_row("It comes and goes, and it's getting worse each time.",
               {"pattern_comes_and_goes": True, "pattern_worsening": True},
               "pattern"))
    add(ex_row("It comes and goes but it hasn't changed at all.",
               {"pattern_comes_and_goes": True, "pattern_worsening": False},
               "pattern"))
    add(ex_row("Constant, and getting steadily worse.",
               {"pattern_comes_and_goes": False, "pattern_worsening": True},
               "pattern"))

    # -- radiation_sites -----------------------------------------------
    for t, v in [("It shoots down my left arm.", ["arm"]),
                 ("The pain spreads into my arm.", ["arm"]),
                 ("It goes up into my jaw.", ["jaw"]),
                 ("My jaw aches with it too.", ["jaw"]),
                 ("I feel it across my shoulders.", ["shoulder"]),
                 ("It runs up the side of my neck.", ["neck"]),
                 ("It goes straight through to my back.", ["back"]),
                 ("It bores through into my upper back.", ["back"]),
                 ("Down my arm and into my jaw at the same time.",
                  ["arm", "jaw"]),
                 ("Into my neck and left shoulder.", ["neck", "shoulder"])]:
        add(ex_row(t, {"radiation_sites": v}, "radiation"))
    for t in ["No, it stays right in the middle and doesn't spread.",
              "It's just in my chest, nowhere else.",
              "It doesn't move anywhere.",
              "Stays put -- no spreading to my arms or jaw."]:
        add(ex_row(t, {"radiation_sites": []}, "radiation"))

    # -- pain_qualities ----------------------------------------------
    for t, v in [("It's a heavy pressure, like something sitting on my chest.",
                  ["pressure", "heavy"]),
                 ("Feels like my chest is being crushed.", ["crushing"]),
                 ("A crushing weight, right in the centre.",
                  ["crushing", "heavy"]),
                 ("A tight band right around my chest.", ["tightness"]),
                 ("Sharp and stabbing.", ["sharp", "stabbing"]),
                 ("Just a sharp jab.", ["sharp"]),
                 ("Kind of a squeezing, like a fist closing.", ["squeezing"]),
                 ("Just a dull ache.", ["aching"]),
                 ("A burning feeling, like bad heartburn.", ["burning"]),
                 ("A tearing feeling, like something ripping.", ["tearing"]),
                 ("Pressure, like a belt cinched too tight.",
                  ["pressure", "tightness"])]:
        add(ex_row(t, {"pain_qualities": v}, "quality"))

    # -- severity ---------------------------------------------------
    for t, v in [("I'd say about an 8 out of 10.", 8),
                 ("Maybe a 3, it's mild.", 3),
                 ("It's a 6 right now.", 6),
                 ("A solid 10 -- worst pain of my life.", 10),
                 ("Around a 5, moderate.", 5),
                 ("Pretty bad -- I'd call it a 9.", 9),
                 ("Low, like a 2.", 2),
                 ("Maybe a 7 when it peaks.", 7)]:
        add(ex_row(t, {"severity_1_10": v}, "severity"))

    # -- onset_hours_ago (stored in hours) ------------------------
    for t, v in [("It started about twenty minutes ago.", 0.33),
                 ("Came on maybe half an hour ago.", 0.5),
                 ("It started this morning, six hours or so ago.", 6),
                 ("About two days ago.", 48),
                 ("Roughly three days back.", 72),
                 ("Sometime last week.", 168),
                 ("Just now, a few minutes ago.", 0.1),
                 ("Yesterday afternoon, about a day ago.", 24),
                 ("Four hours ago, give or take.", 4)]:
        add(ex_row(t, {"onset_hours_ago": v}, "onset"))

    # -- nitroglycerin_status --------------------------------------
    for t, v in [("I don't have nitroglycerin prescribed.", "not_prescribed"),
                 ("Never been given nitro.", "not_prescribed"),
                 ("I've got a nitro script but I haven't taken any for this.",
                  "prescribed_not_taken"),
                 ("I have nitro tablets, didn't use one this time.",
                  "prescribed_not_taken"),
                 ("I took my nitro and the pain went away completely.",
                  "taken_resolved"),
                 ("One nitro under the tongue and it cleared right up.",
                  "taken_resolved"),
                 ("I took two nitro tablets and it still hurts.",
                  "taken_not_resolved"),
                 ("Used my nitro spray twice, no relief.",
                  "taken_not_resolved")]:
        add(ex_row(t, {"nitroglycerin_status": v}, "nitro"))

    # -- cardiac risk / history -------------------------------------
    for t, v in [("I've got diabetes and high blood pressure.",
                  ["diabetes", "hypertension"]),
                 ("I smoke, and my cholesterol runs high.",
                  ["smoker", "high_cholesterol"]),
                 ("I'm diabetic.", ["diabetes"]),
                 ("My blood pressure's high and I'm quite overweight.",
                  ["hypertension", "obesity_bmi_30_plus"]),
                 ("I've got peripheral vascular disease.", ["pvd"]),
                 ("Strong family history -- my dad and brother both had early "
                  "heart attacks.", ["strong_family_history"])]:
        add(ex_row(t, {"cardiac_risk_factors": v}, "cardiac_history"))
    for t in ["None of those risk factors apply to me.",
              "No diabetes, no blood pressure problems, none of that."]:
        add(ex_row(t, {"cardiac_risk_factors": []}, "cardiac_history"))
    for t in ["I had a stent put in two years ago.",
              "Yes, I've had a heart attack before.",
              "I've had bypass surgery.",
              "I'm on medication for heart failure."]:
        add(ex_row(t, {"history_of_heart_disease": True}, "cardiac_history"))
    for t in ["No heart problems that I know of.",
              "My heart's always been fine, as far as I know."]:
        add(ex_row(t, {"history_of_heart_disease": False}, "cardiac_history"))
    add(ex_row("I was diagnosed with angina a few years back.",
               {"known_angina_history": True}, "cardiac_history"))
    add(ex_row("I've never been told I have angina.",
               {"known_angina_history": False}, "cardiac_history"))

    # -- pe_risk_factors -----------------------------------------
    for t, v in [("I had knee surgery about three weeks ago.",
                  ["recent_major_surgery_1mo"]),
                 ("Major abdominal op last month.",
                  ["recent_major_surgery_1mo"]),
                 ("My leg's been in a cast for the last month.",
                  ["leg_or_hip_fracture_or_cast_1mo"]),
                 ("Broke my hip a few weeks ago, still on crutches.",
                  ["leg_or_hip_fracture_or_cast_1mo"]),
                 ("I was stuck in bed with the flu for two weeks.",
                  ["prolonged_bedrest_1mo"]),
                 ("Just got off a ten-hour flight yesterday.",
                  ["long_travel_6h_1mo"]),
                 ("Long car trip last week, about eight hours straight.",
                  ["long_travel_6h_1mo"]),
                 ("I've had a blood clot in my leg before.",
                  ["prior_dvt_or_pe"]),
                 ("Had a pulmonary embolism a couple of years ago.",
                  ["prior_dvt_or_pe"]),
                 ("I have Factor V Leiden -- an inherited clotting problem.",
                  ["inherited_clotting_disorder"]),
                 ("I'm on chemo for breast cancer right now.",
                  ["cancer_active_or_treated_6mo"]),
                 ("Finished radiotherapy for lung cancer three months ago.",
                  ["cancer_active_or_treated_6mo"])]:
        add(ex_row(t, {"pe_risk_factors": v}, "pe_history"))
    for t in ["No surgery, no travel, no clot history -- none of that.",
              "None of those apply to me."]:
        add(ex_row(t, {"pe_risk_factors": []}, "pe_history"))

    # -- other_symptoms -----------------------------------------
    for t, v in [("I feel dizzy and a bit sick to my stomach.",
                  ["dizziness", "nausea"]),
                 ("I've been sweating a lot.", ["sweating"]),
                 ("I threw up once.", ["vomiting"]),
                 ("I'm running a fever.", ["fever"]),
                 ("A bit short of breath.", ["difficulty_breathing"]),
                 ("I've got a nagging cough.", ["cough"]),
                 ("I coughed up some blood.", ["hemoptysis"]),
                 ("There was blood when I coughed.", ["hemoptysis"]),
                 ("My heart's been racing and skipping beats.",
                  ["palpitations"]),
                 ("Lightheaded and clammy, and a bit nauseous.",
                  ["dizziness", "sweating", "nausea"])]:
        add(ex_row(t, {"other_symptoms": v}, "other_symptoms"))
    for t in ["No other symptoms at all.",
              "No -- no dizziness, nausea, fever, breathing trouble, none of it.",
              "Nothing else, just the chest pain."]:
        add(ex_row(t, {"other_symptoms": []}, "other_symptoms"))

    # -- GERD screen -- and the anti-default negatives (EXTRACT_SYS rule 7) --
    add(ex_row("It feels exactly like the heartburn I've been diagnosed with "
               "before.", {"heartburn_exact_match": True}, "gerd"))
    add(ex_row("It's a bit like heartburn but not really the same.",
               {"heartburn_exact_match": False}, "gerd"))
    add(ex_row("There's a burning feeling right behind my breastbone.",
               {"burning_in_chest": True}, "gerd"))
    add(ex_row("No burning, more of a pressure.",
               {"burning_in_chest": False}, "gerd"))
    add(ex_row("I've got a sour, acid taste in my mouth.",
               {"sour_taste_in_mouth": True}, "gerd"))
    add(ex_row("No sour or acid taste, no.",
               {"sour_taste_in_mouth": False}, "gerd"))
    add(ex_row("Exactly like my usual heartburn -- burning, and that sour "
               "taste too.",
               {"heartburn_exact_match": True, "burning_in_chest": True,
                "sour_taste_in_mouth": True}, "gerd"))
    for t, cause in [("It's probably just my heartburn acting up.", "heartburn"),
                     ("Honestly I think it's just acid reflux.", "acid reflux"),
                     ("Feels like indigestion to me.", "indigestion"),
                     ("My neighbour said it's probably stress.", "stress"),
                     ("I reckon it's a pulled muscle from the gym.",
                      "pulled muscle")]:
        add(ex_row(t, {"suspected_cause": cause}, "gerd_anti_default"))

    # -- focal screens --------------------------------------------
    for t, k, v in [
        ("Taking a deep breath in makes it much worse.",
         "pain_worse_with_deep_breath", True),
        ("It catches sharply when I breathe in.",
         "pain_worse_with_deep_breath", True),
        ("Breathing deeply doesn't change it.",
         "pain_worse_with_deep_breath", False),
        ("It's worse when I twist or press on the spot.",
         "pain_worse_with_movement", True),
        ("If I push on my chest, that's exactly where it hurts.",
         "pain_worse_with_movement", True),
        ("Moving around doesn't affect it at all.",
         "pain_worse_with_movement", False),
        ("It only hurts when I cough.", "pain_caused_by_coughing", True),
        ("Every cough sets off a jab.", "pain_caused_by_coughing", True),
        ("It's got nothing to do with coughing.",
         "pain_caused_by_coughing", False),
        ("There's a rash with little blisters right where it hurts.",
         "rash_at_pain_site", True),
        ("No rash anywhere.", "rash_at_pain_site", False),
    ]:
        add(ex_row(t, {k: v}, "focal_screen"))

    # -- misc bools ---------------------------------------------
    for t, k, v in [
        ("It started right after I got hit in the chest playing hockey.",
         "followed_chest_injury", True),
        ("This came on after a fall onto my chest.",
         "followed_chest_injury", True),
        ("No injury -- it just came on by itself.",
         "followed_chest_injury", False),
        ("I used some cocaine last night.", "cocaine_use_within_3_days", True),
        ("No, no drug use.", "cocaine_use_within_3_days", False),
        ("I blacked out for a few seconds earlier.", "passed_out", True),
        ("No, I haven't fainted or anything.", "passed_out", False),
        ("There's a chance I could be pregnant.", "pregnant", True),
        ("I'm about twelve weeks along.", "pregnant", True),
        ("No chance -- I'm not pregnant.", "pregnant", False),
        ("Yes, the pain is happening right now.",
         "chest_pain_present_now", True),
        ("It's not hurting right now, it passed a while ago.",
         "chest_pain_present_now", False),
        ("My pulse feels like it's about 130.", "heart_rate_bpm", 130),
        ("I took my temperature, it was 101.2.", "temperature_f", 101.2),
    ]:
        add(ex_row(t, {k: v}, "misc_bool"))

    # -- ANTI-HALLUCINATION / precision (drives baseline failure #5) --------
    grp = ["cardiac_symptoms_present_now", "other_symptoms"]
    for t in ["I do have some of those right now.",
              "Yeah, a couple of those, I suppose.",
              "Some of that, yes.",
              "A few of them, right now."]:
        add(ex_row(t, {"cardiac_symptoms_present_now": True},
                   "anti_overextract", asked=grp,
                   known={"duration": "under_5_min"}))
    for t in ["None of those right now.",
              "No, nothing like that at the moment."]:
        add(ex_row(t, {"cardiac_symptoms_present_now": False},
                   "anti_overextract", asked=grp))
    for t in ["I'm not really sure how to answer that.",
              "Hard to say, honestly.",
              "I don't know.",
              "Could you say that again?",
              "Hmm, maybe? I can't tell.",
              "I'd only be guessing.",
              "That's a difficult one to answer."]:
        add(ex_row(t, {}, "underspecified",
                   asked=["pattern_comes_and_goes", "pattern_worsening"]))
    add(ex_row("Yeah, I'm still 58.", {}, "no_new_info",
               known={"age": 58}, asked=["age"]))
    add(ex_row("Like I said, just a few minutes.", {}, "no_new_info",
               known={"duration": "under_5_min"}, asked=["duration"]))
    add(ex_row("Same as I told you -- it's in the centre of my chest.", {},
               "no_new_info", known={"location": "centre of chest"},
               asked=["location"]))
    add(ex_row("Actually, scrap that -- it's more like ten minutes, I "
               "misspoke.", {"duration": "over_5_min"}, "correction",
               known={"duration": "under_5_min"}, asked=["duration"]))
    add(ex_row("Wait, I got that wrong -- it does spread to my left arm.",
               {"radiation_sites": ["arm"]}, "correction",
               known={"radiation_sites": []}, asked=["radiation_sites"]))
    add(ex_row("It's probably nothing, I just want to be safe.", {},
               "underspecified", asked=["suspected_cause"]))

    # -- v2: value-accuracy misses seen in eval_extraction.py --------------
    # bystander age / bystander history is NOT the patient's -- "my dad had a
    # heart attack at 50" pulled age=50 + history_of_heart_disease in v1.
    for t in ["My dad had a heart attack at 50.",
              "My brother had a stent put in at 45.",
              "Heart disease runs in my family -- father and both uncles.",
              "My mum died of a heart attack young, in her fifties."]:
        add(ex_row(t, {"cardiac_risk_factors": ["strong_family_history"]},
                   "bystander_not_patient", asked=["cardiac_risk_factors"]))

    # a stent / bypass / prior MI is history_of_heart_disease, NOT a diagnosis
    # of angina (v1 added known_angina_history here); and being told you do NOT
    # have angina says nothing about heart disease.
    for t in ["I had a stent put in two years ago.",
              "I've had a triple bypass.",
              "I had a heart attack back in 2019."]:
        add(ex_row(t, {"history_of_heart_disease": True},
                   "stent_not_angina", asked=["history_of_heart_disease",
                                              "known_angina_history"]))
    for t in ["I've never been told I have angina.",
              "No, no one has ever said the word angina to me."]:
        add(ex_row(t, {"known_angina_history": False},
                   "stent_not_angina", asked=["history_of_heart_disease",
                                              "known_angina_history"]))

    # central chest location stated in passing is not extracted as `location`
    # (v1 hallucinated location on "middle of my chest" phrasings)
    for t, gold in [
        ("It stays right in the middle of my chest and doesn't go anywhere.",
         {"radiation_sites": []}),
        ("It's a crushing pain right in the middle of my chest.",
         {"pain_qualities": ["crushing"]}),
        ("Dead centre of my chest, and it's been constant.",
         {"pattern_comes_and_goes": False}),
    ]:
        add(ex_row(t, gold, "central_location_not_extracted"))

    # a bare denial to the associated-symptoms question is other_symptoms=[]
    # only (v1 also flipped chest_pain_present_now)
    for t in ["No, nothing like that, just the chest pain.",
              "None of those, only the chest pain itself."]:
        add(ex_row(t, {"other_symptoms": []}, "anti_overextract",
                   asked=["other_symptoms"]))

    # onset value precision (v1 got the key but the wrong number)
    for t, v in [("About ninety minutes ago.", 1.5),
                 ("Two and a half hours ago.", 2.5),
                 ("Since about eight this morning, so four or five hours.", 4.5),
                 ("A day and a half ago.", 36),
                 ("Roughly ten days back now.", 240),
                 ("It came on around 45 minutes ago.", 0.75)]:
        add(ex_row(t, {"onset_hours_ago": v}, "onset_precision",
                   asked=["onset_hours_ago"]))

    # "still there / easing" is chest_pain_present_now, not a pattern claim
    for t in ["It's easing off but still there a bit.",
              "Still going, though a little less intense now."]:
        add(ex_row(t, {"chest_pain_present_now": True},
                   "anti_overextract", asked=["chest_pain_present_now"]))

    return R


# ===========================================================================
# DISTRESS DATA
# ===========================================================================
def distress_rows() -> list[dict]:
    R: list[dict] = []
    add = R.append

    # -- life_threatening = true  (drives baseline failure #1) -------------
    for t in ["This feels like a real emergency, please help.",
              "I think I'm dying -- something is really wrong.",
              "Please, I need help right now, I'm scared it's my heart.",
              "My chest is being crushed and I can't take it.",
              "Oh god, this is bad. Really bad.",
              "I feel like something terrible is about to happen.",
              "Somebody get me an ambulance, I can't do this.",
              "I'm terrified -- my whole chest is seizing up.",
              "Help me, please, this doesn't feel survivable.",
              "I've never felt anything like this, I think it's serious.",
              "It's an emergency, I know it is, please hurry.",
              "I'm frightened this is a heart attack."]:
        add(di_row(t, "distress_life_threatening", lt=True,
                   rationale="Explicit plea for help / frames it as an "
                             "emergency; err toward flagging."))
    add(di_row("I think I'm dying and I can barely stay upright.",
               "distress_life_threatening", lt=True, vsw=True,
               rationale="Sense of impending doom plus weakness."))

    # -- very_sick_or_weak = true  (drives baseline failure #4) -----------
    for t in ["I feel extremely sick and weak all over.",
              "I can barely hold myself up, I'm so weak.",
              "I feel awful -- genuinely awful, worse than I've ever felt.",
              "Completely wiped out, something is really off with me.",
              "I'm so drained I can't get off the couch.",
              "Everything aches and I just feel really unwell.",
              "My whole body feels wrong, like it's shutting down.",
              "I feel horrendous, no energy at all, really shaky.",
              "So weak and rough -- worse by the hour.",
              "I just feel very, very sick."]:
        add(di_row(t, "distress_very_sick", vsw=True,
                   rationale="Reports feeling seriously unwell / profound "
                             "weakness."))
    add(di_row("I feel deeply sick, like my body is packing up on me.",
               "distress_very_sick", vsw=True, lt=True,
               rationale="Profound illness with a 'shutting down' quality -- "
                         "flag both on the ambiguity."))

    # -- severe_difficulty_breathing = true ----------------------------
    for t in ["I can only get a few words out before I have to stop for air.",
              "I'm gasping between every word.",
              "I can't catch my breath at all.",
              "I'm struggling for every single breath here.",
              "I can't... talk properly... too breathless.",
              "It's like breathing through a straw, I can't get enough air."]:
        add(di_row(t, "distress_severe_dyspnea", sdb=True, lt=True,
                   rationale="Speech broken by breathlessness / cannot get a "
                             "full breath."))

    # -- severe_difficulty_breathing = FALSE  (drives failures #2 and #3) --
    for t in ["I have some mild shortness of breath, nothing like gasping.",
              "A bit winded, but I can talk in full sentences fine.",
              "Only slightly out of breath after the stairs -- back to normal "
              "now.",
              "Breathing's fine, it's just the chest pain bothering me.",
              "Maybe a little puffed, but honestly nothing dramatic.",
              "I can breathe okay, it just aches when I do."]:
        add(di_row(t, "distress_mild_dyspnea_negative",
                   rationale="Mild or resolved breathlessness; patient "
                             "explicitly denies severe dyspnea."))
    for t in ["Taking a deep breath makes the pain worse, but my breathing "
              "itself is okay.",
              "It hurts to breathe in deeply, though I'm not short of breath.",
              "Sharp catch when I inhale -- breathing is otherwise normal.",
              "Deep breaths hurt, but I'm not gasping or anything.",
              "The pain spikes on inhaling; air's getting in fine though."]:
        add(di_row(t, "distress_pleuritic_negative",
                   rationale="Pleuritic chest pain (pain on inspiration), not "
                             "respiratory distress -- no breathing emergency."))

    # -- shock / confusion / diaphoresis = true ----------------------
    for t in ["My skin's gone cold and clammy and I feel like I'll collapse.",
              "I'm cold and sweaty and my legs won't hold me.",
              "Clammy all over and about to keel over."]:
        add(di_row(t, "distress_shock", shock=True, lt=True,
                   rationale="Cold, clammy skin with near-collapse."))
    add(di_row("I've gone grey and sweaty and I'm about to go down.",
               "distress_shock", shock=True, diap=True, lt=True,
               rationale="Grey, diaphoretic, near-syncope."))
    for t in ["I keep drifting off and my partner says I'm not making sense.",
              "My husband says I'm slurring and confused."]:
        add(di_row(t, "distress_confusion", conf=True, lt=True,
                   rationale="Reduced responsiveness / incoherent per a "
                             "bystander."))
    add(di_row("Sweat is absolutely pouring down my face, I'm drenched.",
               "distress_diaphoresis", diap=True,
               rationale="Profuse, visible facial sweating."))

    # -- ALL FALSE -- calm, stable (keeps the adapter from over-escalating) --
    for t in ["It's a mild ache that comes and goes. I feel fine otherwise, "
              "just want to know what to do.",
              "Two out of ten, not really worried -- figured I'd check.",
              "Bit of chest tightness earlier, gone now, feeling normal.",
              "Just a quick twinge when I move. No other issues.",
              "I'm calm, breathing's fine, no dizziness -- just being careful.",
              "It's annoying but I'm okay, sitting here comfortably.",
              "Had this on and off for years, same as always.",
              "Honestly I feel pretty good, just this odd flutter now and then.",
              "Mild, dull, and steady. Nothing else going on.",
              "I feel completely normal apart from a little soreness.",
              "No big deal really, I just wanted a second opinion.",
              "Chest feels a bit tight but I've been walking around fine.",
              "It's there but it's not stopping me doing anything."]:
        add(di_row(t, "distress_calm_negative",
                   rationale="Calm, stable account with no red-flag content."))

    # -- ambiguous -> flag anyway (matches DISTRESS_SYS calibration) -------
    for t in ["Honestly I feel really off and I can't explain why.",
              "Something doesn't feel right -- more than my usual.",
              "I'm a bit lightheaded and uneasy about this one.",
              "My gut is telling me this is serious.",
              "I can't put my finger on it, but I feel wrong.",
              "This is different from my normal aches and it worries me."]:
        add(di_row(t, "distress_ambiguous_flag", vsw=True,
                   rationale="Vague but genuine 'something is wrong' signal; "
                             "flag on uncertainty per calibration."))
    add(di_row("My gut says this could be my heart and I'm frightened.",
               "distress_ambiguous_flag", lt=True, vsw=True,
               rationale="Fear plus a cardiac concern -- err toward flagging."))

    # -- with conversation history, to teach that framing ---------------
    add(di_row(
        "It's a lot worse now and I feel like I might pass out.",
        "distress_life_threatening", shock=True, lt=True,
        history="Patient: I've had some chest pressure for about an hour.\n"
                "Assistant: On a scale of 1 to 10, how bad is it right now?",
        rationale="Escalating pain with pre-syncope."))
    add(di_row(
        "No change really, still just the mild ache.",
        "distress_calm_negative",
        history="Patient: Mild chest ache, comes and goes.\n"
                "Assistant: Any change since we started talking?",
        rationale="Explicitly stable, mild, no new features."))
    add(di_row(
        "Now I'm sweating buckets and feel sick with it.",
        "distress_very_sick", diap=True, vsw=True,
        history="Patient: Chest tightness since lunch.\n"
                "Assistant: Any other symptoms alongside the pain?",
        rationale="New profuse sweating and nausea -- unwell, though not "
                  "clearly life-threatening on its own."))

    return R


def build() -> list[dict]:
    return extraction_rows() + distress_rows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=_DEFAULT_OUT)
    args = ap.parse_args()

    rows = build()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    fam = {"extract": 0, "distress": 0}
    cats: dict[str, int] = {}
    for r in rows:
        sys_txt = r["messages"][0]["content"]
        fam["extract" if sys_txt == EXTRACT_SYS else "distress"] += 1
        cats[r["category"]] = cats.get(r["category"], 0) + 1

    print(f"wrote {len(rows)} rows -> {args.out}")
    print(f"  extract  : {fam['extract']}")
    print(f"  distress : {fam['distress']}")
    print("  by category:")
    for c in sorted(cats):
        print(f"    {c:32} {cats[c]}")


if __name__ == "__main__":
    main()

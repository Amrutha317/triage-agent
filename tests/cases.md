# Decision-engine test cases — eyeball against the protocol

Every row is a `pytest` case in `test_decision_engine.py`. "Rule" is the
`rules.yaml` id that must fire. All cases start from a benign baseline
(`age 30`, fleeting pains, resolved, nothing alarming) and perturb the
fields listed.

`pytest -q` green == `rules.yaml` is considered finalized.

## CALL EMS 911 NOW

| # | Scenario | Key slots | Rule |
|---|---|---|---|
| 1 | Severe difficulty breathing | `severe_difficulty_breathing=T` | `ems_severe_dyspnea` |
| 2 | Confused / hard to awaken | `confused_or_hard_to_awaken=T` | `ems_altered_mental_status` |
| 3 | Shock signs | `shock_signs=T` | `ems_shock` |
| 4 | Passed out | `passed_out=T` | `ems_syncope` |
| 5 | >5 min pain, present now, age 45 | `chest_pain_present_now=T, duration=over_5_min, age=45` | `ems_prolonged_pain_age` |
| 6 | >5 min pain, present now, age 35 + diabetes | `... age=35, cardiac_risk_factors=[diabetes]` | `ems_prolonged_pain_risk_factors` |
| 7 | >5 min pain, present now, known heart disease | `... history_of_heart_disease=T` | `ems_prolonged_pain_known_heart_disease` |
| 8 | >5 min pain, present now, crushing quality | `... pain_qualities=[crushing]` | `ems_prolonged_pain_crushing_quality` |
| 9 | Bradycardia 44 bpm | `heart_rate_bpm=44` | `ems_abnormal_heart_rate` |
| 10 | Tachycardia 165 bpm | `heart_rate_bpm=165` | `ems_abnormal_heart_rate` |
| 11 | Visible facial diaphoresis | `visible_facial_diaphoresis=T` | `ems_visible_diaphoresis` |
| 12 | Distress classifier: life-threatening | `triager_assessment_life_threatening=T` | `ems_triager_life_threatening` |
| 13 | **A1 check:** >5 min pain **resolved**, age 60, 10 h ago | `chest_pain_present_now=F, duration=over_5_min, age=60, onset_hours_ago=10` | → `educ_recent_prolonged_pain` (NOT 911) |

## SEE MORE APPROPRIATE GUIDELINE

| # | Scenario | Key slots | Rule |
|---|---|---|---|
| 14 | Pain followed a chest injury | `followed_chest_injury=T` | `injury_redirect` |
| 15 | **Ordering:** injury + real 911 sign → 911 wins | `followed_chest_injury=T, severe_difficulty_breathing=T` | `ems_severe_dyspnea` |

## GO TO ED NOW

| # | Scenario | Key slots | Rule |
|---|---|---|---|
| 16 | Severe pain 9/10 | `severity_1_10=9` | `ed_severe_pain` |
| 17 | Comes-and-goes angina, worsening, not fleeting | `duration=under_5_min, pattern_comes_and_goes=T, pattern_worsening=T` | `ed_worsening_angina` |
| 18 | **Exception:** worsening angina lasting only seconds | `duration=few_seconds, pattern_comes_and_goes=T, pattern_worsening=T` | → `home_fleeting` (exception blocks ED) |
| 19 | Radiation to arm | `radiation_sites=[arm]` | `ed_radiation_to_shoulder_arm_jaw` |
| 20 | **Exception:** radiation but clearly worse with movement | `radiation_sites=[arm], pain_worse_with_movement=T` | → `home_fleeting` (exception blocks ED) |
| 21 | Difficulty breathing (non-severe) | `other_symptoms=[difficulty_breathing]` | `ed_difficulty_breathing` |
| 22 | Coughing up blood | `other_symptoms=[hemoptysis]` | `ed_hemoptysis` |
| 23 | Cocaine use in last 3 days | `cocaine_use_within_3_days=T` | `ed_cocaine_use` |
| 24 | PE risk: recent major surgery | `pe_risk_factors=[recent_major_surgery_1mo]` | `ed_pe_risk_surgery_or_clotting_disorder` |
| 24b | PE risk: inherited clotting disorder | `pe_risk_factors=[inherited_clotting_disorder]` | `ed_pe_risk_surgery_or_clotting_disorder` |
| 25 | PE risk: hip/leg fracture or cast | `pe_risk_factors=[leg_or_hip_fracture_or_cast_1mo]` | `ed_pe_risk_immobility_or_clot_history` |
| 25b | PE risk: prolonged bedrest | `pe_risk_factors=[prolonged_bedrest_1mo]` | `ed_pe_risk_immobility_or_clot_history` |
| 25c | PE risk: long-distance travel | `pe_risk_factors=[long_travel_6h_1mo]` | `ed_pe_risk_immobility_or_clot_history` |
| 25d | PE risk: prior DVT/PE | `pe_risk_factors=[prior_dvt_or_pe]` | `ed_pe_risk_immobility_or_clot_history` |
| 26 | PE risk: active/recent cancer | `pe_risk_factors=[cancer_active_or_treated_6mo]` | `ed_pe_risk_cancer` |

> The protocol's seven PE-risk lines share a disposition (GO_TO_ED_NOW) and
> R/O (pulmonary_embolism) but print **three different care-advice lists**.
> An earlier single merged rule always emitted the same CA list, which was
> wrong for two lines and badly wrong for the cancer line (added "another
> adult should drive" / "nothing by mouth" / "call EMS if" that line doesn't
> call for). Caught by an independent conformance check against the source
> PDF; fixed by splitting into 3 rules by CA group. Each of the 7 members is
> now tested individually, including a CA-content assertion
> (`test_pe_risk_factor_care_advice_matches_protocol_exactly`).

## GO TO ED/UCC NOW (or PCP Triage)

| # | Scenario | Key slots | Rule |
|---|---|---|---|
| 26 | Pleuritic pain (worse on deep breath) | `pain_worse_with_deep_breath=T` | `educ_pleuritic_pain` |
| 27 | >5 min pain within 72 h, low risk, resolved | `chest_pain_present_now=F, duration=over_5_min, age=25, onset_hours_ago=40` | `educ_recent_prolonged_pain` |
| 28 | **Exception:** recent >5 min pain BUT exact heartburn + sour taste | `... heartburn_exact_match=T, sour_taste_in_mouth=T, burning_in_chest=F` | → `None` (exception blocks educ; GERD needs 3rd condition) |
| 29 | Distress classifier: very sick / weak | `triager_assessment_very_sick_weak=T` | `educ_triager_very_sick` |

## SEE PCP WITHIN 24 HOURS

| # | Scenario | Key slots | Rule |
|---|---|---|---|
| 30 | >5 min pain >3 days ago, nothing now | `chest_pain_present_now=F, duration=over_5_min, onset_hours_ago=120, cardiac_symptoms_present_now=F` | `pcp24_old_resolved_prolonged_pain` |
| 31 | Brief (<5 min) pain fully resolved | `chest_pain_present_now=F, duration=under_5_min, cardiac_symptoms_present_now=F, nitroglycerin_status=not_prescribed` | `pcp24_brief_pain_resolved` |
| 32 | Fever > 100.4 °F | `temperature_f=101.3` | `pcp24_fever` |
| 33 | Rash at pain site | `rash_at_pain_site=T` | `pcp24_rash_at_pain_site` |

## CALL PCP WITHIN 24 HOURS — the only GERD path

| # | Scenario | Key slots | Rule |
|---|---|---|---|
| 34 | All three GERD conditions present | `heartburn_exact_match=T, burning_in_chest=T, sour_taste_in_mouth=T` | `cpcp24_gerd` |
| 35 | Only two of three (no sour taste) → NOT GERD | `heartburn_exact_match=T, burning_in_chest=T, sour_taste_in_mouth=F` | → `home_fleeting` |
| 36 | Patient just describes "burning" quality → NOT GERD | `pain_qualities=[burning]` | → `home_fleeting` |

## SEE PCP WITHIN 3 DAYS

| # | Scenario | Key slots | Rule |
|---|---|---|---|
| 37 | <5 min pain, nitro prescribed but not taken | `chest_pain_present_now=T, duration=under_5_min, nitroglycerin_status=prescribed_not_taken` | `pcp3_stable_angina_nitro_not_taken` |
| 38 | **Shadow:** <5 min pain fully resolved after nitro | `chest_pain_present_now=F, duration=under_5_min, nitroglycerin_status=taken_resolved, cardiac_symptoms_present_now=F` | → `pcp24_brief_pain_resolved` (24 h rule wins, protocol order) |
| 39 | Nitro-resolved rule reachable only while a cardiac symptom persists | `... cardiac_symptoms_present_now=T` | `pcp3_stable_angina_nitro_resolved` |
| 40 | Known angina, comes and goes, not worsening | `duration=under_5_min, pattern_comes_and_goes=T, known_angina_history=T, pattern_worsening=F` | `pcp3_known_stable_angina` |
| 41 | Fleeting cough pains persisting > 3 days | `duration=few_seconds, pain_caused_by_coughing=T, onset_hours_ago=96` | `pcp3_cough_fleeting_persistent` |
| 42 | Fleeting pains persisting > 3 days (no cough) | `duration=few_seconds, onset_hours_ago=96` | `pcp3_fleeting_persistent` |

## HOME CARE

| # | Scenario | Key slots | Rule |
|---|---|---|---|
| 43 | Fleeting cough pains, recent | `duration=few_seconds, pain_caused_by_coughing=T, onset_hours_ago=6` | `home_cough_fleeting` |
| 44 | Fleeting pains, recent | `duration=few_seconds, onset_hours_ago=6` | `home_fleeting` |

## Ordering / priority

| # | Scenario | Key slots | Rule |
|---|---|---|---|
| 45 | EMS outranks ED when both match | `chest_pain_present_now=T, duration=over_5_min, age=70, severity_1_10=9` | `ems_prolonged_pain_age` |
| 46 | ED NOW outranks ED/UCC when both match | `severity_1_10=9, pain_worse_with_deep_breath=T` | `ed_severe_pain` |

## Three-valued logic — "not enough information yet"

| # | Scenario | Key slots | Result |
|---|---|---|---|
| 47 | Only age known | `{age: 50}` | `None` |
| 48 | >5 min pain present now, but age/risk/hx/quality all unknown | `{chest_pain_present_now: T, duration: over_5_min}` | `None` |
| 49 | Duration unknown → fleeting home rule cannot fire | benign minus `duration` | `None` |

Plus a meta-test (`test_every_rule_id_is_covered_by_a_positive_case`) that
fails if any rule in `rules.yaml` has no positive case firing it.

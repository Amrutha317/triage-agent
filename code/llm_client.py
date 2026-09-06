"""
llm_client.py -- the ONLY module that calls the language model.

Three jobs, none of which is choosing a disposition:

  1. extract_slots(text, ...)   free patient text  -> {slot: value}   (schema-constrained)
  2. classify_distress(text)    how the patient presents -> the observed_only red-flag
                                slots + the two subjective triager-judgment flags (A4)
  3. render(turn)               a state_machine Turn -> one natural sentence for the patient
                                (the disposition instruction is TEMPLATED, never generated)

Every call is streamed so we can record TTFT and total latency the same way for
baseline vs LoRA and GPU vs CPU (the eval harness reuses `LLMResult`).

Talks to any OpenAI-compatible server:
    BASE_URL   env, default http://localhost:8000/v1
    MODEL      env, default meta-llama/Llama-3.1-8B-Instruct
Point MODEL at "triage-lora" to hit the fine-tuned adapter served alongside the base.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

import slots as S

BASE_URL = os.environ.get("TRIAGE_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("TRIAGE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


# ============================================================================
# Prompts
# ============================================================================
def _slot_field_guide() -> str:
    lines = []
    for s in S.ALL_SLOTS:
        if s.observed_only:
            continue
        t = s.type
        if s.values:
            t = f"{s.type} one of {list(s.values)}"
        lines.append(f"- {s.id} ({t}): {s.question}  [{s.notes or ''}]".rstrip(" []"))
    return "\n".join(lines)


EXTRACT_SYS = (
    "You are a clinical intake information EXTRACTOR for a chest-pain triage "
    "service. You do not give advice, opinions, or diagnoses. You only convert "
    "what the patient SAYS into structured fields.\n\n"
    "Rules:\n"
    "1. Output JSON only, matching the given schema. No prose.\n"
    "2. Include a field ONLY if the patient stated or clearly implied its "
    "value in their message. If they did not mention something, OMIT the key. "
    "Never guess.\n"
    "3. Do not infer emotional state or severity of illness -- that is handled "
    "elsewhere. Only extract concrete facts.\n"
    "4. Use the enum values exactly as listed. Map the patient's words to the "
    "closest enum (e.g. 'a couple minutes' -> duration=under_5_min; 'few "
    "seconds' -> few_seconds; 'ten minutes' -> over_5_min).\n"
    "5. onset_hours_ago is in HOURS ('this morning' ~ 6, '2 days ago' = 48, "
    "'20 minutes ago' = 0.33).\n"
    "6. For list fields, return every value the patient mentioned; return [] "
    "only if they explicitly denied all of them.\n"
    "7. A patient saying 'it's probably just my heartburn' is a value for "
    "suspected_cause, NOT a reason to fill heartburn_exact_match / "
    "burning_in_chest / sour_taste_in_mouth -- only fill those if they "
    "describe each one.\n\n"
    "Field guide:\n" + _slot_field_guide()
)

DISTRESS_SYS = (
    "You assess HOW a patient with chest pain is presenting in a text chat -- "
    "their apparent distress and stability -- not the facts of their symptoms. "
    "You never give advice or a diagnosis. Output JSON only.\n\n"
    "CALIBRATION -- read this first: a false alarm here costs one extra "
    "escalation step that a human then reviews. A missed emergency costs a "
    "life. These are not symmetric mistakes. When a message is genuinely "
    "ambiguous between concerning and not concerning, flag it. You do not "
    "need certainty -- a plausible textual basis is enough. Do not require "
    "the patient to use clinical language; read plain, scared, or informal "
    "phrasing generously in the direction of flagging.\n\n"
    "Set a flag true if the message gives ANY real indication, even partial "
    "or hedged, of:\n"
    "- severe_difficulty_breathing: gasping, broken/one-word sentences from "
    "breathlessness, 'can't get a full breath', or anything suggesting real "
    "trouble breathing -- not just being a little winded.\n"
    "- confused_or_hard_to_awaken: disoriented, incoherent, rambling, someone "
    "else reports they're hard to wake or not making sense.\n"
    "- shock_signs: cold/clammy/grey skin, near-collapse, very weak, or "
    "lightheaded to the point of nearly passing out.\n"
    "- visible_facial_diaphoresis: sweating profusely, drenched, pouring "
    "sweat -- described explicitly, not just 'sweating a bit'.\n"
    "- life_threatening: taken together, the message could plausibly be a "
    "life-threatening emergency. Err toward true on real uncertainty.\n"
    "- very_sick_or_weak: sounds seriously unwell, exhausted, or "
    "deteriorating, even if not clearly an emergency.\n\n"
    "Leave a flag false only when the message gives no real indication either "
    "way -- e.g. a calm, stable description with nothing above present. "
    "Include a one-line 'rationale' explaining your calls, especially any "
    "flag you set on a hedged or ambiguous signal."
)

QUESTION_NLG_SYS = (
    "You are the voice of a calm, warm chest-pain triage assistant. You are "
    "given 1-3 things the assistant needs to ask. Rewrite them as ONE short, "
    "plain-language question a worried adult can answer easily. Do NOT add any "
    "medical advice, reassurance, diagnosis, or new questions. Keep it under 40 "
    "words. Output only the question."
)


# ============================================================================
# Deterministic disposition scripts (the load-bearing text -- never LLM'd)
# ============================================================================
DISPOSITION_SCRIPT: dict[str, str] = {
    "CALL_EMS_911_NOW":
        "Please stop and call 911 (or your local emergency number) now, or "
        "have someone call for you. This needs emergency help right away.",
    "GO_TO_ED_NOW":
        "You need to be seen in an Emergency Department now. Have someone else "
        "drive you or call for a ride -- do not drive yourself -- and bring a "
        "list of your medicines.",
    "GO_TO_ED_UCC_NOW_OR_PCP_TRIAGE":
        "You need to be seen within the next hour. Go to an Emergency "
        "Department or urgent care center now, or call your primary care "
        "provider's on-call line immediately to be told where to go.",
    "SEE_PCP_WITHIN_24_HOURS":
        "Contact your primary care provider to be seen within the next 24 "
        "hours. If their office is closed, an urgent care center is a good "
        "option.",
    "CALL_PCP_WITHIN_24_HOURS":
        "Call your primary care provider within the next 24 hours to talk "
        "through this.",
    "SEE_PCP_WITHIN_3_DAYS":
        "Make an appointment to see your primary care provider within the next "
        "2 to 3 days.",
    "HOME_CARE":
        "This can usually be looked after at home for now.",
    "SEE_MORE_APPROPRIATE_GUIDELINE":
        "Because this pain followed an injury to the chest, this tool is not "
        "the right fit. Please have the injury evaluated in person.",
}

_FIRST_AID_TEXT = {
    1610: "If you are not allergic to aspirin and have not been told to avoid "
          "it, you may chew one adult aspirin (about 325 mg) while you wait for "
          "the ambulance.",
    1045: "While you wait, lie down with your legs raised.",
}

_WORSEN_TAIL = (
    "If things get worse before then -- the pain lasts more than 5 minutes, "
    "spreads to your arm, jaw, neck or back, you get short of breath, break "
    "out in a sweat, or feel faint -- call 911 right away."
)

# Always-on second check, independent of the LLM classifier -- not just an
# outage fallback. Two independent paths to escalation is cheap insurance:
# the failure mode this guards against is the classifier under-calling an
# unambiguous phrase (temperature, prompt drift, a bad turn) mid-conversation,
# not just the API being down. Deliberately narrow and literal -- this is a
# belt-and-suspenders net for unmistakable phrases, not a substitute for the
# classifier's judgment on ambiguous ones. Maps to the MOST SPECIFIC slot it
# can (so the engine still fires the precise EMS rule, e.g.
# ems_severe_dyspnea rather than only the softer triager-judgment rule) and
# always also sets the general life-threatening flag.
_REDFLAG_PATTERNS: dict[str, re.Pattern] = {
    "severe_difficulty_breathing": re.compile(
        r"\b(can'?t breathe|cannot breathe|not breathing|can'?t catch my breath|"
        r"gasping for (air|breath)|struggling to breathe)\b", re.I),
    "passed_out": re.compile(
        r"\b(passed out|passing out|fainted|blacked out|unconscious|"
        r"lost consciousness)\b", re.I),
    "shock_signs": re.compile(
        r"\b(cold and clammy|clammy skin|too weak to stand|about to collapse)\b",
        re.I),
}
_REDFLAG_GENERAL = re.compile(
    r"\b(call 911|call an? ambulance|help me|going to die|dying|"
    r"chest is being crushed|this is an emergency)\b",
    re.I,
)


def keyword_redflags(text: str) -> dict[str, bool]:
    """Pure, deterministic, no LLM. Scans for unmistakable emergency phrases
    and returns the slots they imply -- always includes
    triager_assessment_life_threatening when anything matches. Called on
    EVERY patient message in classify_distress, not only when the LLM call
    fails; union'd with the classifier's own output (never turns a True back
    to False)."""
    text = text or ""
    hits: dict[str, bool] = {}
    for slot, pattern in _REDFLAG_PATTERNS.items():
        if pattern.search(text):
            hits[slot] = True
    if hits or _REDFLAG_GENERAL.search(text):
        hits["triager_assessment_life_threatening"] = True
    return hits


# ============================================================================
# Result type
# ============================================================================
@dataclass
class LLMResult:
    data: Any = None                  # parsed/validated payload (dict for extract/classify)
    text: str = ""                    # raw model text
    ttft_seconds: float | None = None
    total_seconds: float | None = None
    ok: bool = True
    error: str = ""
    meta: dict = field(default_factory=dict)


# ============================================================================
# Client
# ============================================================================
class LLMClient:
    def __init__(
        self,
        base_url: str = BASE_URL,
        model: str = MODEL,
        *,
        llm_questions: bool = True,     # soften questions with the LLM
        llm_final: bool = False,        # keep finals template-only by default
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        self.model = model
        self.llm_questions = llm_questions
        self.llm_final = llm_final
        self.max_retries = max_retries
        self.client = OpenAI(base_url=base_url, api_key="not-needed", timeout=timeout, max_retries=0)
        self._extract_schema = S.json_schema_for_extractor()

    # -- low-level: one streamed chat call with timing + retry ---------------
    def _chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        guided_json: dict | None = None,
    ) -> LLMResult:
        last_err = ""
        # NOTE: guided (schema-enforced) decoding is currently NOT sent. It was
        # removed in b4916df to resolve request timeouts / mid-stream drops on
        # the distress schema. `guided_json` is still accepted and threaded from
        # the call sites, but not applied here, so requests use free-form
        # generation + _parse_json() cleanup. vLLM's --guided-decoding-backend
        # flag in bootstrap.sh is therefore inert. To re-enable, add
        #   if guided_json is not None:
        #       extra["extra_body"] = {"guided_json": guided_json}
        # below AND confirm the serving backend handles DISTRESS_SYS's schema
        # (7 props, mixed types, a required array) without stalling.
        extra: dict = {}

        for attempt in range(self.max_retries + 1):
            try:
                start = time.perf_counter()
                first: float | None = None
                chunks: list[str] = []
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra,
                )
                for ch in stream:
                    if ch.choices and ch.choices[0].delta.content:
                        if first is None:
                            first = time.perf_counter()
                        chunks.append(ch.choices[0].delta.content)
                end = time.perf_counter()
                return LLMResult(
                    text="".join(chunks),
                    ttft_seconds=(first - start) if first else None,
                    total_seconds=end - start,
                    ok=True,
                    meta={"retried": attempt > 0, "attempts": attempt + 1},
                )
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
        return LLMResult(ok=False, error=last_err)

    # -- 1. slot extraction ------------------------------------------------
    def extract_slots(
        self,
        patient_text: str,
        known_slots: dict | None = None,
        asked_slots: list[str] | None = None,
    ) -> LLMResult:
        ctx = ""
        if known_slots:
            ctx += "Already known (do not repeat unless the patient corrects it): " \
                   + json.dumps(known_slots, default=str) + "\n"
        if asked_slots:
            ctx += "The assistant just asked about: " + ", ".join(asked_slots) + "\n"
        user = ctx + f'Patient message:\n"""{patient_text.strip()}"""\n\n' \
               "Return JSON with only the fields the patient gave."

        r = self._chat(
            [{"role": "system", "content": EXTRACT_SYS},
             {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=512,
            guided_json=self._extract_schema,
        )
        raw = _parse_json(r.text) if r.ok else {}
        r.meta["raw_json"] = raw
        r.data = validate_slots(raw)
        return r

    # -- 2. distress / red-flag classification ---------------------------
    def classify_distress(self, patient_text: str, history: str = "") -> LLMResult:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "severe_difficulty_breathing": {"type": "boolean"},
                "confused_or_hard_to_awaken": {"type": "boolean"},
                "shock_signs": {"type": "boolean"},
                "visible_facial_diaphoresis": {"type": "boolean"},
                "life_threatening": {"type": "boolean"},
                "very_sick_or_weak": {"type": "boolean"},
                "rationale": {"type": "string"},
            },
            "required": ["life_threatening", "very_sick_or_weak", "rationale"],
        }
        user = (f"Conversation so far:\n{history}\n\n" if history else "") + \
               f'Latest patient message:\n"""{patient_text.strip()}"""'
        r = self._chat(
            [{"role": "system", "content": DISTRESS_SYS},
             {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=200,
            guided_json=schema,
        )
        raw = _parse_json(r.text) if r.ok else {}
        out: dict = {}
        for k in ("severe_difficulty_breathing", "confused_or_hard_to_awaken",
                  "shock_signs", "visible_facial_diaphoresis"):
            if isinstance(raw.get(k), bool):
                out[k] = raw[k]
        if isinstance(raw.get("life_threatening"), bool):
            out["triager_assessment_life_threatening"] = raw["life_threatening"]
        if isinstance(raw.get("very_sick_or_weak"), bool):
            out["triager_assessment_very_sick_weak"] = raw["very_sick_or_weak"]

        # Always-on second check (not just on API failure) -- union only, so
        # it can add a True the classifier missed but never remove one it set.
        kw_hits = keyword_redflags(patient_text)
        if kw_hits:
            r.meta["keyword_redflag_hits"] = [
                k for k in kw_hits if not out.get(k)
            ]
            out.update(kw_hits)
        if not r.ok:
            r.meta["classifier_call_failed"] = True

        r.data = out
        r.meta["rationale"] = raw.get("rationale", "")
        return r

    # -- 3. natural language out ----------------------------------------
    def render(self, turn) -> str:
        if turn.kind == "final":
            return self.render_final(turn.decision)
        return self.render_question(turn)

    def render_question(self, turn) -> str:
        base = build_question_text(turn)
        prompts = [q for q in (turn.questions or []) if q]
        if not self.llm_questions or not prompts:
            return base
        r = self._chat(
            [{"role": "system", "content": QUESTION_NLG_SYS},
             {"role": "user", "content": "Ask the patient:\n- " + "\n- ".join(prompts)}],
            temperature=0.4,
            max_tokens=90,
        )
        out = (r.text or "").strip().strip('"')
        return out or base

    def render_final(self, decision) -> str:
        text = build_final_text(decision)
        if not self.llm_final:
            return text
        r = self._chat(
            [{"role": "system", "content": QUESTION_NLG_SYS.replace(
                "1-3 things the assistant needs to ask",
                "instructions the assistant must deliver")},
             {"role": "user", "content":
                 "Rephrase warmly for a worried patient. Keep EVERY instruction, "
                 "number and time frame exactly. Add nothing.\n\n" + text}],
            temperature=0.3,
            max_tokens=200,
        )
        return (r.text or "").strip().strip('"') or text


# ============================================================================
# Pure template builders -- no LLM. These are what render_question /
# render_final fall back to when the LLM pass is off, fails, or -- via
# guardrails.py -- gets rejected. Kept module-level (not methods) so
# guardrails.py can import them directly as the known-safe text to substitute
# on a violation, with no client/model dependency.
# ============================================================================
def build_question_text(turn) -> str:
    prompts = [q for q in (turn.questions or []) if q]
    return " ".join(prompts) if prompts else "Can you tell me a bit more?"


def build_final_text(decision) -> str:
    disp = decision.disposition
    parts = [DISPOSITION_SCRIPT.get(disp, DISPOSITION_SCRIPT["GO_TO_ED_UCC_NOW_OR_PCP_TRIAGE"])]
    for ca in getattr(decision, "first_aid", []) or []:
        if ca in _FIRST_AID_TEXT:
            parts.append(_FIRST_AID_TEXT[ca])
    if getattr(decision, "pregnant_flag", False) and disp in (
        "CALL_EMS_911_NOW", "GO_TO_ED_NOW", "GO_TO_ED_UCC_NOW_OR_PCP_TRIAGE"
    ):
        parts.append("Because you may be pregnant, go to a hospital with a "
                     "labor-and-delivery unit if you can.")
    if disp != "CALL_EMS_911_NOW":
        parts.append(_WORSEN_TAIL)
    parts.append("I can't tell you what is causing this -- a clinician needs "
                 "to assess you in person.")
    return " ".join(parts)


# ============================================================================
# Validation helpers (pure, unit-testable, no LLM)
# ============================================================================
def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            v = json.loads(m.group(0))
            return v if isinstance(v, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _coerce(slot: S.Slot, value: Any) -> Any:
    """Return a schema-valid value for `slot`, or None to drop it."""
    if value is None:
        return None
    t = slot.type
    if t == S.BOOL:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "yes"):
            return True
        if isinstance(value, str) and value.lower() in ("false", "no"):
            return False
        return None
    if t in (S.INT, S.FLOAT):
        try:
            n = float(value)
        except (TypeError, ValueError):
            return None
        return int(round(n)) if t == S.INT else n
    if t == S.ENUM:
        return value if value in slot.values else None
    if t == S.ENUM_SET:
        if not isinstance(value, (list, tuple, set)):
            value = [value]
        clean = [v for v in value if v in slot.values]
        return clean            # [] is meaningful ("asked, none")
    # TEXT
    return str(value) if not isinstance(value, str) else value


def validate_slots(raw: dict) -> dict:
    """Drop unknown keys, drop observed_only keys (A4 -- those come from the
    classifier), coerce each value to its schema. Empty/None values dropped
    except [] for list slots."""
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        slot = S.SLOTS_BY_ID.get(k)
        if slot is None or slot.observed_only:
            continue
        cv = _coerce(slot, v)
        if cv is None:
            continue
        if cv == "" :
            continue
        out[k] = cv
    return out


# ============================================================================
if __name__ == "__main__":
    c = LLMClient()
    demo = ("It's a heavy pressure right in the middle of my chest, been going "
            "about 20 minutes, and it's spreading into my left arm. I feel a "
            "bit sick and sweaty.")
    print("--- extract ---")
    r = c.extract_slots(demo)
    print("ttft=%.2fs total=%.2fs ok=%s" % (r.ttft_seconds or -1, r.total_seconds or -1, r.ok))
    print(json.dumps(r.data, indent=2))
    print("--- distress ---")
    d = c.classify_distress(demo)
    print(json.dumps(d.data, indent=2), "|", d.meta.get("rationale"))

"""
guardrails.py -- the last check before any text reaches the patient.

Deterministic, no LLM. (A guardrail that itself needs the model to check the
model is weaker, slower, and adds its own failure mode -- exactly what a
guardrail exists to remove.) Runs on every rendered message, questions and
finals alike, and fails CLOSED: if a check trips, the message is replaced
with the pure template text from llm_client's build_question_text /
build_final_text -- never delivered with a warning bolted on, never "mostly
right." Those builders are LLM-free by construction, so the fallback is
always safe regardless of why the check tripped.

This is the layer that makes it acceptable to let the LLM touch wording at
all: nothing it writes reaches the patient unchecked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from llm_client import build_final_text, build_question_text

# ============================================================================
# Banned patterns
# ============================================================================
# Diagnostic / conclusive language the LLM must never state as fact about the
# patient -- the engine's rule_out list is a differential for a clinician,
# never a conclusion to hand the patient. Deliberately broad: a false
# positive here just means "fall back to the template," which is always
# safe. A false negative is the actual risk, so err toward over-matching.
_DIAGNOSIS_PATTERNS = [
    r"you\s*('re|\s+are)\s+having\s+a\s+(heart attack|stroke)",
    r"this\s+is\s+(a\s+|an\s+)?(heart attack|stroke|myocardial infarction|"
    r"pulmonary embolism|aortic dissection|panic attack)",
    r"you\s+have\s+(gerd|acid reflux|heartburn|angina|a blood clot)",
    r"it'?s\s+(probably\s+|just\s+|only\s+)?(reflux|heartburn|nothing|anxiety|stress)",
    r"don'?t\s+worry",
    r"you'?re\s+(probably\s+)?fine",
    r"no\s+need\s+to\s+(worry|be concerned|see (a doctor|anyone))",
]

# Hedging that must never appear in an EMS-911 message -- that instruction has
# to stay an unambiguous imperative, never a suggestion.
_HEDGE_PATTERNS = [
    r"you\s+(might|may)\s+(want|wish)\s+to",
    r"consider\s+calling",
    r"if\s+you\s+feel\s+like\s+it",
    r"when\s+you\s+get\s+a\s+chance",
    r"no\s+rush",
    r"it'?s\s+up\s+to\s+you",
    r"whenever\s+is\s+convenient",
]

# Drug/dosing language beyond the one templated aspirin instruction --
# guards against the LLM inventing medication advice it isn't licensed to
# give. Aspirin is allowed since that's the actual templated first-aid text.
_UNAPPROVED_DRUG_PATTERN = re.compile(
    r"\b(ibuprofen|acetaminophen|tylenol|advil|motrin|naproxen|nitroglycerin|"
    r"morphine)\b",
    re.I,
)

_DIAGNOSIS_RE = re.compile("|".join(_DIAGNOSIS_PATTERNS), re.I)
_HEDGE_RE = re.compile("|".join(_HEDGE_PATTERNS), re.I)

_EMS_DISPOSITION = "CALL_EMS_911_NOW"
_EMERGENCY_CUE_RE = re.compile(r"\b911\b|\bemergency\b", re.I)


# ============================================================================
@dataclass
class GuardrailResult:
    ok: bool                                   # False = a violation was caught
    text: str                                  # what to actually show the patient
    violations: list[str] = field(default_factory=list)
    original: str = ""                         # what the LLM/template produced, for logging


def check_text(
    text: str, *, disposition: str | None = None, is_final: bool = False
) -> list[str]:
    """Pure check -- no fallback substitution, just the list of violations
    (empty = clean). Used by both guard_question/guard_final and directly by
    tests.

    `is_final` gates the medication check: asking ABOUT a drug ("did you take
    your nitroglycerin?") is a normal, protocol-driven question -- several
    slot questions legitimately name a drug. Only a FINAL message could
    contain a drug *directive*, which is the actual risk (an invented dose or
    medication beyond the one templated aspirin instruction), so the check
    only runs there."""
    t = text or ""
    violations: list[str] = []

    if _DIAGNOSIS_RE.search(t):
        violations.append("diagnostic or falsely-reassuring language")

    if is_final:
        # Aspirin isn't in the banned pattern -- it's the one templated
        # first-aid drug (CA 1610), so any match here is unapproved by
        # definition.
        drug_hit = _UNAPPROVED_DRUG_PATTERN.search(t)
        if drug_hit:
            violations.append(f"unapproved medication mention: '{drug_hit.group(0)}'")

    if disposition == _EMS_DISPOSITION:
        if not _EMERGENCY_CUE_RE.search(t):
            violations.append("911 disposition text doesn't mention 911/emergency")
        if _HEDGE_RE.search(t):
            violations.append("hedging language on a 911 disposition")

    return violations


# ============================================================================
# Fail-closed wrappers -- these are what agent.py actually calls
# ============================================================================
def guard_question(turn, rendered_text: str) -> GuardrailResult:
    violations = check_text(rendered_text, disposition=None)
    if violations:
        return GuardrailResult(
            ok=False, text=build_question_text(turn),
            violations=violations, original=rendered_text,
        )
    return GuardrailResult(ok=True, text=rendered_text, original=rendered_text)


def guard_final(decision, rendered_text: str) -> GuardrailResult:
    violations = check_text(rendered_text, disposition=decision.disposition, is_final=True)
    if violations:
        return GuardrailResult(
            ok=False, text=build_final_text(decision),
            violations=violations, original=rendered_text,
        )
    return GuardrailResult(ok=True, text=rendered_text, original=rendered_text)


if __name__ == "__main__":
    from decision_engine import Decision

    d = Decision(disposition="CALL_EMS_911_NOW", rule_id="ems_severe_dyspnea")
    bad = "You might want to consider calling someone if you feel like it."
    r = guard_final(d, bad)
    print("ok:", r.ok, "violations:", r.violations)
    print("delivered text:", r.text)

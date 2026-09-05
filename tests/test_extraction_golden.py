"""
test_extraction_golden.py -- the turn-level extraction golden set is
well-formed. Doesn't test model quality (there's no model here) -- tests that
the GOLD ITSELF is internally consistent, so a bad row can't silently poison
the bake-off or the extraction-quality report later.

Run:  pytest -q
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import slots as S  # noqa: E402
from llm_client import validate_slots  # noqa: E402

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "eval", "extraction_golden.jsonl"
)


def _load():
    with open(_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_file_exists_and_has_rows():
    rows = _load()
    assert len(rows) >= 50, f"only {len(rows)} rows -- expected a real golden set"


def test_every_row_has_the_required_fields():
    for i, r in enumerate(_load()):
        for field in ("tag", "patient_text", "known_slots", "gold_slots"):
            assert field in r, f"row {i} missing '{field}'"
        assert isinstance(r["patient_text"], str) and r["patient_text"].strip()


def test_gold_slots_are_schema_valid_and_unchanged_by_validation():
    """If validate_slots() would alter a gold row, the gold row itself is
    wrong (bad enum member, wrong type, etc.) -- fix the data, not the code."""
    for i, r in enumerate(_load()):
        gold = r["gold_slots"]
        cleaned = validate_slots(gold)
        assert cleaned == gold, (
            f"row {i} ({r['patient_text']!r}): gold_slots {gold} is not "
            f"schema-valid -- validate_slots would produce {cleaned}"
        )


def test_known_slots_are_also_schema_valid():
    for i, r in enumerate(_load()):
        known = r["known_slots"]
        assert validate_slots(known) == known, f"row {i}: known_slots {known} invalid"


def test_no_observed_only_slot_ever_appears_in_gold():
    """A4: observed_only slots are set exclusively by the distress classifier,
    never by extraction. A gold row containing one would silently teach the
    bake-off / eval the wrong contract."""
    observed_only = {s.id for s in S.ALL_SLOTS if s.observed_only}
    for i, r in enumerate(_load()):
        leaked = observed_only & set(r["gold_slots"])
        assert not leaked, f"row {i}: observed_only slot(s) in gold: {leaked}"


def test_every_askable_slot_has_at_least_one_example():
    askable = {s.id for s in S.ALL_SLOTS if not s.observed_only}
    covered = set()
    for r in _load():
        covered.update(r["gold_slots"])
        covered.update(r["known_slots"])
    missing = askable - covered
    assert not missing, f"no golden example touches: {sorted(missing)}"


def test_includes_the_gerd_trap_as_a_negative_case():
    """The brief's core safety requirement -- 'must not assume chest
    discomfort is heartburn' -- must have a golden case where the patient
    says 'it's probably my heartburn' and the gold does NOT fill any of the
    three GERD slots."""
    rows = _load()
    trap = [r for r in rows if r["tag"] == "gerd_trap"]
    assert trap, "no gerd_trap row found"
    for r in trap:
        assert "heartburn_exact_match" not in r["gold_slots"]
        assert "burning_in_chest" not in r["gold_slots"]
        assert "sour_taste_in_mouth" not in r["gold_slots"]


def test_includes_conservative_no_guess_cases():
    """At least one row where the gold is deliberately empty -- an honest
    'nothing extractable here', to catch a model that hallucinates values on
    vague input rather than abstaining."""
    rows = _load()
    empty = [r for r in rows if r["gold_slots"] == {}]
    assert empty, "no row with an intentionally empty gold_slots"

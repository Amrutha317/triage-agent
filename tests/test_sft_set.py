"""
test_sft_set.py -- the generated SFT training file is well-formed and matches
the live serving prompts. Doesn't test model quality (no model here) -- tests
that a bad row can't silently reach the trainer.

Run:  pytest -q
Regenerate the file first:  python code/generate_sft_set.py
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from llm_client import DISTRESS_SYS, EXTRACT_SYS, validate_slots  # noqa: E402

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "train", "sft_v1.jsonl"
)


def _load():
    if not os.path.exists(_PATH):
        pytest.skip("data/train/sft_v1.jsonl missing -- run code/generate_sft_set.py")
    with open(_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_has_a_real_number_of_rows():
    assert len(_load()) >= 200


def test_shape_matches_finetune_lora_loader():
    """finetune_lora.to_text does messages + [{assistant: response}]."""
    for i, r in enumerate(_load()):
        assert set(r) >= {"messages", "response"}, f"row {i}"
        assert [m["role"] for m in r["messages"]] == ["system", "user"], f"row {i}"
        assert isinstance(r["response"], str) and r["response"].strip(), f"row {i}"


def test_system_prompt_is_verbatim_one_of_the_two_live_prompts():
    for i, r in enumerate(_load()):
        assert r["messages"][0]["content"] in (EXTRACT_SYS, DISTRESS_SYS), (
            f"row {i}: system prompt drifted from llm_client.py"
        )


def test_every_response_is_strict_json():
    for i, r in enumerate(_load()):
        json.loads(r["response"])  # raises on the sloppy-JSON we must not train on


def test_extraction_labels_are_schema_valid():
    for i, r in enumerate(_load()):
        if r["messages"][0]["content"] != EXTRACT_SYS:
            continue
        gold = json.loads(r["response"])
        assert validate_slots(gold) == gold, (
            f"row {i}: extraction label {gold} is not canonical-schema valid"
        )


def test_distress_labels_have_the_required_keys():
    for i, r in enumerate(_load()):
        if r["messages"][0]["content"] != DISTRESS_SYS:
            continue
        obj = json.loads(r["response"])
        for k in ("life_threatening", "very_sick_or_weak", "rationale"):
            assert k in obj, f"row {i}: distress label missing {k!r}"
        for k in ("life_threatening", "very_sick_or_weak"):
            assert isinstance(obj[k], bool), f"row {i}: {k} not a bool"


def test_both_prompt_families_are_represented():
    fams = {r["messages"][0]["content"] for r in _load()}
    assert EXTRACT_SYS in fams and DISTRESS_SYS in fams

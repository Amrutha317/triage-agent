"""
probe_distress.py -- side-by-side distress-classifier output for the base model
vs a served LoRA adapter. Diagnostic only -- not part of the pipeline.

Two modes:
  (default)      a fixed list of hard single phrases.
  --scenarios    the REAL first-turn utterance patient_sim builds for each
                 named eval scenario (scary phrase + mild details concatenated,
                 exactly what classify_distress sees at eval time). This is the
                 one that reproduces the eval misses.

Needs a vLLM server with BOTH models up:
    LORA_DIR=adapters/triage-lora bash bootstrap.sh

Run:
    python code/probe_distress.py --scenarios
    python code/probe_distress.py
    python code/probe_distress.py --phrase "I think I'm dying"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))

from llm_client import DISTRESS_SYS  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCENARIOS = os.path.join(_HERE, "..", "data", "eval", "scenarios.jsonl")

_DEFAULT_PHRASES = [
    "This feels like a real emergency, please help.",
    "I feel extremely sick and weak all over.",
    "I'm drenched in sweat, it's pouring down my face.",
    "Taking a deep breath makes the pain worse.",
    "I have some mild shortness of breath, nothing like gasping.",
]

# eval scenarios whose gold needs a distress flag and which baseline/LoRA missed
_MISS_SCENARIOS = [
    "ems: triager judges life-threatening",
    "educ: triager judges very sick/weak",
    "ems: visible facial diaphoresis",
    "ed: difficulty breathing (non-severe)",
    "educ: pleuritic pain (worse on deep breath)",
]

_FLAGS = ("severe_difficulty_breathing", "confused_or_hard_to_awaken",
          "shock_signs", "visible_facial_diaphoresis",
          "life_threatening", "very_sick_or_weak")


def classify(client: OpenAI, model: str, phrase: str) -> dict:
    r = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=220,
        messages=[{"role": "system", "content": DISTRESS_SYS},
                  {"role": "user",
                   "content": f'Latest patient message:\n"""{phrase}"""'}],
    )
    txt = r.choices[0].message.content or ""
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0)) if m else {"_raw": txt[:200]}


def fmt(d: dict) -> str:
    if "_raw" in d:
        return f"UNPARSEABLE: {d['_raw']}"
    on = [k.replace("_", " ") for k in _FLAGS if d.get(k) is True]
    return "all false" if not on else ", ".join(on)


def scenario_utterances() -> list[tuple[str, str]]:
    from patient_sim import make_patient_sim
    by_name = {}
    with open(_SCENARIOS, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                o = json.loads(line)
                by_name[o["name"]] = o
    out = []
    for name in _MISS_SCENARIOS:
        s = by_name.get(name)
        if s:
            out.append((name, make_patient_sim(s["facts"])(None)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=os.environ.get(
        "TRIAGE_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--lora", default="triage-lora")
    ap.add_argument("--scenarios", action="store_true",
                    help="use patient_sim's real first-turn text for the missed "
                         "eval scenarios")
    ap.add_argument("--phrase", action="append", dest="phrases")
    args = ap.parse_args()

    if args.scenarios:
        items = scenario_utterances()
    else:
        items = [(None, p) for p in (args.phrases or _DEFAULT_PHRASES)]

    client = OpenAI(base_url=args.url, api_key="not-needed")
    for name, text in items:
        if name:
            print(f"\n[{name}]")
        print(f"  {text}")
        for label, model in (("base", args.base), ("lora", args.lora)):
            print(f"    {label:5} ->  {fmt(classify(client, model, text))}")


if __name__ == "__main__":
    main()

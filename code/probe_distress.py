"""
probe_distress.py -- side-by-side distress-classifier output for the base model
vs a served LoRA adapter, on a fixed set of hard phrases (the ones behind the
baseline / LoRA eval misses). Diagnostic only -- not part of the pipeline.

Needs a vLLM server up with BOTH models available, e.g.:
    LORA_DIR=adapters/triage-lora bash bootstrap.sh

Run:
    python code/probe_distress.py
    python code/probe_distress.py --base meta-llama/Llama-3.1-8B-Instruct --lora triage-lora
    python code/probe_distress.py --phrase "I think I'm dying" --phrase "mild ache, I'm fine"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))

from llm_client import DISTRESS_SYS  # noqa: E402

# The phrases behind the eval misses: 2 subjective-triager slots + diaphoresis
# (LoRA regressed these), then 2 true negatives (LoRA should stay calm on these).
_DEFAULT_PHRASES = [
    "This feels like a real emergency, please help.",          # -> life_threatening
    "I feel extremely sick and weak all over.",                # -> very_sick_or_weak
    "I'm drenched in sweat, it's pouring down my face.",       # -> visible_facial_diaphoresis
    "Taking a deep breath makes the pain worse.",              # -> all false (pleuritic)
    "I have some mild shortness of breath, nothing like gasping.",  # -> all false
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
    return ("all false" if not on else ", ".join(on))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=os.environ.get(
        "TRIAGE_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--lora", default="triage-lora")
    ap.add_argument("--phrase", action="append", dest="phrases")
    args = ap.parse_args()

    phrases = args.phrases or _DEFAULT_PHRASES
    client = OpenAI(base_url=args.url, api_key="not-needed")

    for p in phrases:
        print(f"\n{p}")
        for label, model in (("base", args.base), ("lora", args.lora)):
            print(f"  {label:5} [{model}]  ->  {fmt(classify(client, model, p))}")


if __name__ == "__main__":
    main()

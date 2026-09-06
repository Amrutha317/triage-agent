# `data/train/` — SFT training data for the triage LoRA

## The three datasets, and why they're different

| file | rows | what one row is | who reads it | format |
|---|---|---|---|---|
| `data/eval/scenarios.jsonl` | 59 | a whole patient case: ground-truth `facts` + the disposition a correct run must reach | `eval_harness.py` (drives a full simulated conversation, scores final disposition + `rule_id`) | `{name, facts, gold_disposition, gold_rule_id}` |
| `data/eval/extraction_golden.jsonl` | 112 | **one patient sentence** + the exact slot dict a perfect extractor should return — no conversation, no rules engine | `tests/test_extraction_golden.py` (well-formedness today); a per-field precision/recall script would run it against `extract_slots()` | `{tag, patient_text, known_slots, gold_slots}` |
| `data/train/sft_v1.jsonl` | ~290 | **one LLM call** (an `extract_slots` *or* a `classify_distress` call) + its gold JSON output, in chat-fine-tuning form | `finetune_lora.py` → the LoRA adapter | `{messages: [system, user], response, category}` |

### "What is the extraction dataset?" — `data/eval/extraction_golden.jsonl`

It is the **ground truth for the NLU layer in isolation**. Each row is a single
utterance a patient might type and the precise `{slot: value}` a flawless
extractor would produce from it — e.g.

```json
{"tag": "duration_under_5", "patient_text": "It lasts maybe 3 or 4 minutes each time.",
 "known_slots": {}, "gold_slots": {"duration": "under_5_min"}}
```

No state machine, no `rules.yaml`, no disposition. It answers one question only:
*did the model read the patient's words into the right fields?* It is an
**evaluation** set (measure quality) — it is **not** training data, and the
trainer never sees it. `sft_v1.jsonl` is a separate, independently-worded set in
the same spirit (utterance → gold JSON) but (a) in chat-SFT format, (b) larger,
and (c) it also covers the **distress classifier**, which `extraction_golden`
does not.

Keeping them separate matters: if training rows were copied from
`extraction_golden.jsonl`, the extraction eval would be measuring memorisation,
not generalisation.

## How `sft_v1.jsonl` is built

`code/generate_sft_set.py` — deterministic, no network, no LLM. It imports
`EXTRACT_SYS` / `DISTRESS_SYS` straight from `code/llm_client.py`, so every
training row's system+user text is **byte-identical to what the model sees at
serving time** (that's what makes a LoRA transfer). The canonical slot schema
is `code/slots.py`; every extraction label is passed through
`llm_client.validate_slots()` at build time and the build aborts on any label
that isn't schema-valid.

Regenerate any time:

```bash
python code/generate_sft_set.py            # -> data/train/sft_v1.jsonl
pytest -q tests/test_sft_set.py            # sanity-checks the file
```

### What it targets

All 5 disposition misses in `outputs/eval_baseline_all.json` are LLM-layer
errors (the rules engine is deterministic and already correct on all 5 in
`--offline` mode). The set is weighted toward them:

| baseline failure | cause | rows that address it |
|---|---|---|
| `ems_triager_life_threatening` missed | distress classifier under-called `life_threatening` on a plain plea for help | `distress_life_threatening` (+ `distress_ambiguous_flag`) |
| `ed_difficulty_breathing` → 911 | distress classifier promoted mild "short of breath" to `severe_difficulty_breathing` | `distress_mild_dyspnea_negative` |
| `educ_pleuritic_pain` → 911 | distress classifier read "hurts to breathe in" as a breathing emergency | `distress_pleuritic_negative` |
| `educ_triager_very_sick` missed | distress classifier under-called `very_sick_or_weak` | `distress_very_sick` |
| `pcp3_stable_angina_nitro_resolved` → ED | extractor invented `other_symptoms=[difficulty_breathing]` from a vague "some of those" | `anti_overextract`, `underspecified`, `no_new_info` |

`distress_calm_negative` (all-flags-false, calm descriptions) is the
counterweight so the adapter doesn't learn "always escalate".

## Running the fine-tune (on the RunPod pod, repo root)

```bash
export HF_TOKEN=hf_...                       # Llama-3.1-8B-Instruct is gated
python code/generate_sft_set.py              # writes data/train/sft_v1.jsonl (~290 rows)
python code/finetune_lora.py --epochs 5      # 5 epochs suits a set this size
python code/finetune_lora.py                 # base model defaults to Llama-3.1-8B-Instruct
                                             # adapter -> adapters/triage-lora/
LORA_DIR=adapters/triage-lora bash bootstrap.sh          # serve base + adapter
TRIAGE_MODEL=triage-lora python code/eval_harness.py \
    --out outputs/eval_lora.json --label triage-lora     # same harness, compare to baseline
```

`finetune_lora.py` globs **every** `*.jsonl` in `data/train/`, so a `sft_v2.jsonl`
added later is picked up automatically — remove or move files you don't want in
the run.

Defaults worth knowing (all overridable): `--epochs 3`, `--batch 2`,
`--grad-accum 8`, `--lr 2e-4`, `--r 16` / `--alpha 32` (matches
`bootstrap.sh --max-lora-rank 16`), `--max-seq-len 2048` (an extraction row is
~1.8–2.0k tokens — the old 1024 default silently truncated the label off every
one).

## Measuring the fine-tune

Judge the adapter on the layer it actually changes — the two NLU calls — not
only end-to-end. End-to-end scenario accuracy runs through the rest of the
pipeline (including a known distress-flag merge bug where a later calm turn
overwrites a turn-1 escalation), which masks what the adapter did. The isolated
evals below don't have that confound.

```bash
# extractor, in isolation, against the 112-row golden set
python code/eval_extraction.py --model meta-llama/Llama-3.1-8B-Instruct --out outputs/extract_base.json
python code/eval_extraction.py --model triage-lora                     --out outputs/extract_lora.json
python code/eval_extraction.py --compare outputs/extract_base.json outputs/extract_lora.json

# distress classifier, in isolation (labels derived from the 59 scenarios' facts)
python code/eval_distress.py --model meta-llama/Llama-3.1-8B-Instruct --out outputs/distress_base.json
python code/eval_distress.py --model triage-lora                     --out outputs/distress_lora.json
python code/eval_distress.py --compare outputs/distress_base.json outputs/distress_lora.json

# end-to-end stays a SECONDARY number (report with the pipeline-bug caveat)
TRIAGE_MODEL=triage-lora python code/eval_harness.py --out outputs/eval_lora.json --label triage-lora
```

`eval_extraction.py` reports row exact-match, key-level P/R/F1 (an FP key =
hallucinated slot — the failure mode behind the `pcp3` scenario), value
accuracy, and a per-slot table. `eval_distress.py` reports per-flag P/R/F1 plus
which scenarios missed a needed flag or raised a spurious one.

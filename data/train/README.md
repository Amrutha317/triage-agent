# `data/train/` — SFT data for the triage LoRA

`sft_v1.jsonl` — **290 rows**, each a single LLM call (an `extract_slots` *or* a
`classify_distress` call) paired with its ideal JSON output, in chat form:
`{messages: [system, user], response, category}`.

Built by `code/generate_sft_set.py` — deterministic, no network, no LLM:

- imports `EXTRACT_SYS` / `DISTRESS_SYS` verbatim from `code/llm_client.py`, so
  every row's system+user text is byte-identical to what the model sees at
  serving time (that's what lets a LoRA transfer);
- validates every extraction label against `code/slots.py` at build time and
  aborts on anything off-schema;
- shares no rows with the eval sets (`data/eval/`), so the isolated evals
  measure generalisation, not memorisation.

The reported **v1** adapter used an earlier **268-row** version of this set (see
the report PDF §6.1); the file here is the later 290-row version.

## What it targets

The 5 disposition misses in `outputs/eval_baseline_all.json` are all LLM-layer
errors (the rules engine is correct on all 5 in `--offline` mode). Rows are
weighted toward them (the `category` field tags each):

| baseline failure | cause | rows |
|---|---|---|
| `ems_triager_life_threatening` missed | distress under-called `life_threatening` on a plain plea for help | `distress_life_threatening`, `distress_ambiguous_flag` |
| `ed_difficulty_breathing` → 911 | distress promoted mild "short of breath" to `severe_difficulty_breathing` | `distress_mild_dyspnea_negative` |
| `educ_pleuritic_pain` → 911 | distress read "hurts to breathe in" as a breathing emergency | `distress_pleuritic_negative` |
| `educ_triager_very_sick` missed | distress under-called `very_sick_or_weak` | `distress_very_sick` |
| `pcp3_stable_angina_nitro_resolved` → ED | extractor invented `other_symptoms=[difficulty_breathing]` from "some of those" | `anti_overextract`, `underspecified`, `no_new_info` |

`distress_calm_negative` (all flags false, calm phrasing) is the counterweight
so the adapter doesn't learn "always escalate".

## Regenerate

```bash
python code/generate_sft_set.py            # -> data/train/sft_v1.jsonl
pytest -q tests/test_sft_set.py            # checks the file is well-formed
```

## Fine-tune + evaluate (GPU + a running server)

```bash
export HF_TOKEN=hf_...                                    # Llama-3.1-8B is gated
python code/finetune_lora.py                              # r16/α32, 3 epochs -> adapters/triage-lora/
LORA_DIR=adapters/triage-lora bash bootstrap.sh           # serve base + adapter

# measure on the layer the adapter changes (isolated), then end-to-end
python code/eval_extraction.py --model triage-lora --out outputs/extract_lora.json
python code/eval_distress.py   --model triage-lora --out outputs/distress_lora.json
TRIAGE_MODEL=triage-lora python code/eval_harness.py --out outputs/eval_lora.json --label triage-lora
python code/eval_ci.py outputs/extract_base.json outputs/extract_lora.json
```

`finetune_lora.py` globs every `*.jsonl` in `data/train/`. Defaults, all
overridable: `--epochs 3`, `--batch 2`, `--grad-accum 8`, `--lr 2e-4`,
`--r 16` / `--alpha 32`, `--max-seq-len 2048`.

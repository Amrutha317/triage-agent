# Fine-tuning the chest-pain triage NLU layer with QLoRA

**Status:** final. v1 is the reported adapter (v2 regressed and was rolled back;
v3 could not be measured — §9.1).
**Branch:** `sft-dataset-and-model-fix` · **PR:** #1
**Base model:** `meta-llama/Llama-3.1-8B-Instruct`, served with vLLM 0.6.3 on a single GPU.

---

## 1. Summary

The triage agent is a deterministic rules engine (`rules.yaml` + `decision_engine.py` +
`state_machine.py`) wrapped around an 8B LLM that does two narrow jobs — **slot
extraction** and **distress classification**. The LLM never picks a disposition.

Baseline end-to-end accuracy is **91.5%** (54/59 scenarios). All five failures are in
the LLM layer; the rules engine is correct on all five in offline mode.

A QLoRA adapter was trained on a 290-row set built specifically against those failures,
and **evaluated in isolation on the two NLU calls** rather than only end-to-end (an
end-to-end number runs through a separate, identified pipeline defect that masks the
adapter's effect).

**Result (paired bootstrap, B = 5000, `code/eval_ci.py`):**

| layer | metric | base | LoRA v1 | Δ | 95% CI on Δ | beyond noise? |
|---|---|---|---|---|---|---|
| extraction (112) | row exact-match | 0.562 | 0.607 | +0.045 | [−0.036, +0.125] | no |
| | key-F1 | 0.779 | 0.807 | +0.028 | [−0.030, +0.098] | no |
| | key recall | 0.790 | 0.833 | +0.043 | [−0.022, +0.115] | no |
| | value accuracy | 0.844 | 0.826 | −0.018 | [−0.080, +0.044] | no |
| distress (59) | micro recall | 0.875 | 1.000 | +0.125 | [0.000, +0.417] | no |
| | micro precision | 0.636 | 0.571 | −0.065 | [−0.208, +0.069] | no |
| | micro F1 | 0.737 | 0.727 | −0.010 | [−0.156, +0.196] | no |

**Every difference-CI includes zero.** No measurable effect on any metric, isolated or
end-to-end (v1 end-to-end triage 0.915, red-flag recall 0.889 — both inside the baseline
CI). This is the predicted outcome for a rules-engine architecture: the deterministic
engine owns the disposition, and the LLM only extracts facts and phrases questions —
tasks the base 8 B already does near-zero-shot. **The adapter is not shipped.**

The one suggestive number is distress recall (0.875 → 1.000 — the LoRA misses no needed
flag in this set), but the CI is [0, +0.417], five of six flags have support = 1, and §7
shows a correctly-raised flag still does not reach a disposition.

Two later iterations: **v2** (true-only distress output format + 5 epochs) regressed the
distress classifier (isolated micro-F1 ≈ 0.29) and was rolled back; **v3** (reverted
format, 3 epochs) could not be measured — the serving environment degraded to where the
*base* model scored 0.57 key-F1 on a previously-0.78 eval (§9.1).

Three latent bugs were fixed along the way (§8): a train/serve model mismatch, a
training-sequence truncation, and a GPU-placement failure.

---

## 2. System under test

```
patient text ──┬─► llm_client.classify_distress()  ─► observed red-flag / triager slots
               └─► llm_client.extract_slots()       ─► {slot: value}
                        │
                        ▼
               state_machine.ingest(merged slots)  ─► Turn (ask | final)
                        │
                        ▼
               decision_engine  ─► first rules.yaml rule that fires  ─► Disposition
```

- **`extract_slots`** — converts a patient utterance into a schema-constrained
  `{slot: value}` dict. System prompt `EXTRACT_SYS` (~1,640 tokens), guided-JSON decoding.
- **`classify_distress`** — reads *how* the patient presents and emits six observed-only
  red-flag / triager-judgment booleans. System prompt `DISTRESS_SYS`, deliberately
  calibrated to over-flag ("a missed emergency costs a life").
- Everything downstream is deterministic. `slots.py` is the single schema authority.

---

## 3. Baseline evaluation

### 3.1 End-to-end (`code/eval_harness.py`, `data/eval/scenarios.jsonl`, 59 scenarios)

| metric | value |
|---|---|
| triage accuracy (exact disposition) | 0.915 |
| workflow accuracy (disposition + rule_id) | 0.915 |
| under-triage rate | 0.034 |
| over-triage rate | 0.051 |
| red-flag recall (gold = 911) | 0.944 |
| red-flag recall (gold ∈ {911, ED-now}) | 0.970 |
| did not terminate | 0 |
| turns / scenario | mean 6.4, p50 7, p90 10 |
| call TTFT (s) | p50 0.32, p90 2.15 |
| call total (s) | p50 3.38, p90 6.51 |

### 3.2 The five failures — all LLM-layer

| scenario | gold → predicted | root cause | call |
|---|---|---|---|
| `ems: triager judges life-threatening` | `CALL_EMS_911` → `HOME_CARE` | distress classifier under-called `life_threatening` on the concatenated chief complaint | distress |
| `educ: triager judges very sick/weak` | `GO_TO_ED_UCC` → `HOME_CARE` | flag set on turn 1, then cleared downstream (§7) | distress |
| `ed: difficulty breathing (non-severe)` | `GO_TO_ED_NOW` → `CALL_EMS_911` | classifier promoted mild "short of breath" to `severe_difficulty_breathing` | distress |
| `educ: pleuritic pain (worse on deep breath)` | `GO_TO_ED_UCC` → `CALL_EMS_911` | classifier read "hurts to breathe in" as a breathing emergency | distress |
| `pcp3: nitro-resolved` | `SEE_PCP_3_DAYS` → `GO_TO_ED_NOW` | extractor hallucinated `other_symptoms=[difficulty_breathing]` from the vague answer "some of those" | extract |

Four distress calibration errors (two under-, two over-), one extractor over-extraction.

---

## 4. Approach

**Fine-tune the two NLU calls; measure them directly.**

End-to-end scenario accuracy only reflects extraction/distress quality *indirectly*, and
it runs through the rest of the pipeline — including the defect in §7. So the primary
metric is two isolated evals built for this work:

- **`code/eval_extraction.py`** — runs the 112-row `data/eval/extraction_golden.jsonl`
  (one hand-labeled utterance → exact slot dict per row) through `extract_slots()`.
  Reports row exact-match, key-level precision/recall/F1 (an FP key = a hallucinated
  slot — the `pcp3` failure mode), value accuracy (enum-sets as sets, floats with
  tolerance), a per-slot table, and every hallucination row.
- **`code/eval_distress.py`** — derives a labeled set from the 59 scenarios' `facts`
  (each chief-complaint utterance + its six observed-flag values), runs
  `classify_distress()` once per row, reports per-flag precision/recall/F1 and which
  scenarios miss a needed flag or raise a spurious one.

Neither touches the state machine or rules engine, so the number reflects the model.

---

## 5. Datasets

Three distinct artifacts — easy to conflate:

| file | rows | one row is | role |
|---|---|---|---|
| `data/eval/scenarios.jsonl` | 59 | a full patient case → correct disposition | end-to-end eval |
| `data/eval/extraction_golden.jsonl` | 112 | one utterance → exact slot dict | isolated NLU eval |
| `data/train/sft_v1.jsonl` | 290 | one LLM call + gold JSON, chat form | **training** |

### 5.1 How the training set is built (`code/generate_sft_set.py`)

Deterministic, no network, no LLM. Two prompt families:

- **extract** — system = `EXTRACT_SYS` imported verbatim from `llm_client.py`; user
  mirrors `LLMClient.extract_slots()` exactly (known-slots line, "asked about" line,
  the triple-quoted message). Response = the gold slot JSON.
- **distress** — system = `DISTRESS_SYS` verbatim; user mirrors
  `LLMClient.classify_distress()`. Response = the gold flag JSON.

Because the prompts are byte-identical to serving, the adapter transfers. The canonical
schema is `slots.py`; every extraction label is passed through
`llm_client.validate_slots()` at build time and the build aborts on any invalid label.
`tests/test_sft_set.py` re-checks shape, verbatim prompts, strict-JSON responses, and
schema-valid labels.

The set is **independent** of `extraction_golden.jsonl` (different wording), so the
isolated eval measures generalisation, not memorisation.

### 5.2 Composition

Weighted toward the five failures. The **v1** set (268 rows, the reported adapter):

| bucket | rows | targets |
|---|---|---|
| `distress_life_threatening` / `distress_very_sick` | 14 / 12 | failures 1, 2 (under-call) |
| `distress_mild_dyspnea_negative` / `distress_pleuritic_negative` | 6 / 5 | failures 3, 4 (over-call) |
| `distress_calm_negative` | 14 | anti-escalation counterweight |
| `anti_overextract` / `underspecified` / `no_new_info` / `correction` | 10 / 8 / 3 / 2 | failure 5 + general precision |
| full slot / enum coverage (duration, pattern, radiation, nitro, PE, GERD, …) | rest | |

The current file (v3) adds 25 rows for the value-accuracy misses v1 showed
(`bystander_not_patient`, `stent_not_angina`, `onset_precision`,
`central_location_not_extracted`) — but the adapter trained on it was never cleanly
measured (§9.1).

### 5.3 The two iterations after v1 (both unsuccessful)

Measured v1 with `eval_extraction.py` / `eval_distress.py`, then tried:

**v2** — two changes at once (this is the experimental-hygiene mistake called out in
§9.2):
1. distress rows emit the four optional flags only when `true` (v1 emitted all six
   booleans every row);
2. 5 epochs instead of 3, plus +25 extraction rows for v1's value-accuracy misses.

Result: the distress classifier collapsed (isolated micro-F1 ≈ 0.29;
`severe_difficulty_breathing` / `confused` recall → 0 — the "true-only" format taught it
the optional flags don't appear) and the extractor over-fit (guard-slot key-F1 that was
1.0 at baseline dropped to 0.67; FP keys 33 → 48). Because two variables moved together
the cause is not cleanly attributable, but both point to over-correction. Rolled back in
the generator.

**v3** — reverted the distress format to all-six, kept the +25 extraction rows, went
back to 3 epochs. Training loss still fell to ~0.04 by the final step (the templated data
memorises fast), and the run could not be measured — the serving environment had
degraded past the point a base-model canary passed (§9.1).

---

## 6. Training setup (`code/finetune_lora.py`)

| | value |
|---|---|
| method | QLoRA (4-bit nf4, double quant, bf16 compute) |
| LoRA | r=16, α=32, dropout 0.05, target `q_proj,k_proj,v_proj,o_proj` |
| trainable params | 13.6 M / 8.04 B (0.17%) |
| optimizer / schedule | paged_adamw_8bit, cosine, warmup 0.03, lr 2e-4 |
| batch | 2 × grad-accum 8 (effective 16) |
| max-seq-len | 2048 (extraction rows peak at ~1,800 tokens; 0/290 truncated) |
| gradient checkpointing | on |
| **v1** (reported) | SFT set @ commit `d2aa28d`, **268 rows**, 3 epochs, ~50 steps, final train loss 0.662 |
| v2 (rolled back) | SFT set @ `af8eac2`, 290 rows, true-only distress format, 5 epochs, mean loss 0.38 |
| v3 (not measured) | SFT set @ `3a3644c`, 290 rows, format reverted, 3 epochs, mean loss 0.59 (final-step loss ~0.04 — fit hard despite the epoch cut) |

The `data/train/sft_v1.jsonl` currently in the repo is the **v3** content (§5). To
reproduce the reported v1 adapter exactly, `git checkout d2aa28d -- code/generate_sft_set.py`
and regenerate before training.

Serving: `bootstrap.sh` starts vLLM with `--enable-lora --lora-modules
triage-lora=adapters/triage-lora --max-lora-rank 16`; the adapter is addressable as
`model="triage-lora"` alongside the base model.

---

## 7. Results

All differences are a **paired percentile bootstrap** (B = 5000) over the shared rows
(`code/eval_ci.py`). "beyond noise?" = the 95% CI on the difference excludes zero. Base
and v1 were both collected on a verified-healthy server (33 / 32 hallucinated keys in
the saved rows — the healthy signature; a degraded run shows ~175, §9.1).

### 7.1 Isolated extraction — `data/eval/extraction_golden.jsonl`, 112 rows

| metric | base | LoRA v1 | Δ | 95% CI on Δ | beyond noise? |
|---|---|---|---|---|---|
| row exact-match | 0.562 | 0.607 | +0.045 | [−0.036, +0.125] | no |
| key precision | 0.768 | 0.782 | +0.015 | [−0.056, +0.090] | no |
| key recall | 0.790 | 0.833 | +0.043 | [−0.022, +0.115] | no |
| key F1 | 0.779 | 0.807 | +0.028 | [−0.030, +0.098] | no |
| value accuracy | 0.844 | 0.826 | −0.018 | [−0.080, +0.044] | no |
| hallucination row-rate | 0.241 | 0.214 | −0.027 | [−0.098, +0.045] | no |

Point estimates lean toward better *recall* (a few more slots recovered) and slightly
worse *precision*, but nothing survives the CI at n = 112.

### 7.2 Isolated distress — derived from the 59 scenarios, 6 flags

| metric | base | LoRA v1 | Δ | 95% CI on Δ | beyond noise? |
|---|---|---|---|---|---|
| micro precision | 0.636 | 0.571 | −0.065 | [−0.208, +0.069] | no |
| micro recall | 0.875 | 1.000 | +0.125 | [0.000, +0.417] | no |
| micro F1 | 0.737 | 0.727 | −0.010 | [−0.156, +0.196] | no |
| flag exact-match | 0.915 | 0.915 | 0.000 | [−0.051, +0.051] | no |

The recall point estimate (v1 misses no needed flag in this set) is the most suggestive
number, but the CI touches zero and five of the six flags have support = 1 — per-flag
P/R is not a stable estimate here. Of the "spurious" `life_threatening` firings, the
ones on `ems: severe dyspnea` / `ems: shock signs` / `ems: syncope` are on presentations
that route to `CALL_EMS_911` regardless, so the precision point estimate overstates real
over-triage risk.

### 7.3 End-to-end (secondary) — `data/eval/scenarios.jsonl`, 59 scenarios

| metric | baseline | LoRA v1 | note |
|---|---|---|---|
| triage accuracy | 0.915 | 0.915 | inside baseline CI [0.83, 0.98] |
| workflow accuracy | 0.915 | 0.915 | " |
| red-flag recall (911) | 0.944 | 0.889 | inside baseline CI [0.81, 1.00] |
| under-triage rate | 0.034 | 0.051 | inside baseline CI [0.00, 0.09] |
| over-triage rate | 0.051 | 0.034 | inside baseline CI [0.00, 0.12] |
| call total (s), p50 | 3.38 | 2.31 | — |

The composition shifts slightly (over-triage down, red-flag recall down) but every
value is inside the baseline confidence interval — **no measurable end-to-end change**.
§7.4 explains why any real distress-recall gain would not reach a disposition anyway.

### 7.4 The pipeline defect that gates the end-to-end result

**`classify_distress` runs on every turn**, and its schema requires `life_threatening` /
`very_sick_or_weak` as booleans — so every turn it emits one. In `agent.py`:

```python
merged.update(distress.data or {})   # {"triager_assessment_life_threatening": False} on a calm turn
turn = self.fsm.ingest(merged)       # ingest keeps any non-None value  →  slot = False
```

A patient who leads with panic ("this is an emergency") and then answers follow-ups
calmly produces `life_threatening=True` on turn 1 and `False` on turn 3, and the later
value **overwrites** the earlier one. By the time the FSM finalizes, the flag is gone →
the scenario falls through to `HOME_CARE`.

**Evidence** (`code/probe_distress.py --scenarios`): on the real turn-1 utterance for
`ems: triager judges life-threatening`, base returns *all-false* but LoRA v1 correctly
returns `life_threatening` — yet the scenario still ends `HOME_CARE`. The flag is raised
and then cleared.

**Attempted fix, reverted.** Making the six observed flags monotonic across turns
(never `True → False`) sent baseline **over-triage from 0.051 to 0.339** — revealing
that the overwrite was silently walking back ~15 classifier false positives per run. The
distress classifier's *precision* is not high enough for stickiness to be safe; the
overwrite is load-bearing. Reverted in `00ff639`; the correct fix (call the classifier
on the chief-complaint turn only, or gate stickiness on classifier confidence) is left
as future work.

This is why two of the five failures (`ems/educ: triager judges …`) cannot be fixed by
any adapter: the schema forces a boolean every turn, and the pipeline treats the last
one as authoritative.

---

## 8. Incidental bugs fixed

| bug | effect | fix |
|---|---|---|
| `finetune_lora.py` / `bootstrap.sh` defaulted `--model` to `Qwen/Qwen2.5-7B-Instruct` while serving `Llama-3.1-8B-Instruct` | a run without `--model` trains an adapter vLLM can't load against the served base | default → `meta-llama/Llama-3.1-8B-Instruct` in all three places (`d2aa28d`) |
| `--max-seq-len` default 1024; `EXTRACT_SYS` alone is ~1,640 tokens | every extraction training row's gold label was truncated off the end of the sequence | default → 2048; added a token-length histogram + truncation warning after dataset build (`d2aa28d`) |
| `device_map="auto"` in `from_pretrained` | when a vLLM server still held the GPU, accelerate spilled layers to CPU and bitsandbytes 4-bit aborted with an opaque error | pin to `{"": 0}` — fits or raises a clean CUDA OOM (`b6f83fc`) |

---

## 9. Limitations & next steps

### 9.1 Evaluation conditions and an unresolved instability

**Decoding was unconstrained for every result in this report.** Guided
(schema-enforced) decoding was removed from `llm_client._chat` in commit `b4916df`,
ahead of this work, to resolve request timeouts and mid-stream connection drops on the
distress-classification schema. The `guided_json` argument is still threaded from the
call sites but is not applied, so vLLM's `--guided-decoding-backend` flag is inert, and
baseline, v1, v2, and the isolated evals all ran with free-form generation plus regex
JSON extraction (`_parse_json`). This is a sound basis for the base-vs-adapter
comparison — both sides run under identical conditions — but schema conformance is not
enforced at serving time. The `_chat` docstring, which still described schema-on-every-
retry from a since-reverted change (`41a0885`), has been corrected to match the code.

**A recurring serving instability, observed three times.**

1. During baseline collection, individual model calls on an otherwise idle,
   correctly-configured server ranged from ~3 s to over 300 s, with no error and no
   config change. A clean vLLM restart cleared it; the final baseline was collected
   immediately after.
2. Similar spikes appeared during the v2 eval runs (`call_total_seconds` p90 elevated).
3. After the v2/v3 training cycles, the *base* model's isolated extraction key-F1 fell
   from **0.78 to 0.57** with a ~5× jump in hallucinated keys, on a byte-identical eval,
   and did **not** recover on a soft restart (`pkill` + re-serve). The pod could not be
   hard-restarted, so **v3 was never measured** and its numbers are not reported.

The cause was not identified. Earlier timeout work (`b4916df`, and an
`lm-format-enforcer` → `outlines` backend switch) does not explain a degradation that
appears on a healthy server, worsens across train/serve cycles, and survives `pkill` —
plausibly GPU-memory fragmentation or KV/prefix-cache state. Recorded as a
reproducibility risk, not a fixed defect. Practice: hard-restart the instance between
training and serving; before trusting a run, check a *base-model canary* (isolated
extraction key-F1 should be ≈ 0.78) and discard/redo anything that has drifted. The
reported baseline and v1 numbers were collected before (3) and verified against the
healthy-server hallucinated-key signature (33/32, not ~175).

### 9.2 Next steps

1. **Fix the distress-merge defect** (§7.4) — the single highest-value change; expected
   to recover 2–3 failures with no fine-tuning and lift red-flag recall toward 1.0.
2. **The eval is thin** — 59 scenarios, 112 golden rows, a *deterministic templated*
   patient (`patient_sim.py`), and several distress flags at support = 1. CIs are wide.
   `code/split_val.py` now carves a template-level `extraction_val.jsonl` for future
   selection, but the real need is a larger, non-templated eval — de-identified real
   transcripts with clinician-adjudicated labels — before any further fine-tuning. That
   is also the setting where the extraction-recall lean might reach significance.
3. **Distress-classifier precision** — the real weakness. A dedicated precision pass
   (more hard negatives) once (1) is done and stickiness is safe.
4. **Re-enable schema-enforced decoding** — once a serving backend handles the distress
   schema without stalling (§9.1), so serving matches the schema the training data was
   written against.
5. **Experimental hygiene for the next fine-tune** — one variable per iteration (v2
   moved format and epochs together); a held-out capability probe for catastrophic
   forgetting; an LR/rank sweep on the val split.
6. **Consider a stronger base model** — with extraction schema-shaped and NLG templated,
   Phi-3.5-mini / Qwen2.5-3B may match Llama-8B at lower latency; a larger or hosted
   model may beat fine-tuning an 8B outright.

---

## 10. Reproduction

```bash
# --- pod setup ---
export HF_TOKEN=hf_...
bash bootstrap.sh                                  # venv + pinned deps + serve base model

# --- baseline (isolated + end-to-end) ---
python code/eval_extraction.py --model meta-llama/Llama-3.1-8B-Instruct --out outputs/extract_base.json
python code/eval_distress.py   --model meta-llama/Llama-3.1-8B-Instruct --out outputs/distress_base.json
python code/eval_harness.py --out outputs/eval_baseline_all.json --label baseline

# --- train ---
pkill -9 -f "vllm serve"; sleep 3                  # training needs the GPU to itself
python code/generate_sft_set.py                    # -> data/train/sft_v1.jsonl (290 rows)
python code/finetune_lora.py --epochs 5            # -> adapters/triage-lora/

# --- serve adapter + re-eval ---
SKIP_INSTALL=1 LORA_DIR=adapters/triage-lora bash bootstrap.sh
python code/eval_extraction.py --model triage-lora --out outputs/extract_lora.json
python code/eval_extraction.py --compare outputs/extract_base.json outputs/extract_lora.json
python code/eval_distress.py   --model triage-lora --out outputs/distress_lora.json
python code/eval_distress.py   --compare outputs/distress_base.json outputs/distress_lora.json
TRIAGE_MODEL=triage-lora python code/eval_harness.py --out outputs/eval_lora.json --label triage-lora
```

---

## 11. File inventory (this branch)

| file | purpose |
|---|---|
| `code/generate_sft_set.py` | deterministic SFT builder |
| `data/train/sft_v1.jsonl` | 290-row training set |
| `data/train/README.md` | the three datasets + run steps |
| `code/eval_extraction.py` | isolated extractor eval (per-slot P/R/F1, hallucinations) |
| `code/eval_distress.py` | isolated distress-classifier eval (per-flag P/R/F1) |
| `code/probe_distress.py` | base-vs-adapter distress diff on hard phrases / real scenarios |
| `tests/test_sft_set.py` | training-file guard tests |
| `code/finetune_lora.py` | QLoRA trainer (model default, max-seq-len, device_map fixes) |
| `bootstrap.sh` | pod setup + vLLM serve (model default fix) |
| `outputs/eval_baseline_all.json` | baseline end-to-end |
| `outputs/extract_*.json`, `outputs/distress_*.json` | isolated eval results |
